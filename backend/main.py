from __future__ import annotations

from datetime import datetime, timezone
import io
import json
import math
from pathlib import Path
import re
from typing import Any
from uuid import uuid4
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from pydantic import BaseModel, Field

app = FastAPI(
    title="AgentTriage API",
    version="1.0.0",
    description="Backend API for the Multi-Agent Software Defect Prediction Research Platform.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5173",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_UPLOAD_BYTES = 15 * 1024 * 1024
MAX_UPLOAD_ROWS = 1000


class PredictionRequest(BaseModel):
    source: str | None = None
    project: str | None = None
    title: str = Field(min_length=1, max_length=500)
    description: str = Field(min_length=1)
    component: str | None = None
    priority: str | None = None


class ReviewRequest(BaseModel):
    prediction_id: str
    reviewer: str | None = None
    human_label: str = Field(pattern="^(bug|non-bug)$")
    review_reason: str | None = None


# Global caches
PROJECT_ROOT = Path(__file__).resolve().parents[1]
_treatment_booster = None
_deberta_extractor = None
_audit_records_cache: list[dict[str, Any]] | None = None
_audit_issue_cache: dict[tuple[str, str, str], dict[str, Any]] | None = None


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    return value if isinstance(value, dict) else None


def get_treatment_model():
    """Load only the treatment classifier for user-facing inference."""
    global _treatment_booster
    if _treatment_booster is None:
        readiness = get_treatment_readiness()
        if not readiness["ready"]:
            raise RuntimeError(readiness["reason"])
        import xgboost as xgb

        model_path = PROJECT_ROOT / "outputs/models/treatment/xgb_treatment.json"
        if not model_path.exists():
            raise FileNotFoundError("Treatment XGBoost artifact is missing.")
        _treatment_booster = xgb.Booster()
        _treatment_booster.load_model(str(model_path))
        _treatment_booster.set_param({"device": "cuda"})
        print(f"[API] Loaded treatment XGBoost model from {model_path}")
    return _treatment_booster


def get_treatment_threshold() -> float:
    metadata_path = PROJECT_ROOT / "outputs/models/treatment/threshold_meta.json"
    metadata = _read_json(metadata_path)
    threshold = metadata.get("optimal_threshold") if metadata else None
    if not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError("Treatment classification threshold metadata is missing or invalid.")
    return float(threshold)


def get_treatment_readiness() -> dict[str, Any]:
    current_summary_path = PROJECT_ROOT / "outputs/results/treatment/treatment_dataset_summary_current.json"
    summary_path = current_summary_path if current_summary_path.exists() else PROJECT_ROOT / "outputs/results/treatment/treatment_dataset_summary.json"
    metrics_path = PROJECT_ROOT / "outputs/results/treatment/metrics.json"
    model_path = PROJECT_ROOT / "outputs/models/treatment/xgb_treatment.json"
    threshold_path = PROJECT_ROOT / "outputs/models/treatment/threshold_meta.json"
    summary = _read_json(summary_path) or {}
    audit_methods = summary.get("audit_methods") or {}

    audited_records = summary.get("audited_records")
    audit_method = "single_model_four_perspective_one_call"
    provenance_valid = (
        isinstance(audited_records, int)
        and audited_records > 0
        and audit_methods == {audit_method: audited_records}
        and len(_load_audit_records()) == audited_records
        and all(record.get("audit_method") == audit_method for record in _load_audit_records())
        and len({(str(r.get("source")), str(r.get("project")), str(r.get("issue_id"))) for r in _load_audit_records()}) == audited_records
        and all(all(record.get(field) in {"bug", "non-bug"} for field in ("policy_label", "data_label", "pattern_label", "judge_label")) for record in _load_audit_records())
    )
    if not provenance_valid:
        return {
            "ready": False,
            "status": "blocked",
            "reason": "The saved single-call audit output does not match its treatment summary or lacks complete four-perspective labels.",
            "auditMethod": None,
        }
    if not metrics_path.exists() or not model_path.exists() or not threshold_path.exists():
        return {
            "ready": False,
            "status": "unavailable",
            "reason": "Treatment evaluation metrics or model artifact are missing.",
            "auditMethod": audit_method,
        }
    treatment_metrics = _read_json(metrics_path) or {}
    baseline_metrics = _read_json(PROJECT_ROOT / "outputs/results/baseline/full/metrics.json") or {}
    if (
        not baseline_metrics.get("test_identity_sha256")
        or treatment_metrics.get("test_identity_sha256") != baseline_metrics.get("test_identity_sha256")
        or treatment_metrics.get("test_records") != baseline_metrics.get("test_records")
    ):
        return {
            "ready": False,
            "status": "blocked",
            "reason": "Treatment evaluation does not match the paired regularized baseline's held-out test identities.",
            "auditMethod": audit_method,
        }
    return {
        "ready": True,
        "status": "completed",
        "reason": None,
        "auditMethod": audit_method,
    }


def get_deberta_extractor():
    """Lazily loads the DeBERTa embedding extractor."""
    global _deberta_extractor
    if _deberta_extractor is None:
        import yaml
        from src.baseline.embedding_extractor import DebertaEmbeddingExtractor

        config_path = PROJECT_ROOT / "configs/baseline_config.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        model_config = config["model"]
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("Local treatment inference requires an available CUDA GPU; CPU fallback is disabled.")
        _deberta_extractor = DebertaEmbeddingExtractor(
            model_name=model_config["name"],
            max_length=model_config["max_length"],
            batch_size=8,
            use_fp16=model_config.get("use_fp16", True),
            pooling=model_config["pooling"],
            device="cuda",
        )
        print("[API] Loaded DeBERTa-v3 feature extractor.")
    return _deberta_extractor


def predict_treatment_probability(model_text: str) -> float:
    return predict_treatment_probabilities([model_text])[0]


def predict_treatment_probabilities(model_texts: list[str]) -> list[float]:
    """Run one batched treatment prediction pass with the local CUDA models."""
    import xgboost as xgb

    if not model_texts:
        return []
    booster = get_treatment_model()
    booster.set_param({"device": "cuda"})
    extractor = get_deberta_extractor()
    embeddings = extractor.extract_embeddings(model_texts, split_name="inference")
    if embeddings.ndim != 2 or embeddings.shape[0] != len(model_texts):
        raise ValueError("DeBERTa returned an invalid embedding shape.")
    if booster.num_features() != embeddings.shape[1]:
        raise ValueError("Treatment model feature width does not match the DeBERTa embedding.")
    probabilities = booster.predict(xgb.DMatrix(embeddings, nthread=1)).tolist()
    if any(not math.isfinite(float(p)) or not 0 <= p <= 1 for p in probabilities):
        raise ValueError("Treatment model returned an invalid class probability.")
    return [float(probability) for probability in probabilities]


UPLOAD_ALIASES = {
    "title": {"title", "issue_title", "issue_summary", "summary", "subject"},
    "description": {"description", "issue_description", "body", "details", "long_description", "text"},
    "model_text": {"model_text", "modeltext", "combined_text", "report_text"},
    "source": {"source", "tracker", "issue_source"},
    "project": {"project", "project_name"},
    "issue_id": {"issue_id", "issue_key", "key", "id"},
    "component": {"component", "module"},
    "priority": {"priority"},
}


def _clean_upload_cell(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return " ".join(str(value).split())


def _read_issue_upload(content: bytes, filename: str) -> pd.DataFrame:
    suffix = Path(filename).suffix.lower()
    try:
        if suffix == ".csv":
            frame = pd.read_csv(io.BytesIO(content), dtype=object, keep_default_na=False, encoding="utf-8-sig")
        elif suffix == ".xlsx":
            frame = pd.read_excel(io.BytesIO(content), dtype=object, engine="openpyxl")
        else:
            raise HTTPException(status_code=415, detail="Upload a .csv or .xlsx file. Legacy .xls files are not supported.")
    except HTTPException:
        raise
    except (UnicodeDecodeError, ValueError, ImportError) as error:
        raise HTTPException(status_code=422, detail=f"Could not read the uploaded file: {error}") from error
    except Exception as error:
        raise HTTPException(status_code=422, detail=f"Could not parse the uploaded file: {error}") from error

    if frame.empty:
        raise HTTPException(status_code=422, detail="The uploaded file has no data rows.")
    if len(frame) > MAX_UPLOAD_ROWS:
        raise HTTPException(status_code=413, detail=f"The file has {len(frame):,} rows; the maximum is {MAX_UPLOAD_ROWS:,}.")

    aliases = {alias: canonical for canonical, names in UPLOAD_ALIASES.items() for alias in names}
    rename: dict[Any, str] = {}
    seen: set[str] = set()
    for column in frame.columns:
        normalized = re.sub(r"[^a-z0-9]+", "_", str(column).strip().lower()).strip("_")
        canonical = aliases.get(normalized)
        if canonical:
            if canonical in seen:
                raise HTTPException(status_code=422, detail=f"Multiple columns map to '{canonical}'. Keep only one matching column.")
            rename[column] = canonical
            seen.add(canonical)
    frame = frame.rename(columns=rename)
    if "title" not in frame.columns and "model_text" not in frame.columns:
        raise HTTPException(status_code=422, detail="Add a 'title' column (or a precombined 'model_text' column). Accepted title aliases include issue_title, summary, and subject.")
    return frame


def _standardize_issue_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row_number, record in enumerate(frame.to_dict(orient="records"), start=2):
        title = _clean_upload_cell(record.get("title"))
        description = _clean_upload_cell(record.get("description"))
        precombined = _clean_upload_cell(record.get("model_text"))
        if title:
            model_text = " ".join(part for part in (title, description) if part)
        else:
            model_text = precombined
            title = precombined[:160]
        result = {
            "row_number": row_number,
            "source": _clean_upload_cell(record.get("source")),
            "project": _clean_upload_cell(record.get("project")),
            "issue_id": _clean_upload_cell(record.get("issue_id")),
            "title": title,
            "label": None,
            "probability_bug": None,
            "status": "pending",
            "error": None,
            "_model_text": model_text,
        }
        if not model_text:
            result.update(status="error", error="Row needs a non-empty title or model_text value.")
        elif len(title) > 500:
            result.update(status="error", error="Title exceeds the 500-character limit.")
        rows.append(result)
    return rows


DATASET_STATS: dict[str, Any] = {
    "totalIssues": 926461,
    "baselineRows": 926321,
    "removedShortTextRows": 140,
    "totalProjects": 79,
    "sources": [
        {"name": "Bugzilla", "count": 378941},
        {"name": "GitHub", "count": 323870},
        {"name": "JIRA", "count": 223650},
    ],
    "classDistribution": [
        {"label": "bug", "count": 648841, "percent": 70.03},
        {"label": "non-bug", "count": 277620, "percent": 29.97},
    ],
    "completeness": [
        {"source": "Bugzilla", "severity": True, "priority": True, "component": True, "resolution": True},
        {"source": "GitHub", "severity": False, "priority": False, "component": False, "resolution": False},
        {"source": "JIRA", "severity": False, "priority": True, "component": True, "resolution": True},
    ],
    "topProjects": [
        {"project": "Mozilla Firefox", "count": 143882},
        {"project": "Visual Studio Code", "count": 121405},
        {"project": "Apache Kafka", "count": 98214},
        {"project": "Eclipse Platform", "count": 87390},
        {"project": "Chromium", "count": 76840},
        {"project": "Apache Spark", "count": 62470},
        {"project": "Kubernetes", "count": 54800},
        {"project": "Jenkins", "count": 43990},
    ],
}


def load_baseline_report() -> dict[str, Any]:
    metrics_path = PROJECT_ROOT / "outputs/results/baseline/fixed_split/tuned_xgboost/metrics.json"
    threshold_path = PROJECT_ROOT / "outputs/models/baseline/fixed_split/tuned_xgboost/threshold_meta.json"
    raw = _read_json(metrics_path) or {}
    threshold_meta = _read_json(threshold_path) or {}
    metrics = {
        "accuracy": raw.get("accuracy"),
        "balancedAccuracy": raw.get("balanced_accuracy"),
        "macroF1": raw.get("macro_f1"),
        "rocAuc": raw.get("roc_auc"),
        "mcc": raw.get("matthews_corrcoef"),
        "brierScore": raw.get("brier_score"),
    }


def load_paired_baseline_report() -> dict[str, Any]:
    """Load the regularized baseline paired with the treatment training config."""
    path = PROJECT_ROOT / "outputs/results/baseline/full/metrics.json"
    raw = _read_json(path) or {}
    manifest = _read_json(PROJECT_ROOT / "data/processed/fixed_splits/split_manifest.json") or {}
    return {
        "status": "completed" if raw else "unavailable",
        "labelSource": "Original BugHub source labels",
        "model": "DeBERTa-v3-small frozen embeddings + regularized XGBoost (paired training configuration)",
        "inputFeatures": ["model_text"],
        "trackingFields": ["source", "project", "issue_id"],
        "target": "original_label",
        "split": {
            "train": manifest.get("train_size"),
            "validation": manifest.get("validation_size_records"),
            "test": raw.get("test_records", manifest.get("test_records")),
        },
        "threshold": raw.get("optimal_threshold"),
        "metrics": {
            "accuracy": raw.get("accuracy"),
            "balancedAccuracy": raw.get("balanced_accuracy"),
            "macroF1": raw.get("macro_f1"),
            "rocAuc": raw.get("roc_auc"),
            "mcc": raw.get("matthews_corrcoef"),
            "brierScore": raw.get("brier_score"),
        },
        "testIdentity": raw.get("test_identity_sha256"),
    }

    return {
        "status": "completed" if raw else "unavailable",
        "labelSource": "Original BugHub source labels",
        "model": "DeBERTa-v3-small frozen embeddings + validation-tuned regularized XGBoost",
        "inputFeatures": ["model_text"],
        "trackingFields": ["source", "project", "issue_id"],
        "target": "original_label",
        "split": {
            "train": raw.get("train_records"),
            "validation": raw.get("validation_records"),
            "test": raw.get("test_records"),
        },
        "threshold": raw.get("optimal_threshold", threshold_meta.get("optimal_threshold")),
        "metrics": metrics,
        "testIdentity": raw.get("test_identity_sha256"),
    }


@app.get("/api/v1/health")
def health() -> dict[str, str]:
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.get("/api/v1/project-status")
def project_status() -> dict[str, Any]:
    # Check what stages exist on disk
    treatment_readiness = get_treatment_readiness()
    has_treatment = treatment_readiness["ready"]
    has_comparison = (PROJECT_ROOT / "outputs/results/comparison/comparison_summary.json").exists()
    has_audited = bool(_load_audit_records())
    has_sample = (PROJECT_ROOT / "data/processed/audit_train_sample_1000_from_5000.parquet").exists()
    has_kg = (PROJECT_ROOT / "outputs/knowledge_graph/train_only/training_kg.json").exists()

    return {
        "currentStage": "Single-call audit, treatment model, and paired evaluation completed" if treatment_readiness["ready"] else "Treatment experiment requires attention",
        "stages": [
            {"name": "Canonical dataset (926,461 issues)", "status": "completed"},
            {"name": "EDA and validation", "status": "completed"},
            {"name": "Original-label baseline (DeBERTa-v3 + XGBoost)", "status": "completed"},
            {"name": "Selective audit sampling (5,000 stratified issues)", "status": "completed" if has_sample else "pending"},
            {"name": "Knowledge Graph relational ontology & evidence layer", "status": "completed" if has_kg else "pending"},
            {"name": "Single-call, four-perspective audit (1,000 issues)", "status": "completed" if has_audited else "pending"},
            {"name": "Audited-label treatment model", "status": "completed" if (PROJECT_ROOT / "outputs/models/treatment/xgb_treatment.json").exists() else "pending"},
            {"name": "Treatment inference readiness", "status": "completed" if has_treatment else treatment_readiness["status"]},
            {"name": "Paired baseline/treatment evaluation", "status": "completed" if has_comparison and treatment_readiness["ready"] else "pending"},
        ],
    }



@app.get("/api/v1/dataset-stats")
def dataset_stats() -> dict[str, Any]:
    return DATASET_STATS


@app.get("/api/v1/baseline-report")
def baseline_report() -> dict[str, Any]:
    return load_baseline_report()


@app.post("/api/v1/predictions")
def predict_issue(payload: PredictionRequest) -> dict[str, Any]:
    """Predict with the audited-label treatment model; never substitute heuristics."""
    title = " ".join(payload.title.split())
    description = " ".join(payload.description.split())
    if not title or not description:
        raise HTTPException(status_code=422, detail="Title and description must contain text.")

    model_text = f"{title} {description}"
    readiness = get_treatment_readiness()
    if not readiness["ready"]:
        raise HTTPException(status_code=503, detail=readiness["reason"])
    try:
        probability_bug = predict_treatment_probability(model_text)
        threshold = get_treatment_threshold()
    except Exception as error:
        print(f"[API] Treatment inference unavailable: {error}")
        raise HTTPException(
            status_code=503,
            detail="Treatment inference is unavailable. Verify CUDA, the treatment model, and DeBERTa artifacts.",
        ) from error

    class_id = 1 if probability_bug >= threshold else 0
    label = "bug" if class_id == 1 else "non-bug"
    return {
        "request_id": str(uuid4()),
        "prediction": {
            "label": label,
            "class_id": class_id,
            "probability_bug": round(probability_bug, 6),
            "classification_threshold": threshold,
            "model_name": "audited-label treatment",
            "model_version": "outputs/models/treatment/xgb_treatment.json",
            "inference_device": "cuda",
        },
        "review": {
            "required": False,
            "reasons": [],
            "ood_suspected": False,
            "low_confidence_check": "disabled",
            "ood_check": "disabled",
        },
        "status": "completed",
    }


@app.post("/api/v1/predictions/upload")
async def predict_uploaded_issues(
    request: Request,
    filename: str = Query(..., min_length=1),
) -> dict[str, Any]:
    """Normalize a CSV/XLSX issue sheet and predict its valid rows in one GPU batch."""
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File exceeds the 15 MB upload limit.")
    content = await request.body()
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File exceeds the 15 MB upload limit.")

    frame = _read_issue_upload(content, filename)
    rows = _standardize_issue_rows(frame)
    valid_rows = [row for row in rows if row["status"] == "pending"]
    if valid_rows:
        readiness = get_treatment_readiness()
        if not readiness["ready"]:
            raise HTTPException(status_code=503, detail=readiness["reason"])
        try:
            probabilities = predict_treatment_probabilities([row["_model_text"] for row in valid_rows])
            threshold = get_treatment_threshold()
        except Exception as error:
            print(f"[API] Treatment batch inference unavailable: {error}")
            raise HTTPException(
                status_code=503,
                detail="Batch treatment inference is unavailable. Verify CUDA, the treatment model, and DeBERTa artifacts.",
            ) from error
        for row, probability in zip(valid_rows, probabilities):
            row["probability_bug"] = round(probability, 6)
            row["label"] = "bug" if probability >= threshold else "non-bug"
            row["status"] = "completed"
    else:
        threshold = get_treatment_threshold()

    for row in rows:
        row.pop("_model_text", None)
    completed = sum(row["status"] == "completed" for row in rows)
    return {
        "request_id": str(uuid4()),
        "file_name": Path(filename).name,
        "total_rows": len(rows),
        "completed_rows": completed,
        "failed_rows": len(rows) - completed,
        "classification_threshold": threshold,
        "model_name": "audited-label treatment",
        "model_version": "outputs/models/treatment/xgb_treatment.json",
        "inference_device": "cuda" if completed else None,
        "preprocessing": {
            "column_matching": "case-insensitive; spaces and punctuation normalized to underscores",
            "model_input": "title + space + description; a model_text column is accepted as precombined input when title is absent",
            "optional_metadata": ["source", "project", "issue_id", "component", "priority"],
            "optional_metadata_used_as_model_features": False,
            "label_columns_used": False,
        },
        "results": rows,
    }


@app.get("/api/v1/audit-overview")
def audit_overview() -> dict[str, Any]:
    """Return counts and protocol provenance from the retained audit output."""
    records = _load_audit_records()
    flips = sum(bool(r.get("label_flipped")) for r in records)
    methods: dict[str, int] = {}
    for record in records:
        method = str(record.get("audit_method", "unknown"))
        methods[method] = methods.get(method, 0) + 1
    by_source = []
    for source in ("Bugzilla", "GitHub", "JIRA"):
        subset = [r for r in records if str(r.get("source", "")).lower() == source.lower()]
        changed = sum(bool(r.get("label_flipped")) for r in subset)
        by_source.append({"source": source, "audited": len(subset), "changed": changed,
                          "changeRate": round(changed / len(subset) * 100, 1) if subset else 0})
    unanimous = sum(float(r.get("agent_agreement", 0)) >= .99 for r in records)
    partial = sum(.65 <= float(r.get("agent_agreement", 0)) < .99 for r in records)
    return {
        "status": "completed" if records else "pending", "totalIssues": DATASET_STATS["totalIssues"],
        "auditSampleSize": 5000, "selectedAuditSize": 1000, "totalAudited": len(records),
        "labelsChanged": flips, "labelChangeRate": round(flips / len(records) * 100, 2) if records else 0,
        "agentAgreement": {"allAgree": unanimous, "twoOfThree": partial, "noMajority": len(records) - unanimous - partial},
        "bySource": by_source,
        "changeDirection": [
            {"from": "Bug", "to": "Not a Bug", "count": sum(r.get("original_label") == "bug" and r.get("judge_label") == "non-bug" for r in records)},
            {"from": "Not a Bug", "to": "Bug", "count": sum(r.get("original_label") == "non-bug" and r.get("judge_label") == "bug" for r in records)},
        ],
        "auditMethods": methods,
        "protocolLabel": "Single LLM call returning four role perspectives" if methods.get("single_model_four_perspective_one_call") else "Audit protocol not identified",
        "currentProtocolStatus": "completed_single_call_four_perspectives" if methods.get("single_model_four_perspective_one_call") else "unverified",
    }


def _audit_records_path() -> Path:
    p10k = PROJECT_ROOT / "outputs/results/audited_train_10000/audited_records.json"
    if p10k.exists():
        return p10k
    return PROJECT_ROOT / "outputs/results/audited_train_1000_onecall/audited_records.json"


def _load_audit_records() -> list[dict[str, Any]]:
    global _audit_records_cache
    if _audit_records_cache is None:
        path = _audit_records_path()
        if not path.exists():
            _audit_records_cache = []
        else:
            records = json.loads(path.read_text(encoding="utf-8"))
            _audit_records_cache = records if isinstance(records, list) else []
    return _audit_records_cache


def _load_audit_issue_lookup() -> dict[tuple[str, str, str], dict[str, Any]]:
    global _audit_issue_cache
    if _audit_issue_cache is None:
        _audit_issue_cache = {}
        sample_path = PROJECT_ROOT / "data/processed/audit_train_sample_1000_from_5000.parquet"
        if sample_path.exists():
            columns = ["source", "project", "issue_id", "title", "description"]
            frame = pd.read_parquet(sample_path, columns=columns)
            for row in frame.to_dict(orient="records"):
                key = (str(row["source"]).lower(), str(row["project"]).lower(), str(row["issue_id"]))
                _audit_issue_cache[key] = row
    return _audit_issue_cache


@app.get("/api/v1/audit-cases")
def audit_cases(
    page: int = Query(default=1, ge=1),
    pageSize: int = Query(default=10, ge=1, le=100),
    source: str = "all",
    changedOnly: bool = False,
    search: str = "",
) -> dict[str, Any]:
    """Returns paginated audited issue records from disk."""
    try:
        records = _load_audit_records()

        cases = []
        term = search.strip().lower()

        for idx, r in enumerate(records):
            c_source = r.get("source", "")
            c_project = r.get("project", "")
            c_issue_id = str(r.get("issue_id", ""))
            c_orig = "Bug" if "bug" in str(r.get("original_label", "")).lower() and "non" not in str(r.get("original_label", "")).lower() else "Non-Bug"
            c_audit = "Bug" if "bug" in str(r.get("judge_label", "")).lower() and "non" not in str(r.get("judge_label", "")).lower() else "Non-Bug"
            c_changed = bool(r.get("label_flipped", False))
            c_conf = float(r.get("judge_confidence", 0.85))
            c_agree = float(r.get("agent_agreement", 1.0))

            # Case title from reasoning or project
            issue = _load_audit_issue_lookup().get((str(c_source).lower(), str(c_project).lower(), c_issue_id), {})
            c_title = issue.get("title") or f"{c_project.capitalize()} #{c_issue_id}"

            # Filters
            if source != "all" and c_source.upper() != source.upper():
                continue
            if changedOnly and not c_changed:
                continue
            if term and (term not in c_title.lower() and term not in c_issue_id.lower() and term not in c_project.lower()):
                continue

            agree_label = "100% consensus" if c_agree >= 0.99 else f"{int(c_agree * 100)}% majority"
            case_id = f"{c_source.lower()}_{c_project.lower()}_{c_issue_id}"

            cases.append({
                "id": case_id,
                "source": c_source.capitalize(),
                "project": c_project,
                "issueId": c_issue_id,
                "title": c_title,
                "originalLabel": c_orig,
                "auditedLabel": c_audit,
                "changed": c_changed,
                "judgeConfidence": c_conf,
                "agentAgreement": agree_label,
                "auditMethod": r.get("audit_method", "unknown"),
            })

        start = (page - 1) * pageSize
        end = start + pageSize
        return {
            "total": len(cases),
            "page": page,
            "pageSize": pageSize,
            "cases": cases[start:end],
        }
    except Exception as e:
        print(f"[API] Error loading audit cases: {e}")
        return {"total": 0, "page": page, "pageSize": pageSize, "cases": []}


@app.get("/api/v1/audit-cases/{case_id}")
def audit_case_detail(case_id: str) -> dict[str, Any]:
    """Returns individual audit breakdown with Policy, Data, Pattern verdicts and Judge rationale."""
    records = _load_audit_records()

    for r in records:
        c_source = r.get("source", "").lower()
        c_project = r.get("project", "").lower()
        c_issue_id = str(r.get("issue_id", ""))
        curr_id = f"{c_source}_{c_project}_{c_issue_id}"

        if curr_id == case_id:
            c_orig = "Bug" if "bug" in str(r.get("original_label", "")).lower() and "non" not in str(r.get("original_label", "")).lower() else "Non-Bug"
            c_audit = "Bug" if "bug" in str(r.get("judge_label", "")).lower() and "non" not in str(r.get("judge_label", "")).lower() else "Non-Bug"

            issue = _load_audit_issue_lookup().get((c_source, c_project, c_issue_id), {})
            kg_evidence = [*r.get("policy_evidence", []), *r.get("data_evidence", []), *r.get("pattern_evidence", []), *r.get("judge_evidence", [])]

            return {
                "id": curr_id,
                "source": r.get("source", "").capitalize(),
                "project": r.get("project", ""),
                "issueId": c_issue_id,
                "title": issue.get("title") or f"Issue #{c_issue_id}",
                "description": issue.get("description") or "Issue text unavailable in the selected audit sample.",
                "originalLabel": c_orig,
                "auditedLabel": c_audit,
                "changed": bool(r.get("label_flipped", False)),
                "judgeConfidence": float(r.get("judge_confidence", 0.90)),
                "agentAgreement": f"{int(float(r.get('agent_agreement', 1.0)) * 100)}% consensus",
                "agents": [
                    {
                        "name": "Policy Agent",
                        "verdict": "Bug" if "bug" in str(r.get("policy_label", "")).lower() and "non" not in str(r.get("policy_label", "")).lower() else "Non-Bug",
                        "confidence": float(r.get("policy_confidence", 0.85)),
                        "reason": r.get("policy_reason", "Assessed against defect policies."),
                    },
                    {
                        "name": "Data Agent",
                        "verdict": "Bug" if "bug" in str(r.get("data_label", "")).lower() and "non" not in str(r.get("data_label", "")).lower() else "Non-Bug",
                        "confidence": float(r.get("data_confidence", 0.85)),
                        "reason": r.get("data_reason", "Verified metadata consistency."),
                    },
                    {
                        "name": "Pattern Agent",
                        "verdict": "Bug" if "bug" in str(r.get("pattern_label", "")).lower() and "non" not in str(r.get("pattern_label", "")).lower() else "Non-Bug",
                        "confidence": float(r.get("pattern_confidence", 0.85)),
                        "reason": r.get("pattern_reason", "Matched textual defect patterns."),
                    },
                ],
                "knowledgeGraphEvidence": kg_evidence,
                "judge": {
                    "finalLabel": c_audit,
                    "confidence": float(r.get("judge_confidence", 0.90)),
                    "rationale": r.get("judge_reasoning", "Synthesized agent consensus."),
                },
                "auditMethod": r.get("audit_method", "unknown"),
            }


    raise HTTPException(status_code=404, detail=f"Audit case {case_id} not found")


@app.get("/api/v1/model-comparison")
def model_comparison() -> dict[str, Any]:
    """Returns baseline vs treatment comparative metrics."""
    base_rep = load_paired_baseline_report()
    readiness = get_treatment_readiness()
    recorded_metrics = _read_json(PROJECT_ROOT / "outputs/results/treatment/metrics.json") or {}
    treatment_metrics = recorded_metrics or None
    treatment_context = _read_json(PROJECT_ROOT / "outputs/results/treatment/treatment_dataset_summary_current.json") or _read_json(PROJECT_ROOT / "outputs/results/treatment/treatment_dataset_summary.json") or {}
    baseline_identity = base_rep.get("testIdentity")
    treatment_identity = treatment_metrics.get("test_identity_sha256") if treatment_metrics else None
    same_test_split = bool(baseline_identity and baseline_identity == treatment_identity)

    metrics_list = [
        {"key": "accuracy", "label": "Accuracy", "baseline": base_rep["metrics"]["accuracy"], "treatment": treatment_metrics.get("accuracy") if treatment_metrics else None},
        {"key": "balancedAccuracy", "label": "Balanced Accuracy", "baseline": base_rep["metrics"]["balancedAccuracy"], "treatment": treatment_metrics.get("balanced_accuracy") if treatment_metrics else None},
        {"key": "macroF1", "label": "Macro F1", "baseline": base_rep["metrics"]["macroF1"], "treatment": treatment_metrics.get("macro_f1") if treatment_metrics else None},
        {"key": "rocAuc", "label": "ROC-AUC", "baseline": base_rep["metrics"]["rocAuc"], "treatment": treatment_metrics.get("roc_auc") if treatment_metrics else None},
        {"key": "mcc", "label": "MCC", "baseline": base_rep["metrics"]["mcc"], "treatment": treatment_metrics.get("matthews_corrcoef") if treatment_metrics else None},
        {"key": "brier", "label": "Brier Score", "baseline": base_rep["metrics"]["brierScore"], "treatment": treatment_metrics.get("brier_score") if treatment_metrics else None, "lowerIsBetter": True},
    ]

    return {
        "setup": {
            "baselineLabelSource": "Original BugHub source labels",
            "treatmentLabelSource": treatment_context.get("label_policy", "Agent-audited training labels"),
            "testSet": "Held-out test split (Seed 42)",
            "sameTestSplit": same_test_split,
        },
        "baseline": base_rep,
        "treatment": {
            "status": readiness["status"],
            "reason": readiness["reason"],
            "auditMethod": readiness["auditMethod"],
            "recordedAuditMethods": treatment_context.get("audit_methods", {}),
            "threshold": treatment_metrics.get("optimal_threshold") if treatment_metrics else None,
            "recordedAuditedRecords": treatment_context.get("audited_records"),
            "recordedLabelChanges": treatment_context.get("label_changes"),
            "trainingRecords": treatment_context.get("training_records"),
        },
        "treatmentStatus": readiness["status"],
        "comparisonStatus": "comparable" if readiness["ready"] and same_test_split else "protocol_blocked" if readiness["status"] == "blocked" else "test_split_mismatch",
        "metrics": metrics_list,
    }


@app.post("/api/v1/reviews")
def submit_review(payload: ReviewRequest) -> dict[str, Any]:
    """Stores human reviewer decisions for low-confidence operational inference cases."""
    rev_dir = Path("outputs/results/reviews")
    rev_dir.mkdir(parents=True, exist_ok=True)
    rev_file = rev_dir / "operational_reviews.json"

    reviews = []
    if rev_file.exists():
        try:
            with open(rev_file, "r", encoding="utf-8") as f:
                reviews = json.load(f)
        except Exception:
            reviews = []

    record = {
        "prediction_id": payload.prediction_id,
        "reviewer": payload.reviewer or "qa_engineer",
        "human_label": payload.human_label,
        "review_reason": payload.review_reason,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    reviews.append(record)
    with open(rev_file, "w", encoding="utf-8") as f:
        json.dump(reviews, f, indent=2)

    return {
        "status": "accepted",
        "prediction_id": payload.prediction_id,
        "human_label": payload.human_label,
        "stored_as": "operational_review_record",
    }
