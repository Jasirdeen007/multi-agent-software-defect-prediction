from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any
from uuid import uuid4
from fastapi import FastAPI, HTTPException, Query
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

    if audit_methods.get("single_model_four_perspective_one_call"):
        return {
            "ready": False,
            "status": "blocked",
            "reason": "The retained treatment artifact used one model call for four perspectives; the repository protocol requires four separate role calls. Rebuild the audit and treatment artifacts before inference.",
            "auditMethod": "single_model_four_perspective_one_call",
        }
    audited_records = summary.get("audited_records")
    verified_records = audit_methods.get("four_separate_agent_calls")
    protocol_verified = (
        summary.get("audit_protocol") == "four_separate_role_calls"
        or (verified_records == audited_records and isinstance(verified_records, int) and verified_records > 0 and len(audit_methods) == 1)
    )
    if not protocol_verified:
        return {
            "ready": False,
            "status": "blocked",
            "reason": "Treatment audit provenance does not verify the required four separate role calls.",
            "auditMethod": None,
        }
    if not metrics_path.exists() or not model_path.exists() or not threshold_path.exists():
        return {
            "ready": False,
            "status": "unavailable",
            "reason": "Treatment evaluation metrics or model artifact are missing.",
            "auditMethod": "four_separate_role_calls",
        }
    treatment_metrics = _read_json(metrics_path) or {}
    baseline_metrics = _read_json(PROJECT_ROOT / "outputs/results/baseline/fixed_split/tuned_xgboost/metrics.json") or {}
    if not baseline_metrics.get("test_identity_sha256") or treatment_metrics.get("test_identity_sha256") != baseline_metrics.get("test_identity_sha256"):
        return {
            "ready": False,
            "status": "blocked",
            "reason": "Treatment evaluation does not verify the current tuned baseline's held-out test identities.",
            "auditMethod": "four_separate_role_calls",
        }
    return {
        "ready": True,
        "status": "completed",
        "reason": None,
        "auditMethod": "four_separate_role_calls",
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
        _deberta_extractor = DebertaEmbeddingExtractor(
            model_name=model_config["name"],
            max_length=model_config["max_length"],
            batch_size=1,
            use_fp16=model_config.get("use_fp16", True),
            pooling=model_config["pooling"],
            device="auto",
        )
        print("[API] Loaded DeBERTa-v3 feature extractor.")
    return _deberta_extractor


def predict_treatment_probability(model_text: str) -> float:
    """Return the model's class-1 probability after validating embedding width."""
    import xgboost as xgb

    booster = get_treatment_model()
    extractor = get_deberta_extractor()
    embeddings = extractor.extract_embeddings([model_text], split_name="inference")
    if embeddings.ndim != 2 or embeddings.shape[0] != 1:
        raise ValueError("DeBERTa returned an invalid embedding shape.")
    if booster.num_features() != embeddings.shape[1]:
        raise ValueError("Treatment model feature width does not match the DeBERTa embedding.")
    probability = float(booster.predict(xgb.DMatrix(embeddings))[0])
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("Treatment model returned an invalid class probability.")
    return probability


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
    has_comparison = Path("outputs/results/comparison/comparison_summary.json").exists()
    has_audited = Path("data/processed/audited_labels.parquet").exists()
    has_sample = Path("data/processed/audit_sample.parquet").exists()
    has_kg = Path("outputs/knowledge_graph/kg_audit_sample.json").exists()

    return {
        "currentStage": "Treatment audit protocol requires rebuild" if treatment_readiness["status"] == "blocked" else "Baseline and treatment experiments completed",
        "stages": [
            {"name": "Canonical dataset (926,461 issues)", "status": "completed"},
            {"name": "EDA and validation", "status": "completed"},
            {"name": "Original-label baseline (DeBERTa-v3 + XGBoost)", "status": "completed"},
            {"name": "Selective audit sampling (5,000 stratified issues)", "status": "completed" if has_sample else "pending"},
            {"name": "Knowledge Graph relational ontology & evidence layer", "status": "completed" if has_kg else "pending"},
            {"name": "Multi-agent audit council (Policy, Data, Pattern, Judge)", "status": "completed" if has_audited else "pending"},
            {"name": "Audited-label treatment model", "status": "completed" if has_treatment else treatment_readiness["status"]},
            {"name": "Baseline vs treatment comparative evaluation", "status": "completed" if has_comparison else "in-progress"},
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
    """Predict with the treatment model; never substitute heuristic scores."""
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
            detail="Treatment inference is unavailable. Verify the treatment model and DeBERTa artifacts.",
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
            "model_name": "treatment",
            "model_version": "outputs/models/treatment/xgb_treatment.json",
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


@app.get("/api/v1/audit-overview")
def audit_overview() -> dict[str, Any]:
    """Returns real multi-agent audit summary and flip rates from audited records."""
    audited_p = Path("data/processed/audited_labels.parquet")
    sample_p = Path("data/processed/audit_sample.parquet")

    total_issues = 926461
    sample_size = 5000

    if not audited_p.exists():
        return {
            "status": "planned",
            "totalIssues": total_issues,
            "auditSampleSize": sample_size,
            "labelsChanged": 0,
            "labelChangeRate": 0.0,
            "agentAgreement": {"allAgree": 0, "twoOfThree": 0, "noMajority": 0},
            "bySource": [
                {"source": "Bugzilla", "audited": 0, "changed": 0, "changeRate": 0.0},
                {"source": "GitHub", "audited": 0, "changed": 0, "changeRate": 0.0},
                {"source": "JIRA", "audited": 0, "changed": 0, "changeRate": 0.0},
            ],
            "changeDirection": [
                {"from": "Bug", "to": "Not a Bug", "count": 0},
                {"from": "Not a Bug", "to": "Bug", "count": 0},
            ],
        }

    try:
        df = pd.read_parquet(audited_p)
        total_audited = len(df)
        flips = int(df["label_flipped"].sum())
        flip_rate = round(float(flips / total_audited * 100.0), 2) if total_audited > 0 else 0.0

        all_agree = int((df["agent_agreement"] >= 0.99).sum())
        two_agree = int(((df["agent_agreement"] >= 0.65) & (df["agent_agreement"] < 0.99)).sum())
        no_maj = int((df["agent_agreement"] < 0.65).sum())

        # Change direction
        bug_to_nonbug = int(((df["original_label"] == "bug") & (df["judge_label"] == "non-bug")).sum())
        nonbug_to_bug = int(((df["original_label"] == "non-bug") & (df["judge_label"] == "bug")).sum())

        # By source
        by_source = []
        for src in ["Bugzilla", "GitHub", "JIRA"]:
            sub = df[df["source"].str.upper() == src.upper()]
            n_sub = len(sub)
            ch_sub = int(sub["label_flipped"].sum()) if n_sub > 0 else 0
            cr_sub = round(float(ch_sub / n_sub * 100.0), 1) if n_sub > 0 else 0.0
            by_source.append({
                "source": src,
                "audited": n_sub,
                "changed": ch_sub,
                "changeRate": cr_sub,
            })

        return {
            "status": "completed",
            "totalIssues": total_issues,
            "auditSampleSize": sample_size,
            "totalAudited": total_audited,
            "labelsChanged": flips,
            "labelChangeRate": flip_rate,
            "agentAgreement": {
                "allAgree": all_agree,
                "twoOfThree": two_agree,
                "noMajority": no_maj,
            },
            "bySource": by_source,
            "changeDirection": [
                {"from": "Bug", "to": "Not a Bug", "count": bug_to_nonbug},
                {"from": "Not a Bug", "to": "Bug", "count": nonbug_to_bug},
            ],
        }
    except Exception as e:
        print(f"[API] Error computing audit overview: {e}")
        return {"status": "error", "error": str(e)}


@app.get("/api/v1/audit-cases")
def audit_cases(
    page: int = Query(default=1, ge=1),
    pageSize: int = Query(default=10, ge=1, le=100),
    source: str = "all",
    changedOnly: bool = False,
    search: str = "",
) -> dict[str, Any]:
    """Returns paginated audited issue records from disk."""
    json_p = Path("outputs/results/audited/audited_records.json")
    if not json_p.exists():
        return {"total": 0, "page": page, "pageSize": pageSize, "cases": []}

    try:
        with open(json_p, "r", encoding="utf-8") as f:
            records = json.load(f)

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
            c_title = r.get("title") or f"{c_project.capitalize()} #{c_issue_id}: {r.get('policy_reason', '')[:80]}..."

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
    json_p = Path("outputs/results/audited/audited_records.json")
    if not json_p.exists():
        raise HTTPException(status_code=404, detail="No audit records available")

    with open(json_p, "r", encoding="utf-8") as f:
        records = json.load(f)

    for r in records:
        c_source = r.get("source", "").lower()
        c_project = r.get("project", "").lower()
        c_issue_id = str(r.get("issue_id", ""))
        curr_id = f"{c_source}_{c_project}_{c_issue_id}"

        if curr_id == case_id or case_id.endswith(c_issue_id):
            c_orig = "Bug" if "bug" in str(r.get("original_label", "")).lower() and "non" not in str(r.get("original_label", "")).lower() else "Non-Bug"
            c_audit = "Bug" if "bug" in str(r.get("judge_label", "")).lower() and "non" not in str(r.get("judge_label", "")).lower() else "Non-Bug"

            from src.knowledge_graph.retriever import get_kg_retriever
            kg_evidence = get_kg_retriever().retrieve_evidence_for_issue(r)

            return {
                "id": curr_id,
                "source": r.get("source", "").capitalize(),
                "project": r.get("project", ""),
                "issueId": c_issue_id,
                "title": r.get("title") or f"Issue #{c_issue_id}",
                "description": r.get("description") or r.get("model_text") or "No description provided.",
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
            }


    raise HTTPException(status_code=404, detail=f"Audit case {case_id} not found")


@app.get("/api/v1/model-comparison")
def model_comparison() -> dict[str, Any]:
    """Returns baseline vs treatment comparative metrics."""
    base_rep = load_baseline_report()
    readiness = get_treatment_readiness()
    recorded_metrics = _read_json(PROJECT_ROOT / "outputs/results/treatment/metrics.json") or {}
    treatment_metrics = recorded_metrics if readiness["ready"] else None
    treatment_context = _read_json(PROJECT_ROOT / "outputs/results/treatment/treatment_dataset_summary.json") or {}
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
            "threshold": treatment_metrics.get("optimal_threshold") if treatment_metrics else None,
            "recordedAuditedRecords": treatment_context.get("audited_records"),
            "recordedLabelChanges": treatment_context.get("label_changes"),
            "trainingRecords": treatment_context.get("training_records"),
        },
        "treatmentStatus": readiness["status"],
        "comparisonStatus": "comparable" if same_test_split else "protocol_blocked" if readiness["status"] == "blocked" else "test_split_mismatch",
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
