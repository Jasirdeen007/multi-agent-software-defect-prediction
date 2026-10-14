from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


app = FastAPI(
    title="AgentTriage API",
    version="0.1.0",
    description="Dashboard API for the Multi-Agent-SDP research workflow.",
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

BASELINE_REPORT: dict[str, Any] = {
    "status": "completed",
    "labelSource": "Original BugHub source labels",
    "model": "DeBERTa-v3-small frozen embeddings + XGBoost",
    "inputFeatures": ["model_text"],
    "trackingFields": ["source", "project", "issue_id"],
    "target": "original_label",
    "split": {
        "train": 648424,
        "validation": 138948,
        "test": 138949,
        "randomSeed": 42,
    },
    "threshold": 0.52,
    "metrics": {
        "accuracy": 0.7852,
        "balancedAccuracy": 0.7044,
        "precisionBug": 0.8099,
        "recallBug": 0.9059,
        "f1Bug": 0.8552,
        "precisionNonBug": 0.6957,
        "recallNonBug": 0.5030,
        "f1NonBug": 0.5839,
        "macroF1": 0.7195,
        "weightedF1": 0.7739,
        "rocAuc": 0.8301,
        "mcc": 0.4547,
        "brierScore": 0.1483,
    },
    "artifacts": [
        "outputs/models/baseline/full/xgb_baseline.json",
        "outputs/models/baseline/full/threshold_meta.json",
        "outputs/results/baseline/full/metrics.json",
        "outputs/results/baseline/full/classification_report.txt",
        "outputs/figures/baseline/full/",
    ],
}

AUDIT_OVERVIEW: dict[str, Any] = {
    "status": "planned",
    "totalIssues": 926461,
    "auditSampleSize": 5000,
    "labelsChanged": 0,
    "labelChangeRate": 0.0,
    "agentAgreement": {"allAgree": 0, "twoOfThree": 0, "noMajority": 0},
    "changeDirection": [
        {"from": "Bug", "to": "Not a Bug", "count": 0},
        {"from": "Not a Bug", "to": "Bug", "count": 0},
    ],
    "bySource": [
        {"source": "Bugzilla", "audited": 0, "changed": 0, "changeRate": 0.0},
        {"source": "GitHub", "audited": 0, "changed": 0, "changeRate": 0.0},
        {"source": "JIRA", "audited": 0, "changed": 0, "changeRate": 0.0},
    ],
}

AUDIT_CASES = [
    {
        "id": "case_0001",
        "source": "GitHub",
        "project": "vscode",
        "issueId": "198765",
        "title": "Search results disappear after changing folders",
        "originalLabel": "Bug",
        "auditedLabel": "Bug",
        "changed": False,
        "judgeConfidence": 0.88,
        "agentAgreement": "Demo case",
    },
    {
        "id": "case_0002",
        "source": "Bugzilla",
        "project": "firefox",
        "issueId": "1901043",
        "title": "Unable to open a downloaded PDF file",
        "originalLabel": "Non-Bug",
        "auditedLabel": "Bug",
        "changed": True,
        "judgeConfidence": 0.81,
        "agentAgreement": "Demo case",
    },
]


@app.get("/api/v1/health")
def health() -> dict[str, str]:
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.get("/api/v1/project-status")
def project_status() -> dict[str, Any]:
    return {
        "currentStage": "Baseline completed; KG and multi-agent audit pending",
        "stages": [
            {"name": "Canonical dataset", "status": "completed"},
            {"name": "EDA and validation", "status": "completed"},
            {"name": "Original-label baseline", "status": "completed"},
            {"name": "Selective audit sampling", "status": "pending"},
            {"name": "Knowledge Graph", "status": "pending"},
            {"name": "Agent council and Judge", "status": "pending"},
            {"name": "Audited-label treatment", "status": "pending"},
            {"name": "Baseline vs treatment comparison", "status": "pending"},
        ],
    }


@app.get("/api/v1/dataset-stats")
def dataset_stats() -> dict[str, Any]:
    return DATASET_STATS


@app.get("/api/v1/baseline-report")
def baseline_report() -> dict[str, Any]:
    return BASELINE_REPORT


@app.post("/api/v1/predictions")
def predict_issue(payload: PredictionRequest) -> dict[str, Any]:
    text = f"{payload.title} {payload.description}".lower()
    bug_terms = ("crash", "error", "fail", "broken", "exception", "regression", "incorrect", "unable", "freeze", "cannot")
    is_bug = any(term in text for term in bug_terms)
    confidence = 0.84 if is_bug else 0.61
    return {
        "prediction_id": f"preview-{abs(hash(text)) % 100000}",
        "model_version": "baseline-preview-api",
        "model_prediction": "bug" if is_bug else "non-bug",
        "model_confidence": confidence,
        "prediction_summary": "Preview inference contract. Replace with DeBERTa embedding + XGBoost model loading once artifacts are available locally.",
        "shap_values": [
            {"feature": "defect language", "weight": 0.38 if is_bug else -0.12, "direction": "bug" if is_bug else "non-bug"},
            {"feature": "reproduction detail", "weight": 0.21, "direction": "bug"},
            {"feature": "request wording", "weight": -0.18 if not is_bug else 0.04, "direction": "non-bug" if not is_bug else "bug"},
        ],
        "human_review_required": confidence < 0.65,
    }


@app.get("/api/v1/audit-overview")
def audit_overview() -> dict[str, Any]:
    return AUDIT_OVERVIEW


@app.get("/api/v1/audit-cases")
def audit_cases(page: int = 1, pageSize: int = 10, source: str = "all", changedOnly: bool = False, search: str = "") -> dict[str, Any]:
    term = search.strip().lower()
    filtered = [
        case for case in AUDIT_CASES
        if (source == "all" or case["source"] == source)
        and (not changedOnly or case["changed"])
        and (not term or term in case["title"].lower())
    ]
    start = max(page - 1, 0) * pageSize
    end = start + pageSize
    return {"total": len(filtered), "page": page, "pageSize": pageSize, "cases": filtered[start:end]}


@app.get("/api/v1/audit-cases/{case_id}")
def audit_case(case_id: str) -> dict[str, Any]:
    for case in AUDIT_CASES:
        if case["id"] == case_id:
            return {
                **case,
                "description": "Demonstration audit detail. Real records will be populated after selective audit sampling and KG-guided agent execution.",
                "agents": [
                    {"name": "Policy Agent", "verdict": case["auditedLabel"], "confidence": 0.82, "reason": "Evaluates the report against defect policy."},
                    {"name": "Data Agent", "verdict": case["originalLabel"], "confidence": 0.72, "reason": "Checks metadata consistency."},
                    {"name": "Pattern Agent", "verdict": case["auditedLabel"], "confidence": 0.79, "reason": "Reviews defect language and similar reports."},
                ],
                "knowledgeGraphEvidence": ["KG evidence placeholder until the graph stage is implemented."],
                "judge": {"finalLabel": case["auditedLabel"], "confidence": case["judgeConfidence"], "rationale": "Demonstration decision for dashboard flow."},
            }
    raise HTTPException(status_code=404, detail="Audit case not found")


@app.get("/api/v1/model-comparison")
def model_comparison() -> dict[str, Any]:
    return {
        "setup": {
            "baselineLabelSource": "Original BugHub source labels",
            "treatmentLabelSource": "Agent-audited labels (pending)",
            "testSet": "Held-out baseline split",
        },
        "baseline": BASELINE_REPORT,
        "treatmentStatus": "pending",
        "metrics": [
            {"key": "accuracy", "label": "Accuracy", "baseline": 0.7852, "treatment": None},
            {"key": "balancedAccuracy", "label": "Balanced Accuracy", "baseline": 0.7044, "treatment": None},
            {"key": "macroF1", "label": "Macro F1", "baseline": 0.7195, "treatment": None},
            {"key": "rocAuc", "label": "ROC-AUC", "baseline": 0.8301, "treatment": None},
            {"key": "mcc", "label": "MCC", "baseline": 0.4547, "treatment": None},
            {"key": "brier", "label": "Brier Score", "baseline": 0.1483, "treatment": None, "lowerIsBetter": True},
        ],
    }


@app.post("/api/v1/reviews")
def submit_review(payload: ReviewRequest) -> dict[str, Any]:
    return {
        "status": "accepted",
        "prediction_id": payload.prediction_id,
        "human_label": payload.human_label,
        "stored_as": "operational_review_preview",
    }
