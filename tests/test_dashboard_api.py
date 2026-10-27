import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))

import backend.main as api


def test_prediction_uses_treatment_probability_and_threshold(monkeypatch):
    model_texts = []
    monkeypatch.setattr(api, "get_treatment_readiness", lambda: {"ready": True, "status": "completed", "reason": None, "auditMethod": "four_separate_role_calls"})
    monkeypatch.setattr(api, "predict_treatment_probability", lambda text: model_texts.append(text) or 0.61)
    monkeypatch.setattr(api, "get_treatment_threshold", lambda: 0.55)

    with TestClient(api.app) as client:
        response = client.post("/api/v1/predictions", json={
            "title": "  Settings page crashes ",
            "description": "  Opening settings closes the application.  ",
        })

    assert response.status_code == 200
    assert model_texts == ["Settings page crashes Opening settings closes the application."]
    result = response.json()
    assert result["prediction"] == {
        "label": "bug",
        "class_id": 1,
        "probability_bug": 0.61,
        "classification_threshold": 0.55,
        "model_name": "treatment",
        "model_version": "outputs/models/treatment/xgb_treatment.json",
    }
    assert result["review"]["low_confidence_check"] == "disabled"
    assert result["review"]["ood_check"] == "disabled"


def test_prediction_fails_closed_when_treatment_inference_is_unavailable(monkeypatch):
    def unavailable(_text):
        raise RuntimeError("model artifacts unavailable")

    monkeypatch.setattr(api, "get_treatment_readiness", lambda: {"ready": True, "status": "completed", "reason": None, "auditMethod": "four_separate_role_calls"})
    monkeypatch.setattr(api, "predict_treatment_probability", unavailable)

    with TestClient(api.app) as client:
        response = client.post("/api/v1/predictions", json={
            "title": "Settings page crashes",
            "description": "Opening settings closes the application.",
        })

    assert response.status_code == 503
    assert "Treatment inference is unavailable" in response.json()["detail"]


def test_model_comparison_blocks_unverified_treatment_protocol():
    result = api.model_comparison()

    assert result["treatmentStatus"] == "blocked"
    assert result["comparisonStatus"] == "protocol_blocked"
    assert result["setup"]["sameTestSplit"] is False
    assert result["treatment"]["recordedAuditedRecords"] == 1000
    assert result["treatment"]["threshold"] is None
    assert "four separate role calls" in result["treatment"]["reason"]
    metric_values = {metric["key"]: metric for metric in result["metrics"]}
    assert metric_values["macroF1"]["baseline"] == 0.7546
    assert metric_values["macroF1"]["treatment"] is None