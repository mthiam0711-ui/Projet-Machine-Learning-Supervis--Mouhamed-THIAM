"""Tests de l'API REST (Mission 5), via le TestClient FastAPI (pas de serveur réel requis)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient

from api.main import app
from tests.test_pipeline import VALID_RECORD

client = TestClient(app)


def test_health_endpoint():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "version" in body


def test_model_info_endpoint():
    r = client.get("/model-info")
    assert r.status_code == 200
    body = r.json()
    assert "expected_features" in body
    assert 0.0 <= body["decision_threshold"] <= 1.0
    assert len(body["expected_features"]) == 19


def test_predict_valid_record():
    r = client.post("/predict", json=VALID_RECORD)
    assert r.status_code == 200
    body = r.json()
    assert body["churn_prediction"] in (0, 1)
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["risk_label"] in (
        "Risque élevé", "Risque modéré (ciblé par le seuil métier)", "Risque faible",
    )


def test_predict_missing_field_returns_422():
    incomplete = dict(VALID_RECORD)
    del incomplete["Contract"]
    r = client.post("/predict", json=incomplete)
    assert r.status_code == 422  # validation Pydantic


def test_predict_without_total_charges_is_handled():
    record = dict(VALID_RECORD)
    del record["TotalCharges"]
    r = client.post("/predict", json=record)
    assert r.status_code == 200
    assert 0.0 <= r.json()["churn_probability"] <= 1.0
