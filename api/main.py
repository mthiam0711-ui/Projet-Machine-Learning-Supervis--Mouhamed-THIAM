"""API REST de prédiction du churn — Projet capstone Supervised Learning.

Endpoints :
    GET  /health       -> statut du service + version
    POST /predict       -> prédiction + probabilité de churn pour un client
    GET  /model-info    -> métadonnées du modèle (features, seuil, performance)

Lancement local :
    uvicorn api.main:app --reload --port 8000

Exemples curl : voir README.md.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI, HTTPException
from pydantic import ValidationError

from api.schemas import (
    CustomerFeatures, PredictionResponse, HealthResponse, ModelInfoResponse,
)
from src.serve import load_pipeline, load_metadata, predict_one, EXPECTED_FEATURES

APP_VERSION = "1.0.0"

app = FastAPI(
    title="Telco Churn Prediction API",
    description="Prédit la probabilité de résiliation (churn) d'un client télécom.",
    version=APP_VERSION,
)

# Chargés une fois au démarrage du service (pas à chaque requête)
_pipeline = load_pipeline()
_metadata = load_metadata()


def _risk_label(proba: float) -> str:
    if proba >= 0.5:
        return "Risque élevé"
    if proba >= _metadata.get("decision_threshold", 0.5):
        return "Risque modéré (ciblé par le seuil métier)"
    return "Risque faible"


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", version=APP_VERSION)


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerFeatures):
    try:
        record = customer.model_dump()
        result = predict_one(_pipeline, _metadata, record)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:  # garde-fou : ne jamais exposer une trace interne
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction : {e}")

    return PredictionResponse(
        churn_prediction=result["churn_prediction"],
        churn_probability=round(result["churn_probability"], 4),
        threshold_used=result["threshold_used"],
        risk_label=_risk_label(result["churn_probability"]),
    )


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info():
    return ModelInfoResponse(
        model_name=_metadata.get("final_model", "unknown"),
        expected_features=EXPECTED_FEATURES,
        decision_threshold=_metadata.get("decision_threshold", 0.5),
        cv_f1_mean=_metadata.get("cv_f1_mean", -1.0),
        best_params=_metadata.get("best_params", {}),
    )
