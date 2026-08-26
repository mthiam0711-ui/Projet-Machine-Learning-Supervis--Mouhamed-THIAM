"""Utilitaires partagés par l'API et les tests : chargement du modèle final,
schéma des features attendues, fonction de prédiction avec seuil métier."""
import json
from pathlib import Path

import joblib
import pandas as pd

MODEL_DIR = Path(__file__).resolve().parents[1] / "model"
PIPELINE_PATH = MODEL_DIR / "final_calibrated_pipeline.joblib"
METADATA_PATH = MODEL_DIR / "metadata.json"

EXPECTED_FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges",
]


def load_metadata() -> dict:
    with open(METADATA_PATH) as f:
        return json.load(f)


def load_pipeline():
    return joblib.load(PIPELINE_PATH)


def predict_one(pipeline, metadata: dict, record: dict) -> dict:
    """Prédit le churn pour un client (dict de features brutes).

    Applique le seuil de décision métier défini en Mission 4 (et non 0.5).
    """
    missing = [c for c in EXPECTED_FEATURES if c not in record]
    if missing:
        raise ValueError(f"Features manquantes : {missing}")

    df = pd.DataFrame([{c: record[c] for c in EXPECTED_FEATURES}])
    proba = float(pipeline.predict_proba(df)[0, 1])
    threshold = metadata.get("decision_threshold", 0.5)
    prediction = int(proba >= threshold)
    return {
        "churn_prediction": prediction,
        "churn_probability": proba,
        "threshold_used": threshold,
    }
