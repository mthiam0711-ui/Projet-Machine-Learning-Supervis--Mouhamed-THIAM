"""Tests du pipeline final sérialisé (Mission 5)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.serve import load_pipeline, load_metadata, predict_one, EXPECTED_FEATURES
from src.data import load_dataset

VALID_RECORD = {
    "gender": "Female", "SeniorCitizen": "No", "Partner": "Yes", "Dependents": "No",
    "tenure": 1, "PhoneService": "No", "MultipleLines": "No phone service",
    "InternetService": "DSL", "OnlineSecurity": "No", "OnlineBackup": "Yes",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
    "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 29.85, "TotalCharges": 29.85,
}


@pytest.fixture(scope="module")
def pipeline():
    return load_pipeline()


@pytest.fixture(scope="module")
def metadata():
    return load_metadata()


def test_pipeline_reload_gives_identical_predictions(pipeline, tmp_path):
    """Après rechargement du pipeline sérialisé, les prédictions sont identiques."""
    import joblib
    df = pd.DataFrame([VALID_RECORD])
    proba_1 = pipeline.predict_proba(df)[:, 1]

    reload_path = tmp_path / "reloaded.joblib"
    joblib.dump(pipeline, reload_path)
    reloaded = joblib.load(reload_path)
    proba_2 = reloaded.predict_proba(df)[:, 1]

    np.testing.assert_array_almost_equal(proba_1, proba_2, decimal=10)


def test_probabilities_are_within_unit_interval(pipeline):
    """Les probabilités prédites sont bien dans [0, 1]."""
    X, y = load_dataset()
    sample = X.sample(50, random_state=0)
    proba = pipeline.predict_proba(sample)[:, 1]
    assert (proba >= 0.0).all() and (proba <= 1.0).all()


def test_output_shape_matches_input(pipeline):
    """La sortie a la bonne forme : une prédiction par ligne en entrée."""
    X, y = load_dataset()
    sample = X.sample(30, random_state=1)
    pred = pipeline.predict(sample)
    proba = pipeline.predict_proba(sample)
    assert pred.shape == (30,)
    assert proba.shape == (30, 2)


def test_pipeline_handles_missing_values(pipeline):
    """Le pipeline gère les valeurs manquantes (ex. TotalCharges d'un client neuf)."""
    record = dict(VALID_RECORD)
    record["TotalCharges"] = None
    df = pd.DataFrame([record])
    proba = pipeline.predict_proba(df)[:, 1]
    assert not np.isnan(proba[0])
    assert 0.0 <= proba[0] <= 1.0


def test_expected_features_are_present_via_predict_one(pipeline, metadata):
    """predict_one lève une erreur explicite si une feature attendue manque."""
    incomplete = dict(VALID_RECORD)
    del incomplete["Contract"]
    with pytest.raises(ValueError):
        predict_one(pipeline, metadata, incomplete)


def test_performance_on_reference_subset(pipeline, metadata):
    """La performance sur un mini jeu de référence (extrait du dataset) reste correcte."""
    from sklearn.metrics import f1_score

    X, y = load_dataset()
    X_ref, y_ref = X.sample(300, random_state=42), None
    y_ref = y.loc[X_ref.index]
    threshold = metadata.get("decision_threshold", 0.5)
    proba = pipeline.predict_proba(X_ref)[:, 1]
    pred = (proba >= threshold).astype(int)
    f1 = f1_score(y_ref, pred)
    # Seuil de non-régression large (le mini-jeu n'est pas le test set officiel)
    assert f1 > 0.30, f"F1 anormalement bas sur le jeu de référence : {f1:.3f}"


def test_all_expected_features_declared():
    """La liste EXPECTED_FEATURES correspond bien aux colonnes du dataset nettoyé."""
    X, y = load_dataset()
    assert set(EXPECTED_FEATURES) == set(X.columns)
