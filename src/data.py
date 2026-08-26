"""Chargement et nettoyage minimal du dataset Telco Customer Churn."""
from pathlib import Path
import pandas as pd

RANDOM_STATE = 42
TARGET = "Churn"
ID_COL = "customerID"

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "Telco-Customer-Churn.csv"


def load_raw(path: Path = DATA_PATH) -> pd.DataFrame:
    """Charge le CSV brut sans aucune transformation apprise."""
    df = pd.read_csv(path)
    return df


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoyage non-appris : typage et normalisation de la cible.

    Ces opérations ne dépendent d'aucune statistique calculée sur les données
    (pas de moyenne, médiane, mode...) : elles peuvent donc être appliquées
    avant le split sans provoquer de fuite de données.
    """
    df = df.copy()
    # TotalCharges est chargé comme texte car 11 lignes contiennent un espace " "
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    # SeniorCitizen est encodé 0/1 mais est sémantiquement catégoriel
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    # Encodage binaire de la cible : Yes -> 1 (churn), No -> 0
    df[TARGET] = df[TARGET].map({"Yes": 1, "No": 0})
    return df


def get_feature_target(df: pd.DataFrame):
    X = df.drop(columns=[TARGET, ID_COL])
    y = df[TARGET]
    return X, y


def load_dataset(path: Path = DATA_PATH):
    df = basic_clean(load_raw(path))
    return get_feature_target(df)


if __name__ == "__main__":
    X, y = load_dataset()
    print(X.shape, y.mean())
