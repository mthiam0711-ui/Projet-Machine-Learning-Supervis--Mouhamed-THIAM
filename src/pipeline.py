"""Construction du pipeline scikit-learn unique (sans fuite de données)."""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import ChurnFeatureEngineer

NUMERIC_FEATURES = [
    "tenure", "MonthlyCharges", "TotalCharges",
    "num_services", "charge_per_tenure",
]

CATEGORICAL_FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod", "tenure_group",
]


def build_preprocessor(with_feature_engineering: bool = True) -> ColumnTransformer:
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    if with_feature_engineering:
        num_cols, cat_cols = NUMERIC_FEATURES, CATEGORICAL_FEATURES
    else:
        num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
        cat_cols = [c for c in CATEGORICAL_FEATURES if c != "tenure_group"]
    return ColumnTransformer([
        ("num", numeric_pipe, num_cols),
        ("cat", categorical_pipe, cat_cols),
    ])


def build_pipeline(model, with_feature_engineering: bool = True) -> Pipeline:
    """Construit le pipeline complet : feature engineering -> preprocessing -> modèle.

    Toutes les étapes (imputation, scaling, encodage) sont `fit` uniquement
    sur les données passées à `.fit()`, c'est-à-dire le train. Rien n'est
    jamais estimé sur validation/test.
    """
    steps = []
    if with_feature_engineering:
        steps.append(("feat_eng", ChurnFeatureEngineer()))
    steps.append(("preprocess", build_preprocessor(with_feature_engineering)))
    steps.append(("model", model))
    return Pipeline(steps)
