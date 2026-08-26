"""Feature engineering pour le churn Telco.

Toutes les transformations ici ne dépendent QUE de la ligne courante
(pas de statistique globale apprise) : elles sont donc "sûres" à appliquer
avant même le split, mais on les place par prudence et par convention dans
le pipeline scikit-learn (étape ColumnTransformer / FunctionTransformer),
fit sur le train uniquement, pour garder une seule source de vérité.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

SERVICE_COLS = [
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies",
]


class ChurnFeatureEngineer(BaseEstimator, TransformerMixin):
    """Ajoute 3 features dérivées :

    1. tenure_group : ancienneté catégorisée (0-1an, 1-2ans, 2-4ans, 4ans+)
    2. num_services : nombre de services actifs souscrits (hors téléphonie de base)
    3. charge_per_tenure : ratio TotalCharges / (tenure + 1), proxy de la
       dépense moyenne mensuelle réelle sur toute la durée de vie du client.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # 1. Ancienneté catégorisée
        bins = [-1, 12, 24, 48, np.inf]
        labels = ["0-1an", "1-2ans", "2-4ans", "4ans+"]
        X["tenure_group"] = pd.cut(X["tenure"], bins=bins, labels=labels).astype(str)

        # 2. Nombre de services actifs
        def count_services(row):
            n = 0
            for col in SERVICE_COLS:
                if col in row and str(row[col]) not in ("No", "No internet service", "No phone service"):
                    n += 1
            return n

        X["num_services"] = X.apply(count_services, axis=1)

        # 3. Charge moyenne par mois d'ancienneté (robuste aux nouveaux clients via +1)
        total = X["TotalCharges"].fillna(X["MonthlyCharges"])  # nouveaux clients: TotalCharges NaN
        X["charge_per_tenure"] = total / (X["tenure"] + 1)

        return X

    def get_feature_names_out(self, input_features=None):
        base = list(input_features) if input_features is not None else []
        return np.array(base + ["tenure_group", "num_services", "charge_per_tenure"])
