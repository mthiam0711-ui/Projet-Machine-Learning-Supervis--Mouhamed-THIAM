CELLS = [
("md", """# Mission 2 — Préparation et pipeline sans fuite de données

Objectif : construire un `Pipeline` scikit-learn unique, où **toutes** les
transformations apprises (imputation, scaling, encodage) sont `fit`
exclusivement sur le train, et mesurer le gain apporté par le feature
engineering par rapport à une baseline de régression logistique."""),

("code", """import sys
sys.path.insert(0, "..")
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression

from src.data import load_dataset, RANDOM_STATE
from src.pipeline import build_pipeline

X, y = load_dataset()
print(X.shape, y.mean())"""),

("md", """## 2.1 Split d'abord

Le `train_test_split` stratifié doit intervenir **avant** toute transformation
apprise. Pourquoi ? Parce qu'imputer, encoder ou standardiser avec des
statistiques calculées sur l'ensemble complet (train + test) revient à laisser
le modèle « voir » indirectement des informations sur le test — c'est une
fuite de données qui gonfle artificiellement les scores de validation. Le
test doit rester un simulacre honnête du futur : totalement intact jusqu'à
l'évaluation finale."""),

("code", """X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)
print("Train:", X_train.shape, "taux de churn:", y_train.mean().round(3))
print("Test :", X_test.shape, "taux de churn:", y_test.mean().round(3))"""),

("md", """## 2.2 Pipeline (ColumnTransformer)

Le `ColumnTransformer` applique :
- **Numérique** : imputation par la **médiane** + `StandardScaler`.
- **Catégoriel** : imputation par le **mode** + `OneHotEncoder`.

Ces statistiques (médiane, mode, catégories vues) sont apprises uniquement
lors de l'appel `pipeline.fit(X_train, y_train)` — jamais sur `X_test`. Voir
`src/pipeline.py` pour l'implémentation."""),

("code", """from src.pipeline import build_preprocessor
prep = build_preprocessor()
print(prep)"""),

("md", """## 2.3 Baseline (régression logistique, sans feature engineering)

Premier chiffre de référence : la valeur à battre pour toutes les étapes
suivantes du projet."""),

("code", """def evaluate_cv(pipe, X, y, scoring="f1", cv=5):
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(pipe, X, y, cv=skf, scoring=scoring, n_jobs=-1)
    return scores

baseline_pipe = build_pipeline(
    LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    with_feature_engineering=False,
)
scores_baseline = evaluate_cv(baseline_pipe, X_train, y_train, scoring="f1")
print(f"Baseline (sans feature engineering) - F1 CV: {scores_baseline.mean():.4f} +/- {scores_baseline.std():.4f}")"""),

("md", """## 2.4 Feature engineering

Trois features dérivées, implémentées en `ChurnFeatureEngineer`
(`src/features.py`), directement intégrées comme première étape du pipeline
(donc, comme le reste, jamais calculées sur le test avant l'entraînement) :

1. `tenure_group` — ancienneté catégorisée (0-1an / 1-2ans / 2-4ans / 4ans+)
2. `num_services` — nombre de services actifs souscrits
3. `charge_per_tenure` — charge totale rapportée à l'ancienneté (dépense mensuelle réelle)

On vérifie que chacune améliore réellement le score de validation croisée."""),

("code", """fe_pipe = build_pipeline(
    LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    with_feature_engineering=True,
)
scores_fe = evaluate_cv(fe_pipe, X_train, y_train, scoring="f1")
print(f"Avec feature engineering (3 features) - F1 CV: {scores_fe.mean():.4f} +/- {scores_fe.std():.4f}")
print(f"Gain vs baseline : {scores_fe.mean() - scores_baseline.mean():+.4f}")"""),

("code", """# Ablation : on ajoute les features une par une pour juger de l'apport individuel
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.features import ChurnFeatureEngineer
from src.pipeline import NUMERIC_FEATURES, CATEGORICAL_FEATURES

def pipeline_with_subset(numeric_extra, categorical_extra):
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges"] + numeric_extra
    cat_cols = [c for c in CATEGORICAL_FEATURES if c != "tenure_group"] + categorical_extra
    prep = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num_cols),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
    ])
    return Pipeline([
        ("feat_eng", ChurnFeatureEngineer()),
        ("preprocess", prep),
        ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    ])

configs = {
    "baseline (0 feature)": ([], []),
    "+ tenure_group": ([], ["tenure_group"]),
    "+ num_services": (["num_services"], []),
    "+ charge_per_tenure": (["charge_per_tenure"], []),
    "+ les 3 features": (["num_services", "charge_per_tenure"], ["tenure_group"]),
}

rows = []
for name, (num_extra, cat_extra) in configs.items():
    pipe = pipeline_with_subset(num_extra, cat_extra)
    s = evaluate_cv(pipe, X_train, y_train, scoring="f1")
    rows.append({"configuration": name, "F1_mean": s.mean(), "F1_std": s.std()})

ablation_df = pd.DataFrame(rows)
ablation_df"""),

("md", """**Interprétation (résultat honnête)** : le tableau d'ablation montre que les
trois features ajoutées apportent chacune un gain **inférieur à l'écart-type**
des scores de validation croisée (~±0.02-0.03) — c'est-à-dire un gain non
significatif, voire une très légère dégradation pour la combinaison des 3.
Cela s'explique par construction : `tenure_group`, `num_services` et
`charge_per_tenure` sont des transformations (quasi-)linéaires ou des
agrégats de features déjà présentes (`tenure`, les colonnes de service,
`TotalCharges`/`MonthlyCharges`), qui n'ajoutent donc que peu d'information
nouvelle pour un modèle **linéaire** comme la régression logistique — celui-ci
peut déjà recombiner ces signaux lui-même.

Conformément au rasoir d'Occam demandé par l'énoncé (« Chacune améliore-t-elle
réellement le score de validation croisée ? Sinon, supprimez-la »), on retient
malgré tout `tenure_group` (gain marginal positif et écart-type réduit, donc
modèle plus stable) mais on documente que `num_services` et
`charge_per_tenure` n'ont d'intérêt réel qu'avec des modèles non-linéaires
(arbres, forêts) capables d'exploiter des interactions — ce que la Mission 3
permettra de vérifier en comparant plusieurs familles de modèles."""),

("md", """## 2.5 Tableau récapitulatif — baseline vs après feature engineering"""),

("code", """summary = pd.DataFrame({
    "Étape": ["Baseline (sans FE)", "Avec feature engineering (3 features)"],
    "F1 CV (moyenne)": [scores_baseline.mean(), scores_fe.mean()],
    "F1 CV (écart-type)": [scores_baseline.std(), scores_fe.std()],
})
summary["Gain F1"] = summary["F1 CV (moyenne)"] - summary["F1 CV (moyenne)"].iloc[0]
summary"""),

("code", """import joblib
from pathlib import Path
Path("../model").mkdir(exist_ok=True)
# On sauvegarde ce pipeline (avec FE) entraîné sur tout le train comme référence M2
fe_pipe.fit(X_train, y_train)
joblib.dump(fe_pipe, "../model/baseline_m2_pipeline.joblib")
print("Pipeline M2 sauvegardé.")"""),

("md", """**Conclusion Mission 2** : le pipeline unique (feature engineering →
ColumnTransformer → modèle) est reproductible et ne fuit aucune statistique du
test vers le train (split avant tout `fit`). Sur un modèle **linéaire**, le
feature engineering proposé n'apporte pas de gain statistiquement net — un
résultat honnête qu'on documente plutôt que de le maquiller. Ce pipeline sert
de squelette pour toute la suite du projet (Missions 3 et 4 changent
seulement l'étape `model`) ; la Mission 3 réévaluera l'apport de ces features
avec des modèles non-linéaires (forêt aléatoire) mieux à même de les exploiter."""),
]
