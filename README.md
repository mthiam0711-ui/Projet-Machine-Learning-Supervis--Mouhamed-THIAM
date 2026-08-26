# Prédiction du churn — Telco Customer Churn

Projet capstone *Supervised Learning* (Master IA) — pipeline complet de la
donnée brute à une API de prédiction en production.

## 1. Problème

Un opérateur télécom veut identifier, parmi ~7 000 clients, ceux qui
présentent un risque élevé de résiliation (**churn**) afin de les cibler en
priorité avec une offre de rétention. Un faux négatif (churner non détecté)
coûte bien plus cher qu'un faux positif (offre envoyée à un client fidèle) —
voir `report/rapport_final.pdf`, section Cadrage (Mission 0), pour le détail
des coûts et de la métrique retenue.

## 2. Données

- Source : [Telco Customer Churn (IBM Sample)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn),
  ~7 043 clients, 19 features (démographie, services souscrits, contrat,
  facturation) + la cible `Churn`.
- `TotalCharges` contient 11 valeurs manquantes correspondant à des clients
  tout juste arrivés (`tenure = 0`) — géré par imputation dans le pipeline.
- Déséquilibre modéré : ~27% de churn.

## 3. Modèle

- **Pipeline unique scikit-learn** (`src/pipeline.py`) : feature engineering
  → `ColumnTransformer` (imputation + scaling/encodage) → modèle. Toutes les
  statistiques apprises le sont exclusivement sur le train (aucune fuite de
  données, cf. rapport Mission 2).
- **Modèle final** : `RandomForestClassifier`, optimisé par Optuna (60
  essais, 7 hyperparamètres, `MedianPruner`), puis calibré par
  `CalibratedClassifierCV` (isotonic).
- **Performance** (validation croisée, train) : F1 ≈ 0.637 (contre 0.552 par
  défaut, et 0.588 pour une régression logistique).
- **Seuil de décision** : fixé à ~0.07 (et non 0.5), pour minimiser le coût
  métier estimé (FN ≫ FP). Voir `model/metadata.json`.
- Modèle sérialisé : `model/final_calibrated_pipeline.joblib`.

## 4. Structure du dépôt

```
.
├── data/                       # dataset (ou script de téléchargement)
├── notebooks/
│   ├── 01_eda.ipynb             # Mission 1 — EDA, détection de fuite, hypothèses
│   ├── 02_pipeline_baseline.ipynb   # Mission 2 — pipeline sans fuite, baseline, FE
│   ├── 03_modeling_comparison.ipynb # Mission 3 — benchmark, CV, test statistique
│   └── 04_tuning_calibration_shap.ipynb # Mission 4 — Optuna, calibration, SHAP
├── src/                         # code réutilisable (data, features, pipeline, serve)
├── model/                       # pipeline sérialisé + métadonnées
├── api/                         # API FastAPI (main.py, schemas.py)
├── tests/                       # tests pytest (pipeline + API)
├── report/                      # rapport PDF (≤12 pages)
├── requirements.txt
└── README.md
```

## 5. Installation

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## 6. Reproduire l'entraînement

Les notebooks `notebooks/01_*.ipynb` à `04_*.ipynb` sont entièrement
ré-exécutables de bout en bout (`random_state` fixé partout) et reproduisent
l'ensemble du pipeline, du EDA jusqu'à la sérialisation du modèle final dans
`model/final_calibrated_pipeline.joblib`.

## 7. Lancer l'API

```bash
uvicorn api.main:app --reload --port 8000
```

### Exemples curl

```bash
# Santé du service
curl http://127.0.0.1:8000/health

# Métadonnées du modèle
curl http://127.0.0.1:8000/model-info

# Prédiction
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female", "SeniorCitizen": "No", "Partner": "Yes", "Dependents": "No",
    "tenure": 1, "PhoneService": "No", "MultipleLines": "No phone service",
    "InternetService": "DSL", "OnlineSecurity": "No", "OnlineBackup": "Yes",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
    "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 29.85, "TotalCharges": 29.85
  }'
```

Réponse attendue :
```json
{
  "churn_prediction": 1,
  "churn_probability": 0.6027,
  "threshold_used": 0.07,
  "risk_label": "Risque élevé"
}
```

`TotalCharges` peut être omis (client sans historique de facturation) — le
pipeline impute cette valeur automatiquement.

## 8. Tests

```bash
pytest tests/ -v
```

12 tests couvrent : la reproductibilité après rechargement du pipeline, les
bornes des probabilités, la forme des sorties, la gestion des valeurs
manquantes, la présence des features attendues, la performance sur un
mini-jeu de référence, ainsi que les 3 endpoints de l'API.

## 9. Documentation complémentaire

- `MODEL_CARD.md` — données d'entraînement, performance par sous-groupe, limites.
- `report/rapport_final.pdf` — cadrage, EDA, pipeline, benchmark, tuning/SHAP,
  déploiement, réponses aux questions de réflexion (≤ 12 pages).

## 10. Monitoring en production (plan)

- **Data drift** : surveiller la distribution des features (ex. via
  Evidently) pour détecter un décalage entre la distribution d'entraînement
  et celle des nouveaux clients (ex. évolution des offres commerciales).
- **Concept drift** : suivre la performance réelle (F1, rappel) sur les
  churns confirmés a posteriori ; une dégradation progressive signale que la
  relation features → churn a changé (ex. nouvelle politique tarifaire d'un
  concurrent).
- **Performance drift** : ré-évaluer périodiquement le modèle sur un
  échantillon récent labellisé, avec alerte si le F1 chute sous un seuil
  défini en Mission 0.
