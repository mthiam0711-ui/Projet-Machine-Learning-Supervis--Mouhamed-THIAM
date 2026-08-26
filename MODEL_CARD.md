# Model Card — Telco Churn Predictor

*Suivant le format proposé par Mitchell et al., « Model Cards for Model
Reporting », FAccT 2019. Une model card documente de façon standardisée les
données, la performance et les limites d'un modèle, pour que quiconque le
déploie ou l'audite comprenne son contexte d'usage prévu — c'est devenu un
standard car il rend explicite ce qu'un simple score d'accuracy cache
(performance par sous-groupe, hypothèses, cas d'échec).*

## 1. Détails du modèle

- **Type** : `RandomForestClassifier` (scikit-learn), optimisé par Optuna
  (60 essais, `MedianPruner`), puis calibré (`CalibratedClassifierCV`,
  méthode isotonique).
- **Hyperparamètres retenus** : voir `model/metadata.json`
  (`n_estimators=600, max_depth=11, min_samples_split=27,
  min_samples_leaf=8, max_features='log2', class_weight='balanced',
  criterion='entropy'`).
- **Version** : 1.0.0 — entraîné le 26/08/2026.
- **Développeur** : projet capstone individuel, Master IA.
- **Licence** : usage pédagogique.

## 2. Usage prévu

- **Cas d'usage principal** : prioriser, pour le service marketing, les
  clients à cibler par une campagne de rétention, à partir de leur profil
  contractuel et de services.
- **Utilisateurs prévus** : équipe marketing / CRM d'un opérateur télécom
  (via l'API `/predict`), pas directement le client final.
- **Hors périmètre** : ce modèle ne doit pas servir à des décisions
  individuelles ayant un impact autre que commercial (ex. tarification
  discriminatoire, refus de service) ; il n'est pas conçu pour un usage
  temps réel critique (latence non testée en charge).

## 3. Données d'entraînement

- **Source** : Telco Customer Churn (IBM Sample), ~7 043 clients, 19
  features (démographie, services, contrat, facturation).
- **Période** : instantané statique (pas d'information temporelle sur les
  clients au-delà de leur ancienneté `tenure`).
- **Split** : 80% train (5 634 clients) / 20% test (1 409 clients),
  stratifié sur la cible.
- **Prétraitement** : imputation médiane/mode, standardisation des
  numériques, one-hot encoding des catégorielles — appris exclusivement sur
  le train (voir `notebooks/02_pipeline_baseline.ipynb`).

## 4. Métriques de performance

Sur le jeu de test (jamais vu pendant l'entraînement ni le tuning),
au seuil de décision métier (**0.07**, et non 0.5 — voir section 6) :

| Métrique | Valeur |
|---|---|
| F1 (classe churn) | 0.541 |
| Rappel (classe churn) | 0.976 |
| Précision (classe churn) | 0.374 |
| Taux de clients ciblés | 69.3% |
| F1 en validation croisée (train, seuil 0.5) | 0.637 ± (voir notebook M4) |

Le seuil bas (0.07) est un choix **délibéré** issu du coût métier estimé en
Mission 0 (un churner manqué coûte ~20x plus cher qu'une offre envoyée à
tort) : il maximise le rappel au prix d'une précision plus faible et d'un
taux de ciblage élevé. Ce compromis doit être revalidé avec les vraies
estimations de coût de l'entreprise avant mise en production.

## 5. Performance par sous-groupe

| Sous-groupe | n (test) | F1 | Rappel | Taux de ciblage | Taux de churn réel |
|---|---|---|---|---|---|
| Genre — Homme | 722 | 0.518 | 0.972 | 69.0% | 25.1% |
| Genre — Femme | 687 | 0.563 | 0.979 | 69.6% | 28.1% |
| Senior — Non | 1187 | 0.507 | 0.971 | 65.8% | 23.3% |
| Senior — Oui | 222 | 0.662 | 0.990 | **87.8%** | 44.1% |
| Contrat — Mensuel | 773 | 0.606 | 1.000 | 97.8% | 42.6% |
| Contrat — 1 an | 300 | 0.308 | 0.833 | 53.0% | 12.0% |
| Contrat — 2 ans | 336 | 0.171 | 0.667 | 18.2% | 2.7% |
| Internet — Fibre | 613 | 0.608 | 0.996 | 93.6% | 41.1% |
| Internet — DSL | 484 | 0.445 | 0.938 | 64.5% | 20.0% |
| Internet — Aucun | 312 | 0.400 | 0.920 | 28.8% | 8.0% |

**Observation notable** : les clients seniors sont ciblés à 87.8% contre
65.8% pour les non-seniors — un écart en partie justifié par un taux de
churn réel plus élevé (44.1% vs 23.3%) mais qui mérite un examen d'équité
(section 7) avant déploiement, notamment si l'âge est une caractéristique
protégée dans le contexte réglementaire visé.

## 6. Seuil de décision et coûts

- Coût estimé d'un faux négatif (churner manqué) : **400 €** (perte de
  valeur client, illustratif — à remplacer par une vraie estimation LTV).
- Coût estimé d'un faux positif (offre envoyée à tort) : **20 €** (coût de
  la promotion).
- Seuil optimal résultant : **0.07** (voir `notebooks/04_tuning_calibration_shap.ipynb`, section 4.4).

## 7. Limites connues et biais potentiels

- **Rappel très élevé au prix d'un fort taux de ciblage (69%)** : en
  pratique, envoyer une offre à 7 clients sur 10 peut ne pas être
  opérationnellement soutenable — le ratio de coûts métier doit être
  raffiné avec l'équipe marketing avant mise en production.
- **Disparité de ciblage par sous-groupe** (seniors, type de contrat,
  service internet) : à auditer avec des métriques d'équité formelles
  (demographic parity, equalized odds — voir rapport, question de réflexion
  n°4) avant tout déploiement à grande échelle.
- **Dataset statique** : aucune information temporelle réelle (pas de
  date de collecte, pas d'historique multi-période) ; la stationnarité de la
  distribution (les clients de demain ressemblent à ceux d'aujourd'hui)
  n'est pas garantie — voir plan de monitoring (README, section 10).
- **Silver-standard sur `Churn`** : la définition exacte de la variable
  cible (fenêtre temporelle de la résiliation) n'est pas documentée dans le
  dataset source ; à clarifier avec l'équipe métier réelle.
- **Faible échantillon sur certains sous-groupes** (ex. `Two year` +
  churn réel) : les métriques par sous-groupe ci-dessus sont sujettes à un
  bruit d'échantillonnage plus élevé sur les catégories peu représentées.

## 8. Considérations éthiques

Le modèle influence une action commerciale (offre de rétention), pas un
refus de service ; le risque direct pour l'individu est donc limité, mais un
usage détourné (ex. tarification personnalisée pénalisante pour les profils
« fidèles » jugés à faible risque) sortirait du cadre prévu et n'est pas
recommandé.
