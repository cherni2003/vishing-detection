# Architecture MLOps complète — Vishing Detection

> Document de référence de l'équipe (3 personnes). Il remplace `docs/architecture.md` et reprend l'architecture du dépôt en y ajoutant **Apache Spark**, l'entraînement sur **Kaggle** et un modèle qui **écoute la voix** (pas seulement le texte).
> Les chiffres cités viennent de l'audit du corpus DS04 (VISH-DARIJA-TTS). Ils sont à mettre à jour après chaque nouvel audit.

---

## 1. Objectif et périmètre

**Objectif.** Détecter le vishing (voice phishing) à partir du **signal audio** d'un appel : l'entrée du modèle est un fichier WAV, la sortie est un score de risque entre 0 et 1.

**Dans le périmètre**
- Audit, validation et harmonisation de plusieurs corpus audio multilingues.
- Extraction de caractéristiques acoustiques et d'embeddings audio.
- Entraînement, comparaison et évaluation de modèles.
- Suivi des expériences, versionnement des données et des modèles.
- Déploiement possible d'une API de prédiction.

**Hors périmètre pour l'instant**
- Détection en temps réel sur flux téléphonique.
- Transcription automatique comme dépendance du modèle (option de phase 2, jamais obligatoire).
- Corpus exclu de la phase actuelle : Zenodo 8342762 (parole normale tunisienne, sans vishing).

---

## 2. Principes d'ingénierie (à respecter par tous)

1. **Le modèle écoute l'audio.** Aucune colonne d'annotation du scénario n'entre dans le modèle (voir §7.4).
2. **Un seul identifiant fiable : `sample_id`** (= `dataset_uid`, ex. `scam_0103`). Le `scenario_id` seul n'est pas unique entre classes : ne jamais joindre dessus.
3. **Pas de fuite de données.** Le split se fait au niveau de l'appel (ou du locuteur/de la conversation si l'information existe). Tous les segments et toutes les variantes bruitées d'un appel restent dans le même split.
4. **Toute expérience bat d'abord les baselines** (§10.1). Un score sous la baseline la plus forte n'est pas un résultat.
5. **Les labels originaux sont conservés** (`label_original`) à côté du label harmonisé.
6. **Reproductibilité** : code dans Git, données et modèles dans DVC, paramètres dans `params.yaml`, seed fixée (42).
7. **Pas de données lourdes ni de secrets dans Git** : jamais de WAV, jamais de clés d'API, jamais de fichiers de credentials.
8. **Une étape = une commande** (`python -m vishing_detection.pipeline.run_stage <étape>`), rejouable par `dvc repro`.

---

## 3. Vue d'ensemble

```text
 SOURCES (DS01…DS06)                         VERSIONNEMENT
        │                                   Git  : code, configs, petits CSV, *.dvc
        ▼                                   DVC  : données, features, modèles
 ┌───────────────┐                          MLflow : expériences
 │ data/raw      │ ◄── DVC (zip ou téléchargement DOI + MD5)
 └──────┬────────┘
        ▼
 ┌─────────────────────────── SPARK (PySpark) ───────────────────────────┐
 │ 1 validate  → data/interim/quality.parquet                            │
 │ 2 prepare   → data/processed/manifest.parquet + segments.parquet      │
 │ 3 features  → data/features/handcrafted.parquet                       │
 └──────┬────────────────────────────────────────────────────────────────┘
        ▼
 ┌────────────────────── KAGGLE (GPU, PyTorch) ──────────────────────────┐
 │ 4 embeddings → data/features/embeddings_<modèle>.parquet              │
 │ 5 train      → models/<run_id>/   (logué dans MLflow)                 │
 │ 6 evaluate   → reports/metrics.json                                   │
 └──────┬────────────────────────────────────────────────────────────────┘
        ▼
 7 export → models/current/ ──► deployment/ (FastAPI + Docker) ──► score
```

**Pourquoi cette répartition.** Spark excelle sur le traitement de données en parallèle (contrôles qualité, agrégations, extraction de features sur des milliers de fichiers). Il gère mal le GPU : les embeddings et l'entraînement restent donc en PyTorch sur Kaggle. Le lien entre les deux est un format commun, **parquet**, avec la clé `sample_id`.

---

## 4. Environnements et stack

| Environnement | Usage | Contenu |
|---|---|---|
| **Local** (poste de chaque membre) | Développement, tests, petits échantillons | Code, `data/metadata`, manifests, mini-jeu de test. Peu d'espace disque : pas d'audio complet. |
| **Kaggle** | Étapes lourdes : Spark à grande échelle, embeddings, entraînement GPU | Notebook `kaggle/train_kaggle.ipynb` qui clone le dépôt et installe le package. |
| **CI GitHub Actions** | Tests automatiques à chaque push | `pytest` + vérifications de schéma sur un mini-jeu. |

| Domaine | Outil |
|---|---|
| Langage | Python 3.11 ou 3.12 (pour PySpark : vérifier la compatibilité avant d'utiliser 3.14) |
| Traitement distribué | Apache Spark / PySpark (Java 17 ou plus requis) |
| Audio | librosa, soundfile |
| ML | scikit-learn (baselines), PyTorch + Hugging Face Transformers (embeddings, fine-tuning) |
| Suivi d'expériences | MLflow |
| Versionnement | Git/GitHub (code), DVC (données et modèles) |
| Tests | pytest |
| Déploiement | FastAPI, Docker |

---

## 5. Arborescence complète du dépôt

```text
vishing-detection/
├── .github/workflows/
│   └── ci.yaml                       # pytest + lint à chaque push / PR
├── .dvc/                             # config DVC (remote S3 Hugging Face)
├── configs/
│   ├── config.yaml                   # chemins, seed, tailles de split
│   ├── spark.yaml                    # session Spark, partitions, mémoire
│   └── kaggle.yaml                   # chemins Kaggle (/kaggle/input, /kaggle/working)
├── data/
│   ├── raw/                          # données originales (DVC)
│   ├── interim/                      # sorties de validation (parquet)
│   ├── processed/                    # manifest unifié, segments (parquet)
│   ├── features/                     # features acoustiques, embeddings (parquet)
│   └── metadata/                     # audio_manifest.csv, datasets_catalog.csv, rapports
├── deployment/
│   ├── app.py                        # API FastAPI
│   ├── Dockerfile
│   └── requirements-serve.txt
├── docs/
│   ├── architecture.md               # ce document
│   ├── datasets_inventory.md         # inventaire des sources
│   └── decisions/                    # une note par décision technique (ADR)
├── kaggle/
│   ├── make_notebook.py              # régénère le notebook
│   └── train_kaggle.ipynb            # lance les étapes GPU
├── models/
│   ├── README.md
│   └── current/                      # modèle retenu (DVC)
├── notebooks/
│   ├── 01_audit_vish_darija.ipynb    # audit DS04 (fait)
│   └── NN_audit_<corpus>.ipynb       # un notebook par corpus
├── reports/
│   ├── audit_vish_darija.md
│   ├── metrics.json
│   └── figures/
├── src/vishing_detection/
│   ├── __init__.py
│   ├── config.py                     # chargement de configs/*.yaml
│   ├── data/
│   │   ├── audit.py                  # fonctions d'audit extraites des notebooks
│   │   ├── ingestion.py              # un lecteur par corpus → schéma commun
│   │   └── schema.py                 # définition et validation du schéma du manifest
│   ├── spark/
│   │   ├── session.py                # crée la SparkSession depuis spark.yaml
│   │   ├── validate.py               # contrôles qualité distribués
│   │   ├── prepare.py                # harmonisation, flags, splits, segments
│   │   └── extract_features.py       # features acoustiques en parallèle
│   ├── audio/
│   │   ├── loader.py                 # lecture, rééchantillonnage, mono
│   │   ├── segmentation.py           # fenêtres fixes
│   │   ├── features.py               # MFCC, prosodie, pauses
│   │   └── augment.py                # bruit, codecs téléphoniques
│   ├── models/
│   │   ├── baselines.py              # majoritaire, durée, nombre de tours
│   │   ├── embeddings.py             # encodeur audio pré-entraîné
│   │   ├── train.py                  # tête de classification, fine-tuning
│   │   ├── evaluate.py               # métriques globales et par sous-ensemble
│   │   └── predict.py                # inférence par segments + agrégation
│   ├── tracking/
│   │   └── mlflow_utils.py
│   └── pipeline/
│       └── run_stage.py              # point d'entrée de chaque étape
├── tests/
│   ├── conftest.py                   # mini-projet temporaire pour les tests
│   ├── test_package.py
│   ├── test_manifest.py              # schéma, colonnes interdites
│   ├── test_segmentation.py          # fenêtres, chargeur audio, bruit
│   ├── test_audit.py                 # fuite entre groupes, baselines, métriques
│   ├── test_spark_stages.py          # Spark local + chaîne complète sur le mini-jeu
│   └── data/
│       ├── make_mini.py              # génère le mini-jeu
│       └── mini/                     # 8 petits WAV synthétiques + manifest (< 1 Mo, versionnés dans Git)
├── .env.example                      # variables à définir (jamais de vraies valeurs)
├── dvc.yaml                          # enchaînement des étapes
├── params.yaml                       # hyperparamètres
├── requirements.txt
├── pyproject.toml
├── .gitignore  .dvcignore
└── README.md
```

---

## 6. Gestion des données

### 6.1 Sources

| ID | Source | Langues annoncées | Statut |
|---|---|---|---|
| DS01 | NCSU TR-2023-1 | Chinois, anglais | À auditer |
| DS02 | Zenodo 20429763 | Français, anglais, arabe standard | À auditer |
| DS03 | Korean Voice Phishing Detection (GitHub) | Coréen | À auditer |
| DS04 | **VISH-DARIJA-TTS** (Zenodo 20039126, CC-BY-4.0) | Darija marocaine | **En cours** (audit fait) |
| DS05 | Hybrid Voice Phishing Detection System (GitHub) | À vérifier | À auditer |
| DS06 | Mendeley p384bgyzz3 | Dialectes arabes | À auditer |

Règle d'intégration : licence vérifiée avant intégration, labels originaux conservés, aucune hypothèse sur la présence d'appels légitimes sans vérification.

### 6.2 Couches de données

| Couche | Dossier | Contenu | Outil | Versionné par |
|---|---|---|---|---|
| Brute | `data/raw/<corpus>/` | Fichiers originaux, jamais modifiés | téléchargement DOI + MD5 | DVC |
| Intermédiaire | `data/interim/` | Rapports de qualité (parquet) | Spark `validate` | DVC (`outs`) |
| Préparée | `data/processed/` | `manifest.parquet`, `segments.parquet` | Spark `prepare` | DVC (`outs`) |
| Features | `data/features/` | `handcrafted.parquet`, `embeddings_<modèle>.parquet` | Spark / Kaggle | DVC (`outs`) |
| Métadonnées | `data/metadata/` | CSV légers (catalogue, manifest d'origine) | — | Git |

### 6.3 Schéma du manifest commun

Fichier `data/metadata/audio_manifest.csv` (et `processed/manifest.parquet`) :

| Colonne | Description |
|---|---|
| `sample_id` | Identifiant unique global (ex. `scam_0103`) |
| `audio_path` | Chemin relatif vers le WAV |
| `label` | 1 = vishing, 0 = légitime (harmonisé) |
| `label_original` | Label d'origine du corpus |
| `language` | Langue ou dialecte |
| `source_dataset` | ID du corpus (ex. `DS04_VISH-DARIJA-TTS`) |
| `speaker_id` | Locuteur si connu, sinon vide |
| `scenario_id` | Identifiant de scénario/conversation (= `sample_id` si non fiable) |
| `is_synthetic` | Booléen |
| `duration_sec`, `sample_rate` | Propriétés audio |
| `split` | `train`, `validation` ou `test` |
| `license` | Licence du corpus |

**Table séparée d'analyse** : `data/metadata/<corpus>_audit_features.csv` (nombre de tours, `hard_scam`, thème, demande sensible). Elle sert à l'évaluation et à l'analyse, **jamais** à l'entraînement.

### 6.4 Table des segments

`data/processed/segments.parquet` : `segment_id`, `sample_id`, `start_sec`, `end_sec`, `label`, `split` (hérité de l'appel).

### 6.5 Splits

- Le split est fixé une fois, au niveau de l'appel, et stocké dans le manifest.
- Les corpus qui fournissent des splits (DS04 : 2 380 train / 510 validation / 510 test, équilibrés) sont utilisés tels quels après contrôle de fuite.
- Pour les corpus sans split : découpage groupé (`GroupShuffleSplit`) par locuteur ou conversation quand l'information existe.
- **Un test hors corpus** (entraîner sur un corpus, tester sur un autre) est conservé pour mesurer la généralisation (§10.3).

---

## 7. Pipeline de traitement

### 7.1 Étapes

| # | Étape | Où | Entrées → sorties |
|---|---|---|---|
| 1 | `validate` | Spark | manifests + en-têtes WAV → `interim/quality.parquet` |
| 2 | `prepare` | Spark | manifests + qualité → `processed/manifest.parquet`, `segments.parquet` |
| 2b | `baselines` | local | manifest → `reports/baselines.json` (majorité, durée, nombre de tours) |
| 3 | `features_handcrafted` | Spark | WAV → `features/handcrafted.parquet` |
| 4 | `embeddings` | Kaggle GPU | segments → `features/embeddings_<modèle>.parquet` |
| 5 | `train` | Kaggle GPU | features/embeddings → `models/<run_id>/` |
| 6 | `evaluate` | Kaggle | modèle + test → `reports/metrics.json` |
| 7 | `export` | local | meilleur run → `models/current/` |

### 7.2 `dvc.yaml`

Le fichier `dvc.yaml` à la racine décrit les étapes `validate`, `prepare`, `baselines`, `features_handcrafted`, `embeddings`, `train` et `evaluate`, avec leurs dépendances, paramètres et sorties. `dvc repro` ne rejoue que les étapes dont une entrée a changé. L'étape `embeddings` s'exécute sur Kaggle (GPU), puis ses sorties sont poussées avec `dvc push`.

### 7.3 Rôle de Spark, étape par étape

| Module | Fonction |
|---|---|
| `session.py` | `get_spark()` : crée la session à partir de `configs/spark.yaml` |
| `validate.py` | En-têtes WAV (fréquence, canaux, durée), hash dupliqués, valeurs manquantes, fichiers absents |
| `prepare.py` | Harmonisation des labels, conservation de `label_original`, contrôle de fuite entre splits, construction des segments |
| `extract_features.py` | Applique `audio/features.py` sur chaque segment en parallèle (`mapInPandas`) |

`configs/spark.yaml` :

```yaml
spark:
  app_name: vishing-detection
  master: "local[*]"          # sur Kaggle aussi : une seule machine
  driver_memory: "8g"
  shuffle_partitions: 64
  parquet_compression: snappy
paths:
  manifest_csv: data/metadata/audio_manifest.csv
  interim_dir: data/interim
  processed_dir: data/processed
  features_dir: data/features
```

**Quand Spark vaut le coup.** Au volume de DS04 seul (3 400 fichiers, 22,6 h), pandas suffirait. Spark prend son intérêt avec les 6 corpus réunis et l'extraction de features sur plusieurs dizaines de milliers de fichiers. Le code est écrit en Spark dès maintenant pour ne pas avoir à le réécrire.

### 7.4 Règle de séparation entrées / annotations

| Autorisé en entrée du modèle | Interdit en entrée du modèle |
|---|---|
| Audio (segments), features acoustiques, embeddings audio | `speaker_role`, `gender`, `contains_sensitive_request`, `request_type`, `persuasion_strategy`, `attack_type`, `requested_asset`, `impersonated_entity`, `victim_role`, `risk_level`, `theme`, nombre de tours, `split` |

Ces colonnes servent à l'analyse d'erreurs, à l'évaluation par sous-ensemble ou comme cibles auxiliaires. `tests/test_manifest.py` vérifie qu'aucune n'apparaît dans `features/`.

---

## 8. Traitement audio

- **Format de référence** : WAV mono, 16 bits. DS04 est en 24 kHz ; les encodeurs pré-entraînés attendent souvent 16 kHz : le rééchantillonnage est fait dans `audio/loader.py`, jamais à la main dans un notebook.
- **Segmentation** : fenêtres fixes (par défaut 10 s, recouvrement 50 %, paramétrables dans `params.yaml`). Chaque segment hérite du label et du split de l'appel. Cette étape neutralise le raccourci « durée = label » détecté à l'audit.
- **Features acoustiques** (baseline) : MFCC (moyenne et écart-type), énergie, débit de pauses, rapport de silence, caractéristiques de hauteur. À utiliser avec prudence : sur DS04 le silence seul donne déjà une AUC de 0,60 parce qu'il suit le nombre de tours.
- **Embeddings audio** (candidat principal) : sortie d'un encodeur pré-entraîné (famille wav2vec 2.0 / XLS-R / WavLM / encodeur Whisper), moyennée par segment. Le modèle exact est un paramètre de `params.yaml`, à choisir par comparaison.
- **Augmentation** (entraînement seulement) : bruit additif à différents SNR, simulation de codec téléphonique (bande étroite 8 kHz), changement léger de vitesse. Les variantes bruitées fournies par DS04 (SNR 20, 10, 5 dB) servent d'abord à **évaluer la robustesse**, et restent dans le split de leur fichier d'origine.

---

## 9. Modèles

Ordre de développement : chaque niveau doit battre le précédent pour mériter sa place.

| Niveau | Modèle | But |
|---|---|---|
| 0 | Classe majoritaire, règle sur la durée, règle sur le nombre de tours | Fixer le plancher |
| 1 | Features acoustiques + régression logistique / arbres (scikit-learn) | Savoir ce que la simple acoustique apporte |
| 2 | Embeddings figés + tête de classification | Candidat principal, entraînement rapide |
| 3 | Fine-tuning de l'encodeur (peu de couches, faible taux d'apprentissage) | Gain éventuel, à justifier par MLflow |
| 4 (phase 2) | Fusion audio + texte (transcription automatique) | Option, jamais la dépendance du système |

**Prédiction au niveau de l'appel** (`predict.py`) : l'appel est découpé en segments, chaque segment reçoit un score, puis les scores sont agrégés (moyenne, maximum ou moyenne des k plus hauts : paramètre à comparer). Le seuil de décision est choisi sur la validation, selon le coût des faux négatifs.

**Classes déséquilibrées.** DS04 est équilibré (1 700 / 1 700), mais un déploiement réel ne le sera pas (très peu d'appels frauduleux). L'évaluation inclut donc la précision et le rappel à plusieurs seuils, et un scénario à faible prévalence.

---

## 10. Évaluation

### 10.1 Baselines obligatoires (DS04, split test)

| Règle | Accuracy test |
|---|---|
| Classe majoritaire | 50 % |
| Durée > 21 s = vishing | 80,6 % |
| Nombre de tours > 4 = vishing | **86,7 %** |

Tout modèle doit être comparé à la dernière ligne. À nombre de tours égal (4), la durée ne sépare plus les classes (AUC 0,52).

### 10.2 Rapports obligatoires pour chaque run

1. Global : precision, recall, F1, ROC-AUC, matrice de confusion.
2. **Scams difficiles** : 178 scams à 4 tours sans demande sensible, comparés aux appels normaux de même durée de structure.
3. Par nombre de tours (2 à 11).
4. Par niveau de bruit : propre, 20, 10 et 5 dB.
5. Par langue et par corpus quand plusieurs corpus sont intégrés.
6. Taux de faux négatifs et de faux positifs à seuil fixé.

### 10.3 Généralisation

- **Entraînement sur un corpus, test sur un autre** (cross-corpus) : le seul test fiable quand un corpus est synthétique avec une seule voix TTS.
- Tout résultat sur DS04 seul est présenté comme un résultat « intra-corpus ».

### 10.4 Critère d'acceptation d'un modèle

Un modèle est retenu si, sur le test : il dépasse la meilleure baseline de façon nette, il tient sur les scams difficiles, il reste robuste à 10 dB, et ses résultats sont reproduits par `dvc repro` à partir d'un clone propre.

---

## 11. Suivi des expériences (MLflow)

Chaque run enregistre :

| Élément | Contenu |
|---|---|
| Paramètres | tout `params.yaml`, modèle d'encodeur, seed |
| Versions | hash Git, hash DVC des données et features |
| Métriques | celles de §10.2, préfixées par sous-ensemble |
| Artefacts | modèle, matrice de confusion, courbes ROC, `metrics.json` |
| Tags | corpus d'entraînement, auteur, `baseline` ou `candidate` |

**Sur Kaggle**, MLflow écrit dans `/kaggle/working/mlruns` : exportez ce dossier en fin de session. Pour comparer les runs à trois, un serveur MLflow distant partagé est préférable (par exemple DagsHub). Cette décision est à prendre en équipe (§17).

`params.yaml` (extrait) :

```yaml
seed: 42
prepare:
  label_map: {normal: 0, scam: 1}
segmentation:
  window_sec: 10
  hop_sec: 5
  target_sample_rate: 16000
features:
  n_mfcc: 20
embeddings:
  model_name: "<à choisir>"
  pooling: mean
train:
  head: logistic        # logistic | mlp
  fine_tune: false
  learning_rate: 1.0e-3
  epochs: 20
  aggregation: mean_topk
  topk: 3
```

---

## 12. Workflow Kaggle

1. Créer un notebook Kaggle avec GPU et accès Internet activé.
2. Les identifiants du remote DVC (Hugging Face) sont stockés dans les **Kaggle Secrets**, jamais dans le notebook. Ils sont exportés en variables d'environnement au démarrage.
3. Cellules de `kaggle/train_kaggle.ipynb` :
   ```bash
   git clone https://github.com/<organisation>/vishing-detection.git
   cd vishing-detection && git checkout <branche>
   pip install -e .
   dvc pull data/processed data/features
   python -m vishing_detection.pipeline.run_stage embeddings
   python -m vishing_detection.pipeline.run_stage train
   python -m vishing_detection.pipeline.run_stage evaluate
   dvc push
   ```
4. Récupérer `mlruns/` (ou logguer vers le serveur distant), puis committer `dvc.lock` et `reports/metrics.json` depuis une machine locale.
5. Limite : sessions de durée limitée. Écrire sur disque des points de reprise (`checkpoints/`) et rendre chaque étape idempotente.

---

## 13. Qualité du code et CI

**Tests**

| Fichier | Vérifie |
|---|---|
| `test_manifest.py` | Colonnes du schéma, `sample_id` unique, splits valides, colonnes d'annotation interdites |
| `test_segmentation.py` | Fenêtres et recouvrement, chargeur audio (segment, rééchantillonnage), mélange de bruit au SNR demandé |
| `test_audit.py` | Fuite entre groupes, règles à seuil, rapport de baselines, agrégation des scores, métriques |
| `test_spark_stages.py` | `validate`, `prepare` (segments, héritage du split, refus des fuites), `features_handcrafted`, `baselines`, puis `train` et `evaluate` sur embeddings factices |
| `test_package.py` | Le package s'importe |

**CI** : `.github/workflows/ci.yaml`

```yaml
name: ci
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - uses: actions/setup-java@v4
        with: { distribution: temurin, java-version: "17" }
      - run: pip install -e . pytest
      - run: pytest -q
```

**Conventions** : formatage `ruff` (ou `black`), types sur les fonctions publiques, docstrings en français ou en anglais mais cohérentes dans tout le dépôt, `pre-commit` pour bloquer l'ajout de fichiers `.wav`, `.zip` ou de secrets.

---

## 14. Déploiement (optionnel)

- **API** `deployment/app.py` (FastAPI) :
  - `POST /predict` : reçoit un fichier WAV, renvoie `{score, label, seuil, scores_par_segment}`.
  - `GET /health` : état du service.
- **Conteneur** `deployment/Dockerfile` : image légère avec le package, le modèle de `models/current/` et un `requirements-serve.txt` réduit (sans Spark).
- **Contraintes d'inférence** : même rééchantillonnage et même segmentation qu'à l'entraînement (partagés dans `src/`, jamais recopiés).
- **Surveillance** (si déploiement réel) : distribution des scores, durée des appels, dérive des features, taux d'alertes revues manuellement.
- **Avertissement** : le système n'est pas une garantie de détection. Il doit rester une aide à la décision, et les alertes sont à vérifier.

---

## 15. Sécurité, licences et éthique

- **Licences** : vérifiées par corpus avant intégration (DS04 : CC-BY-4.0, citation obligatoire). La licence est stockée dans le manifest.
- **Secrets** : jamais dans Git. Variables d'environnement ou Kaggle Secrets. `.gitignore` bloque les fichiers de credentials.
- **Contrôle du jeu de données** : l'auteur de DS04 signale dans son rapport de validation un contrôle de sécurité en échec sur la détection de matériel de clé privée (résolution documentée dans ses rapports). À confirmer localement sur chaque corpus téléchargé **avant tout `dvc push`** vers un remote partagé.
- **Données synthétiques** : DS04 est généré par TTS (voix ar-EG, un seul moteur pour les deux classes). Documenter cette limite dans tout résultat publié.
- **Contenu sensible** : les dialogues décrivent des arnaques. Ne pas publier d'audio ni de texte soumis à restriction, et suivre `ethics_statement.md` de chaque corpus.
- **Usage** : recherche académique en cybersécurité.

---

## 16. DVC : ce qui est suivi et comment

| Contenu | Géré par |
|---|---|
| Code, configs, `data/metadata/*.csv`, `reports/`, `dvc.yaml`, `dvc.lock`, `*.dvc` | Git |
| `data/raw/`, `data/interim/`, `data/processed/`, `data/features/`, `models/` | DVC (remote S3 Hugging Face déjà configuré) |
| Modèles pré-entraînés téléchargés (encodeurs) | Cache Hugging Face, hors Git et hors DVC |

Commandes courantes :

```bash
dvc pull                 # récupérer données/modèles correspondant au commit
dvc repro                # rejouer les étapes dont les entrées ont changé
dvc push                 # envoyer les nouveaux artefacts
dvc metrics show         # afficher reports/metrics.json
dvc metrics diff main    # comparer avec la branche main
```

**Disque limité** : ne pas extraire les corpus complets en local. Pour le brut, versionner le zip ou ne conserver que le téléchargement par DOI avec MD5, et laisser Kaggle faire l'extraction.

---

## 17. Organisation de l'équipe et Git

| Rôle | Responsabilités | Modules |
|---|---|---|
| **Data Engineering** | Inventaire, métadonnées, qualité, Spark | `data/`, `spark/`, `tests/test_manifest.py`, `test_spark_stages.py` |
| **Audio Processing** | Chargement, segmentation, features, augmentation | `audio/`, `spark/extract_features.py`, `test_segmentation.py` |
| **ML & MLOps** | Embeddings, entraînement, évaluation, MLflow, déploiement | `models/`, `tracking/`, `kaggle/`, `deployment/`, CI |

**Contrats entre rôles** : le schéma du manifest (§6.3), la table des segments (§6.4) et la clé `sample_id` sont les interfaces. Tout changement passe par une note dans `docs/decisions/` et une PR relue par les deux autres.

**Git**
- `main` : toujours vert (tests passent), protégé.
- Une branche par fonctionnalité (`feature/<sujet>`), merge par pull request après `pytest`.
- Commits courts et explicites. Jamais de WAV, zip ou secret.

**Décisions d'équipe encore ouvertes** : serveur MLflow partagé (oui/non, où), encodeur audio retenu, durée de fenêtre, méthode d'agrégation au niveau appel, stratégie de test cross-corpus.

---

## 17 bis. État du code livré dans ce dépôt

| Élément | État |
|---|---|
| `config`, `data/schema`, `data/audit`, `data/ingestion` (DS04) | Implémentés et testés |
| `spark/` : `session`, `validate`, `prepare`, `extract_features` | Implémentés, testés sur le mini-jeu (Spark local) |
| `audio/` : `loader`, `segmentation`, `features`, `augment` | Implémentés et testés |
| `models/` : `baselines`, `train` (logistique / forêt aléatoire), `evaluate`, `predict` | Implémentés, testés sur embeddings factices |
| `pipeline/run_stage.py`, `dvc.yaml`, `params.yaml`, CI | Écrits, YAML validés ; `dvc repro` à lancer sur ta machine |
| `models/embeddings.py` | **Squelette non testé** : demande `torch`, `transformers` et un GPU (Kaggle) |
| `deployment/app.py` | **Squelette** : la fonction `embed()` doit être branchée sur l'encodeur retenu |
| `kaggle/train_kaggle.ipynb` | Modèle de notebook : renseigner l'URL du dépôt et les Kaggle Secrets |
| Ingestion de DS01–DS03, DS05, DS06 | À écrire après l'audit de chaque corpus |

---

## 18. Feuille de route

| Phase | Contenu | Livrables |
|---|---|---|
| 0. Fondations | Repo, DVC, CI, schéma du manifest | `schema.py`, `ci.yaml`, mini-jeu de test |
| 1. Audit des corpus | DS04 fait ; auditer DS01 à DS06 un par un | `reports/audit_<corpus>.md`, `datasets_catalog.csv` mis à jour |
| 2. Pipeline de données (Spark) | `validate`, `prepare`, `features_handcrafted` | parquet versionnés, tests verts |
| 3. Baselines | Niveaux 0 et 1, premiers runs MLflow | `metrics.json`, tableau des baselines |
| 4. Modèle audio | Embeddings, tête de classification, comparaison d'encodeurs | Runs MLflow comparés |
| 5. Évaluation sérieuse | Sous-ensembles, bruit, cross-corpus | Rapport d'évaluation |
| 6. Export et API | `models/current/`, FastAPI, Docker | Image Docker, démo |
| 7. Phase 2 (optionnelle) | Transcription + fusion texte/audio | Comparaison avec l'audio seul |

---

## 19. Risques connus

| Risque | Effet | Parade |
|---|---|---|
| Raccourcis dans DS04 (nombre de tours, annotations) | Scores gonflés | Baselines, §7.4, évaluation par sous-ensemble |
| Corpus synthétique à voix unique | Mauvaise généralisation | Cross-corpus, augmentation, corpus réels si disponibles |
| Voix TTS ar-EG pour de la darija | Accent éloigné du réel | Documenter, tester sur d'autres corpus |
| Disque local limité | Impossible d'extraire les corpus | Étapes lourdes sur Kaggle, téléchargement par DOI |
| Sessions Kaggle limitées | Entraînement interrompu | Points de reprise, étapes idempotentes |
| Compatibilité Python/PySpark | Spark ne démarre pas | Python 3.11 ou 3.12, Java 17 |
| Fuite entre splits | Résultats trompeurs | Split au niveau de l'appel, tests automatiques |

---

## 20. Annexe — `requirements.txt` cible

```text
# Données et Spark
pandas
numpy
pyarrow
pyspark

# Audio
librosa
soundfile

# ML
scikit-learn
torch
transformers

# Suivi et versionnement
mlflow
dvc
dvc-s3

# Visualisation et rapports
matplotlib
scipy
tabulate

# Configuration
PyYAML

# Déploiement (déjà isolé dans deployment/requirements-serve.txt)
# fastapi, uvicorn, python-multipart

# Tests
pytest
```

---

*À mettre à jour à chaque audit de corpus et à chaque décision d'équipe. Dernier audit pris en compte : DS04 (VISH-DARIJA-TTS).*
