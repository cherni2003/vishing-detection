# Vishing Detection

## 1. Présentation du projet

Le **vishing** (*voice phishing*) est une forme de fraude qui utilise les appels téléphoniques pour tromper les victimes et obtenir des informations personnelles ou bancaires.

Ce projet vise à développer un système capable de détecter les appels frauduleux à partir de fichiers audio.

Le système utilise des techniques de traitement audio et de machine learning pour distinguer deux catégories :

- **Legitimate :** appel normal.
- **Vishing :** appel frauduleux.

Le projet étudie également les données multilingues afin de pouvoir travailler avec différentes langues et dialectes.

## 2. Objectifs

- Collecter et analyser des datasets audio.
- Vérifier la qualité des données et des labels.
- Préparer les fichiers audio pour l'entraînement.
- Extraire des caractéristiques à partir des signaux audio.
- Entraîner et comparer plusieurs modèles de machine learning.
- Évaluer les performances des modèles.
- Améliorer la détection des appels frauduleux.
- Préparer le modèle pour une future utilisation dans une application.

## 3. Technologies utilisées

| Technologie | Utilisation |
|---|---|
| Python | Développement du projet |
| Git et GitHub | Gestion du code et collaboration |
| Apache Spark / PySpark | Préparation et traitement des données |
| Librosa | Traitement audio et extraction MFCC |
| Scikit-learn | Entraînement et évaluation des modèles |
| PyTorch | Extraction des embeddings audio |
| Wav2Vec2 XLS-R | Extraction des caractéristiques audio |
| Kaggle | Utilisation du GPU pour les expériences |
| DVC | Gestion du pipeline et versionnement des données |
| MLflow | Suivi des expériences et des résultats |

## 4. Architecture du repository

```text
vishing-detection/
├── configs/
├── docs/
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   ├── features/
│   └── metadata/
├── notebooks/
├── src/
│   └── vishing_detection/
│       ├── data/
│       ├── audio/
│       ├── spark/
│       ├── models/
│       ├── tracking/
│       └── pipeline/
├── kaggle/
├── tests/
├── models/
├── reports/
├── deployment/
├── .github/workflows/
├── dvc.yaml
├── params.yaml
├── requirements.txt
└── README.md
```

## 5. Pipeline de traitement

Le projet utilise plusieurs étapes pour préparer les données et entraîner les modèles.

### 5.1. Collecte et préparation des données

Plusieurs datasets ont été étudiés :

- KorCCVi
- FTC Robocalls
- TeleAntiFraud-28k
- VISHGUARD

Le dataset **VISHGUARD** a été choisi pour les premières expériences.

Il contient des appels en anglais et en français, avec deux catégories : fraude et appel légitime.

Au total, 2 061 appels ont été utilisés.

### 5.2. Validation des données

Une première étape avec PySpark permet de vérifier :

- Les fichiers audio.
- Les durées des appels.
- Les labels.
- Les doublons.
- Les ensembles d'entraînement, de validation et de test.

### 5.3. Préparation audio

Les fichiers audio sont préparés et découpés en segments de 10 secondes, avec un chevauchement de 5 secondes.

Les fichiers sont rééchantillonnés à 16 kHz.

### 5.4. Extraction des caractéristiques audio

Deux approches ont été testées :

**Approche 1 : MFCC**

Les caractéristiques audio MFCC et d'autres informations acoustiques sont extraites avec Librosa et PySpark.

**Approche 2 : Wav2Vec2 XLS-R**

Le modèle pré-entraîné Wav2Vec2 XLS-R 300M est utilisé pour extraire des embeddings audio.

L'extraction est réalisée sur Kaggle avec un GPU.

### 5.5. Entraînement des modèles

Plusieurs approches ont été testées :

- Classe majoritaire.
- Classification basée sur la durée de l'appel.
- MFCC avec Logistic Regression.
- Embeddings Wav2Vec2 avec Logistic Regression.

Les résultats sont comparés pour identifier l'approche la plus intéressante.

### 5.6. Évaluation

Les modèles sont évalués à l'aide de plusieurs métriques :

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Matrice de confusion

Une analyse supplémentaire est également réalisée pour vérifier l'influence de la durée des appels sur les résultats.

## 6. Résultats préliminaires

Les premières expériences ont été réalisées sur le dataset VISHGUARD.

### 6.1. Informations sur les données

| Élément | Valeur |
|---|---|
| Dataset | VISHGUARD |
| Langues | Anglais et français |
| Nombre d'appels | 2 061 |
| Entraînement | 1 442 |
| Validation | 206 |
| Test | 413 |
| Durée des segments | 10 secondes |
| Chevauchement | 5 secondes |

Les données sont séparées en trois ensembles : entraînement, validation et test.

La séparation est réalisée au niveau de l'appel afin d'éviter que des segments d'un même appel soient présents dans plusieurs ensembles.

### 6.2. Comparaison des modèles

Les résultats obtenus sur les 413 appels de test sont les suivants :

| Modèle | Accuracy | F1-score | AUC |
|---|---:|---:|---:|
| Classe majoritaire | 49,9 % | - | - |
| Durée de l'appel | 95,6 % | - | 0,993 |
| MFCC + Logistic Regression | 85,0 % | 0,861 | 0,936 |
| Wav2Vec2 XLS-R 300M + Logistic Regression | 96,6 % | 0,967 | 0,994 |

Le modèle Wav2Vec2 avec Logistic Regression obtient les meilleurs résultats globaux parmi les approches testées.

### 6.3. Analyse de la durée des appels

Une limite importante a été identifiée pendant les expériences.

Les appels frauduleux sont généralement plus longs que les appels légitimes dans le dataset utilisé.

La durée seule permet donc d'obtenir une accuracy de 95,6 %.

Pour mieux comprendre ce problème, une analyse supplémentaire a été réalisée sur 90 appels dont la durée est proche du seuil de 18,9 secondes.

Les résultats sont les suivants :

| Approche | AUC |
|---|---:|
| Durée seule | 0,883 |
| Wav2Vec2 + Logistic Regression | 0,979 |

Cette expérience montre que le modèle utilise également des informations audio autres que la durée de l'appel.

Cependant, cette analyse reste préliminaire, car elle repose sur un nombre limité d'appels.

## 7. État actuel du projet

Plusieurs étapes importantes ont déjà été réalisées :

- Étude et sélection du dataset VISHGUARD.
- Préparation des données et création du manifest.
- Vérification de la qualité des fichiers audio.
- Création des ensembles Train, Validation et Test.
- Traitement des données avec Apache Spark.
- Extraction des caractéristiques MFCC.
- Création de modèles de référence.
- Extraction des embeddings avec Wav2Vec2 XLS-R 300M sur Kaggle.
- Entraînement de Logistic Regression.
- Évaluation et comparaison des modèles.
- Analyse de l'influence de la durée des appels.
- Intégration des étapes dans le pipeline du projet.
- Suivi des expériences avec MLflow.
- Collaboration avec Git et GitHub.

Le meilleur résultat obtenu jusqu'à présent est de **96,6 % d'accuracy** avec Wav2Vec2 XLS-R 300M et Logistic Regression.

## 8. Limites et prochaines étapes

Malgré les bons résultats, plusieurs améliorations sont nécessaires.

### Limites actuelles

- Le dataset contient uniquement des appels en anglais et en français.
- Les appels sont simulés avec des voix synthétiques.
- Une seule voix est utilisée pour chaque langue.
- Les scripts sont écrits par les mêmes auteurs.
- Les résultats n'ont pas encore été validés sur des appels réels.
- La durée des appels influence fortement les résultats.
- L'analyse à durée contrôlée repose sur seulement 90 appels.


## 9. Collaboration et versionnement

Le projet est développé en équipe avec Git et GitHub.

- La branche `main` contient la version stable du projet.
- Chaque nouvelle fonctionnalité est développée dans une branche séparée.
- Les modifications sont intégrées avec des Pull Requests.
- Les fichiers volumineux ne sont pas ajoutés directement dans Git.
- DVC est utilisé pour organiser les étapes du pipeline et gérer les paramètres.

Le stockage distant DVC reste à configurer pour partager les données et les modèles entre les membres de l'équipe.

## 10. Démarrage rapide

### Prérequis

- Python 3.11 ou 3.12
- Java 17 ou une version supérieure
- Git

### Installation

```bash
python -m venv .venv
```

Sous Windows :

```bash
.venv\Scripts\activate
```

Installer les dépendances :

```bash
pip install -r requirements.txt
pip install -e .
```

Lancer les tests :

```bash
pytest -q
```

### Exécution du pipeline

Pour lancer une étape :

```bash
python -m vishing_detection.pipeline.run_stage <etape>
```

Étapes disponibles :

- `validate`
- `prepare`
- `features_handcrafted`
- `baselines`
- `embeddings`
- `train`
- `evaluate`
- `analyse_duree`

Pour exécuter le pipeline avec DVC :

```bash
dvc repro
```

Les étapes d'extraction des embeddings nécessitent un environnement avec GPU, comme Kaggle.

Pour plus de détails sur l'architecture et l'organisation du projet, consultez [docs/architecture.md](docs/architecture.md).

## 11. Avertissement

Ce projet est réalisé dans un cadre académique et de recherche en cybersécurité.

Les résultats dépendent de la qualité et des caractéristiques des données utilisées.

Le système ne garantit pas une détection parfaite de toutes les tentatives de vishing et nécessite des tests supplémentaires sur des appels réels avant toute utilisation pratique.
