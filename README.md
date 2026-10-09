# Vishing Detection

## 1. Présentation du projet

Le **vishing** (*voice phishing*) est une forme de cyberattaque dans laquelle un attaquant utilise un appel vocal pour tromper une victime, obtenir des informations confidentielles ou l'inciter à effectuer une action frauduleuse.

Ce projet vise à étudier et développer un système de **détection du vishing à partir de données audio**, en exploitant des techniques de traitement du signal et de machine learning.

Le projet s'appuie sur plusieurs sources de données multilingues. Une attention particulière sera accordée à la qualité des données, à l'harmonisation des labels et à l'évaluation des modèles sur différentes langues et dialectes.

## 2. Objectifs

* Collecter et analyser des datasets audio adaptés à la détection du vishing.
* Vérifier la qualité, les labels, les formats et les licences des données.
* Construire un pipeline commun de préparation et de traitement des données audio.
* Extraire des caractéristiques pertinentes à partir des signaux audio.
* Entraîner et comparer plusieurs modèles de machine learning.
* Évaluer les performances à l'aide de métriques adaptées à la détection des fraudes.
* Mettre en place des pratiques MLOps pour améliorer la traçabilité et la reproductibilité des expériences.
* Étudier la possibilité de déployer le modèle sélectionné.

## 3. Technologies envisagées

| Technologie            | Utilisation prévue                                           |
| ---------------------- | ------------------------------------------------------------ |
| Python                 | Développement du pipeline et des modèles                     |
| Git et GitHub          | Gestion du code et collaboration                             |
| Apache Spark / PySpark | Traitement distribué des métadonnées et contrôles de qualité |
| DVC                    | Versionnement des données et des artefacts volumineux        |
| Librosa et SoundFile   | Chargement et traitement audio                               |
| Scikit-learn           | Modèles de référence et évaluation                           |
| Kaggle                 | Environnement d'expérimentation et d'entraînement            |
| MLflow                 | Suivi des expériences et des métriques                       |
| Pytest                 | Tests automatisés                                            |
| FastAPI et Docker      | Déploiement éventuel du modèle                               |

Ces technologies constituent la stack envisagée. Leur utilisation effective sera confirmée au fur et à mesure de l'avancement du projet.

## 4. Architecture du repository

```text
vishing-detection/
├── configs/                 # Configurations du projet
├── docs/                    # Documentation et décisions techniques
├── data/
│   ├── raw/                 # Données originales
│   ├── interim/             # Données intermédiaires
│   ├── processed/           # Données nettoyées et préparées
│   ├── features/            # Caractéristiques audio extraites
│   └── metadata/            # Manifest et rapports de qualité
├── notebooks/               # Exploration et expérimentations
├── src/
│   └── vishing_detection/
│       ├── data/             # Ingestion et validation des données
│       ├── audio/            # Prétraitement et extraction des features
│       ├── spark/            # Traitements PySpark
│       ├── models/           # Entraînement, prédiction et évaluation
│       ├── tracking/         # Suivi des expériences
│       └── pipeline/         # Orchestration du pipeline
├── tests/                    # Tests automatisés
├── models/                   # Documentation et références des modèles
├── reports/                  # Rapports, métriques et figures
├── deployment/               # Éléments éventuels de déploiement
├── .github/workflows/        # Automatisation des tests
├── requirements.txt          # Dépendances Python
├── .gitignore                # Fichiers exclus de Git
└── README.md                 # Présentation du projet
```

## 5. Pipeline envisagé

Le pipeline de traitement suivra progressivement les étapes suivantes :

1. **Collecte et audit :** inventaire des datasets, vérification des licences, formats, langues et labels.
2. **Validation :** contrôle des fichiers audio, des métadonnées, des doublons et des données manquantes.
3. **Prétraitement :** préparation des fichiers audio et harmonisation des labels.
4. **Extraction des caractéristiques :** calcul de caractéristiques audio ou utilisation de représentations apprises.
5. **Entraînement :** création d'un modèle de référence et comparaison de plusieurs approches.
6. **Évaluation :** analyse des performances globales et par langue ou source de données.
7. **Versionnement et traçabilité :** conservation des versions des données, paramètres et résultats.
8. **Export et intégration :** préparation du modèle pour la prédiction et, si nécessaire, son déploiement.

## 6. Évaluation des modèles

Les modèles seront évalués à l'aide de plusieurs métriques, notamment :

* Precision
* Recall
* F1-score
* Matrice de confusion
* ROC-AUC, lorsque cette métrique est applicable

Une attention particulière sera portée aux faux négatifs et aux faux positifs. Les résultats seront également analysés par langue et par source de données lorsque les volumes disponibles le permettent.

La séparation entre les ensembles d'entraînement, de validation et de test devra limiter les fuites de données, notamment entre segments provenant d'un même locuteur ou d'une même conversation.

## 7. Organisation du travail

Le projet est développé par une équipe de trois membres. Le travail sera réparti entre :

* **Data Engineering :** inventaire des datasets, métadonnées, qualité des données et organisation du pipeline.
* **Audio Processing :** chargement, prétraitement et extraction des caractéristiques audio.
* **Machine Learning & MLOps :** entraînement, évaluation, suivi des expériences et préparation du modèle final.

Les interfaces et formats communs seront définis pour permettre l'intégration des contributions.

## 8. Collaboration et versionnement

Le code source est géré avec Git et GitHub.

* La branche `main` est destinée à conserver une version stable du projet.
* Les nouvelles fonctionnalités sont développées sur des branches dédiées.
* Les modifications sont intégrées à `main` après vérification.
* Les datasets volumineux ne doivent pas être ajoutés directement au dépôt Git.
* DVC pourra être utilisé pour versionner les données et les artefacts associés, avec un espace de stockage partagé configuré pour l'équipe.

## 9. État actuel du projet

Le projet est en phase de préparation de l'architecture et d'audit des sources de données. Les modèles, les caractéristiques audio définitives et les étapes d'entraînement seront sélectionnés après l'analyse des datasets disponibles.

## 10. Avertissement

Ce projet est réalisé à des fins académiques et de recherche en cybersécurité. Les résultats dépendront de la qualité, de la représentativité et des limites des données utilisées. Le système développé ne doit pas être considéré comme une garantie absolue de détection du vishing.
