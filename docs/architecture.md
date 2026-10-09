# Architecture technique — Vishing Detection

## 1. Vue d'ensemble

Le projet Vishing Detection vise à développer un pipeline de détection du voice phishing à partir de données audio multilingues.

L'architecture est organisée en plusieurs modules indépendants afin de faciliter le développement en équipe, les tests, la reproductibilité des expériences et l'évolution du système.

## 2. Architecture logique

```text
Sources de données audio
          |
          v
Ingestion et inventaire des datasets
          |
          v
Validation et contrôle qualité
          |
          v
Harmonisation des labels et métadonnées
          |
          v
Prétraitement audio
          |
          v
Extraction des caractéristiques
          |
          v
Entraînement et comparaison des modèles
          |
          v
Évaluation sur des données indépendantes
          |
          v
Export du modèle et prédiction
```

## 3. Organisation des composants

### 3.1 Data Engineering

Responsabilités :

* Inventorier les datasets et documenter leurs sources.
* Vérifier les formats, les licences, les labels et les métadonnées disponibles.
* Construire un manifest commun.
* Détecter les fichiers manquants, les doublons et les valeurs invalides.
* Préparer les données pour les étapes suivantes.

Apache Spark / PySpark pourra être utilisé pour les traitements distribués de métadonnées, les contrôles de qualité et les agrégations lorsque le volume le justifie.

### 3.2 Audio Processing

Responsabilités :

* Charger les fichiers audio.
* Contrôler la durée, la fréquence d'échantillonnage et le nombre de canaux.
* Appliquer les transformations nécessaires.
* Segmenter les enregistrements si cela est pertinent.
* Extraire les caractéristiques audio requises par le modèle.

Les bibliothèques Python seront sélectionnées en fonction des formats et des modèles retenus.

### 3.3 Machine Learning

Responsabilités :

* Construire une baseline.
* Entraîner et comparer les modèles candidats.
* Enregistrer les paramètres et les résultats des expériences.
* Évaluer les performances globales et par langue ou source.
* Analyser les erreurs et sélectionner un modèle final.

L'entraînement pourra être réalisé sur Kaggle en fonction des ressources nécessaires.

### 3.4 MLOps

Les pratiques MLOps envisagées sont :

* Git et GitHub pour le versionnement du code et la collaboration.
* DVC pour le versionnement des données et des artefacts volumineux.
* MLflow pour le suivi des expériences et des métriques.
* Des tests automatisés pour vérifier les composants critiques.
* Une procédure d'export permettant de reproduire les prédictions.

Le déploiement via FastAPI et Docker sera envisagé si le périmètre du projet le nécessite.

## 4. Gestion des données

Les données seront organisées en plusieurs niveaux :

* `data/raw/` : données originales.
* `data/interim/` : données intermédiaires.
* `data/processed/` : données validées et préparées.
* `data/features/` : caractéristiques extraites.
* `data/metadata/` : manifest, inventaire et rapports de qualité.

Les fichiers audio ne seront pas versionnés directement dans Git. Un stockage adapté sera utilisé pour les datasets et les artefacts volumineux.

Chaque échantillon devra être associé, lorsque l'information est disponible, à son dataset source, sa langue, son label, son locuteur ou sa conversation, et son statut synthétique ou réel.

## 5. Harmonisation des labels

Les labels originaux des datasets peuvent être différents. Une étape explicite d'harmonisation sera donc nécessaire.

Les catégories communes seront définies après inspection des données. Un dataset contenant uniquement des exemples frauduleux ne sera pas considéré comme suffisant pour apprendre, à lui seul, la distinction entre appels frauduleux et appels légitimes.

Les données dont les labels sont ambigus ou non vérifiables devront être documentées avant leur utilisation.

## 6. Stratégie d'évaluation

Le jeu de test devra rester indépendant des données utilisées pour l'entraînement et le choix du modèle.

Lorsque les métadonnées le permettent, les échantillons issus d'un même locuteur, d'une même conversation ou d'un même scénario seront regroupés dans un seul split afin de limiter les fuites de données.

Les métriques envisagées comprennent :

* Precision
* Recall
* F1-score
* Matrice de confusion
* ROC-AUC lorsque cela est pertinent

Les performances devront être examinées globalement et par langue ou source lorsque les données disponibles permettent une comparaison fiable.

## 7. Flux de travail de l'équipe

Le travail sera développé sur des branches Git dédiées.

Chaque modification devra être testée avant d'être intégrée dans `main`. Les interfaces communes, notamment le format du manifest, les labels harmonisés et les entrées attendues par le modèle, seront documentées afin de faciliter l'intégration entre les membres.

## 8. Principes de conception

L'architecture suivra les principes suivants :

* Modularité : chaque composant aura une responsabilité claire.
* Reproductibilité : les configurations et les expériences seront documentées.
* Traçabilité : les versions des données, du code et du modèle seront identifiables.
* Qualité : les composants critiques seront testés.
* Sécurité : les secrets et données sensibles ne seront pas publiés dans le dépôt.
* Évolutivité : les outils distribués seront utilisés lorsque le volume de données le justifie.

## 9. État de l'architecture

Cette architecture est une proposition initiale. Les choix définitifs de modèles, de caractéristiques audio, de stockage DVC et de déploiement seront confirmés après l'audit des datasets et les premières expérimentations.
