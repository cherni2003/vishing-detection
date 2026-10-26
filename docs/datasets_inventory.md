# Inventaire des datasets — Vishing Detection

## 1. Objectif

Ce document centralise les sources de données envisagées pour le projet de détection du voice phishing (vishing).

Chaque source doit être vérifiée avant son intégration au pipeline : disponibilité des fichiers, format, langues, labels, licence, volume et caractéristiques des enregistrements.

**Important :** la présence d'une source dans cet inventaire ne signifie pas que ses données ont déjà été téléchargées, validées ou approuvées pour l'entraînement.

## 2. Sources identifiées

| ID   | Source                                                                                                          | Langues ou dialectes annoncés       | Statut initial |
| ---- | --------------------------------------------------------------------------------------------------------------- | ----------------------------------- | -------------- |
| DS01 | [NCSU Technical Report](https://techrep.csc.ncsu.edu/2023/TR-2023-1.pdf)                                        | Chinois et anglais                  | À auditer      |
| DS02 | [Dataset Zenodo — 20429763](https://zenodo.org/records/20429763)                                                | Français, anglais et arabe standard | À auditer      |
| DS03 | [Korean Voice Phishing Detection](https://github.com/selfcontrol7/Korean_Voice_Phishing_Detection)              | Coréen                              | À auditer      |
| DS04 | [Dataset Zenodo — 20039126](https://zenodo.org/records/20039126)                                                | Darija marocaine                    | À auditer      |
| DS05 | [Hybrid Voice Phishing Detection System](https://github.com/kishore2k05/Hybrid-Voice-Phishing-Detection-System) | À vérifier                          | À auditer      |
| DS06 | [Dataset Mendeley — p384bgyzz3](https://data.mendeley.com/datasets/p384bgyzz3/4)                                | Dialectes arabes                    | À auditer      |

## 3. Dataset exclu de la phase actuelle

Le dataset suivant est volontairement exclu du périmètre actuel :

* [Dataset Zenodo — 8342762](https://zenodo.org/records/8342762)
* Description provisoire : données de parole normale en tunisien, sans exemples de vishing selon l'inventaire initial.
* Décision : ne pas l'intégrer à la phase actuelle.
* Réévaluation éventuelle : une phase ultérieure, après définition du protocole expérimental et des données négatives nécessaires.

## 4. Informations à vérifier pour chaque source

Pour chaque dataset, documenter les éléments suivants :

* Disponibilité et méthode d'accès aux fichiers.
* Licence et conditions d'utilisation.
* Nombre d'enregistrements et durée totale.
* Formats audio et fréquences d'échantillonnage.
* Langues et dialectes réellement présents.
* Labels originaux et leur signification.
* Présence de données frauduleuses et légitimes.
* Présence de données synthétiques ou réelles, si connue.
* Disponibilité des identifiants de locuteurs, conversations ou scénarios.
* Doublons, données manquantes et limites connues.

## 5. Règles d'intégration

1. Ne pas intégrer un dataset avant d'avoir vérifié sa licence et ses conditions d'utilisation.
2. Conserver les labels originaux pour assurer la traçabilité.
3. Harmoniser les labels uniquement après compréhension de leur signification.
4. Ne pas supposer qu'un dataset contient des appels légitimes simplement parce qu'il contient des fichiers audio.
5. Éviter les fuites de données entre entraînement, validation et test.
6. Documenter les différences entre les sources afin d'identifier les biais potentiels.
7. Ne pas publier de données sensibles ou de fichiers audio soumis à des restrictions.

## 6. Statuts possibles

* `À auditer` : source identifiée, mais pas encore vérifiée.
* `En cours` : vérification ou préparation commencée.
* `Validé` : licence, fichiers, labels et caractéristiques principales vérifiés.
* `Intégré` : dataset préparé et référencé dans le manifest commun.
* `Exclu` : source non utilisée dans la phase actuelle.

## 7. Prochaine étape

Auditer les sources une par une, puis créer un manifest commun dans `data/metadata/audio_manifest.csv` pour référencer les échantillons effectivement retenus.

Les informations de cet inventaire devront être mises à jour à mesure que les vérifications sont réalisées.
