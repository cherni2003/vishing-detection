# Audit du corpus VISH-DARIJA-TTS (DS04)

Source : https://zenodo.org/records/20039126 (v1, licence CC-BY-4.0). Corpus **synthétique** (voix TTS générées).

## 1. Volumétrie
- 3400 fichiers audio propres (22.6 h, 3.91 Go) et 10200 variantes bruitées (3 par fichier : SNR 20, 10 et 5 dB).
- 17351 tours de parole, avec texte latin et texte arabe alignés.
- Identifiant fiable : `dataset_uid`. Le `scenario_id` seul n'est pas unique entre classes (normal_0103 et scam_0103 partagent le même).

## 2. Langues
- Darija marocaine, en écriture latine (100.0% des tours ont du texte latin) et arabe (100.0% ont du texte arabe).
- Les voix sont générées avec le code de langue `ar-EG` (arabe égyptien) : accent éloigné de la darija réelle.

## 3. Labels et déséquilibre des classes
- Équilibre parfait : 1700 scam / 1700 normal, et identique dans chaque split.

| split      |   normal |   scam |
|:-----------|---------:|-------:|
| test       |      255 |    255 |
| train      |     1190 |   1190 |
| validation |      255 |    255 |

- Aucun `dataset_uid` n'apparaît dans plusieurs splits. Les variantes bruitées restent dans le split de leur fichier propre.

## 4. Qualité audio
- Tous les fichiers sont lisibles, en 24 kHz, mono, 16 bits. Aucun doublon de hash.
- Vérification directe sur 300 fichiers : formats et durées conformes au manifest, aucun écrêtage, niveau sonore identique entre classes (AUC du volume : 0,47).
- Les 10 200 variantes bruitées sont équilibrées par classe et par niveau de SNR.

## 5. Raccourcis dans les données (risque de surévaluation)
| Règle triviale | Accuracy sur test |
|---|---|
| Classe majoritaire | 50 % |
| Durée > 21 s = scam | 80.6% |
| Nombre de tours > 4 = scam | 86.7% |

- Les appels normaux ont presque tous 4 tours. À nombre de tours égal, la durée ne sépare plus les classes (AUC 0,52).
- **178 scams « difficiles »** (4 tours, sans demande sensible) : ils ressemblent aux appels normaux par la structure.
- Colonnes qui donnent directement le label, à ne jamais utiliser comme features : `speaker_role`, `gender`, `contains_sensitive_request`, `request_type`, `persuasion_strategy`. Par précaution : `attack_type`, `requested_asset`, `impersonated_entity`, `victim_role`, `risk_level`.

## 6. Qualité du texte
- 9384 lignes sont marquées `needs_manual_review` : le détecteur de l'auteur signale tout texte contenant un `?`. Ce sont des questions légitimes (faux positifs), non des caractères détruits.
- Exception réelle : 6 textes arabes (scams 324 et 333, tours 0 à 2) sont remplacés par des `?`.

## 7. Limites
- Données synthétiques : pas de garantie de généralisation à de vrais appels.
- Une seule voix TTS (même modèle pour les deux classes) : possible sur-apprentissage du style de voix.
- Les splits fournis sont utilisés tels quels.

## 8. Protocole d'évaluation recommandé
1. Baseline obligatoire : règle sur le nombre de tours (et sur la durée).
2. Métriques globales et sur le sous-ensemble des scams difficiles.
3. Résultats par nombre de tours et par niveau de bruit.
4. Aucune colonne d'annotation du scénario en entrée du modèle.
