# Modèles

Les modèles entraînés ne sont **pas** versionnés dans Git : ils sont suivis par DVC (`dvc.yaml`, sortie `models/latest`).

- `models/latest/` : dernier modèle entraîné (`model.joblib`, `meta.json`).
- Pour récupérer le modèle correspondant au commit courant : `dvc pull models/latest`.
- Les runs et leurs métriques sont tracés dans MLflow ; les métriques finales dans `reports/metrics.json`.
