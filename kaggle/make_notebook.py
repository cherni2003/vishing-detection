"""Régénère kaggle/train_kaggle.ipynb. Usage : python kaggle/make_notebook.py"""
import json
from pathlib import Path

cells = [
    ("markdown", "# Entraînement sur Kaggle (GPU)\nActiver le GPU et Internet dans les paramètres du notebook. "
                 "Les identifiants du remote DVC sont dans les **Kaggle Secrets** (jamais dans ce notebook)."),
    ("code", "import os\nfrom kaggle_secrets import UserSecretsClient\n"
             "s = UserSecretsClient()\n"
             "os.environ['AWS_ACCESS_KEY_ID'] = s.get_secret('AWS_ACCESS_KEY_ID')\n"
             "os.environ['AWS_SECRET_ACCESS_KEY'] = s.get_secret('AWS_SECRET_ACCESS_KEY')"),
    ("code", "BRANCH = 'main'   # branche à exécuter\n"
             "!git clone https://github.com/<organisation>/vishing-detection.git /kaggle/working/vishing-detection\n"
             "%cd /kaggle/working/vishing-detection\n"
             "!git checkout {BRANCH}\n"
             "!pip install -q -e . dvc dvc-s3 mlflow pyspark"),
    ("code", "!dvc pull data/processed data/metadata"),
    ("code", "os.environ['MLFLOW_TRACKING_URI'] = 'file:///kaggle/working/mlruns'   # ou serveur distant partagé\n"
             "!python -m vishing_detection.pipeline.run_stage embeddings"),
    ("code", "!python -m vishing_detection.pipeline.run_stage train\n"
             "!python -m vishing_detection.pipeline.run_stage evaluate"),
    ("code", "!dvc push\n!cat reports/metrics.json"),
    ("markdown", "Ensuite, en local : récupérer `mlruns/` si besoin, puis committer `dvc.lock` et `reports/metrics.json`."),
]

def cell(kind, src):
    c = {"cell_type": kind, "metadata": {}, "source": src.splitlines(keepends=True)}
    if kind == "code":
        c.update({"execution_count": None, "outputs": []})
    return c

nb = {"cells": [cell(*c) for c in cells],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
Path(__file__).with_name("train_kaggle.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("notebook écrit")
