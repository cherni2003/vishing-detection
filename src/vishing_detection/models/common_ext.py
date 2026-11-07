"""Chemins et utilitaires communs aux étapes de validation externe."""
from pathlib import Path

import numpy as np
import yaml

PROCESSED = Path("data/processed")
EXT = PROCESSED / "external"
REPORTS = Path("reports")
MODEL_PATH = Path("models/model.joblib")           # À ADAPTER (sortie de train)
EMBEDDINGS = PROCESSED / "embeddings.parquet"      # À ADAPTER (call_id, split, label, embedding)
CALLS_MANIFEST = PROCESSED / "manifest.parquet"    # À ADAPTER (call_id, duration_sec)


def load_params() -> dict:
    with open("params.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def stack(df, col: str = "embedding") -> np.ndarray:
    return np.stack(df[col].to_numpy()).astype("float32")