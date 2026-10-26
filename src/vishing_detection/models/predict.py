"""Agrégation des scores de segments en score d'appel."""
from __future__ import annotations

import numpy as np


def aggregate_scores(scores, method: str = "mean", topk: int = 3) -> float:
    s = np.asarray(scores, dtype=float)
    if s.size == 0:
        raise ValueError("aucun score à agréger")
    if method == "mean":
        return float(s.mean())
    if method == "max":
        return float(s.max())
    if method == "topk":
        return float(np.sort(s)[-topk:].mean())
    raise ValueError(f"méthode inconnue : {method}")
