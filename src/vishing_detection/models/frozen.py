"""Appels de test et seuils figés, calculés exactement comme dans analyse_duree."""
from __future__ import annotations

import pandas as pd

from vishing_detection.config import resolve
from vishing_detection.data.audit import fit_threshold_rule
from vishing_detection.models.analyse_duree import best_threshold
from vishing_detection.models.evaluate import call_scores
from vishing_detection.models.train import load_xy


def load_calls(model, cfg, params, split: str) -> pd.DataFrame:
    ev = params["evaluate"]
    df, X, _ = load_xy(resolve(cfg["outputs"]["embeddings"]),
                       resolve(cfg["outputs"]["segments"]), split)
    df = df.assign(score=model.predict_proba(X)[:, 1])
    man = pd.read_parquet(resolve(cfg["outputs"]["manifest"]))
    return call_scores(df, ev["aggregation"], ev.get("topk", 3)).merge(
        man[["sample_id", "duration_sec", "language"]], on="sample_id")


def frozen_thresholds(model, cfg, params) -> tuple[float, float]:
    """(seuil de score choisi sur la validation, seuil de durée appris sur le train)."""
    t_score = best_threshold(load_calls(model, cfg, params, "validation"))
    man = pd.read_parquet(resolve(cfg["outputs"]["manifest"]))
    tr = man[man["split"] == "train"]
    t_dur, _ = fit_threshold_rule(tr["duration_sec"].to_numpy(), tr["label"].to_numpy())
    return t_score, t_dur