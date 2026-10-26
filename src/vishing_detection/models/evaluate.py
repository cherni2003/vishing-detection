"""Évaluation : global + sous-ensembles (scams difficiles, nombre de tours...)."""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)

from vishing_detection.config import get_config, get_params, resolve
from vishing_detection.models.predict import aggregate_scores
from vishing_detection.models.train import load_xy
from vishing_detection.tracking.mlflow_utils import log_metrics, start_run


def compute_metrics(y_true, y_score, threshold: float = 0.5) -> dict:
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out = {
        "n": int(len(y_true)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float((y_true == y_pred).mean()),
        "fn": int(fn), "fp": int(fp),
    }
    if len(np.unique(y_true)) == 2:
        out["roc_auc"] = float(roc_auc_score(y_true, y_score))
    return out


def call_scores(df: pd.DataFrame, method: str, topk: int) -> pd.DataFrame:
    """Score d'appel = agrégation des scores de ses segments."""
    g = df.groupby("sample_id")
    return pd.DataFrame({
        "score": g["score"].apply(lambda s: aggregate_scores(s, method, topk)),
        "label": g["label"].first(),
    }).reset_index()


def run(params: dict | None = None) -> dict:
    cfg, params = get_config(), params or get_params()
    ev = params["evaluate"]
    model = joblib.load(resolve(cfg["outputs"]["model_dir"]) / "model.joblib")
    df, X, _ = load_xy(resolve(cfg["outputs"]["embeddings"]), resolve(cfg["outputs"]["segments"]), ev["split"])
    df = df.assign(score=model.predict_proba(X)[:, 1])
    calls = call_scores(df, ev["aggregation"], ev.get("topk", 3))

    result = {"global": compute_metrics(calls["label"], calls["score"], ev["threshold"])}

    audit_path = resolve(cfg["data"]["audit_features_path"])
    if audit_path.exists():
        audit = pd.read_csv(audit_path).rename(columns={"dataset_uid": "sample_id"})
        calls = calls.merge(audit, on="sample_id", how="left", suffixes=("", "_audit"))
        subsets = {}
        if "hard_scam" in calls:
            mask = (calls["hard_scam"] == True) | (calls["label"] == 0)   # noqa: E712
            subsets["hard_scam_vs_normal"] = calls[mask]
        if "n_turns" in calls:
            for n, sub in calls.groupby("n_turns"):
                if len(sub) >= 20:
                    subsets[f"n_turns_{int(n)}"] = sub
        result["subsets"] = {k: compute_metrics(s["label"], s["score"], ev["threshold"])
                             for k, s in subsets.items()}

    out = resolve(cfg["outputs"]["metrics"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    with start_run("evaluate", params=ev, tags={"stage": "evaluate"}) as mlflow:
        log_metrics(mlflow, result["global"], prefix="global")
    print("evaluate :", json.dumps(result["global"]))
    return result
