"""Entraînement d'une tête de classification sur les embeddings (niveau 2)."""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from vishing_detection.config import get_config, get_params, resolve
from vishing_detection.data.schema import forbidden_columns_present
from vishing_detection.tracking.mlflow_utils import log_metrics, start_run


def load_xy(embeddings_path, segments_path, split: str):
    emb = pd.read_parquet(embeddings_path)
    seg = pd.read_parquet(segments_path)[["segment_id", "label", "split"]]
    df = emb.merge(seg, on="segment_id")
    df = df[df["split"] == split]
    return df, np.stack(df["embedding"].to_numpy()), df["label"].to_numpy()


def build_model(train_params: dict, seed: int):
    if train_params.get("head", "logistic") == "rf":
        return RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=train_params.get("C", 1.0),
                           max_iter=train_params.get("max_iter", 1000), random_state=seed),
    )


def run(params: dict | None = None) -> None:
    cfg, params = get_config(), params or get_params()
    seed = int(params["seed"])
    emb_path, seg_path = resolve(cfg["outputs"]["embeddings"]), resolve(cfg["outputs"]["segments"])

    assert not forbidden_columns_present(pd.read_parquet(emb_path).columns), \
        "colonnes d'annotation interdites dans les features"

    _, X_tr, y_tr = load_xy(emb_path, seg_path, "train")
    _, X_va, y_va = load_xy(emb_path, seg_path, "validation")
    model = build_model(params["train"], seed).fit(X_tr, y_tr)
    val_auc = float(roc_auc_score(y_va, model.predict_proba(X_va)[:, 1]))

    out = resolve(cfg["outputs"]["model_dir"])
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out / "model.joblib")
    (out / "meta.json").write_text(json.dumps({"params": params["train"], "val_auc_segment": val_auc}, indent=2))

    with start_run("train", params=params["train"], tags={"stage": "train"}) as mlflow:
        log_metrics(mlflow, {"val_auc_segment": val_auc})
    print(f"train : AUC validation (segments) = {val_auc:.3f} -> {out}")
