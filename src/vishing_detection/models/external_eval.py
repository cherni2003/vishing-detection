"""Évaluation externe : rappel au seuil figé, comparé à la règle de durée."""
from __future__ import annotations

import json

import joblib
import mlflow
import numpy as np
import pandas as pd
from scipy.stats import binomtest

from vishing_detection.config import get_config, get_params, resolve
from vishing_detection.models.evaluate import call_scores
from vishing_detection.models.frozen import frozen_thresholds


def _recall(hit) -> dict:
    k, n = int(np.sum(hit)), int(len(hit))
    ci = binomtest(k, n).proportion_ci(confidence_level=0.95, method="wilson")
    return {"recall": k / n, "ic95": [ci.low, ci.high], "n": n}


def run(params: dict | None = None) -> dict:
    cfg, params = get_config(), params or get_params()
    ev, ext = params["evaluate"], resolve("data/processed/external")   # À ADAPTER
    model = joblib.load(resolve(cfg["outputs"]["model_dir"]) / "model.joblib")
    t_score, t_dur = frozen_thresholds(model, cfg, params)

    emb = pd.read_parquet(ext / "embeddings.parquet")
    X = np.stack(emb["embedding"].to_numpy()).astype("float32")
    emb = emb.assign(sample_id=emb["call_id"], label=1,
                     score=model.predict_proba(X)[:, 1])
    calls = call_scores(emb, ev["aggregation"], ev.get("topk", 3)).merge(
        pd.read_parquet(ext / "calls.parquet")[["call_id", "duration_sec"]]
        .rename(columns={"call_id": "sample_id"}), on="sample_id")

    short = calls.duration_sec <= t_dur 
    calls.assign(hit=calls.score >= t_score, short=short).to_parquet(
    ext / "scores.parquet", index=False)  # la règle de durée les classe « légitimes » par construction
    res = {"seuil_score": t_score, "seuil_duree_s": t_dur, "n_appels": len(calls),
           "modele": _recall(calls.score >= t_score),
           "regle_duree": _recall(calls.duration_sec > t_dur),
           "modele_appels_courts": _recall(calls.loc[short, "score"] >= t_score) if short.any() else None,
           "part_appels_courts": float(short.mean()),
           "quantiles_score": calls.score.quantile([.05, .25, .5, .75, .95]).round(3).to_dict(),
           "note": "Aucune classe légitime : rappel uniquement, ni AUC ni accuracy."}
    out = resolve(cfg["paths"]["reports_dir"]) / "external_eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    with mlflow.start_run(run_name="external_eval_ftc"):
        mlflow.log_metric("ext_recall_model", res["modele"]["recall"])
        mlflow.log_metric("ext_recall_duree", res["regle_duree"]["recall"])
        mlflow.log_artifact(str(out))
    print("external_eval :", json.dumps(res["modele"]))
    return res