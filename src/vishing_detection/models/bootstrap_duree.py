"""IC 95 % par bootstrap apparié, stratifié par classe : modèle vs durée seule."""
from __future__ import annotations

import json

import joblib
import mlflow
import numpy as np
from sklearn.metrics import roc_auc_score

from vishing_detection.config import get_config, get_params, resolve
from vishing_detection.models.frozen import frozen_thresholds, load_calls

NAMES = ["auc_modele", "auc_duree", "auc_diff", "acc_modele", "acc_duree", "acc_diff"]


def _bootstrap(d, t_score, t_dur, n_iter, ci, rng) -> dict:
    y, s, dur = d.label.to_numpy(), d.score.to_numpy(), d.duration_sec.to_numpy()
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    draws = np.empty((n_iter, 6))
    for b in range(n_iter):
        i = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        am, ad = roc_auc_score(y[i], s[i]), roc_auc_score(y[i], dur[i])
        cm, cd = ((s[i] >= t_score) == y[i]).mean(), ((dur[i] > t_dur) == y[i]).mean()
        draws[b] = (am, ad, am - ad, cm, cd, cm - cd)
    q = [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100]
    lo, hi = np.percentile(draws, q, axis=0)
    out = {n: {"ic": [round(float(a), 4), round(float(b), 4)], "moyenne": round(float(m), 4)}
           for n, a, b, m in zip(NAMES, lo, hi, draws.mean(axis=0))}
    out |= {"n_appels": int(len(y)), "n_fraudes": int(y.sum())}
    return out


def run(params: dict | None = None) -> dict:
    cfg, params = get_config(), params or get_params()
    B = params["bootstrap"]
    model = joblib.load(resolve(cfg["outputs"]["model_dir"]) / "model.joblib")
    t_score, t_dur = frozen_thresholds(model, cfg, params)
    te = load_calls(model, cfg, params, "test")
    band = te[(te.duration_sec - t_dur).abs() <= B["band_halfwidth"]]
    rng = np.random.default_rng(params["seed"])

    res = {"seuil_score": t_score, "seuil_duree_s": t_dur,
           "bande": _bootstrap(band, t_score, t_dur, B["n_iter"], B["ci"], rng),
           "test_complet": _bootstrap(te, t_score, t_dur, B["n_iter"], B["ci"], rng)}
    out = resolve(cfg["paths"]["reports_dir"]) / "bootstrap.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    with mlflow.start_run(run_name="bootstrap_duree"):
        mlflow.log_artifact(str(out))
        mlflow.log_metric("band_auc_diff_lo", res["bande"]["auc_diff"]["ic"][0])
    print("bootstrap_duree :", json.dumps(res["bande"]["auc_diff"]))
    return res