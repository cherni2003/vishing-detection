"""Analyse à durée contrôlée : le modèle fait-il mieux que la durée de l'appel ?

Étape `analyse_duree` -> reports/analyse_duree.json. Compare le modèle à la règle
« durée > seuil » (seuil appris sur le train), globalement et dans une bande de durées
autour de ce seuil, là où la durée seule hésite.
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from vishing_detection.config import get_config, get_params, resolve
from vishing_detection.data.audit import fit_threshold_rule
from vishing_detection.models.evaluate import call_scores
from vishing_detection.models.train import load_xy


def best_threshold(calls: pd.DataFrame) -> float:
    """Seuil de score maximisant l'accuracy (à calculer sur la validation)."""
    ths = np.linspace(0.05, 0.95, 91)
    return float(ths[np.argmax([accuracy_score(calls.label, calls.score >= x) for x in ths])])


def band_analysis(calls: pd.DataFrame, duration_threshold: float, score_threshold: float,
                  widths=(2, 3, 4, 6), min_calls: int = 30) -> dict:
    """Modèle vs durée seule, dans la plus petite bande ±W autour du seuil de durée
    qui contient au moins `min_calls` appels des deux classes."""
    for w in widths:
        b = calls[(calls.duration_sec - duration_threshold).abs() <= w]
        if len(b) >= min_calls and b.label.nunique() == 2:
            return {"largeur_s": w, "n_appels": int(len(b)), "n_fraude": int(b.label.sum()),
                    "auc_modele": float(roc_auc_score(b.label, b.score)),
                    "auc_duree": float(roc_auc_score(b.label, b.duration_sec)),
                    "acc_modele": float(accuracy_score(b.label, b.score >= score_threshold)),
                    "acc_regle_duree": float(accuracy_score(b.label, b.duration_sec > duration_threshold))}
    return {"erreur": f"moins de {min_calls} appels dans les bandes testées"}


def run(params: dict | None = None) -> dict:
    cfg, params = get_config(), params or get_params()
    ev, an = params["evaluate"], params.get("analyse", {})
    model = joblib.load(resolve(cfg["outputs"]["model_dir"]) / "model.joblib")
    man = pd.read_parquet(resolve(cfg["outputs"]["manifest"]))

    def calls(split: str) -> pd.DataFrame:
        df, X, _ = load_xy(resolve(cfg["outputs"]["embeddings"]), resolve(cfg["outputs"]["segments"]), split)
        df = df.assign(score=model.predict_proba(X)[:, 1])
        return call_scores(df, ev["aggregation"], ev.get("topk", 3)).merge(
            man[["sample_id", "duration_sec", "language"]], on="sample_id")

    va, te = calls("validation"), calls("test")
    train = man[man["split"] == "train"]
    thr_dur, _ = fit_threshold_rule(train["duration_sec"].to_numpy(), train["label"].to_numpy())
    t = best_threshold(va)

    res = {"seuil_score_validation": t, "seuil_duree_s": thr_dur,
           "test_accuracy": float(accuracy_score(te.label, te.score >= t)),
           "test_f1": float(f1_score(te.label, te.score >= t)),
           "test_accuracy_regle_duree": float(accuracy_score(te.label, te.duration_sec > thr_dur)),
           "test_auc_modele": float(roc_auc_score(te.label, te.score)),
           "test_auc_duree_seule": float(roc_auc_score(te.label, te.duration_sec)),
           "bande_autour_du_seuil": band_analysis(te, thr_dur, t, an.get("band_widths", (2, 3, 4, 6)),
                                                  an.get("min_calls", 30)),
           "par_langue": {l: {"n": int(len(g)), "accuracy": float(accuracy_score(g.label, g.score >= t)),
                              "auc": float(roc_auc_score(g.label, g.score))}
                          for l, g in te.groupby("language")}}
    out = resolve(cfg["paths"]["reports_dir"]) / "analyse_duree.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("analyse_duree :", json.dumps(res))
    return res