"""Baselines obligatoires : classe majoritaire et règles à un seuil.

Étape `baselines` -> reports/baselines.json. Le nombre de tours vient de la table
d'analyse (cfg.data.audit_features_path) et sert uniquement de plancher de comparaison.
"""
from __future__ import annotations

import json

import pandas as pd

from vishing_detection.config import get_config, resolve
from vishing_detection.data.audit import baseline_report


def run(params: dict | None = None) -> dict:
    cfg = get_config()
    df = pd.read_parquet(resolve(cfg["outputs"]["manifest"]))
    audit_path = resolve(cfg["data"]["audit_features_path"])
    if audit_path.exists():
        audit = pd.read_csv(audit_path).rename(columns={"dataset_uid": "sample_id"})
        df = df.merge(audit[["sample_id", "n_turns"]], on="sample_id", how="left")
    report = baseline_report(df, ["duration_sec", "n_turns"])
    out = resolve(cfg["outputs"]["baselines"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("baselines :", json.dumps(report))
    return report
