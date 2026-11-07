"""Rappel par campagne (case_details) et bootstrap par groupes."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from vishing_detection.config import get_config, get_params, resolve


def _cluster_ci(hit, n, rng, n_iter=10000):
    k = len(hit)
    draws = np.empty(n_iter)
    for b in range(n_iter):
        i = rng.integers(0, k, k)
        draws[b] = hit[i].sum() / max(n[i].sum(), 1)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return [round(float(lo), 4), round(float(hi), 4)]


def run(params: dict | None = None) -> dict:
    cfg, params = get_config(), params or get_params()
    E = params["external"]
    ext = resolve("data/processed/external")

    sc = pd.read_parquet(ext / "scores.parquet")
    meta = pd.read_csv(Path(E["root"]) / E["metadata"])
    meta = meta[meta["language"] == E["language"]].drop_duplicates("file_name")
    meta["sample_id"] = "ftc_" + meta["file_name"].map(
        lambda f: Path(f).stem.replace("_normalized", ""))
    df = sc.merge(meta[["sample_id", "case_details"]], on="sample_id", how="left")
    df["case_details"] = df["case_details"].fillna("inconnue")

    g = df.groupby("case_details")
    tab = pd.DataFrame({
        "n": g.size(),
        "rappel": g["hit"].mean(),
        "n_courts": g["short"].sum(),
        "rappel_courts": df[df.short].groupby("case_details")["hit"].mean(),
        "duree_mediane": g["duration_sec"].median(),
    }).sort_values("n", ascending=False)

    rng = np.random.default_rng(params["seed"])
    hit = g["hit"].sum().to_numpy().astype(float)
    n = g.size().to_numpy().astype(float)
    hit_s = df[df.short].groupby("case_details")["hit"].sum().reindex(g.size().index, fill_value=0).to_numpy().astype(float)
    n_s = g["short"].sum().to_numpy().astype(float)

    res = {
        "n_appels": int(len(df)), "n_campagnes": int(len(tab)),
        "part_plus_grosse_campagne": round(float(tab.n.iloc[0] / len(df)), 3),
        "part_top5_campagnes": round(float(tab.n.head(5).sum() / len(df)), 3),
        "rappel_global": round(float(df.hit.mean()), 4),
        "ic95_bootstrap_par_campagne": _cluster_ci(hit, n, rng),
        "rappel_appels_courts": round(float(df[df.short].hit.mean()), 4),
        "ic95_courts_bootstrap_par_campagne": _cluster_ci(hit_s, n_s, rng),
        "campagnes_rappel_inf_90": tab[tab.rappel < 0.9].round(3).reset_index().to_dict("records"),
    }
    out = resolve(cfg["paths"]["reports_dir"]) / "external_campaigns.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    tab.round(3).to_csv(out.with_name("external_campaigns_table.csv"))
    print(json.dumps({k: v for k, v in res.items() if k != "campagnes_rappel_inf_90"}, indent=1))
    return res