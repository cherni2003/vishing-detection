"""Ingestion des robocalls FTC : durée par appel, puis segments de 10 s (pas de 5 s)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from pyspark.sql import functions as F

from vishing_detection.models.common_ext import EXT, load_params

MIN_DURATION = 1.0


def _probe(batches):
    import librosa

    for pdf in batches:
        durs, oks = [], []
        for p in pdf["path"]:
            try:
                durs.append(float(librosa.get_duration(path=p)))
                oks.append(True)
            except Exception:
                durs.append(0.0)
                oks.append(False)
        yield pdf.assign(duration_sec=durs, ok=oks)[["call_id", "path", "duration_sec", "ok"]]


def run(spark) -> None:
    P = load_params()
    E = P["external"]
    root = Path(E["root"])
    win = float(P["segmentation"]["window_sec"])
    hop = float(P["segmentation"]["hop_sec"])

    meta = pd.read_csv(root / E["metadata"])
    meta = meta[meta["language"] == E["language"]].drop_duplicates("file_name")
    if E.get("limit"):
        meta = meta.sort_values("file_name").head(int(E["limit"]))
    meta["path"] = [str((root / f).resolve()) for f in meta["file_name"]]
    missing = [p for p in meta["path"] if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} fichiers absents, ex. {missing[0]}")
    meta["call_id"] = ["ftc_" + Path(f).stem.replace("_normalized", "") for f in meta["file_name"]]
    listing = meta[["call_id", "path"]]

    probe = (
        spark.createDataFrame(listing)
        .repartition(max(2, len(listing) // 200))
        .mapInPandas(_probe, schema="call_id string, path string, duration_sec double, ok boolean")
    )
    # Un seul passage librosa : on ramène le résultat en pandas (sinon chaque action relance le calcul)
    probed = probe.toPandas()
    bad = int((~probed["ok"] | (probed["duration_sec"] < MIN_DURATION)).sum())
    calls_pdf = probed[probed["ok"] & (probed["duration_sec"] >= MIN_DURATION)].drop(columns="ok")

    segments = (
        spark.createDataFrame(calls_pdf)
        .withColumn("n", F.greatest(F.lit(1), F.floor((F.col("duration_sec") - win) / hop).cast("int") + 1))
        .withColumn("k", F.explode(F.sequence(F.lit(0), F.col("n") - 1)))
        .withColumn("start_sec", F.col("k") * F.lit(hop))
        .withColumn("end_sec", F.least(F.col("start_sec") + F.lit(win), F.col("duration_sec")))
        .drop("n", "k")
        .toPandas()
    )

    EXT.mkdir(parents=True, exist_ok=True)
    calls_pdf.to_parquet(EXT / "calls.parquet", index=False)
    segments.to_parquet(EXT / "segments.parquet", index=False)
    print(f"{len(calls_pdf)} appels, {len(segments)} segments, {bad} fichiers écartés")