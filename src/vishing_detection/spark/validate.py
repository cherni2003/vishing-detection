"""Étape 1 : contrôles qualité distribués sur le manifest -> interim/quality.parquet."""
from __future__ import annotations

from pyspark.sql import functions as F
from pyspark.sql.window import Window

from vishing_detection.config import get_config, resolve

REQUIRED = ["sample_id", "audio_path", "label", "duration_sec", "split"]


def run(spark, params: dict | None = None) -> None:
    cfg = get_config()
    df = spark.read.csv(str(resolve(cfg["data"]["manifest_path"])), header=True, inferSchema=True)

    dup = F.count("*").over(Window.partitionBy("sample_id")) > 1
    missing = F.lit(False)
    for c in REQUIRED:
        missing = missing | F.col(c).isNull()
    bad_duration = ~(F.col("duration_sec") > 0)
    bad_split = ~F.col("split").isin("train", "validation", "test")
    bad_label = ~F.col("label").isin(0, 1)

    quality = df.select(
        "sample_id",
        dup.alias("duplicate_id"),
        missing.alias("missing_field"),
        bad_duration.alias("bad_duration"),
        bad_split.alias("bad_split"),
        bad_label.alias("bad_label"),
    ).withColumn(
        "ok",
        ~(F.col("duplicate_id") | F.col("missing_field") | F.col("bad_duration")
          | F.col("bad_split") | F.col("bad_label")),
    )
    out = resolve(cfg["outputs"]["quality"])
    out.parent.mkdir(parents=True, exist_ok=True)
    quality.write.mode("overwrite").parquet(str(out))

    n, n_ok = quality.count(), quality.filter("ok").count()
    print(f"validate : {n_ok}/{n} échantillons valides -> {out}")
