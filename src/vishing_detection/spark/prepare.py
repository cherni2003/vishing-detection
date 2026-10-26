"""Étape 2 : manifest propre + segments (fenêtres fixes) -> processed/."""
from __future__ import annotations

from pyspark.sql import functions as F

from vishing_detection.config import get_config, get_params, resolve


def run(spark, params: dict | None = None) -> None:
    cfg, params = get_config(), params or get_params()
    window = float(params["segmentation"]["window_sec"])
    hop = float(params["segmentation"]["hop_sec"])
    group_col = params["prepare"]["group_col"]

    manifest = spark.read.csv(str(resolve(cfg["data"]["manifest_path"])), header=True, inferSchema=True)
    quality = spark.read.parquet(str(resolve(cfg["outputs"]["quality"])))
    clean = manifest.join(quality.filter("ok").select("sample_id"), "sample_id", "inner")

    # Aucune fuite : un groupe (conversation/scénario) ne doit apparaître que dans un seul split.
    leaks = (clean.groupBy(group_col).agg(F.countDistinct("split").alias("n"))
             .filter("n > 1").count())
    if leaks:
        raise ValueError(f"{leaks} groupes ({group_col}) présents dans plusieurs splits")

    # Segments : même logique que audio.segmentation.make_segments.
    n_seg = (F.when(F.col("duration_sec") > window,
                    F.floor((F.col("duration_sec") - window) / hop) + 1)
             .otherwise(1).cast("int"))
    segments = (
        clean.withColumn("i", F.explode(F.sequence(F.lit(0), n_seg - 1)))
        .withColumn("start_sec", F.col("i") * hop)
        .withColumn("end_sec", F.least(F.col("start_sec") + window, F.col("duration_sec")))
        .withColumn("segment_id", F.concat_ws("_", "sample_id", F.lpad(F.col("i").cast("string"), 3, "0")))
        .select("segment_id", "sample_id", "audio_path", "start_sec", "end_sec", "label", "split")
    )

    out_manifest, out_segments = resolve(cfg["outputs"]["manifest"]), resolve(cfg["outputs"]["segments"])
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    clean.write.mode("overwrite").parquet(str(out_manifest))
    segments.write.mode("overwrite").parquet(str(out_segments))
    print(f"prepare : {clean.count()} appels, {segments.count()} segments")
