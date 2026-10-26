"""Étape 3 : features acoustiques par segment, en parallèle (mapInPandas)."""
from __future__ import annotations

import os

from vishing_detection.audio.features import feature_names
from vishing_detection.config import get_config, get_params, project_root, resolve


def run(spark, params: dict | None = None) -> None:
    cfg, params = get_config(), params or get_params()
    sr = int(params["segmentation"]["target_sample_rate"])
    n_mfcc = int(params["features"]["n_mfcc"])
    root = str(project_root())
    names = feature_names(n_mfcc)
    schema = "segment_id string, sample_id string, " + ", ".join(f"{n} double" for n in names)

    def extract(batches):
        import pandas as pd
        from vishing_detection.audio.features import acoustic_features
        from vishing_detection.audio.loader import load_audio

        for pdf in batches:
            rows = []
            for r in pdf.itertuples():
                path = r.audio_path if os.path.isabs(r.audio_path) else os.path.join(root, r.audio_path)
                y, file_sr = load_audio(path, sr=sr, start_sec=r.start_sec, end_sec=r.end_sec)
                feats = acoustic_features(y, file_sr, n_mfcc=n_mfcc)
                rows.append({"segment_id": r.segment_id, "sample_id": r.sample_id, **feats})
            yield pd.DataFrame(rows, columns=["segment_id", "sample_id"] + names)

    segments = spark.read.parquet(str(resolve(cfg["outputs"]["segments"])))
    feats = segments.select("segment_id", "sample_id", "audio_path", "start_sec", "end_sec") \
                    .mapInPandas(extract, schema=schema)
    out = resolve(cfg["outputs"]["handcrafted"])
    out.parent.mkdir(parents=True, exist_ok=True)
    feats.write.mode("overwrite").parquet(str(out))
    print(f"features_handcrafted -> {out}")
