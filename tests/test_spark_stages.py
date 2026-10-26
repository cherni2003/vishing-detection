"""Étapes Spark + chaîne complète sur le mini-jeu (Spark en local, Java 17+ requis)."""
import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pyspark")
pytest.importorskip("librosa")

from vishing_detection.audio.features import feature_names
from vishing_detection.audio.segmentation import make_segments
from vishing_detection.config import get_config, resolve
from vishing_detection.data.schema import forbidden_columns_present
from vishing_detection.models import baselines, evaluate, train
from vishing_detection.spark import extract_features, prepare, validate
from vishing_detection.spark.session import get_spark


@pytest.fixture()
def spark():
    s = get_spark()
    yield s
    s.stop()


def test_validate_prepare_features_and_models(mini_project, spark):
    cfg = get_config()

    validate.run(spark)
    quality = pd.read_parquet(resolve(cfg["outputs"]["quality"]))
    assert quality["ok"].all() and len(quality) == 8

    prepare.run(spark)
    seg = pd.read_parquet(resolve(cfg["outputs"]["segments"]))
    man = pd.read_parquet(resolve(cfg["outputs"]["manifest"]))
    expected = sum(len(make_segments(d, 4, 2)) for d in man["duration_sec"])
    assert len(seg) == expected == 4 * 1 + 4 * 2          # normaux 3 s : 1 segment, scams 7 s : 2
    # Chaque segment hérite du split et du label de son appel
    merged = seg.merge(man[["sample_id", "split", "label"]], on="sample_id", suffixes=("", "_m"))
    assert (merged["split"] == merged["split_m"]).all() and (merged["label"] == merged["label_m"]).all()

    extract_features.run(spark)
    feats = pd.read_parquet(resolve(cfg["outputs"]["handcrafted"]))
    assert len(feats) == len(seg)
    assert list(feats.columns) == ["segment_id", "sample_id"] + feature_names(5)
    assert not forbidden_columns_present(feats.columns)
    assert feats.drop(columns=["segment_id", "sample_id"]).notna().all().all()

    # Baselines
    rep = baselines.run()
    assert rep["majority"] == 0.5 and rep["duration_sec"]["acc_test"] == 1.0

    # Embeddings factices (le vrai encodeur tourne sur Kaggle) -> train -> evaluate
    rng = np.random.default_rng(0)
    lab = seg.set_index("segment_id")["label"]
    emb = pd.DataFrame({
        "segment_id": seg["segment_id"], "sample_id": seg["sample_id"],
        "embedding": [(rng.normal(size=16) + 3 * lab[s]).astype("float32").tolist() for s in seg["segment_id"]],
    })
    emb.to_parquet(resolve(cfg["outputs"]["embeddings"]), index=False)
    train.run()
    result = evaluate.run()
    assert result["global"]["n"] == 2 and result["global"]["accuracy"] == 1.0
    assert json.loads(resolve(cfg["outputs"]["metrics"]).read_text())["global"]["n"] == 2


def test_prepare_rejects_group_split_leakage(mini_project, spark, mini_manifest):
    bad = mini_manifest.copy()
    bad.loc[1, "scenario_id"] = bad.loc[0, "scenario_id"]     # même groupe, splits différents
    bad.loc[0, "split"], bad.loc[1, "split"] = "train", "test"
    bad.to_csv(mini_project / "data" / "metadata" / "audio_manifest.csv", index=False)
    validate.run(spark)
    with pytest.raises(ValueError, match="plusieurs splits"):
        prepare.run(spark)
