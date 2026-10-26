"""Fixture : un mini-projet temporaire (configs + manifest) pointant vers tests/data/mini."""
import shutil
from pathlib import Path

import pandas as pd
import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
MINI = REPO / "tests" / "data" / "mini"


@pytest.fixture()
def mini_manifest() -> pd.DataFrame:
    m = pd.read_csv(MINI / "mini_manifest.csv")
    return pd.DataFrame({
        "sample_id": m["sample_id"],
        "audio_path": [str(MINI / f) for f in m["filename"]],     # chemins absolus
        "label": m["label"], "label_original": m["label_original"],
        "language": "darija", "source_dataset": "MINI", "speaker_id": "",
        "scenario_id": m["sample_id"], "is_synthetic": True,
        "duration_sec": m["duration_sec"], "sample_rate": 8000,
        "split": m["split"], "license": "CC-BY-4.0",
    })


@pytest.fixture()
def mini_project(tmp_path, monkeypatch, mini_manifest):
    (tmp_path / "configs").mkdir()
    shutil.copy(REPO / "configs" / "config.yaml", tmp_path / "configs")
    spark = yaml.safe_load((REPO / "configs" / "spark.yaml").read_text())
    spark["spark"].update({"driver_memory": "1g", "shuffle_partitions": 2})
    (tmp_path / "configs" / "spark.yaml").write_text(yaml.safe_dump(spark))
    params = yaml.safe_load((REPO / "params.yaml").read_text())
    params["segmentation"].update({"window_sec": 4, "hop_sec": 2, "target_sample_rate": 8000})
    params["features"]["n_mfcc"] = 5
    (tmp_path / "params.yaml").write_text(yaml.safe_dump(params))
    (tmp_path / "data" / "metadata").mkdir(parents=True)
    mini_manifest.to_csv(tmp_path / "data" / "metadata" / "audio_manifest.csv", index=False)
    monkeypatch.setenv("VISHING_ROOT", str(tmp_path))
    return tmp_path
