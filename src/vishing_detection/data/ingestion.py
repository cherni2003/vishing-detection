"""Un lecteur par corpus -> manifest commun (schéma de data/schema.py).

Ajouter une fonction par nouveau corpus (DS01...DS06) ; toutes retournent un
DataFrame avec exactement MANIFEST_COLUMNS.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from vishing_detection.data.schema import MANIFEST_COLUMNS

def ingest_vish_darija(manifest_with_splits_csv: str, path_prefix: str) -> pd.DataFrame:
    """VISH-DARIJA-TTS (DS04) : audio_manifest_normalized_with_splits.csv -> manifest commun.

    sample_id = dataset_uid (scenario_id seul n'est pas unique entre classes).
    """
    df = pd.read_csv(manifest_with_splits_csv)
    out = pd.DataFrame({
        "sample_id": df["dataset_uid"],
        "audio_path": path_prefix + df["relative_path"].str.replace("\\", "/", regex=False),
        "label": (df["label"] == "scam").astype(int),
        "label_original": df["label"],
        "language": "darija",
        "source_dataset": "DS04_VISH-DARIJA-TTS",
        "speaker_id": "",
        "scenario_id": df["dataset_uid"],
        "is_synthetic": True,
        "duration_sec": df["duration_sec"],
        "sample_rate": df["sample_rate_hz"],
        "split": df["split"],
        "license": "CC-BY-4.0",
    })
    return out[MANIFEST_COLUMNS]


def _load_vishguard_records(annotations_json) -> list[dict]:
    data = json.loads(Path(annotations_json).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        for key in ("records", "annotations", "data", "samples"):
            if isinstance(data.get(key), list):
                return data[key]
        return [{"id": k, **v} if isinstance(v, dict) else {"id": k} for k, v in data.items()]
    return list(data)


def ingest_vishguard(annotations_json, path_prefix: str, *, seed: int = 42,
                     test_size: float = 0.2, validation_size: float = 0.1,
                     languages: list[str] | None = None,
                     license_: str = "CC-BY-4.0") -> pd.DataFrame:
    """VISHGUARD (DS02) -> manifest commun. Split stratifié langue x classe, au niveau de l'appel."""
    rec = pd.DataFrame(_load_vishguard_records(annotations_json))
    meta = rec["metadata"].apply(lambda m: m if isinstance(m, dict) else {})
    meta_path = meta.apply(lambda m: m.get("audio_path"))
    rec["audio_rel"] = rec["audio_path"].fillna(meta_path) if "audio_path" in rec.columns else meta_path
    if languages:
        rec = rec[rec["language"].isin(languages)]
    rec = rec.reset_index(drop=True)

    strata = rec["language"] + "_" + rec["type"]
    rest, test = train_test_split(rec.index, test_size=test_size, stratify=strata, random_state=seed)
    val_share = validation_size / (1 - test_size)
    train, val = train_test_split(rest, test_size=val_share, stratify=strata[rest], random_state=seed)
    split = pd.Series("train", index=rec.index)
    split[val], split[test] = "validation", "test"

    out = pd.DataFrame({
        "sample_id": rec["id"],
        "audio_path": path_prefix + rec["audio_rel"].str.replace("\\", "/", regex=False),
        "label": (rec["type"] == "fraudulent").astype(int),
        "label_original": rec["type"],
        "language": rec["language"],
        "source_dataset": "DS02_VISHGUARD",
        "speaker_id": "tts_" + rec["language"],
        "scenario_id": rec["id"],
        "is_synthetic": True,
        "duration_sec": rec["duration"],
        "sample_rate": 16000,
        "split": split,
        "license": license_,
    })
    return out[MANIFEST_COLUMNS]