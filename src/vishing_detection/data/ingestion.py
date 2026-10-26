"""Un lecteur par corpus -> manifest commun (schéma de data/schema.py).

Ajouter une fonction par nouveau corpus (DS01...DS06) ; toutes retournent un
DataFrame avec exactement MANIFEST_COLUMNS.
"""
from __future__ import annotations

import pandas as pd

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
