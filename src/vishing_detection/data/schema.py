"""Schéma du manifest commun et règles de validation."""
from __future__ import annotations

import pandas as pd

MANIFEST_COLUMNS = [
    "sample_id", "audio_path", "label", "label_original", "language",
    "source_dataset", "speaker_id", "scenario_id", "is_synthetic",
    "duration_sec", "sample_rate", "split", "license",
]

SPLITS = {"train", "validation", "test"}

# Colonnes d'annotation qui donnent (presque) le label : analyse seulement,
# jamais en entrée du modèle (voir docs/architecture.md, section 7.4).
FORBIDDEN_FEATURE_COLUMNS = {
    "speaker_role", "gender", "contains_sensitive_request", "request_type",
    "persuasion_strategy", "attack_type", "requested_asset",
    "impersonated_entity", "victim_role", "risk_level", "theme",
    "n_turns", "num_turns", "hard_scam", "any_sensitive",
}


def validate_manifest(df: pd.DataFrame) -> list[str]:
    """Retourne la liste des erreurs (liste vide = manifest valide)."""
    errors: list[str] = []
    missing = [c for c in MANIFEST_COLUMNS if c not in df.columns]
    if missing:
        return [f"colonnes manquantes : {missing}"]
    if df["sample_id"].duplicated().any():
        errors.append("sample_id dupliqués")
    if not df["label"].isin([0, 1]).all():
        errors.append("label hors {0, 1}")
    bad_split = set(df["split"].dropna().unique()) - SPLITS
    if bad_split or df["split"].isna().any():
        errors.append(f"split invalide : {sorted(map(str, bad_split))}")
    if (df["duration_sec"] <= 0).any():
        errors.append("duration_sec <= 0")
    return errors


def forbidden_columns_present(columns) -> set[str]:
    return FORBIDDEN_FEATURE_COLUMNS & set(columns)
