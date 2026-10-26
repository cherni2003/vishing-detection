from vishing_detection.data.schema import (MANIFEST_COLUMNS, forbidden_columns_present,
                                           validate_manifest)


def test_mini_manifest_is_valid(mini_manifest):
    assert list(mini_manifest.columns) == MANIFEST_COLUMNS
    assert validate_manifest(mini_manifest) == []


def test_duplicate_ids_and_bad_split_are_detected(mini_manifest):
    bad = mini_manifest.copy()
    bad.loc[1, "sample_id"] = bad.loc[0, "sample_id"]
    bad.loc[2, "split"] = "dev"
    errors = validate_manifest(bad)
    assert any("dupliqu" in e for e in errors)
    assert any("split" in e for e in errors)


def test_missing_columns_are_reported(mini_manifest):
    assert "colonnes manquantes" in validate_manifest(mini_manifest.drop(columns=["label"]))[0]


def test_forbidden_annotation_columns():
    assert forbidden_columns_present(["segment_id", "speaker_role", "n_turns"]) == {"speaker_role", "n_turns"}
    assert forbidden_columns_present(["segment_id", "mfcc_mean_0"]) == set()
