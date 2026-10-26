import pandas as pd

from vishing_detection.data.audit import baseline_report, fit_threshold_rule, group_leakage
from vishing_detection.models.evaluate import compute_metrics
from vishing_detection.models.predict import aggregate_scores


def test_group_leakage_counts_groups_in_several_splits():
    df = pd.DataFrame({"g": ["a", "a", "b"], "split": ["train", "test", "train"]})
    assert group_leakage(df, "g") == 1


def test_threshold_rule_and_baseline_report(mini_manifest):
    t, acc = fit_threshold_rule(mini_manifest["duration_sec"], mini_manifest["label"])
    assert t == 3.0 and acc == 1.0
    rep = baseline_report(mini_manifest, ["duration_sec"])
    assert rep["majority"] == 0.5 and rep["duration_sec"]["acc_test"] == 1.0


def test_aggregation_methods():
    s = [0.1, 0.9, 0.5]
    assert aggregate_scores(s, "mean") == 0.5
    assert aggregate_scores(s, "max") == 0.9
    assert abs(aggregate_scores(s, "topk", topk=2) - 0.7) < 1e-9


def test_compute_metrics():
    m = compute_metrics([0, 0, 1, 1], [0.1, 0.6, 0.4, 0.9])
    assert m["fp"] == 1 and m["fn"] == 1 and abs(m["roc_auc"] - 0.75) < 1e-9
