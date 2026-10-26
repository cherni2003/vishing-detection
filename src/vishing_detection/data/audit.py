"""Fonctions d'audit réutilisables (extraites du notebook 01_audit)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def class_balance(df: pd.DataFrame, label: str = "label", split: str = "split") -> pd.DataFrame:
    """Effectifs par split et par classe."""
    return pd.crosstab(df[split], df[label], margins=True)


def group_leakage(df: pd.DataFrame, group_col: str, split_col: str = "split") -> int:
    """Nombre de groupes présents dans plus d'un split (0 attendu)."""
    return int((df.groupby(group_col)[split_col].nunique() > 1).sum())


def fit_threshold_rule(x: np.ndarray, y: np.ndarray, candidates=None) -> tuple[float, float]:
    """Meilleur seuil « x > t => classe 1 » sur le train. Retourne (seuil, accuracy)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y)
    candidates = np.unique(x) if candidates is None else np.asarray(candidates)
    best_t, best_acc = float(candidates[0]), -1.0
    for t in candidates:
        acc = float(((x > t).astype(int) == y).mean())
        if acc > best_acc:
            best_t, best_acc = float(t), acc
    return best_t, best_acc


def baseline_report(df: pd.DataFrame, feature_cols: list[str], label: str = "label",
                    split: str = "split") -> dict:
    """Accuracy test de la classe majoritaire et de règles à un seuil, ajustées sur train."""
    train, test = df[df[split] == "train"], df[df[split] == "test"]
    majority = int(train[label].mode().iloc[0])
    report = {"majority": float((test[label] == majority).mean())}
    for col in feature_cols:
        if col not in df.columns or df[col].isna().all():
            continue
        t, acc_train = fit_threshold_rule(train[col].to_numpy(), train[label].to_numpy())
        acc_test = float(((test[col] > t).astype(int) == test[label]).mean())
        report[col] = {"threshold": t, "acc_train": acc_train, "acc_test": acc_test}
    return report
