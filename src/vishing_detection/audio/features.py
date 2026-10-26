"""Caractéristiques acoustiques de base (baseline niveau 1)."""
from __future__ import annotations

import numpy as np


def feature_names(n_mfcc: int) -> list[str]:
    names = [f"mfcc_mean_{i}" for i in range(n_mfcc)] + [f"mfcc_std_{i}" for i in range(n_mfcc)]
    return names + ["rms_mean", "rms_std", "zcr_mean", "silence_ratio"]


def acoustic_features(y: np.ndarray, sr: int, n_mfcc: int = 20,
                      silence_db: float = -40.0) -> dict[str, float]:
    import librosa

    if len(y) < 2048:                      # segment trop court : pad
        y = np.pad(y, (0, 2048 - len(y)))
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    rms = librosa.feature.rms(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    feats = {f"mfcc_mean_{i}": float(mfcc[i].mean()) for i in range(n_mfcc)}
    feats.update({f"mfcc_std_{i}": float(mfcc[i].std()) for i in range(n_mfcc)})
    feats["rms_mean"] = float(rms.mean())
    feats["rms_std"] = float(rms.std())
    feats["zcr_mean"] = float(zcr.mean())
    feats["silence_ratio"] = float((20 * np.log10(rms + 1e-9) < silence_db).mean())
    return feats
