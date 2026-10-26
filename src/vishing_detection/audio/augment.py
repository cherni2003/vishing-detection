"""Augmentations (entraînement seulement). Les variantes bruitées fournies par
les corpus servent d'abord à évaluer la robustesse."""
from __future__ import annotations

import numpy as np


def add_noise(y: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Mélange `noise` à `y` au rapport signal/bruit demandé."""
    noise = np.resize(noise, y.shape)
    p_y, p_n = np.mean(y ** 2) + 1e-12, np.mean(noise ** 2) + 1e-12
    scale = np.sqrt(p_y / (p_n * 10 ** (snr_db / 10)))
    return (y + scale * noise).astype(np.float32)


def telephone_band(y: np.ndarray, sr: int, low: float = 300.0, high: float = 3400.0) -> np.ndarray:
    """Simule une bande téléphonique (filtre passe-bande)."""
    from scipy.signal import butter, sosfilt

    sos = butter(4, [low, min(high, sr / 2 - 1)], btype="band", fs=sr, output="sos")
    return sosfilt(sos, y).astype(np.float32)
