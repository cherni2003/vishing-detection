"""Lecture audio : mono, rééchantillonnage, extraction d'un segment."""
from __future__ import annotations

import numpy as np
import soundfile as sf


def load_audio(path, sr: int | None = None, start_sec: float = 0.0,
               end_sec: float | None = None) -> tuple[np.ndarray, int]:
    """Charge (une partie d') un WAV en mono float32, rééchantillonné à `sr` si demandé."""
    info = sf.info(str(path))
    start = int(start_sec * info.samplerate)
    stop = None if end_sec is None else int(end_sec * info.samplerate)
    y, file_sr = sf.read(str(path), start=start, stop=stop, dtype="float32", always_2d=True)
    y = y.mean(axis=1)
    if sr is not None and sr != file_sr:
        import librosa
        y = librosa.resample(y, orig_sr=file_sr, target_sr=sr)
        return y.astype(np.float32), sr
    return y, file_sr
