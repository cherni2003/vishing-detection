"""Génère le mini-jeu de test (8 petits WAV 8 kHz + manifest). Usage : python tests/data/make_mini.py"""
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

OUT = Path(__file__).parent / "mini"
OUT.mkdir(exist_ok=True)
SR = 8000
rng = np.random.default_rng(42)
splits = ["train", "train", "validation", "test"]

rows = []
for label, name, dur in [(0, "normal", 3.0), (1, "scam", 7.0)]:
    for i, split in enumerate(splits):
        t = np.arange(int(dur * SR)) / SR
        f0 = 140 + 40 * label + 10 * i
        y = 0.3 * np.sin(2 * np.pi * f0 * t) + 0.02 * rng.standard_normal(len(t))
        fname = f"{name}_{i:02d}.wav"
        sf.write(OUT / fname, y.astype(np.float32), SR, subtype="PCM_16")
        rows.append({"sample_id": f"{name}_{i:02d}", "filename": fname, "label": label,
                     "label_original": name, "duration_sec": dur, "split": split})
pd.DataFrame(rows).to_csv(OUT / "mini_manifest.csv", index=False)
print("mini-jeu écrit dans", OUT)
