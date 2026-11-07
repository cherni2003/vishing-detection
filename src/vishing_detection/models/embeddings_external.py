"""Embeddings des segments externes (robocalls). Même encodeur, même pooling que `embeddings`.

À exécuter sur Kaggle. Sortie : data/processed/external/embeddings.parquet
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import re

from vishing_detection.audio.loader import load_audio
from vishing_detection.config import get_params
from vishing_detection.models.common_ext import EXT

# Les chemins du parquet sont ceux de Windows : on ne garde que le nom du fichier.
AUDIO_DIR = Path(os.environ.get("ROBOCALL_AUDIO_DIR", "data/raw/ftc_robocalls/audio-wav-16khz"))


def run(params: dict | None = None) -> None:
    import torch
    from transformers import AutoFeatureExtractor, AutoModel

    params = params or get_params()
    sr = int(params["segmentation"]["target_sample_rate"])
    emb = params["embeddings"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    extractor = AutoFeatureExtractor.from_pretrained(emb["model_name"])
    model = AutoModel.from_pretrained(emb["model_name"]).to(device).eval()

    segs = pd.read_parquet(EXT / "segments.parquet")
    rows, skipped = [], 0
    with torch.no_grad():
        for r in segs.itertuples():
            path = AUDIO_DIR / Path(r.path.replace("\\", "/")).name
            y, _ = load_audio(path, sr=sr, start_sec=r.start_sec, end_sec=r.end_sec)
            if len(y) < 1600:                       # < 0,1 s : l'encodeur plante
                skipped += 1
                continue
            inputs = extractor(y, sampling_rate=sr, return_tensors="pt").to(device)
            hidden = model(**inputs).last_hidden_state          # (1, T, D)
            vec = hidden.mean(dim=1).squeeze(0).cpu().numpy()    # pooling = mean
            rows.append({"segment_id": f"{r.call_id}_{r.start_sec:g}",
                         "sample_id": r.call_id, "call_id": r.call_id,
                         "embedding": vec.astype(np.float32).tolist()})
    out = EXT / "embeddings.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out, index=False)
    print(f"embeddings externes : {len(rows)} segments -> {out} ({skipped} ignorés)")


import re
...
def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)
...
path = AUDIO_DIR / _safe(Path(r.path.replace("\\", "/")).name)