"""Embeddings audio par segment avec un encodeur pré-entraîné (GPU conseillé).

À exécuter sur Kaggle. Sortie : data/features/embeddings.parquet
(segment_id, sample_id, embedding: liste de float).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from vishing_detection.audio.loader import load_audio
from vishing_detection.config import get_config, get_params, resolve


def run(params: dict | None = None) -> None:
    import torch
    from transformers import AutoFeatureExtractor, AutoModel

    cfg, params = get_config(), params or get_params()
    sr = int(params["segmentation"]["target_sample_rate"])
    emb = params["embeddings"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    extractor = AutoFeatureExtractor.from_pretrained(emb["model_name"])
    model = AutoModel.from_pretrained(emb["model_name"]).to(device).eval()

    segs = pd.read_parquet(resolve(cfg["outputs"]["segments"]))
    rows = []
    with torch.no_grad():
        for r in segs.itertuples():
            y, _ = load_audio(resolve(r.audio_path), sr=sr, start_sec=r.start_sec, end_sec=r.end_sec)
            inputs = extractor(y, sampling_rate=sr, return_tensors="pt").to(device)
            hidden = model(**inputs).last_hidden_state          # (1, T, D)
            vec = hidden.mean(dim=1).squeeze(0).cpu().numpy()    # pooling = mean
            rows.append({"segment_id": r.segment_id, "sample_id": r.sample_id,
                         "embedding": vec.astype(np.float32).tolist()})
    out = resolve(cfg["outputs"]["embeddings"])
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out, index=False)
    print(f"embeddings : {len(rows)} segments -> {out}")
