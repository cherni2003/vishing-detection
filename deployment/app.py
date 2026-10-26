"""API de prédiction : un WAV en entrée, un score de vishing en sortie.

Lancer : uvicorn deployment.app:app --reload   (modèle dans models/latest)
La segmentation et le rééchantillonnage sont ceux de l'entraînement (params.yaml).
"""
from __future__ import annotations

import io
import os

import joblib
import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, UploadFile

from vishing_detection.audio.segmentation import make_segments
from vishing_detection.config import get_params
from vishing_detection.models.predict import aggregate_scores

app = FastAPI(title="Vishing Detection")
MODEL_PATH = os.environ.get("MODEL_PATH", "models/latest/model.joblib")
_model = None


def get_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


def embed(segment: np.ndarray, sr: int) -> np.ndarray:
    """À brancher sur le même encodeur que models/embeddings.py (params.yaml: embeddings)."""
    raise NotImplementedError("brancher l'encodeur d'embeddings utilisé à l'entraînement")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    params = get_params()
    seg_p, ev = params["segmentation"], params["evaluate"]
    y, sr = sf.read(io.BytesIO(await file.read()), dtype="float32", always_2d=True)
    y = y.mean(axis=1)
    windows = make_segments(len(y) / sr, seg_p["window_sec"], seg_p["hop_sec"])
    X = np.stack([embed(y[int(a * sr):int(b * sr)], sr) for a, b in windows])
    scores = get_model().predict_proba(X)[:, 1]
    score = aggregate_scores(scores, ev["aggregation"], ev.get("topk", 3))
    return {"score": score, "is_vishing": bool(score >= ev["threshold"]),
            "threshold": ev["threshold"], "segment_scores": scores.tolist()}
