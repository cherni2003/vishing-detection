"""Enveloppe MLflow : sans effet si mlflow n'est pas installé."""
from __future__ import annotations

import contextlib
import os


@contextlib.contextmanager
def start_run(run_name: str, params: dict | None = None, tags: dict | None = None):
    try:
        import mlflow
    except ImportError:
        yield None
        return
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if uri:
        mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("vishing-detection")
    with mlflow.start_run(run_name=run_name, tags=tags or {}) as run:
        if params:
            mlflow.log_params(_flatten(params))
        yield mlflow


def log_metrics(mlflow, metrics: dict, prefix: str = "") -> None:
    if mlflow is None:
        return
    for k, v in _flatten(metrics, prefix).items():
        if isinstance(v, (int, float)):
            mlflow.log_metric(k, float(v))


def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out
