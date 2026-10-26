"""Découpage d'un appel en fenêtres fixes (neutralise le raccourci « durée = label »)."""
from __future__ import annotations


def make_segments(duration_sec: float, window_sec: float, hop_sec: float) -> list[tuple[float, float]]:
    """Liste de (début, fin). Un appel plus court que la fenêtre donne un seul segment."""
    if duration_sec <= window_sec:
        return [(0.0, float(duration_sec))]
    n = int((duration_sec - window_sec) // hop_sec) + 1
    return [(i * hop_sec, i * hop_sec + window_sec) for i in range(n)]
