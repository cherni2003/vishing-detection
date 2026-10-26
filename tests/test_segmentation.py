import numpy as np

from vishing_detection.audio.augment import add_noise
from vishing_detection.audio.loader import load_audio
from vishing_detection.audio.segmentation import make_segments
from conftest import MINI


def test_short_call_gives_one_segment():
    assert make_segments(3.0, 10, 5) == [(0.0, 3.0)]


def test_windows_cover_call_with_overlap():
    assert make_segments(20.0, 10, 5) == [(0, 10), (5, 15), (10, 20)]
    assert make_segments(7.0, 4, 2) == [(0, 4), (2, 6)]


def test_loader_extracts_and_resamples_segment():
    y, sr = load_audio(MINI / "scam_00.wav", sr=16000, start_sec=1.0, end_sec=3.0)
    assert sr == 16000 and abs(len(y) - 2 * 16000) <= 2


def test_add_noise_reaches_target_snr():
    rng = np.random.default_rng(0)
    y, n = rng.standard_normal(16000).astype("float32"), rng.standard_normal(16000)
    noisy = add_noise(y, n, snr_db=10)
    snr = 10 * np.log10((y ** 2).mean() / (((noisy - y) ** 2).mean()))
    assert abs(snr - 10) < 0.5
