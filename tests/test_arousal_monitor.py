import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SERVER = ROOT / "server"
if str(SERVER) not in sys.path:
    sys.path.insert(0, str(SERVER))

from biosignal_analyzers import ProfileWindowAnalyzer
from biosignal_profiles import get_profile
from config import config


def _run_analyzer(profile_key: str, signal: np.ndarray):
    config.set_signal_profile(profile_key)
    analyzer = ProfileWindowAnalyzer(get_profile(profile_key))
    result = None
    step = config.window_increment_samples
    window = config.window_size_samples
    for end in range(window, signal.shape[1] + 1, step):
        result = analyzer.on_window(signal[:, end - window:end])
    return result


def _ecg_signal(bpm: float, seconds: float = 10.0) -> np.ndarray:
    n = int(config.sample_rate * seconds)
    signal = np.zeros(n, dtype=np.float64)
    interval = max(1, int(config.sample_rate * 60.0 / bpm))
    peak_width = 14
    for peak in range(30, n, interval):
        end = min(n, peak + peak_width)
        signal[peak:end] += np.hanning(end - peak) * 1200.0
    return np.tile(signal, (config.n_channels, 1))


def _eda_signal(rise: bool, seconds: float = 10.0) -> np.ndarray:
    n = int(config.sample_rate * seconds)
    t = np.linspace(0.0, 1.0, n)
    if rise:
        signal = 0.2 + 0.5 * t + 0.18 * np.maximum(0, np.sin(2 * np.pi * 2.2 * t))
    else:
        signal = np.full(n, 0.2)
    return np.tile(signal, (config.n_channels, 1))


def test_ecg_elevated_rate_reports_higher_arousal_than_steady_rate():
    steady = _run_analyzer("ecg", _ecg_signal(72.0))
    elevated = _run_analyzer("ecg", _ecg_signal(130.0))

    assert steady["arousal"]["score"] < elevated["arousal"]["score"]
    assert elevated["arousal"]["level"] in {"elevated", "high"}
    assert "not deception" in elevated["arousal"]["disclaimer"].lower()
    assert elevated["metrics"]["arousal_score"] == elevated["arousal"]["score"]


def test_eda_rising_arousal_reports_higher_arousal_than_stable_tonic():
    stable = _run_analyzer("eda", _eda_signal(False))
    rising = _run_analyzer("eda", _eda_signal(True))

    assert stable["arousal"]["score"] < rising["arousal"]["score"]
    assert rising["arousal"]["level"] in {"elevated", "high"}
    assert "skin conductance" in " ".join(rising["arousal"]["drivers"]).lower()
    assert "not deception" in rising["arousal"]["disclaimer"].lower()
