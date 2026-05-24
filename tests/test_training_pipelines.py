import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SERVER = ROOT / "server"
if str(SERVER) not in sys.path:
    sys.path.insert(0, str(SERVER))

from biosignal_pipeline import BiosignalPipeline
from biosignal_profiles import list_profiles
from config import config


def _synthetic_stream(profile_key: str, label_idx: int, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(n_samples, dtype=np.float32) / float(config.sample_rate)
    base = np.zeros((config.n_channels, n_samples), dtype=np.float32)

    if profile_key == "ecg":
        signal = np.zeros(n_samples, dtype=np.float32)
        step = max(60, int(config.sample_rate * (0.55 + 0.12 * label_idx)))
        for peak in range(30, n_samples, step):
            end = min(n_samples, peak + 12)
            signal[peak:end] += np.hanning(end - peak).astype(np.float32) * (1.0 + label_idx)
        base[:] = signal
    elif profile_key == "eda":
        trend = (label_idx + 1) * 0.01 * t
        phasic = (0.2 + label_idx * 0.2) * np.maximum(0, np.sin(2 * np.pi * (0.15 + label_idx * 0.1) * t))
        base[:] = trend + phasic
    elif profile_key == "ppg":
        base[:] = 0.5 + (0.2 + label_idx * 0.2) * np.sin(2 * np.pi * (1.0 + label_idx * 0.4) * t)
    elif profile_key == "resp":
        base[:] = (0.2 + label_idx * 0.2) * np.sin(2 * np.pi * (0.15 + label_idx * 0.1) * t + label_idx)
    elif profile_key == "temp":
        base[:] = 0.02 * label_idx + (label_idx - 1) * 0.02 * t
    else:
        base[:] = (0.2 + label_idx * 0.2) * np.sin(2 * np.pi * (2.0 + label_idx) * t)

    channel_scale = np.linspace(0.8, 1.2, config.n_channels, dtype=np.float32)[:, None]
    return base * channel_scale + rng.normal(0, 0.01, base.shape).astype(np.float32)


@pytest.mark.parametrize("profile", [profile.key for profile in list_profiles()])
def test_profile_lda_training_smoke(profile: str) -> None:
    rng = np.random.default_rng(456)
    config.set_signal_profile(profile)
    pipeline = BiosignalPipeline()

    labels = list(config.class_labels)[: min(3, len(config.class_labels))]
    seconds = max(float(getattr(pipeline._impl, "_segment_seconds", 0.2)) + 8.0, 12.0)
    n_steps = int(seconds * 1000 / config.window_increment_ms)
    n_samples = config.window_size_samples + max(0, n_steps - 1) * config.window_increment_samples

    for label_idx, label in enumerate(labels):
        signal = _synthetic_stream(profile, label_idx, n_samples, rng)
        pipeline.start_recording(label)
        for step in range(n_steps):
            end = config.window_size_samples + step * config.window_increment_samples
            pipeline.on_window(signal[:, end - config.window_size_samples : end])
        pipeline.stop_recording()

    result = pipeline.train("LDA")

    assert result["success"], result
    assert pipeline.is_trained
    assert pipeline.get_training_summary()["total_windows"] >= 8
    assert len(result["class_counts"]) >= 2
