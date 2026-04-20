"""Live signal diagnostics and safety watchdog helpers."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


def _safe_db(value: float) -> float:
    return float(10.0 * np.log10(max(value, 1e-12)))


@dataclass
class SafetyWatchdog:
    enabled: bool = True
    stream_timeout_ms: int = 1500
    auto_estop_on_stale: bool = True
    last_signal_monotonic: float = 0.0
    stale: bool = False
    trip_count: int = 0
    last_trip_reason: str = ""
    last_trip_ts: float = 0.0

    def note_signal(self) -> None:
        self.last_signal_monotonic = time.monotonic()
        self.stale = False

    def check(self, *, stream_running: bool) -> bool:
        if not self.enabled or not stream_running:
            self.stale = False
            return False

        if self.last_signal_monotonic <= 0:
            return False

        age_ms = (time.monotonic() - self.last_signal_monotonic) * 1000.0
        if age_ms <= self.stream_timeout_ms:
            self.stale = False
            return False

        if not self.stale:
            self.stale = True
            self.trip_count += 1
            self.last_trip_reason = "signal_timeout"
            self.last_trip_ts = time.time()
            return True

        return False

    def status(self, *, stream_running: bool) -> Dict[str, object]:
        age_ms = 0.0
        if self.last_signal_monotonic > 0:
            age_ms = max(0.0, (time.monotonic() - self.last_signal_monotonic) * 1000.0)
        return {
            "enabled": self.enabled,
            "stream_timeout_ms": self.stream_timeout_ms,
            "auto_estop_on_stale": self.auto_estop_on_stale,
            "stream_running": stream_running,
            "signal_age_ms": round(age_ms, 1),
            "stale": bool(stream_running and age_ms > self.stream_timeout_ms),
            "trip_count": self.trip_count,
            "last_trip_reason": self.last_trip_reason,
            "last_trip_ts": self.last_trip_ts,
        }


class LiveDiagnostics:
    def __init__(self, *, sample_rate: int, window_increment_ms: int, full_scale: float) -> None:
        self.sample_rate = int(sample_rate)
        self.expected_interval_ms = float(window_increment_ms)
        self.full_scale = max(float(full_scale), 1e-6)
        self.last_payload: Dict[str, object] = self._empty_payload()
        self._last_window_monotonic = 0.0
        self._process_last_ms = 0.0
        self._process_avg_ms = 0.0
        self._process_max_ms = 0.0
        self._interval_last_ms = 0.0
        self._interval_jitter_ms = 0.0
        self._interval_avg_ms = 0.0
        self._dropped_windows = 0
        self._window_count = 0

    def reset(self, *, full_scale: Optional[float] = None) -> None:
        if full_scale is not None:
            self.full_scale = max(float(full_scale), 1e-6)
        self.last_payload = self._empty_payload()
        self._last_window_monotonic = 0.0
        self._process_last_ms = 0.0
        self._process_avg_ms = 0.0
        self._process_max_ms = 0.0
        self._interval_last_ms = 0.0
        self._interval_jitter_ms = 0.0
        self._interval_avg_ms = 0.0
        self._dropped_windows = 0
        self._window_count = 0

    def mark_pipeline_start(self) -> float:
        return time.perf_counter()

    def mark_pipeline_end(self, started_at: float) -> None:
        elapsed_ms = max(0.0, (time.perf_counter() - started_at) * 1000.0)
        self._process_last_ms = elapsed_ms
        if self._window_count <= 1:
            self._process_avg_ms = elapsed_ms
        else:
            self._process_avg_ms = (self._process_avg_ms * 0.9) + (elapsed_ms * 0.1)
        self._process_max_ms = max(self._process_max_ms, elapsed_ms)

    def on_window(self, window: np.ndarray, recent_samples: Optional[np.ndarray] = None) -> Dict[str, object]:
        now = time.monotonic()
        if self._last_window_monotonic > 0:
            interval_ms = max(0.0, (now - self._last_window_monotonic) * 1000.0)
            self._interval_last_ms = interval_ms
            self._interval_jitter_ms = interval_ms - self.expected_interval_ms
            if self._window_count <= 1:
                self._interval_avg_ms = interval_ms
            else:
                self._interval_avg_ms = (self._interval_avg_ms * 0.9) + (interval_ms * 0.1)
            if interval_ms > self.expected_interval_ms * 1.75:
                missed = max(int(round(interval_ms / max(self.expected_interval_ms, 1.0))) - 1, 1)
                self._dropped_windows += missed
        self._last_window_monotonic = now
        self._window_count += 1

        segment = recent_samples if recent_samples is not None and recent_samples.size else window
        spectrum = self._compute_spectrum(segment)
        quality = self._compute_quality(segment)

        self.last_payload = {
            "spectrum": spectrum,
            "noise": quality,
            "timing": {
                "process_last_ms": round(self._process_last_ms, 3),
                "process_avg_ms": round(self._process_avg_ms, 3),
                "process_max_ms": round(self._process_max_ms, 3),
                "interval_last_ms": round(self._interval_last_ms, 3),
                "interval_avg_ms": round(self._interval_avg_ms, 3),
                "interval_jitter_ms": round(self._interval_jitter_ms, 3),
                "dropped_windows": int(self._dropped_windows),
                "window_count": int(self._window_count),
                "expected_interval_ms": round(self.expected_interval_ms, 3),
            },
        }
        return self.last_payload

    def status(self) -> Dict[str, object]:
        age_ms = 0.0
        if self._last_window_monotonic > 0:
            age_ms = max(0.0, (time.monotonic() - self._last_window_monotonic) * 1000.0)
        payload = dict(self.last_payload)
        timing = dict(payload.get("timing") or {})
        timing["signal_age_ms"] = round(age_ms, 1)
        payload["timing"] = timing
        return payload

    def _empty_payload(self) -> Dict[str, object]:
        return {
            "spectrum": {
                "freq_hz": [],
                "mag_db": [],
                "segment_ms": 0.0,
            },
            "noise": {
                "hum_50_db": 0.0,
                "hum_60_db": 0.0,
                "drift_db": 0.0,
                "clip_pct": 0.0,
                "crest_factor": 0.0,
            },
            "timing": {
                "process_last_ms": 0.0,
                "process_avg_ms": 0.0,
                "process_max_ms": 0.0,
                "interval_last_ms": 0.0,
                "interval_avg_ms": 0.0,
                "interval_jitter_ms": 0.0,
                "dropped_windows": 0,
                "window_count": 0,
                "expected_interval_ms": round(self.expected_interval_ms, 3),
                "signal_age_ms": 0.0,
            },
        }

    def _compute_spectrum(self, segment: np.ndarray) -> Dict[str, object]:
        if segment.ndim != 2 or segment.size == 0 or segment.shape[1] < 8:
            return {"freq_hz": [], "mag_db": [], "segment_ms": 0.0}

        n_samples = int(segment.shape[1])
        centered = segment.astype(np.float64) - np.mean(segment, axis=1, keepdims=True)
        win = np.hanning(n_samples)
        spectrum = np.fft.rfft(centered * win, axis=1)
        power = np.mean(np.abs(spectrum) ** 2, axis=0)
        freq = np.fft.rfftfreq(n_samples, d=1.0 / max(self.sample_rate, 1))
        max_hz = min(self.sample_rate / 2.0, float(freq[-1]) if freq.size else 0.0)
        if max_hz <= 0:
            return {"freq_hz": [], "mag_db": [], "segment_ms": 0.0}

        freq_target = np.linspace(0.0, max_hz, 96)
        power_interp = np.interp(freq_target, freq, power)
        ref = max(float(np.max(power_interp)), 1e-12)
        mag_db = 10.0 * np.log10(np.maximum(power_interp / ref, 1e-12))
        return {
            "freq_hz": [round(float(v), 3) for v in freq_target.tolist()],
            "mag_db": [round(float(v), 3) for v in mag_db.tolist()],
            "segment_ms": round((n_samples / max(self.sample_rate, 1)) * 1000.0, 2),
        }

    def _compute_quality(self, segment: np.ndarray) -> Dict[str, float]:
        if segment.ndim != 2 or segment.size == 0:
            return {
                "hum_50_db": 0.0,
                "hum_60_db": 0.0,
                "drift_db": 0.0,
                "clip_pct": 0.0,
                "crest_factor": 0.0,
            }

        centered = segment.astype(np.float64) - np.mean(segment, axis=1, keepdims=True)
        n_samples = int(segment.shape[1])
        win = np.hanning(n_samples)
        spectrum = np.fft.rfft(centered * win, axis=1)
        power = np.mean(np.abs(spectrum) ** 2, axis=0)
        freq = np.fft.rfftfreq(n_samples, d=1.0 / max(self.sample_rate, 1))
        total_power = float(np.sum(power))

        def band_power(center_hz: float, half_width_hz: float = 1.5) -> float:
            if freq.size == 0:
                return 0.0
            mask = np.abs(freq - center_hz) <= half_width_hz
            if not np.any(mask):
                idx = int(np.argmin(np.abs(freq - center_hz)))
                return float(power[idx])
            return float(np.sum(power[mask]))

        hum50 = band_power(50.0)
        hum60 = band_power(60.0)
        drift = float(np.sum(power[freq <= 1.0])) if freq.size else 0.0

        abs_segment = np.abs(segment.astype(np.float64))
        clip_pct = float(np.mean(abs_segment >= (0.95 * self.full_scale)) * 100.0)
        rms = float(np.sqrt(np.mean(segment.astype(np.float64) ** 2)))
        peak = float(np.max(abs_segment)) if abs_segment.size else 0.0
        crest_factor = peak / max(rms, 1e-9)

        return {
            "hum_50_db": round(_safe_db(hum50 / max(total_power, 1e-12)), 3),
            "hum_60_db": round(_safe_db(hum60 / max(total_power, 1e-12)), 3),
            "drift_db": round(_safe_db(drift / max(total_power, 1e-12)), 3),
            "clip_pct": round(clip_pct, 3),
            "crest_factor": round(float(crest_factor), 3),
        }
