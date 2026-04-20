"""Profile-specific live analyzers for non-EMG biosignals."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from biosignal_profiles import BiosignalProfile
from config import config


class ProfileWindowAnalyzer:
    """Maintain short signal history and emit a decoded state per profile."""

    def __init__(self, profile: BiosignalProfile):
        self.profile = profile
        self.sample_rate = config.sample_rate
        self.n_channels = config.n_channels
        self.history_size = max(self.sample_rate * 10, config.window_size_samples)
        self._buf = np.zeros((self.n_channels, self.history_size), dtype=np.float64)
        self._buf_head = 0
        self._total_samples = 0
        self._initialized = False

    def on_window(self, window: np.ndarray) -> Optional[Dict]:
        if not self._initialized:
            self._append(window)
            self._initialized = True
        else:
            inc = min(config.window_increment_samples, window.shape[1])
            self._append(window[:, -inc:])

        if self._total_samples < config.window_size_samples:
            return None

        analysis = {
            "eeg": self._analyze_eeg,
            "ecg": self._analyze_ecg,
            "eog": self._analyze_eog,
            "eda": self._analyze_eda,
            "ppg": self._analyze_ppg,
            "resp": self._analyze_resp,
            "temp": self._analyze_temp,
        }.get(self.profile.key, self._analyze_generic)

        return analysis()

    def _append(self, samples: np.ndarray) -> None:
        n_new = samples.shape[1]
        for i in range(n_new):
            self._buf[:, self._buf_head % self.history_size] = samples[:, i]
            self._buf_head += 1
        self._total_samples += n_new

    def _get_recent(self, seconds: float) -> np.ndarray:
        n = min(int(seconds * self.sample_rate), self.history_size, self._total_samples)
        if n <= 0:
            return self._buf[:, :0].copy()
        end = self._buf_head
        idx = np.arange(end - n, end) % self.history_size
        return self._buf[:, idx].copy()

    def _label_result(self, label: str, confidence: float, metrics: Dict[str, float], summary: str) -> Dict:
        labels = list(self.profile.class_labels)
        class_idx = labels.index(label) if label in labels else 0
        return {
            "label": label,
            "class_idx": class_idx,
            "confidence": float(np.clip(confidence, 0.0, 0.99)),
            "metrics": metrics,
            "summary": summary,
            "profile": self.profile.key,
        }

    def _dominant_freq(self, signal: np.ndarray, fmin: float, fmax: float) -> Tuple[float, float]:
        if signal.size < 8:
            return 0.0, 0.0
        x = signal.astype(np.float64) - float(np.mean(signal))
        spec = np.fft.rfft(x)
        freqs = np.fft.rfftfreq(x.size, d=1.0 / self.sample_rate)
        power = np.abs(spec) ** 2
        mask = (freqs >= fmin) & (freqs <= fmax)
        if not np.any(mask):
            return 0.0, 0.0
        mf = freqs[mask]
        mp = power[mask]
        idx = int(np.argmax(mp))
        return float(mf[idx]), float(mp[idx])

    def _band_powers(self, signal: np.ndarray) -> Dict[str, float]:
        x = signal.astype(np.float64) - float(np.mean(signal))
        spec = np.fft.rfft(x)
        freqs = np.fft.rfftfreq(x.size, d=1.0 / self.sample_rate)
        power = np.abs(spec) ** 2
        bands = {
            "delta dominant": (1.0, 4.0),
            "theta dominant": (4.0, 8.0),
            "alpha dominant": (8.0, 12.0),
            "beta dominant": (12.0, 30.0),
            "gamma dominant": (30.0, 45.0),
        }
        out = {}
        for name, (lo, hi) in bands.items():
            mask = (freqs >= lo) & (freqs < hi)
            out[name] = float(power[mask].sum()) if np.any(mask) else 0.0
        return out

    def _find_peaks(self, signal: np.ndarray, min_interval_s: float, threshold: Optional[float] = None) -> np.ndarray:
        if signal.size < 3:
            return np.array([], dtype=int)
        x = signal.astype(np.float64)
        candidates = np.where((x[1:-1] > x[:-2]) & (x[1:-1] >= x[2:]))[0] + 1
        if threshold is None:
            threshold = float(np.mean(x) + 0.35 * np.std(x))
        candidates = candidates[x[candidates] >= threshold]
        if candidates.size == 0:
            return candidates

        min_interval = max(1, int(min_interval_s * self.sample_rate))
        selected = []
        last = -min_interval
        for idx in candidates:
            if idx - last >= min_interval:
                selected.append(idx)
                last = idx
            elif x[idx] > x[selected[-1]]:
                selected[-1] = idx
                last = idx
        return np.array(selected, dtype=int)

    def _analyze_eeg(self) -> Dict:
        hist = self._get_recent(2.0)
        signal = np.mean(hist, axis=0)
        bands = self._band_powers(signal)
        ordered = sorted(bands.items(), key=lambda kv: kv[1], reverse=True)
        total = sum(bands.values()) + 1e-9
        top_label, top_power = ordered[0]
        second_power = ordered[1][1] if len(ordered) > 1 else 0.0
        label = top_label if top_power / total > 0.28 else "mixed activity"
        confidence = max(top_power - second_power, 0.0) / total + top_power / total * 0.35
        dom_freq, _ = self._dominant_freq(signal, 1.0, 45.0)
        metrics = {
            "dominant_frequency_hz": round(dom_freq, 2),
            "alpha_ratio": round(bands["alpha dominant"] / total, 3),
            "beta_ratio": round(bands["beta dominant"] / total, 3),
        }
        summary = f"Dominant band {label} at {dom_freq:.1f} Hz"
        return self._label_result(label, confidence, metrics, summary)

    def _analyze_ecg(self) -> Dict:
        hist = self._get_recent(8.0)
        signal = np.mean(hist[: min(3, hist.shape[0])], axis=0)
        signal = signal - np.mean(signal)
        threshold = float(np.mean(signal) + 0.55 * np.std(signal))
        peaks = self._find_peaks(signal, min_interval_s=0.4, threshold=threshold)
        if peaks.size < 2:
            return self._label_result(
                "artifact",
                0.25,
                {"heart_rate_bpm": 0.0, "peak_count": float(peaks.size)},
                "Insufficient R-peaks in the current buffer",
            )

        rr = np.diff(peaks) / self.sample_rate
        bpm = 60.0 / max(float(np.mean(rr)), 1e-6)
        variability = float(np.std(rr) / max(np.mean(rr), 1e-6))
        if variability > 0.18:
            label = "irregular rhythm"
        elif bpm > 105.0:
            label = "elevated rate"
        elif bpm < 55.0:
            label = "slow rate"
        else:
            label = "steady rhythm"
        confidence = min(0.95, 0.45 + 0.35 * min(peaks.size / 6.0, 1.0) + 0.2 * max(0.0, 1.0 - variability))
        metrics = {
            "heart_rate_bpm": round(bpm, 1),
            "rr_variability": round(variability, 3),
            "peak_count": float(peaks.size),
        }
        summary = f"{bpm:.0f} bpm, RR variability {variability:.2f}"
        return self._label_result(label, confidence, metrics, summary)

    def _analyze_eog(self) -> Dict:
        hist = self._get_recent(1.5)
        horizontal = hist[0] - hist[1] if hist.shape[0] > 1 else hist[0]
        vertical = hist[2] - hist[3] if hist.shape[0] > 3 else hist[min(1, hist.shape[0] - 1)]
        h_mean = float(np.mean(horizontal))
        v_mean = float(np.mean(vertical))
        h_peak = float(np.max(np.abs(horizontal)))
        v_peak = float(np.max(np.abs(vertical)))

        if v_peak > 140.0 and v_peak > h_peak * 1.1:
            label = "blink"
            confidence = min(0.98, v_peak / 220.0)
        elif h_mean > 35.0:
            label = "right saccade"
            confidence = min(0.95, abs(h_mean) / 120.0)
        elif h_mean < -35.0:
            label = "left saccade"
            confidence = min(0.95, abs(h_mean) / 120.0)
        elif v_mean > 30.0:
            label = "up gaze"
            confidence = min(0.9, abs(v_mean) / 100.0)
        elif v_mean < -30.0:
            label = "down gaze"
            confidence = min(0.9, abs(v_mean) / 100.0)
        else:
            label = "fixation"
            confidence = 0.7
        metrics = {
            "horizontal_bias_uv": round(h_mean, 1),
            "vertical_bias_uv": round(v_mean, 1),
            "blink_peak_uv": round(v_peak, 1),
        }
        summary = f"H {h_mean:.0f} uV, V {v_mean:.0f} uV"
        return self._label_result(label, confidence, metrics, summary)

    def _analyze_eda(self) -> Dict:
        hist = self._get_recent(8.0)
        signal = np.mean(hist, axis=0)
        slope = float(signal[-1] - signal[0])
        span = float(np.max(signal) - np.min(signal))
        recent = float(np.mean(signal[-min(signal.size, self.sample_rate):]))
        baseline = float(np.mean(signal))

        if span > 0.35 and recent > baseline + 0.08:
            label = "phasic peak"
            confidence = min(0.95, span / 0.8)
        elif slope > 0.08:
            label = "rising arousal"
            confidence = min(0.9, slope / 0.25)
        elif slope < -0.08:
            label = "recovering"
            confidence = min(0.9, abs(slope) / 0.25)
        else:
            label = "stable tonic"
            confidence = 0.7
        metrics = {
            "tonic_level": round(baseline, 3),
            "phasic_span": round(span, 3),
            "slope": round(slope, 3),
        }
        summary = f"Tonic {baseline:.2f} {self.profile.units}, span {span:.2f}"
        return self._label_result(label, confidence, metrics, summary)

    def _analyze_ppg(self) -> Dict:
        hist = self._get_recent(8.0)
        signal = np.mean(hist, axis=0)
        signal = signal - np.mean(signal)
        span = float(np.max(signal) - np.min(signal))
        threshold = float(np.mean(signal) + 0.25 * np.std(signal))
        peaks = self._find_peaks(signal, min_interval_s=0.4, threshold=threshold)

        if peaks.size < 2 or span < 0.1:
            return self._label_result(
                "weak pulse",
                min(0.7, span / 0.2 if span > 0 else 0.2),
                {"pulse_rate_bpm": 0.0, "pulse_amplitude": round(span, 3)},
                "Pulse waveform is weak in the current buffer",
            )

        rr = np.diff(peaks) / self.sample_rate
        bpm = 60.0 / max(float(np.mean(rr)), 1e-6)
        if bpm > 105.0:
            label = "elevated pulse"
        elif bpm < 55.0:
            label = "slow pulse"
        else:
            label = "pulse stable"
        confidence = min(0.96, 0.45 + 0.25 * min(peaks.size / 6.0, 1.0) + 0.25 * min(span / 0.9, 1.0))
        metrics = {
            "pulse_rate_bpm": round(bpm, 1),
            "pulse_amplitude": round(span, 3),
            "peak_count": float(peaks.size),
        }
        summary = f"{bpm:.0f} bpm, amplitude {span:.2f}"
        return self._label_result(label, confidence, metrics, summary)

    def _analyze_resp(self) -> Dict:
        hist = self._get_recent(10.0)
        signal = np.mean(hist, axis=0)
        dom_freq, dom_power = self._dominant_freq(signal, 0.05, 1.2)
        rate = dom_freq * 60.0
        phase = "inhalation phase" if signal[-1] - signal[-min(signal.size, 10)] >= 0 else "exhalation phase"
        total_power = float(np.sum(np.abs(np.fft.rfft(signal - np.mean(signal))) ** 2)) + 1e-9
        confidence = min(0.95, 0.4 + 0.45 * (dom_power / total_power))
        if rate > 24.0:
            label = "fast breathing"
        elif rate < 8.0 and rate > 0.0:
            label = "slow breathing"
        elif rate == 0.0:
            label = "steady breathing"
            confidence = 0.25
        else:
            label = phase
        metrics = {
            "resp_rate_bpm": round(rate, 1),
            "cycle_frequency_hz": round(dom_freq, 3),
        }
        summary = f"{rate:.1f} breaths/min"
        return self._label_result(label, confidence, metrics, summary)

    def _analyze_temp(self) -> Dict:
        hist = self._get_recent(10.0)
        signal = np.mean(hist, axis=0)
        slope = float(signal[-1] - signal[0])
        mean_temp = float(np.mean(signal))
        if slope > 0.004:
            label = "warming"
            confidence = min(0.9, slope / 0.02)
        elif slope < -0.004:
            label = "cooling"
            confidence = min(0.9, abs(slope) / 0.02)
        else:
            label = "stable temperature"
            confidence = 0.75
        metrics = {
            "mean_level": round(mean_temp, 4),
            "drift": round(slope, 4),
        }
        summary = f"Drift {slope:+.4f} {self.profile.units}"
        return self._label_result(label, confidence, metrics, summary)

    def _analyze_generic(self) -> Dict:
        hist = self._get_recent(2.0)
        rms = np.sqrt(np.mean(hist.astype(np.float64) ** 2, axis=1))
        mean_rms = float(np.mean(rms))
        label = self.profile.class_labels[0] if self.profile.class_labels else "active"
        return self._label_result(
            label,
            0.5,
            {"mean_rms": round(mean_rms, 4)},
            f"Mean activity {mean_rms:.4f} {self.profile.units}",
        )
