"""Selected-chunk signal analysis for KYMA."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from scipy import signal as scipy_signal

    _SCIPY_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - optional runtime dependency
    scipy_signal = None  # type: ignore[assignment]
    _SCIPY_ERROR = exc


def _round_scalar(value: Any, digits: int = 6) -> float:
    return round(float(value), digits)


def _round_list(values: np.ndarray, digits: int = 6) -> List[float]:
    return [round(float(v), digits) for v in np.asarray(values, dtype=np.float64).tolist()]


def _round_grid(values: np.ndarray, digits: int = 5) -> List[List[float]]:
    arr = np.asarray(values, dtype=np.float64)
    return [[round(float(v), digits) for v in row] for row in arr.tolist()]


class SignalWorkshop:
    """Backend DSP analysis for a selected biosignal chunk."""

    MAX_CHANNELS = 16
    MAX_SAMPLES = 4096
    LAPLACE_SIGMA = (0.0, 0.5, 1.0, 2.0, 4.0, 6.0)

    @property
    def available(self) -> bool:
        return scipy_signal is not None

    @property
    def error_message(self) -> str:
        if self.available:
            return ""
        return f"{type(_SCIPY_ERROR).__name__}: {_SCIPY_ERROR}" if _SCIPY_ERROR else "SciPy is not available"

    def status(self) -> Dict[str, Any]:
        return {
            "available": self.available,
            "last_error": self.error_message,
            "views": [
                "fft",
                "psd",
                "spectrogram",
                "autocorrelation",
                "histogram",
                "envelope",
                "correlation",
                "laplace",
            ],
        }

    def analyze(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if scipy_signal is None:
            raise RuntimeError(f"Signal workshop requires SciPy: {self.error_message}")

        channels, sample_rate, labels, focus_idx = self._normalize_payload(payload)
        focus = channels[focus_idx]
        duration_s = focus.size / sample_rate

        fft_freq, fft_mag = self._fft_payload(focus, sample_rate)
        psd_freq, psd_db, band_metrics = self._psd_payload(focus, sample_rate)
        spec_time, spec_freq, spec_db = self._spectrogram_payload(focus, sample_rate)
        auto_lags, auto_vals = self._autocorrelation_payload(focus, sample_rate)
        hist_bins, hist_counts = self._histogram_payload(focus)
        env_time, env_signal, env_curve = self._envelope_payload(focus, sample_rate)
        corr_matrix = self._correlation_payload(channels)
        laplace_sigma, laplace_freq, laplace_db = self._laplace_payload(focus, sample_rate)
        summary = self._summary_payload(
            channels=channels,
            focus=focus,
            labels=labels,
            focus_idx=focus_idx,
            sample_rate=sample_rate,
            duration_s=duration_s,
            band_metrics=band_metrics,
            fft_freq=fft_freq,
            fft_mag=fft_mag,
        )

        return {
            "ok": True,
            "available": True,
            "profile": str(payload.get("profile") or "").strip().lower() or "signal",
            "sample_rate": int(round(sample_rate)),
            "channel_labels": labels,
            "focus_channel": int(focus_idx),
            "selection_label": str(payload.get("selection_label") or "").strip(),
            "selection_start_s": payload.get("selection_start_s"),
            "selection_end_s": payload.get("selection_end_s"),
            "summary": summary,
            "fft": {
                "freq_hz": _round_list(fft_freq, 4),
                "mag_db": _round_list(fft_mag, 4),
            },
            "psd": {
                "freq_hz": _round_list(psd_freq, 4),
                "psd_db": _round_list(psd_db, 4),
                "bands": band_metrics,
            },
            "spectrogram": {
                "time_s": _round_list(spec_time, 4),
                "freq_hz": _round_list(spec_freq, 4),
                "mag_db_grid": _round_grid(spec_db, 3),
            },
            "autocorrelation": {
                "lags_ms": _round_list(auto_lags * 1000.0, 3),
                "values": _round_list(auto_vals, 5),
            },
            "histogram": {
                "bin_centers": _round_list(hist_bins, 5),
                "counts": [int(v) for v in hist_counts.tolist()],
            },
            "envelope": {
                "time_ms": _round_list(env_time * 1000.0, 3),
                "signal": _round_list(env_signal, 5),
                "envelope": _round_list(env_curve, 5),
            },
            "correlation": {
                "labels": labels,
                "matrix": _round_grid(corr_matrix, 4),
            },
            "laplace": {
                "sigma": _round_list(laplace_sigma, 3),
                "freq_hz": _round_list(laplace_freq, 4),
                "mag_db_grid": _round_grid(laplace_db, 3),
            },
        }

    def _normalize_payload(self, payload: Dict[str, Any]) -> Tuple[np.ndarray, float, List[str], int]:
        raw_channels = payload.get("channels") or []
        if not isinstance(raw_channels, list) or not raw_channels:
            raise ValueError("Signal workshop requires one or more channel arrays")
        if len(raw_channels) > self.MAX_CHANNELS:
            raise ValueError(f"Signal workshop supports up to {self.MAX_CHANNELS} channels per request")

        arrays = []
        for row in raw_channels:
            arr = np.asarray(row, dtype=np.float64).flatten()
            if arr.size < 8:
                raise ValueError("Selected chunk is too short for analysis")
            arrays.append(arr[: self.MAX_SAMPLES])

        n_samples = min(arr.size for arr in arrays)
        channels = np.vstack([arr[:n_samples] for arr in arrays])
        sample_rate = float(payload.get("sample_rate") or 250.0)
        if sample_rate <= 1.0:
            raise ValueError("Invalid sample rate for signal workshop")

        labels = [str(x or "").strip() for x in (payload.get("channel_labels") or [])[: channels.shape[0]]]
        while len(labels) < channels.shape[0]:
            labels.append(f"CH{len(labels) + 1}")

        focus_idx = int(payload.get("focus_channel") or 0)
        focus_idx = max(0, min(focus_idx, channels.shape[0] - 1))
        return channels, sample_rate, labels, focus_idx

    def _fft_payload(self, signal: np.ndarray, sample_rate: float) -> Tuple[np.ndarray, np.ndarray]:
        centered = signal - np.mean(signal)
        spec = np.fft.rfft(centered * np.hanning(centered.size))
        freq = np.fft.rfftfreq(centered.size, d=1.0 / sample_rate)
        mag = np.abs(spec)
        ref = max(float(np.max(mag)), 1e-12)
        mag_db = 20.0 * np.log10(np.maximum(mag / ref, 1e-12))
        return freq, mag_db

    def _psd_payload(self, signal: np.ndarray, sample_rate: float) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        nperseg = min(max(64, int(signal.size // 3)), int(signal.size))
        freq, psd = scipy_signal.welch(signal, fs=sample_rate, nperseg=nperseg, scaling="density")
        psd_db = 10.0 * np.log10(np.maximum(psd, 1e-12))

        def band(low: float, high: float) -> float:
            mask = (freq >= low) & (freq < high)
            return float(np.trapz(psd[mask], freq[mask])) if np.any(mask) else 0.0

        bands = {
            "delta": _round_scalar(band(1.0, 4.0), 6),
            "theta": _round_scalar(band(4.0, 8.0), 6),
            "alpha": _round_scalar(band(8.0, 12.0), 6),
            "beta": _round_scalar(band(12.0, 30.0), 6),
            "gamma": _round_scalar(band(30.0, min(45.0, sample_rate / 2.0)), 6),
        }
        return freq, psd_db, bands

    def _spectrogram_payload(self, signal: np.ndarray, sample_rate: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        nperseg = min(128, max(32, int(signal.size // 4)))
        noverlap = max(0, int(nperseg * 0.75))
        freq, time_s, spec = scipy_signal.spectrogram(
            signal,
            fs=sample_rate,
            nperseg=nperseg,
            noverlap=noverlap,
            scaling="density",
            mode="magnitude",
        )
        spec_db = 20.0 * np.log10(np.maximum(spec, 1e-12))
        return time_s, freq, spec_db

    def _autocorrelation_payload(self, signal: np.ndarray, sample_rate: float) -> Tuple[np.ndarray, np.ndarray]:
        centered = signal - np.mean(signal)
        corr = np.correlate(centered, centered, mode="full")
        corr = corr[corr.size // 2 :]
        if corr[0] != 0:
            corr = corr / corr[0]
        max_lag = min(int(sample_rate * 1.0), corr.size)
        lags = np.arange(max_lag, dtype=np.float64) / sample_rate
        return lags, corr[:max_lag]

    def _histogram_payload(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        counts, edges = np.histogram(signal, bins=24)
        centers = (edges[:-1] + edges[1:]) * 0.5
        return centers, counts

    def _envelope_payload(self, signal: np.ndarray, sample_rate: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        analytic = scipy_signal.hilbert(signal)
        envelope = np.abs(analytic)
        time_axis = np.arange(signal.size, dtype=np.float64) / sample_rate
        return time_axis, signal, envelope

    def _correlation_payload(self, channels: np.ndarray) -> np.ndarray:
        if channels.shape[0] == 1:
            return np.asarray([[1.0]], dtype=np.float64)
        with np.errstate(invalid="ignore"):
            corr = np.corrcoef(channels)
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        return corr

    def _laplace_payload(self, signal: np.ndarray, sample_rate: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        time_axis = np.arange(signal.size, dtype=np.float64) / sample_rate
        max_hz = min(sample_rate / 2.0, 60.0)
        freq = np.linspace(0.0, max_hz, 72)
        sigma = np.asarray(self.LAPLACE_SIGMA, dtype=np.float64)
        values = np.zeros((sigma.size, freq.size), dtype=np.float64)

        for i, damp in enumerate(sigma):
            damp_vec = np.exp(-damp * time_axis)
            for j, hz in enumerate(freq):
                kernel = damp_vec * np.exp(-1j * 2.0 * np.pi * hz * time_axis)
                values[i, j] = np.abs(np.sum(signal * kernel) / max(signal.size, 1))

        ref = max(float(np.max(values)), 1e-12)
        mag_db = 20.0 * np.log10(np.maximum(values / ref, 1e-12))
        return sigma, freq, mag_db

    def _summary_payload(
        self,
        *,
        channels: np.ndarray,
        focus: np.ndarray,
        labels: List[str],
        focus_idx: int,
        sample_rate: float,
        duration_s: float,
        band_metrics: Dict[str, float],
        fft_freq: np.ndarray,
        fft_mag: np.ndarray,
    ) -> Dict[str, Any]:
        centered = focus - np.mean(focus)
        derivative = np.diff(focus) if focus.size > 1 else np.zeros(1, dtype=np.float64)
        area = np.trapz(focus, dx=1.0 / sample_rate)
        dominant_idx = int(np.argmax(fft_mag)) if fft_mag.size else 0
        spectral_centroid = (
            float(np.sum(fft_freq * np.maximum(fft_mag, 0.0)) / np.maximum(np.sum(np.maximum(fft_mag, 0.0)), 1e-12))
            if fft_mag.size
            else 0.0
        )
        zero_crossings = int(np.sum(np.diff(np.signbit(centered)).astype(int) != 0))
        return {
            "channels": int(channels.shape[0]),
            "samples": int(focus.size),
            "duration_ms": _round_scalar(duration_s * 1000.0, 3),
            "focus_channel": int(focus_idx),
            "focus_label": labels[focus_idx],
            "mean": _round_scalar(np.mean(focus), 6),
            "rms": _round_scalar(np.sqrt(np.mean(np.square(focus))), 6),
            "std": _round_scalar(np.std(focus), 6),
            "min": _round_scalar(np.min(focus), 6),
            "max": _round_scalar(np.max(focus), 6),
            "peak_to_peak": _round_scalar(np.ptp(focus), 6),
            "area": _round_scalar(area, 6),
            "derivative_rms": _round_scalar(np.sqrt(np.mean(np.square(derivative))), 6),
            "zero_crossings": int(zero_crossings),
            "dominant_frequency_hz": _round_scalar(fft_freq[dominant_idx] if fft_freq.size else 0.0, 4),
            "spectral_centroid_hz": _round_scalar(spectral_centroid, 4),
            "band_metrics": band_metrics,
        }
