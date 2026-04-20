"""Profile-aware calibration routines for KYMA biosignal modes."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from config import config

logger = logging.getLogger(__name__)


class CalibStage(str, Enum):
    IDLE = "idle"
    CHANNEL_CHECK = "channel_check"
    BASELINE = "baseline"
    TASK_CHECK = "task_check"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass(frozen=True)
class CalibrationProtocol:
    key: str
    title: str
    summary: str
    baseline_seconds: float
    task_seconds: float
    baseline_prompt: str
    task_prompt: str
    instructions: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "title": self.title,
            "summary": self.summary,
            "baseline_seconds": self.baseline_seconds,
            "task_seconds": self.task_seconds,
            "baseline_prompt": self.baseline_prompt,
            "task_prompt": self.task_prompt,
            "instructions": list(self.instructions),
        }


PROTOCOLS: Dict[str, CalibrationProtocol] = {
    "emg": CalibrationProtocol(
        key="emg",
        title="EMG contact and activation",
        summary="Checks channel contact, resting baseline, and multi-channel muscle activation.",
        baseline_seconds=3.0,
        task_seconds=2.5,
        baseline_prompt="Relax your forearm and stay still for 3 seconds.",
        task_prompt="Make a firm fist and hold it briefly.",
        instructions=(
            "Stay still so channel contact can be checked.",
            "Keep your forearm relaxed for the baseline capture.",
            "Produce one clear contraction during the task check.",
        ),
    ),
    "eeg": CalibrationProtocol(
        key="eeg",
        title="EEG baseline and alpha response",
        summary="Checks channel activity, quiet baseline, and eyes-closed alpha increase.",
        baseline_seconds=3.0,
        task_seconds=3.0,
        baseline_prompt="Keep your eyes open, stay relaxed, and minimize movement.",
        task_prompt="Close your eyes and relax until the check completes.",
        instructions=(
            "Stay still and keep facial muscles relaxed.",
            "Record a quiet eyes-open baseline.",
            "Close your eyes to check posterior alpha response.",
        ),
    ),
    "ecg": CalibrationProtocol(
        key="ecg",
        title="ECG contact and rhythm check",
        summary="Checks lead contact, baseline stability, and heartbeat visibility.",
        baseline_seconds=4.0,
        task_seconds=5.0,
        baseline_prompt="Stand or sit still and breathe normally.",
        task_prompt="Keep still for a short rhythm check.",
        instructions=(
            "Reduce motion so the baseline is clean.",
            "Capture a stable resting segment.",
            "Verify that R-peaks are visible and regular enough to track.",
        ),
    ),
    "eog": CalibrationProtocol(
        key="eog",
        title="EOG fixation and eye motion",
        summary="Checks baseline drift, horizontal eye motion, and blink visibility.",
        baseline_seconds=2.5,
        task_seconds=2.5,
        baseline_prompt="Look straight ahead and keep your eyes still.",
        task_prompt="Look left, then right, and blink once.",
        instructions=(
            "Hold a steady fixation for the baseline.",
            "Move your eyes without moving your head.",
            "Include one clear blink during the task window.",
        ),
    ),
    "eda": CalibrationProtocol(
        key="eda",
        title="EDA contact and response check",
        summary="Checks skin contact, tonic stability, and a small phasic response.",
        baseline_seconds=4.0,
        task_seconds=4.0,
        baseline_prompt="Keep both sensors in steady contact and stay relaxed.",
        task_prompt="Take one deep breath or perform one brief mental stressor.",
        instructions=(
            "Hold the electrodes still during the tonic baseline.",
            "Keep contact pressure constant.",
            "Produce one small arousal response during the task window.",
        ),
    ),
    "ppg": CalibrationProtocol(
        key="ppg",
        title="PPG pulse check",
        summary="Checks optical contact, pulse amplitude, and beat tracking.",
        baseline_seconds=3.0,
        task_seconds=4.0,
        baseline_prompt="Keep the sensor still and avoid squeezing.",
        task_prompt="Continue holding still while the pulse check runs.",
        instructions=(
            "Minimize finger motion.",
            "Capture a stable baseline amplitude.",
            "Verify that several pulse peaks are visible.",
        ),
    ),
    "resp": CalibrationProtocol(
        key="resp",
        title="Respiration cycle check",
        summary="Checks baseline noise and one clear inhale/exhale cycle.",
        baseline_seconds=3.0,
        task_seconds=4.0,
        baseline_prompt="Breathe normally and stay relaxed.",
        task_prompt="Take one slow inhale and exhale.",
        instructions=(
            "Keep the belt or sensor steady.",
            "Capture a normal breathing baseline.",
            "Produce one clear respiratory cycle during the task window.",
        ),
    ),
    "temp": CalibrationProtocol(
        key="temp",
        title="Temperature contact check",
        summary="Checks sensor contact, drift stability, and noise floor.",
        baseline_seconds=4.0,
        task_seconds=4.0,
        baseline_prompt="Maintain steady contact with the temperature sensor.",
        task_prompt="Continue holding steady while stability is measured.",
        instructions=(
            "Keep the sensor in constant contact.",
            "Allow the baseline to settle.",
            "Verify that the temperature trace remains stable.",
        ),
    ),
}


class CalibrationManager:
    def __init__(self):
        self.stage = CalibStage.IDLE
        self.results: Dict[str, Any] = {}
        self._status_cb: Optional[Callable[[Dict], None]] = None

    def set_status_callback(self, fn: Callable[[Dict], None]) -> None:
        self._status_cb = fn

    def reset(self) -> None:
        self.stage = CalibStage.IDLE
        self.results = {}

    def describe_protocol(self, profile_key: Optional[str] = None) -> Dict[str, Any]:
        key = (profile_key or config.signal_profile_name or "emg").lower()
        protocol = PROTOCOLS.get(key, PROTOCOLS["emg"])
        return protocol.to_dict()

    def status(self) -> Dict[str, Any]:
        return {
            "stage": self.stage.value,
            "profile": config.signal_profile_name,
            "protocol": self.describe_protocol(),
            "results": self.results,
        }

    async def run_calibration(self, stream) -> Dict[str, Any]:
        profile = config.signal_profile
        protocol = PROTOCOLS.get(profile.key, PROTOCOLS["emg"])
        self.reset()
        self.results.update(
            {
                "profile": profile.key,
                "profile_name": profile.display_name,
                "protocol": protocol.to_dict(),
            }
        )

        self.stage = CalibStage.CHANNEL_CHECK
        self._emit(
            f"{protocol.title}: {protocol.instructions[0]}",
            {"instruction": protocol.instructions[0]},
        )
        await asyncio.sleep(0.4)

        window = stream.get_window()
        if window is None:
            return self._fail("No data received. Start the stream and try again.")

        channel_report = self._assess_channel_quality(window)
        self.results.update(channel_report)
        self._emit(
            f"Channels: {channel_report['good_channels']}/{config.n_channels} look usable",
            {
                "quality": channel_report["channel_quality"],
                "issues": channel_report["channel_issues"],
                "channel_rms": channel_report["channel_rms"],
            },
        )
        if channel_report["good_channels"] < 2:
            return self._fail("Too few usable channels. Check contact and wiring.", channel_report)

        self.stage = CalibStage.BASELINE
        self._emit(
            protocol.baseline_prompt,
            {"instruction": protocol.instructions[1], "duration_s": protocol.baseline_seconds},
        )
        baseline_windows = await self._collect_windows(stream, protocol.baseline_seconds)
        if not baseline_windows:
            return self._fail("Could not collect baseline data.", {"duration_s": protocol.baseline_seconds})

        baseline = self._summarize_baseline(baseline_windows)
        self.results.update(baseline)
        self._emit(baseline["message"], baseline["metrics"])

        self.stage = CalibStage.TASK_CHECK
        self._emit(
            protocol.task_prompt,
            {"instruction": protocol.instructions[2], "duration_s": protocol.task_seconds},
        )
        task_windows = await self._collect_windows(stream, protocol.task_seconds)
        if not task_windows:
            return self._fail("Could not collect task-check data.", {"duration_s": protocol.task_seconds})

        task_report = self._analyze_task_window(profile.key, task_windows, baseline)
        self.results["task_check"] = task_report
        self._emit(task_report["message"], task_report["metrics"])
        if not task_report["success"]:
            return self._fail(task_report["message"], task_report["metrics"])
        if task_report.get("warning"):
            self._emit(task_report["warning"], {"warning": task_report["warning"], **task_report["metrics"]})

        self.stage = CalibStage.COMPLETE
        self.results["success"] = True
        self._emit("Calibration complete.", self.results)
        return self.results

    async def _collect_windows(self, stream, duration_s: float) -> List[np.ndarray]:
        windows: List[np.ndarray] = []
        end = time.monotonic() + max(duration_s, 0.1)
        while time.monotonic() < end:
            window = stream.get_window()
            if window is not None:
                windows.append(window)
            await asyncio.sleep(0.05)
        return windows

    def _assess_channel_quality(self, window: np.ndarray) -> Dict[str, Any]:
        profile = config.signal_profile
        rms = np.sqrt(np.mean(window.astype(np.float64) ** 2, axis=1))
        quiet_floor = max(float(profile.mute_floor) * 0.2, float(profile.metric_full_scale) * 0.005, 1e-6)
        warn_floor = max(float(profile.mute_floor) * 2.0, quiet_floor * 3.0)
        saturation = max(float(profile.display_full_scale) * 0.95, warn_floor * 20.0)

        quality: List[float] = []
        issues: List[str] = []
        for idx, value in enumerate(rms):
            label = config.channel_labels[idx] if idx < len(config.channel_labels) else f"CH{idx + 1}"
            if value < quiet_floor:
                quality.append(0.0)
                issues.append(f"{label}: very low activity or disconnected")
            elif value > saturation:
                quality.append(0.25)
                issues.append(f"{label}: saturated or clipping")
            elif value > warn_floor * 6.0:
                quality.append(0.7)
                issues.append(f"{label}: elevated baseline activity")
            else:
                quality.append(1.0)

        good_channels = sum(q >= 0.7 for q in quality)
        return {
            "channel_quality": quality,
            "channel_issues": issues,
            "channel_rms": [round(float(v), 5) for v in rms],
            "good_channels": int(good_channels),
        }

    def _flatten_windows(self, windows: List[np.ndarray]) -> np.ndarray:
        return np.concatenate([w.astype(np.float64, copy=False) for w in windows], axis=1)

    def _signal_band_ratios(self, signal: np.ndarray) -> Dict[str, float]:
        centered = signal.astype(np.float64, copy=False) - float(np.mean(signal))
        if centered.size < 8:
            return {"delta": 0.0, "theta": 0.0, "alpha": 0.0, "beta": 0.0, "gamma": 0.0}

        spec = np.fft.rfft(centered)
        freqs = np.fft.rfftfreq(centered.size, d=1.0 / config.sample_rate)
        power = np.abs(spec) ** 2
        total_mask = (freqs >= 1.0) & (freqs <= 45.0)
        total = float(power[total_mask].sum()) + 1e-9

        def band(low_hz: float, high_hz: float) -> float:
            mask = (freqs >= low_hz) & (freqs < high_hz)
            return float(power[mask].sum()) / total if np.any(mask) else 0.0

        return {
            "delta": band(1.0, 4.0),
            "theta": band(4.0, 8.0),
            "alpha": band(8.0, 12.0),
            "beta": band(12.0, 30.0),
            "gamma": band(30.0, 45.0),
        }

    def _dominant_frequency(self, signal: np.ndarray, low_hz: float, high_hz: float) -> float:
        centered = signal.astype(np.float64, copy=False) - float(np.mean(signal))
        if centered.size < 8:
            return 0.0
        spec = np.fft.rfft(centered)
        freqs = np.fft.rfftfreq(centered.size, d=1.0 / config.sample_rate)
        power = np.abs(spec) ** 2
        mask = (freqs >= low_hz) & (freqs <= high_hz)
        if not np.any(mask):
            return 0.0
        return float(freqs[mask][int(np.argmax(power[mask]))])

    def _find_peaks(self, signal: np.ndarray, min_interval_s: float, threshold: Optional[float] = None) -> np.ndarray:
        if signal.size < 3:
            return np.array([], dtype=int)
        x = signal.astype(np.float64, copy=False)
        candidates = np.where((x[1:-1] > x[:-2]) & (x[1:-1] >= x[2:]))[0] + 1
        if threshold is None:
            threshold = float(np.mean(x) + 0.35 * np.std(x))
        candidates = candidates[x[candidates] >= threshold]
        if candidates.size == 0:
            return candidates

        min_interval = max(1, int(min_interval_s * config.sample_rate))
        selected: List[int] = []
        last = -min_interval
        for idx in candidates:
            if idx - last >= min_interval:
                selected.append(int(idx))
                last = int(idx)
            elif x[idx] > x[selected[-1]]:
                selected[-1] = int(idx)
                last = int(idx)
        return np.asarray(selected, dtype=int)

    def _summarize_baseline(self, windows: List[np.ndarray]) -> Dict[str, Any]:
        flat = self._flatten_windows(windows)
        channel_mean = np.mean(flat, axis=1)
        channel_std = np.std(flat, axis=1)
        channel_rms = np.sqrt(np.mean(flat ** 2, axis=1))
        baseline_rms = float(np.sqrt(np.mean(flat ** 2)))

        metrics: Dict[str, Any] = {
            "baseline_rms": round(baseline_rms, 5),
            "channel_rms": [round(float(v), 5) for v in channel_rms],
        }
        message = f"Baseline RMS {baseline_rms:.5f} {config.signal_profile.units}"

        if config.signal_profile_name == "eeg":
            mean_signal = np.mean(flat, axis=0)
            bands = self._signal_band_ratios(mean_signal)
            dom_freq = self._dominant_frequency(mean_signal, 1.0, 45.0)
            metrics.update(
                {
                    "alpha_ratio": round(bands["alpha"], 3),
                    "beta_ratio": round(bands["beta"], 3),
                    "dominant_frequency_hz": round(dom_freq, 2),
                }
            )
            message = f"Baseline alpha ratio {bands['alpha']:.3f} at {dom_freq:.1f} Hz"
        elif config.signal_profile_name == "ecg":
            lead = np.mean(flat[: min(3, flat.shape[0])], axis=0)
            peaks = self._find_peaks(lead, min_interval_s=0.4, threshold=float(np.mean(lead) + 0.55 * np.std(lead)))
            bpm = 60.0 / max(float(np.mean(np.diff(peaks) / config.sample_rate)), 1e-6) if peaks.size >= 2 else 0.0
            metrics.update({"heart_rate_bpm": round(bpm, 1), "peak_count": int(peaks.size)})
            message = f"Baseline heart rate estimate {bpm:.1f} bpm"
        elif config.signal_profile_name == "eog":
            horiz = flat[0] - flat[1] if flat.shape[0] > 1 else flat[0]
            vert = flat[2] - flat[3] if flat.shape[0] > 3 else flat[min(1, flat.shape[0] - 1)]
            metrics.update(
                {
                    "horizontal_bias_uv": round(float(np.mean(horiz)), 1),
                    "vertical_bias_uv": round(float(np.mean(vert)), 1),
                }
            )
            message = "Baseline fixation captured"
        elif config.signal_profile_name == "ppg":
            signal = np.mean(flat, axis=0)
            metrics["amplitude"] = round(float(np.max(signal) - np.min(signal)), 3)
        elif config.signal_profile_name == "resp":
            signal = np.mean(flat, axis=0)
            metrics["resp_span"] = round(float(np.max(signal) - np.min(signal)), 3)
        elif config.signal_profile_name == "temp":
            signal = np.mean(flat, axis=0)
            metrics["drift"] = round(float(signal[-1] - signal[0]), 4)
            message = f"Baseline drift {signal[-1] - signal[0]:+.4f} {config.signal_profile.units}"

        return {
            "baseline_mean": channel_mean.tolist(),
            "baseline_std": channel_std.tolist(),
            "baseline_channel_rms": channel_rms.tolist(),
            "baseline_rms": baseline_rms,
            "message": message,
            "metrics": metrics,
        }

    def _analyze_task_window(self, profile_key: str, windows: List[np.ndarray], baseline: Dict[str, Any]) -> Dict[str, Any]:
        analysis = {
            "emg": self._analyze_emg_task,
            "eeg": self._analyze_eeg_task,
            "ecg": self._analyze_ecg_task,
            "eog": self._analyze_eog_task,
            "eda": self._analyze_eda_task,
            "ppg": self._analyze_ppg_task,
            "resp": self._analyze_resp_task,
            "temp": self._analyze_temp_task,
        }.get(profile_key, self._analyze_generic_task)
        return analysis(windows, baseline)

    def _analyze_emg_task(self, windows: List[np.ndarray], baseline: Dict[str, Any]) -> Dict[str, Any]:
        flat = self._flatten_windows(windows)
        active_rms = float(np.sqrt(np.mean(flat ** 2)))
        channel_rms = np.sqrt(np.mean(flat ** 2, axis=1))
        baseline_rms = max(float(baseline.get("baseline_rms", 1e-6)), 1e-6)
        baseline_channel_rms = np.asarray(baseline.get("baseline_channel_rms", channel_rms.tolist()))
        threshold = np.maximum(baseline_channel_rms * 2.0, config.signal_profile.mute_floor * 2.0)
        active_channels = int(np.sum(channel_rms > threshold))
        snr = active_rms / baseline_rms
        return {
            "success": active_channels >= 2 and snr >= 1.25,
            "warning": None if snr >= 2.0 else "Activation was weak. Consider improving electrode contact.",
            "message": f"Activation SNR {snr:.1f}x with {active_channels} active channels",
            "metrics": {
                "active_rms": round(active_rms, 5),
                "snr_estimate": round(snr, 2),
                "active_channels": active_channels,
            },
        }

    def _analyze_eeg_task(self, windows: List[np.ndarray], baseline: Dict[str, Any]) -> Dict[str, Any]:
        flat = self._flatten_windows(windows)
        mean_signal = np.mean(flat, axis=0)
        bands = self._signal_band_ratios(mean_signal)
        dom_freq = self._dominant_frequency(mean_signal, 1.0, 45.0)
        base_alpha = float(baseline.get("metrics", {}).get("alpha_ratio", 0.0))
        alpha_gain = bands["alpha"] / max(base_alpha, 1e-6)
        success = bands["alpha"] >= 0.08
        warning = None if alpha_gain >= 1.15 else "Alpha increase was small. Try relaxing more or reducing eye movement."
        return {
            "success": success,
            "warning": warning,
            "message": f"Eyes-closed alpha ratio {bands['alpha']:.3f} ({alpha_gain:.2f}x baseline)",
            "metrics": {
                "alpha_ratio": round(bands["alpha"], 3),
                "beta_ratio": round(bands["beta"], 3),
                "alpha_gain": round(alpha_gain, 2),
                "dominant_frequency_hz": round(dom_freq, 2),
            },
        }

    def _analyze_ecg_task(self, windows: List[np.ndarray], baseline: Dict[str, Any]) -> Dict[str, Any]:
        flat = self._flatten_windows(windows)
        signal = np.mean(flat[: min(3, flat.shape[0])], axis=0)
        threshold = float(np.mean(signal) + 0.55 * np.std(signal))
        peaks = self._find_peaks(signal, min_interval_s=0.4, threshold=threshold)
        bpm = 60.0 / max(float(np.mean(np.diff(peaks) / config.sample_rate)), 1e-6) if peaks.size >= 2 else 0.0
        success = peaks.size >= 2 and 35.0 <= bpm <= 180.0
        return {
            "success": success,
            "warning": None,
            "message": f"Detected {peaks.size} peaks at {bpm:.1f} bpm",
            "metrics": {
                "heart_rate_bpm": round(bpm, 1),
                "peak_count": int(peaks.size),
            },
        }

    def _analyze_eog_task(self, windows: List[np.ndarray], baseline: Dict[str, Any]) -> Dict[str, Any]:
        flat = self._flatten_windows(windows)
        horiz = flat[0] - flat[1] if flat.shape[0] > 1 else flat[0]
        vert = flat[2] - flat[3] if flat.shape[0] > 3 else flat[min(1, flat.shape[0] - 1)]
        blink = 0.5 * (np.abs(flat[2]) + np.abs(flat[3])) if flat.shape[0] > 3 else np.abs(flat[0])
        h_ptp = float(np.ptp(horiz))
        v_ptp = float(np.ptp(vert))
        blink_peak = float(np.max(blink))
        success = max(h_ptp, v_ptp, blink_peak) >= 45.0
        return {
            "success": success,
            "warning": None if success else "Eye motion was small. Repeat with a larger saccade or blink.",
            "message": f"Horizontal span {h_ptp:.1f} uV, vertical span {v_ptp:.1f} uV, blink peak {blink_peak:.1f} uV",
            "metrics": {
                "horizontal_span_uv": round(h_ptp, 1),
                "vertical_span_uv": round(v_ptp, 1),
                "blink_peak_uv": round(blink_peak, 1),
            },
        }

    def _analyze_eda_task(self, windows: List[np.ndarray], baseline: Dict[str, Any]) -> Dict[str, Any]:
        flat = self._flatten_windows(windows)
        signal = np.mean(flat, axis=0)
        span = float(np.max(signal) - np.min(signal))
        slope = float(signal[-1] - signal[0])
        success = span >= 0.02
        return {
            "success": success,
            "warning": None if span >= 0.06 else "EDA response was small. Increase contact quality or repeat the task.",
            "message": f"EDA span {span:.3f} {config.signal_profile.units}, slope {slope:+.3f}",
            "metrics": {
                "response_span": round(span, 3),
                "slope": round(slope, 3),
            },
        }

    def _analyze_ppg_task(self, windows: List[np.ndarray], baseline: Dict[str, Any]) -> Dict[str, Any]:
        flat = self._flatten_windows(windows)
        signal = np.mean(flat, axis=0)
        signal = signal - float(np.mean(signal))
        span = float(np.max(signal) - np.min(signal))
        threshold = float(np.mean(signal) + 0.25 * np.std(signal))
        peaks = self._find_peaks(signal, min_interval_s=0.4, threshold=threshold)
        bpm = 60.0 / max(float(np.mean(np.diff(peaks) / config.sample_rate)), 1e-6) if peaks.size >= 2 else 0.0
        success = peaks.size >= 2 and span >= 0.05
        return {
            "success": success,
            "warning": None if span >= 0.1 else "Pulse amplitude is weak. Improve sensor contact and keep still.",
            "message": f"Pulse amplitude {span:.3f} with {peaks.size} peaks at {bpm:.1f} bpm",
            "metrics": {
                "pulse_rate_bpm": round(bpm, 1),
                "pulse_amplitude": round(span, 3),
                "peak_count": int(peaks.size),
            },
        }

    def _analyze_resp_task(self, windows: List[np.ndarray], baseline: Dict[str, Any]) -> Dict[str, Any]:
        flat = self._flatten_windows(windows)
        signal = np.mean(flat, axis=0)
        span = float(np.max(signal) - np.min(signal))
        rate_hz = self._dominant_frequency(signal, 0.05, 1.2)
        rate_bpm = rate_hz * 60.0
        success = span >= 0.03
        return {
            "success": success,
            "warning": None if 4.0 <= rate_bpm <= 30.0 else "Breathing rate estimate is unusual; repeat with one slower cycle.",
            "message": f"Respiration span {span:.3f}, estimated rate {rate_bpm:.1f} breaths/min",
            "metrics": {
                "resp_span": round(span, 3),
                "resp_rate_bpm": round(rate_bpm, 1),
            },
        }

    def _analyze_temp_task(self, windows: List[np.ndarray], baseline: Dict[str, Any]) -> Dict[str, Any]:
        flat = self._flatten_windows(windows)
        signal = np.mean(flat, axis=0)
        drift = float(signal[-1] - signal[0])
        noise = float(np.std(signal - np.linspace(signal[0], signal[-1], signal.size)))
        success = noise <= max(float(config.signal_profile.metric_full_scale) * 0.2, 0.01)
        return {
            "success": success,
            "warning": None if success else "Temperature trace is noisy. Improve contact stability.",
            "message": f"Temperature drift {drift:+.4f} {config.signal_profile.units}, noise {noise:.4f}",
            "metrics": {
                "drift": round(drift, 4),
                "noise": round(noise, 4),
            },
        }

    def _analyze_generic_task(self, windows: List[np.ndarray], baseline: Dict[str, Any]) -> Dict[str, Any]:
        flat = self._flatten_windows(windows)
        rms = float(np.sqrt(np.mean(flat ** 2)))
        return {
            "success": rms > 0.0,
            "warning": None,
            "message": f"Mean activity {rms:.4f} {config.signal_profile.units}",
            "metrics": {"mean_rms": round(rms, 4)},
        }

    def _emit(self, message: str, data: Dict[str, Any] = None) -> None:
        payload = {
            "stage": self.stage.value,
            "message": message,
            "data": data or {},
            "profile": config.signal_profile_name,
            "protocol": self.describe_protocol(),
        }
        logger.info("[Calibration/%s] %s", self.stage.value, message)
        if self._status_cb:
            self._status_cb(payload)

    def _fail(self, reason: str, data: Dict[str, Any] = None) -> Dict[str, Any]:
        self.stage = CalibStage.FAILED
        self.results["success"] = False
        self.results["error"] = reason
        if data:
            self.results["failure_context"] = data
        self._emit(reason, data)
        return self.results
