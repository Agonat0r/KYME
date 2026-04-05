"""
Calibration manager — async, multi-stage.

Stages
──────
1. CHANNEL_CHECK  : measure each channel's RMS; flag saturated / dead channels
2. BASELINE       : 3-second resting baseline (used to set noise floor)
3. GESTURE_CHECK  : prompt user to make a fist; verify signal amplitude
4. COMPLETE / FAILED

Status updates are delivered via the registered callback as plain dicts so
main.py can broadcast them over WebSocket without circular imports.
"""

import asyncio
import logging
import time
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from config import config

logger = logging.getLogger(__name__)


class CalibStage(str, Enum):
    IDLE          = "idle"
    CHANNEL_CHECK = "channel_check"
    BASELINE      = "baseline"
    GESTURE_CHECK = "gesture_check"
    COMPLETE      = "complete"
    FAILED        = "failed"


class CalibrationManager:
    def __init__(self):
        self.stage = CalibStage.IDLE
        self.results: Dict[str, Any] = {}
        self._status_cb: Optional[Callable[[Dict], None]] = None

    def set_status_callback(self, fn: Callable[[Dict], None]) -> None:
        self._status_cb = fn

    # ── Main calibration coroutine ────────────────────────────────────────

    async def run_calibration(self, stream) -> Dict[str, Any]:
        self.results = {}

        # ── Stage 1: Channel quality ──────────────────────────────────────
        self.stage = CalibStage.CHANNEL_CHECK
        self._emit("Checking channels — relax your arm …")
        await asyncio.sleep(1.5)

        window = stream.get_window()
        if window is None:
            return self._fail("No data received. Check BrainFlow connection.")

        rms = np.sqrt(np.mean(window ** 2, axis=1))          # (n_ch,)
        quality: List[float] = []
        issues: List[str] = []
        for i, r in enumerate(rms):
            if r < 1e-7:
                quality.append(0.0)
                issues.append(f"CH{i+1}: no signal (electrode disconnected?)")
            elif r > 1.0:
                quality.append(0.3)
                issues.append(f"CH{i+1}: saturated ({r:.3f} V) — check contact")
            elif r > 0.3:
                quality.append(0.7)
            else:
                quality.append(1.0)

        good = sum(q > 0.5 for q in quality)
        self.results["channel_quality"] = quality
        self.results["channel_issues"] = issues
        self._emit(
            f"Channels: {good}/{config.n_channels} good",
            {"quality": quality, "issues": issues},
        )

        if good < 2:
            return self._fail("Too few good channels. Check electrode placement.")

        # ── Stage 2: Baseline ─────────────────────────────────────────────
        self.stage = CalibStage.BASELINE
        self._emit("Measuring baseline — keep arm relaxed for 3 s …")
        await asyncio.sleep(0.3)

        baseline_wins: List[np.ndarray] = []
        t0 = time.monotonic()
        while time.monotonic() - t0 < 3.0:
            w = stream.get_window()
            if w is not None:
                baseline_wins.append(w)
            await asyncio.sleep(0.05)

        if baseline_wins:
            stacked = np.stack(baseline_wins, axis=0)        # (n_win, n_ch, n_s)
            b_mean = stacked.mean(axis=(0, 2)).tolist()      # per-channel
            b_std  = stacked.std(axis=(0, 2)).tolist()
            b_rms  = float(np.sqrt(np.mean(stacked ** 2)))
            self.results["baseline_mean"] = b_mean
            self.results["baseline_std"]  = b_std
            self.results["baseline_rms"]  = b_rms
            self._emit(f"Baseline RMS = {b_rms:.5f}", {"baseline_rms": b_rms})
        else:
            return self._fail("Could not collect baseline data.")

        # ── Stage 3: Gesture check ────────────────────────────────────────
        self.stage = CalibStage.GESTURE_CHECK
        self._emit("Now make a FIST and hold it …")
        await asyncio.sleep(1.5)

        w = stream.get_window()
        if w is not None:
            active_rms = float(np.sqrt(np.mean(w ** 2)))
            baseline_rms = self.results.get("baseline_rms", 1e-6)
            snr = active_rms / max(baseline_rms, 1e-9)
            self.results["active_rms"] = active_rms
            self.results["snr_estimate"] = round(snr, 2)
            self._emit(
                f"Active RMS = {active_rms:.5f}  (SNR ≈ {snr:.1f}×)",
                {"active_rms": active_rms, "snr": snr},
            )
            if snr < 2.0:
                self._emit(
                    "Warning: low SNR — consider repositioning electrodes",
                    {"warning": "low_snr"},
                )

        # ── Done ──────────────────────────────────────────────────────────
        self.stage = CalibStage.COMPLETE
        self.results["success"] = True
        self._emit("Calibration complete!", self.results)
        return self.results

    # ── Helpers ───────────────────────────────────────────────────────────

    def _emit(self, message: str, data: Dict = None) -> None:
        payload = {"stage": self.stage.value, "message": message, "data": data or {}}
        logger.info(f"[Calibration/{self.stage.value}] {message}")
        if self._status_cb:
            self._status_cb(payload)

    def _fail(self, reason: str) -> Dict:
        self.stage = CalibStage.FAILED
        self.results["success"] = False
        self.results["error"] = reason
        self._emit(reason)
        return self.results
