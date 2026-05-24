"""
Guided Recording Flow — interactive, gated recording sessions.

Each training step PAUSES and waits for user confirmation before recording.
After each step, shows quality feedback and offers "Redo" or "Next".

Flow:
  Step N: "You will now record [gesture]"
          → user reads instructions → clicks "I'm Ready"
          → recording runs for duration
          → quality check → "Looks good!" or "Signal noisy — redo?"
          → user confirms → next step
"""
import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
import numpy as np
from config import config

logger = logging.getLogger(__name__)

class StepType(str, Enum):
    RECORD = "record"
    REST = "rest"
    CALIBRATE = "calibrate"
    COMPLETE = "complete"

class StepPhase(str, Enum):
    IDLE = "idle"
    WAITING = "waiting"          # waiting for user to click "Start"
    COUNTDOWN = "countdown"      # 3-2-1 before recording
    RECORDING = "recording"      # actively recording
    QUALITY_CHECK = "quality"    # showing quality after recording
    RESTING = "resting"          # rest period between steps
    COMPLETE = "complete"        # all done

@dataclass
class StepQuality:
    """Post-recording quality assessment for a single step."""
    rms_mean: float = 0.0
    rms_std: float = 0.0
    snr_estimate: float = 0.0
    windows_recorded: int = 0
    grade: str = "unknown"       # good | fair | poor
    message: str = ""
    pass_threshold: bool = True

@dataclass
class RecordingStep:
    step: int
    instruction: str
    gesture: str = ""
    duration_s: float = 5.0
    type: str = "record"
    reps: int = 1
    current_rep: int = 0
    completed: bool = False
    activation_target: float = 0.0
    requires_confirmation: bool = True
    user_prompt: str = ""       # rich instruction shown in the "Ready?" gate
    quality: Optional[StepQuality] = None

@dataclass
class GuidedSessionState:
    active: bool = False
    current_step: int = 0
    total_steps: int = 0
    steps: List[RecordingStep] = field(default_factory=list)
    progress: float = 0.0
    model_readiness: float = 0.0
    windows_collected: int = 0
    windows_target: int = 200
    elapsed_s: float = 0.0
    live_activation: float = 0.0
    live_stability: float = 0.0
    message: str = ""
    phase: str = "idle"
    # ── Gate state ──────────────────────────────────────────────────
    waiting_for_user: bool = False
    current_gesture: str = ""
    current_instruction: str = ""
    current_user_prompt: str = ""
    current_tips: List[str] = field(default_factory=list)
    step_quality: Optional[Dict] = None
    can_redo: bool = False


class GuidedRecordingFlow:
    def __init__(self):
        self._state = GuidedSessionState()
        self._status_cb: Optional[Callable] = None
        self._start_time: float = 0.0
        self._step_start: float = 0.0
        self._cancel = False
        self._activation_history = []
        # ── Gate synchronization ────────────────────────────────────
        self._user_confirmed = asyncio.Event()
        self._confirm_seq = 0
        self._user_redo = False
        self._step_rms_values: List[float] = []

    def set_status_callback(self, fn: Callable[[Dict], None]):
        self._status_cb = fn

    def create_session(self, protocol: List[Dict], target_windows: int = 200,
                       training_instructions: List[Dict] = None) -> GuidedSessionState:
        steps = []
        for i, p in enumerate(protocol):
            steps.append(RecordingStep(
                step=i+1, instruction=p.get("instruction", ""),
                gesture=p.get("gesture", ""), duration_s=p.get("duration_s", 5),
                type=p.get("type", "record"), reps=p.get("reps", 1),
                requires_confirmation=p.get("requires_confirmation", True),
                user_prompt=p.get("user_prompt", p.get("instruction", ""))))
        self._state = GuidedSessionState(
            active=True, current_step=0, total_steps=len(steps),
            steps=steps, windows_target=target_windows, phase="idle")
        self._cancel = False
        self._confirm_seq = 0
        self._user_redo = False
        self._user_confirmed.clear()
        self._training_instructions = training_instructions or []
        return self._state

    def confirm_step(self):
        """Called when the user clicks 'I'm Ready' / 'Start Recording'."""
        self._user_redo = False
        self._confirm_seq += 1
        self._user_confirmed.set()
        logger.info("Guided recording: user confirmed step")

    def redo_step(self):
        """Called when the user wants to redo the last step."""
        self._user_redo = True
        self._confirm_seq += 1
        self._user_confirmed.set()
        logger.info("Guided recording: user requested redo")

    async def run(self, stream, pipeline) -> Dict:
        self._cancel = False
        self._start_time = time.monotonic()
        self._state.active = True
        results = {"gestures_recorded": {}, "total_windows": 0, "success": False}

        for i, step in enumerate(self._state.steps):
            if self._cancel:
                self._emit("Session cancelled by user")
                break
            self._state.current_step = i
            self._state.elapsed_s = time.monotonic() - self._start_time

            for rep in range(step.reps):
                if self._cancel: break
                step.current_rep = rep + 1

                if step.type == "record":
                    should_redo = True
                    while should_redo:
                        should_redo = False

                        # ── GATE: Wait for user confirmation ──────────
                        if step.requires_confirmation:
                            self._state.phase = "waiting"
                            self._state.waiting_for_user = True
                            self._state.current_gesture = step.gesture
                            self._state.current_instruction = step.instruction
                            self._state.current_user_prompt = step.user_prompt
                            self._state.can_redo = False
                            wait_seq = self._confirm_seq

                            # Find matching training instruction for tips
                            tips = []
                            for ti in self._training_instructions:
                                if ti.get("gesture") == step.gesture:
                                    tips = ti.get("tips", [])
                                    break
                            self._state.current_tips = tips

                            rep_label = f" (rep {rep+1}/{step.reps})" if step.reps > 1 else ""
                            self._emit(f"Get ready: {step.gesture}{rep_label}")

                            # Wait for user to click "Start"
                            while self._confirm_seq == wait_seq and not self._cancel:
                                await asyncio.sleep(0.1)
                            if self._cancel: break
                            self._state.waiting_for_user = False

                        # ── COUNTDOWN: 3-2-1 ─────────────────────────
                        self._state.phase = "countdown"
                        for countdown in [3, 2, 1]:
                            if self._cancel: break
                            self._emit(f"Starting in {countdown}...")
                            await asyncio.sleep(1.0)
                        if self._cancel: break

                        # ── RECORD ────────────────────────────────────
                        self._state.phase = "recording"
                        rep_label = f" (rep {rep+1}/{step.reps})" if step.reps > 1 else ""
                        self._emit(f"Recording '{step.gesture}'{rep_label} - hold steady")
                        self._step_rms_values = []

                        pipeline.start_recording(step.gesture)
                        self._step_start = time.monotonic()

                        windows_before = self._get_window_count(pipeline, step.gesture)

                        while time.monotonic() - self._step_start < step.duration_s:
                            if self._cancel: break
                            w = stream.get_window()
                            if w is not None:
                                rms = float(np.sqrt(np.mean(w**2)))
                                self._state.live_activation = rms
                                self._step_rms_values.append(rms)
                                self._activation_history.append(rms)
                                if len(self._activation_history) > 20:
                                    self._activation_history.pop(0)
                                self._state.live_stability = 1.0 - min(1.0,
                                    float(np.std(self._activation_history[-10:])) /
                                    max(float(np.mean(self._activation_history[-10:])), 1e-8))
                            remaining = step.duration_s - (time.monotonic() - self._step_start)
                            self._state.elapsed_s = time.monotonic() - self._start_time
                            self._emit_progress(remaining)
                            await asyncio.sleep(0.1)

                        pipeline.stop_recording()

                        # ── QUALITY CHECK ─────────────────────────────
                        quality = self._assess_step_quality(pipeline, step, windows_before)
                        step.quality = quality

                        self._state.phase = "quality"
                        self._state.step_quality = {
                            "grade": quality.grade,
                            "message": quality.message,
                            "windows": quality.windows_recorded,
                            "snr": round(quality.snr_estimate, 2),
                            "rms_mean": round(quality.rms_mean, 4),
                        }
                        self._state.can_redo = True

                        # Update counts
                        summary = pipeline.get_training_summary()
                        count = summary.get("per_gesture", {}).get(step.gesture, 0)
                        results["gestures_recorded"][step.gesture] = count
                        self._state.windows_collected = summary.get("total_windows", 0)
                        self._state.model_readiness = min(1.0,
                            self._state.windows_collected / max(self._state.windows_target, 1))
                        self._state.progress = (i + 1) / max(len(self._state.steps), 1)

                        if quality.grade == "poor":
                            self._emit(f"Signal quality was poor for '{step.gesture}'. Redo recommended.")
                        else:
                            self._emit(f"'{step.gesture}' recorded - {quality.message}")

                        # Wait for user to confirm quality or redo
                        if step.requires_confirmation:
                            wait_seq = self._confirm_seq
                            while self._confirm_seq == wait_seq and not self._cancel:
                                await asyncio.sleep(0.1)
                            if self._cancel: break
                            if self._user_redo:
                                should_redo = True
                                self._user_redo = False
                                logger.info(f"Redoing step: {step.gesture}")
                                continue

                elif step.type == "rest":
                    self._state.phase = "resting"
                    self._emit(step.instruction)
                    self._step_start = time.monotonic()
                    while time.monotonic() - self._step_start < step.duration_s:
                        if self._cancel: break
                        remaining = step.duration_s - (time.monotonic() - self._step_start)
                        self._emit_progress(remaining)
                        await asyncio.sleep(0.2)

            step.completed = True

        self._state.phase = "complete"
        self._state.active = False
        results["total_windows"] = self._state.windows_collected
        results["success"] = not self._cancel and self._state.windows_collected > 0
        self._emit(f"Recording complete - {self._state.windows_collected} windows collected")
        return results

    def _assess_step_quality(self, pipeline, step, windows_before) -> StepQuality:
        """Assess signal quality after recording a step."""
        summary = pipeline.get_training_summary()
        windows_after = summary.get("per_gesture", {}).get(step.gesture, 0)
        n_new = windows_after - windows_before

        rms_arr = np.array(self._step_rms_values) if self._step_rms_values else np.array([0.0])
        rms_mean = float(np.mean(rms_arr))
        rms_std = float(np.std(rms_arr))
        snr = rms_mean / max(rms_std, 1e-8)

        if n_new < 3:
            return StepQuality(rms_mean, rms_std, snr, n_new, "poor",
                "Very few windows recorded - try again", False)
        if snr > 3.0:
            return StepQuality(rms_mean, rms_std, snr, n_new, "good",
                f"{n_new} windows - SNR {snr:.1f} - Clean signal", True)
        if snr > 1.5:
            return StepQuality(rms_mean, rms_std, snr, n_new, "fair",
                f"{n_new} windows - SNR {snr:.1f} - Acceptable", True)
        return StepQuality(rms_mean, rms_std, snr, n_new, "poor",
            f"{n_new} windows - SNR {snr:.1f} - Noisy - consider redoing", False)

    def _get_window_count(self, pipeline, gesture) -> int:
        try:
            return pipeline.get_training_summary().get("per_gesture", {}).get(gesture, 0)
        except Exception:
            return 0

    def cancel(self):
        self._cancel = True
        self._user_confirmed.set()  # unblock any waiting gate

    def get_state(self) -> Dict:
        s = self._state
        return {
            "active": s.active, "current_step": s.current_step,
            "total_steps": s.total_steps, "progress": round(s.progress, 3),
            "model_readiness": round(s.model_readiness, 3),
            "windows_collected": s.windows_collected,
            "windows_target": s.windows_target,
            "elapsed_s": round(s.elapsed_s, 1),
            "live_activation": round(s.live_activation, 5),
            "live_stability": round(s.live_stability, 3),
            "message": s.message, "phase": s.phase,
            "waiting_for_user": s.waiting_for_user,
            "current_gesture": s.current_gesture,
            "current_instruction": s.current_instruction,
            "current_user_prompt": s.current_user_prompt,
            "current_tips": s.current_tips,
            "step_quality": s.step_quality,
            "can_redo": s.can_redo,
            "steps": [{"step": st.step, "instruction": st.instruction,
                "gesture": st.gesture, "duration_s": st.duration_s,
                "type": st.type, "reps": st.reps, "current_rep": st.current_rep,
                "completed": st.completed,
                "quality": {"grade": st.quality.grade, "message": st.quality.message}
                    if st.quality else None,
                } for st in s.steps],
        }

    def _emit(self, message: str, data: Dict = None):
        self._state.message = message
        if self._status_cb:
            self._status_cb({"type": "guided_step", "data": {
                "message": message, "phase": self._state.phase,
                **(data or {}), **self.get_state()}})

    def _emit_progress(self, remaining: float):
        if self._status_cb:
            self._status_cb({"type": "guided_progress", "data": {
                "remaining_s": round(remaining, 1), **self.get_state()}})
