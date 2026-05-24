"""
KYMA - Biosignal Control Server
===============================
FastAPI + WebSocket backend.
"""

import asyncio
import base64
import csv
import importlib.util
import json
import logging
import lzma
import os
import pickle
import py_compile
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
import zlib
from contextlib import asynccontextmanager, suppress
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
from fastapi import BackgroundTasks, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from arduino_bridge import ArduinoBridge
from biosignal_pipeline import BiosignalPipeline
from biosignal_profiles import get_profile, list_profile_dicts
from brainflow_stream import CytonStream, PlaybackCytonStream, SimulatedCytonStream
from calibration import CalibrationManager
from config import config
from eeg_brain_viz import EEGBrainVisualizer
from eeg_experiment_presets import list_eeg_experiment_presets
from filter_lab import FilterLab
from firmware_workspace import FirmwareWorkspace
from ai_workflow import AIWorkflow
from live_diagnostics import LiveDiagnostics, SafetyWatchdog
from lsl_bridge import LSLBridge
from lsl_input import LSLInletStream, list_lsl_streams
from osc_bridge import OSCBridge
from signal_workshop import SignalWorkshop
from signal_confidence import SignalConfidenceLayer
from signal_recovery import SignalRecoveryEngine
from intent_engine import IntentEngine
from guided_recording import GuidedRecordingFlow
from code_generator import CodeGenerator
from lie_detector import LieDetectorSession
from models import (
    AICopilotRequest,
    AIConfigRequest,
    AnalogWriteCommand,
    ApiKeyConfig,
    ArduinoSerialWriteRequest,
    CodegenRefineRequest,
    CodegenProjectRequest,
    ConnectRequest,
    ControlSensitivityRequest,
    DatasetCreateRequest,
    DigitalWriteCommand,
    ExperimentRunRequest,
    FilterActivateRequest,
    FilterDesignRequest,
    FirmwareCompileRequest,
    FirmwareFileRequest,
    FirmwareGeneratedRequest,
    FirmwareSaveRequest,
    FirmwareUploadRequest,
    GuidedRecordingStartRequest,
    IntentRequest,
    LSLMarkerRequest,
    LSLStartRequest,
    MoveCommand,
    OSCSendRequest,
    OSCStartRequest,
    PipelineAcquisitionRunRequest,
    PipelineActionRequest,
    PipelineBaselineTrainRequest,
    PipelineDatasetIngestRequest,
    PipelineDatasetInspectRequest,
    PipelineModelRequest,
    PipelinePlanRequest,
    PipelineProjectRequest,
    PipelineQARequest,
    PipelineThroughputRequest,
    ProfileRequest,
    ReviewMarkerRequest,
    SessionStartRequest,
    SubjectUpsertRequest,
    SystemState,
    TrainRequest,
    WebhookRequest,
    WorkshopAnalyzeRequest,
    WorkshopSaveRequest,
    XDFImportRequest,
    XDFInspectRequest,
)
from protocol_templates import list_protocol_templates
from research_manager import DatasetManager, ExperimentManager
from session_export import SessionExporter
from session_recorder import SessionRecorder
from subject_registry import SubjectRegistry
from xdf_import import XDFImporter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


class AppState:
    def __init__(self):
        self.system_state: SystemState = SystemState.IDLE
        self.stream: Optional[CytonStream] = None
        self.stream_source: str = "synthetic" if config.board_id == -1 else "hardware"
        self.stream_details: dict = {}
        self.playback_session_id: Optional[str] = None
        self.pipeline = BiosignalPipeline()
        self.arduino: Optional[ArduinoBridge] = None
        self.lsl = LSLBridge()
        self.osc = OSCBridge()
        self.recorder = SessionRecorder()
        self.subjects = SubjectRegistry()
        self.exporter = SessionExporter(self.recorder)
        self.datasets = DatasetManager(self.recorder)
        self.experiments = ExperimentManager(self.recorder, self.datasets)
        self.xdf = XDFImporter()
        self.recorder_chunk_callback = self.recorder.record_chunk
        self.calibration = CalibrationManager()
        self.filter_lab = FilterLab()
        self.ai = AIWorkflow()
        self.diagnostics = LiveDiagnostics(
            sample_rate=config.sample_rate,
            window_increment_ms=config.window_increment_ms,
            full_scale=config.signal_profile.display_full_scale,
        )
        self.watchdog = SafetyWatchdog()
        self.workshop = SignalWorkshop(export_root=Path(config.data_dir) / "workshop_exports")
        self.firmware = FirmwareWorkspace(
            Path(config.data_dir) / "firmware_workspace",
            seed_paths=[Path(__file__).resolve().parents[1] / "firmware"],
        )
        self.eeg_brain = EEGBrainVisualizer(
            sample_rate=config.sample_rate,
            channel_labels=list(get_profile("eeg").channel_labels),
            root_dir=Path(__file__).resolve().parents[1],
        )
        self.websockets: Set[WebSocket] = set()
        self.last_prediction: Optional[dict] = None
        self.last_vis_window: Optional[np.ndarray] = None
        self.last_vis_timestamp: float = 0.0
        self.last_live_metrics: dict = {}
        self.last_diagnostics_broadcast_ts: float = 0.0
        self.last_ml_broadcast_ts: float = 0.0
        self.last_safety_broadcast_ts: float = 0.0
        self.pipeline_embedding_timeline: List[dict] = []
        self.pipeline_acquisition_run: Dict[str, Any] = {}
        self.pipeline_acquisition_windows: List[dict] = []
        self.pipeline_acquisition_last_accept_ts: float = 0.0
        self.pipeline_acquisition_model: Dict[str, Any] = {}
        self.pipeline_acquisition_runtime: Dict[str, Any] = {}
        self.pipeline_acquisition_last_predict_ts: float = 0.0
        self.pipeline_rebuild_jobs: Dict[str, dict] = {}

        # ── Intelligence layer ────────────────────────────────────────
        self.signal_confidence = SignalConfidenceLayer(n_channels=config.n_channels)
        self.signal_recovery = SignalRecoveryEngine(n_channels=config.n_channels)
        self.intent_engine = IntentEngine()
        self.guided_recording = GuidedRecordingFlow()
        self.lie_detector = LieDetectorSession()
        self.last_confidence_broadcast_ts: float = 0.0
        self.last_recovery_broadcast_ts: float = 0.0
        self._guided_task: Optional[asyncio.Task] = None

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._bcast_queue: Optional[asyncio.Queue] = None
        self.broadcast_dropped: int = 0

    def queue_broadcast(self, msg: dict) -> None:
        if self._loop and self._bcast_queue:
            self._loop.call_soon_threadsafe(self._enqueue_broadcast, msg)

    def _enqueue_broadcast(self, msg: dict) -> None:
        q = self._bcast_queue
        if not q:
            return
        if q.full():
            try:
                q.get_nowait()
                q.task_done()
                self.broadcast_dropped += 1
            except asyncio.QueueEmpty:
                pass
        try:
            q.put_nowait(msg)
        except asyncio.QueueFull:
            self.broadcast_dropped += 1

    def set_state(self, state: SystemState) -> None:
        self.system_state = state
        self.queue_broadcast(
            {
                "type": "state",
                "data": {"state": state.value},
                "timestamp": time.time(),
            }
        )
        self.lsl.push_marker(
            "state",
            {
                "state": state.value,
                "profile": config.signal_profile_name,
                "source": self.stream_source,
            },
        )
        self.osc.send_state(
            state=state.value,
            profile=config.signal_profile_name,
            source=self.stream_source,
        )


app_state = AppState()


def _profile_payload() -> dict:
    profile = config.signal_profile
    return {
        "active": profile.to_dict(),
        "available": list_profile_dicts(),
    }


def _protocol_payload() -> list:
    return list_protocol_templates(config.signal_profile_name)


def _eeg_experiment_payload() -> list:
    return list_eeg_experiment_presets()


def _active_session_payload() -> dict:
    recorder_meta = dict(getattr(app_state.recorder, "_session_metadata", {}) or {})
    return {
        "session_id": app_state.recorder.session_id or "",
        "label": str(getattr(app_state.recorder, "_session_label", "") or "").strip(),
        "subject_id": str(recorder_meta.get("subject_id") or "").strip(),
        "condition": str(recorder_meta.get("condition") or "").strip(),
        "notes": str(recorder_meta.get("notes") or "").strip(),
    }


def _window_stats_for_ai(window: np.ndarray) -> dict:
    if window.ndim != 2 or window.size == 0:
        return {}
    arr = window.astype(np.float64)
    rms_by_channel = np.sqrt(np.mean(np.square(arr), axis=1))
    focus_channel = int(np.argmax(rms_by_channel)) if rms_by_channel.size else 0
    focus = arr[focus_channel] if arr.shape[0] else arr.reshape(-1)
    return {
        "samples": int(arr.shape[1]),
        "duration_ms": round((arr.shape[1] / max(config.sample_rate, 1)) * 1000.0, 3),
        "mean": round(float(np.mean(arr)), 6),
        "rms": round(float(np.sqrt(np.mean(np.square(arr)))), 6),
        "peak_to_peak": round(float(np.ptp(arr)), 6),
        "focus_channel": focus_channel,
        "focus_label": str(config.channel_labels[focus_channel]) if focus_channel < len(config.channel_labels) else f"CH{focus_channel + 1}",
        "focus_rms": round(float(rms_by_channel[focus_channel]) if rms_by_channel.size else 0.0, 6),
        "min": round(float(np.min(focus)) if focus.size else 0.0, 6),
        "max": round(float(np.max(focus)) if focus.size else 0.0, 6),
    }


def _downsample_window_for_ai(window: np.ndarray, target_samples: int = 256) -> list:
    if window.ndim != 2 or window.size == 0:
        return []
    n_samples = int(window.shape[1])
    if n_samples <= target_samples:
        sampled = window
    else:
        idx = np.linspace(0, n_samples - 1, target_samples).round().astype(int)
        sampled = window[:, idx]
    return [
        [round(float(value), 6) for value in sampled[ch].tolist()]
        for ch in range(min(sampled.shape[0], config.n_channels))
    ]


def _live_ml_snapshot(window: np.ndarray, diagnostics: dict) -> dict:
    stats = _window_stats_for_ai(window)
    samples = int(window.shape[1]) if window.ndim == 2 else 0
    return {
        "captured_at": time.time(),
        "state": app_state.system_state.value,
        "stream_running": bool(app_state.stream and app_state.stream.is_running),
        "is_recording": app_state.recorder.is_recording,
        "profile": config.signal_profile.to_dict(),
        "channel_labels": list(config.channel_labels),
        "diagnostics": diagnostics or app_state.diagnostics.status(),
        "review": {
            "paused": False,
            "stats": stats,
            "selection": {
                "duration_ms": stats.get("duration_ms", 0.0),
            },
            "artifacts": [],
            "window": {
                "sample_rate_hz": config.sample_rate,
                "start_sample": 0,
                "end_sample": max(samples - 1, 0),
                "samples": samples,
                "step_samples": 1,
                "focus_channel": int(stats.get("focus_channel") or 0),
                "focus_label": str(stats.get("focus_label") or "CH1"),
                "channels": _downsample_window_for_ai(window),
            },
        },
        "protocol": {},
        "workshop": {},
        "last_prediction": dict(app_state.last_prediction or {}),
        "filter_chain": {},
        "session": _active_session_payload(),
    }


def _ai_snapshot(req: AICopilotRequest) -> dict:
    filter_status = app_state.filter_lab.status(config.signal_profile_name)
    session_payload = _active_session_payload()
    session_payload.update(dict(req.session or {}))
    return {
        "captured_at": time.time(),
        "state": app_state.system_state.value,
        "stream_running": bool(app_state.stream and app_state.stream.is_running),
        "is_recording": app_state.recorder.is_recording,
        "profile": config.signal_profile.to_dict(),
        "channel_labels": req.channel_labels or list(config.channel_labels),
        "diagnostics": req.diagnostics or app_state.diagnostics.status(),
        "review": dict(req.review or {}),
        "protocol": dict(req.protocol or {}),
        "workshop": dict(req.workshop or {}),
        "last_prediction": dict(req.last_prediction or app_state.last_prediction or {}),
        "filter_chain": {
            "label": str((req.filter_chain or {}).get("label") or "").strip(),
            "active_filter": filter_status.get("active_filter"),
        },
        "session": session_payload,
    }


def _pipeline_signal_profile(req: PipelinePlanRequest) -> dict:
    prompt = str(req.prompt or "").lower()
    aliases = {
        "emg": "emg",
        "muscle": "emg",
        "eeg": "eeg",
        "brain": "eeg",
        "ecg": "ecg",
        "ekg": "ecg",
        "heart": "ecg",
        "eog": "eog",
        "eye": "eog",
        "blink": "eog",
        "eda": "eda",
        "gsr": "eda",
        "skin conductance": "eda",
        "ppg": "ppg",
        "pulse": "ppg",
        "resp": "resp",
        "breath": "resp",
        "temperature": "temp",
        "temp": "temp",
    }
    inferred = ""
    for token, profile_key in aliases.items():
        if token in prompt:
            inferred = profile_key
            break
    if "erg" in prompt or "retina" in prompt or "retinal" in prompt:
        return {
            "key": "erg",
            "display_name": "ERG / Custom",
            "family": "electroretinography",
            "description": "Custom retinal or analog biosignal workflow.",
            "units": "uV",
            "channel_labels": [f"ERG {i + 1}" for i in range(config.n_channels)],
            "class_labels": ["baseline", "response", "recovery", "artifact"],
            "default_features": ["peak_latency", "peak_amplitude", "area_under_curve", "slope"],
            "filters": [
                {"kind": "bandpass", "low_hz": 0.3, "high_hz": 100.0, "cutoff_hz": None, "order": 2},
                {"kind": "bandstop", "low_hz": 58.0, "high_hz": 62.0, "cutoff_hz": None, "order": 2},
            ],
        }
    key = inferred or str(req.signal_profile or config.signal_profile_name or "emg").strip().lower()
    try:
        profile = get_profile(key).to_dict()
    except ValueError:
        profile = config.signal_profile.to_dict()
    if req.channel_labels and str(req.signal_profile or "").lower() in {"", str(profile.get("key") or "").lower()}:
        profile["channel_labels"] = [str(label) for label in req.channel_labels[: config.n_channels]]
    return profile


def _pipeline_task(prompt: str, profile: dict) -> dict:
    q = prompt.lower()
    profile_key = str(profile.get("key") or "").lower()
    class_labels = [str(label) for label in profile.get("class_labels") or []]
    task_type = "classification"
    target = "Train a supervised decoder from labeled biosignal windows."
    labels: List[str] = class_labels[:5] or ["baseline", "active", "artifact"]

    if any(word in q for word in ["fatigue", "tired", "exhaust", "endurance"]):
        task_type = "fatigue_tracking"
        target = "Estimate fatigue or recovery from changing spectral and amplitude features."
        labels = ["fresh", "fatiguing", "fatigued", "recovery"]
    elif any(word in q for word in ["blink", "gaze", "saccade"]) or profile_key == "eog":
        task_type = "event_classification"
        target = "Detect blink and gaze events with timestamps."
        labels = ["fixation", "blink", "left saccade", "right saccade", "up gaze", "down gaze"]
    elif any(word in q for word in ["gesture", "hand", "open", "close", "pinch", "fist"]) or profile_key == "emg":
        task_type = "gesture_classification"
        target = "Classify hand or muscle states from short EMG windows."
        labels = ["rest", "open", "close", "pinch", "point"]
    elif any(word in q for word in ["ecg", "ekg", "heart", "beat", "rhythm"]) or profile_key == "ecg":
        task_type = "event_detection"
        target = "Detect beats, quality issues, and rhythm-state candidates for review."
        labels = ["steady rhythm", "elevated rate", "slow rate", "irregular candidate", "artifact"]
    elif any(word in q for word in ["ppg", "pulse", "perfusion"]) or profile_key == "ppg":
        task_type = "pulse_quality"
        target = "Track pulse rate, amplitude, and signal quality."
        labels = ["pulse stable", "elevated pulse", "slow pulse", "weak pulse", "artifact"]
    elif any(word in q for word in ["artifact", "noise", "clean", "filter", "denoise"]):
        task_type = "artifact_cleanup"
        target = "Find noisy spans, suggest filters, and export clean windows."
        labels = ["clean", "hum", "drift", "motion", "clip"]
    elif any(word in q for word in ["export", "dataset", "bids", "csv", "matlab"]):
        task_type = "dataset_export"
        target = "Package windows, labels, metadata, and exports for reproducible research."
        labels = ["accepted", "reject_artifact", "needs_review"]
    elif any(word in q for word in ["control", "robot", "software", "cursor", "keyboard", "game"]):
        task_type = "realtime_control"
        target = "Map decoded states to low-latency software or hardware actions."
        labels = class_labels[:5] or ["rest", "intent_a", "intent_b", "intent_c"]

    return {
        "type": task_type,
        "target": target,
        "labels": labels,
    }


def _pipeline_windowing(task: dict, profile: dict) -> dict:
    task_type = str(task.get("type") or "")
    profile_key = str(profile.get("key") or "").lower()
    if profile_key == "eeg":
        size_ms, step_ms = (1000, 250)
    elif profile_key == "ecg":
        size_ms, step_ms = (1200, 200)
    elif profile_key == "eog":
        size_ms, step_ms = (500, 100)
    elif profile_key in {"eda", "ppg", "resp", "temp"}:
        size_ms, step_ms = (2000, 500)
    else:
        size_ms, step_ms = (200, 50)
    if task_type == "fatigue_tracking":
        size_ms, step_ms = (1000, 250)
    elif task_type == "artifact_cleanup":
        size_ms = max(size_ms, 500)
    samples = max(1, int(round((size_ms / 1000.0) * config.sample_rate)))
    step_samples = max(1, int(round((step_ms / 1000.0) * config.sample_rate)))
    return {
        "sample_rate_hz": config.sample_rate,
        "window_size_ms": size_ms,
        "window_step_ms": step_ms,
        "window_samples": samples,
        "step_samples": step_samples,
        "normalization": "per-session baseline z-score plus robust amplitude scale",
    }


def _pipeline_label_protocol(task: dict) -> dict:
    labels = [str(label) for label in task.get("labels") or []]
    task_type = str(task.get("type") or "")

    def cue_for(label: str) -> str:
        clean = label.replace("_", " ").strip()
        lower = clean.lower()
        if task_type == "fatigue_tracking":
            if "rest" in lower or "baseline" in lower:
                return "Relax the target muscle and stay still."
            if "fresh" in lower:
                return "Make a normal contraction while the muscle is fresh."
            if "fatigue" in lower or "tired" in lower:
                return "Hold the contraction until it starts to feel tired."
            if "recover" in lower:
                return "Relax and let the muscle recover."
        if "rest" in lower or "baseline" in lower:
            return "Relax and stay still."
        if "open" in lower:
            return "Open your hand and hold it steady."
        if "close" in lower:
            return "Close your hand into a firm fist."
        if "pinch" in lower:
            return "Pinch your fingers together and hold."
        if "blink" in lower:
            return "Blink naturally when prompted."
        return f"Do the {clean} task and hold it steady."

    return {
        "labels": [
            {
                "name": label,
                "min_seconds": 8,
                "target_windows": 25,
                "cue": cue_for(label),
            }
            for label in labels
        ],
        "minimum_classes": 2,
        "minimum_windows_total": 10,
        "recommended_windows_per_label": 25,
        "split": "temporal holdout with artifact gap when a session has enough data",
    }


def _detect_duplicate_signal_channels(arr: np.ndarray) -> List[tuple[int, int]]:
    if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 12:
        return []
    rms = np.sqrt(np.mean(np.square(arr), axis=1))
    p2p = np.ptp(arr, axis=1)
    flat = {idx for idx, value in enumerate(rms) if float(value) < 1.0 or float(p2p[idx]) < 0.01}
    pairs: List[tuple[int, int]] = []
    for i in range(arr.shape[0]):
        if i in flat:
            continue
        for j in range(i + 1, arr.shape[0]):
            if j in flat:
                continue
            ratio = min(float(rms[i]), float(rms[j])) / max(float(rms[i]), float(rms[j]), 1e-9)
            if ratio < 0.85:
                continue
            corr = np.corrcoef(arr[i], arr[j])[0, 1]
            if np.isfinite(corr) and abs(float(corr)) > 0.995:
                pairs.append((i, j))
    return pairs


def _pipeline_model_candidates(task: dict, features: list) -> list:
    task_type = str(task.get("type") or "")
    candidates = [
        {
            "id": "baseline_lda",
            "name": "Baseline LDA",
            "type": "classical",
            "features": features,
            "fit_time": "seconds",
            "status": "ready",
            "notes": "First model to train because it exposes feature failures quickly.",
        },
        {
            "id": "foundation_linear_head",
            "name": "Foundation embedding + linear head",
            "type": "foundation_adapter",
            "features": ["pooled_embedding", "qa_score", "artifact_flags"],
            "fit_time": "seconds to minutes",
            "status": "ready_when_embedding_available",
            "notes": "Uses local foundation features as inputs for small labeled datasets.",
        },
        {
            "id": "temporal_verifier",
            "name": "Temporal verifier",
            "type": "sequence",
            "features": ["raw_window", "prediction_history"],
            "fit_time": "minutes",
            "status": "queued_after_baseline",
            "notes": "Only worth training after the baseline shows separable classes.",
        },
    ]
    if task_type == "artifact_cleanup":
        candidates.insert(0, {
            "id": "artifact_quality_head",
            "name": "Artifact quality head",
            "type": "quality_classifier",
            "features": ["hum_power", "drift_power", "clip_percent", "crest_factor"],
            "fit_time": "seconds",
            "status": "ready",
            "notes": "Scores per-channel clean, hum, drift, motion, and clip spans.",
        })
    return candidates


def _pipeline_plan(req: PipelinePlanRequest) -> dict:
    prompt = str(req.prompt or "").strip()
    if not prompt:
        raise ValueError("Enter a short goal for the pipeline.")
    profile = _pipeline_signal_profile(req)
    task = _pipeline_task(prompt, profile)
    q = prompt.lower()
    source = str(req.source or "live").strip() or "live"
    output = str(req.output or "live_model").strip() or "live_model"
    channels = [str(label) for label in profile.get("channel_labels") or []][: config.n_channels]
    if not channels:
        channels = [f"CH{i + 1}" for i in range(config.n_channels)]
    filters = [dict(stage) for stage in profile.get("filters") or []]
    feature_names = [str(name) for name in profile.get("default_features") or []]
    if task["type"] == "fatigue_tracking":
        feature_names = list(dict.fromkeys(feature_names + ["median_frequency", "mean_frequency", "rms_slope"]))
    elif task["type"] in {"event_detection", "event_classification"}:
        feature_names = list(dict.fromkeys(feature_names + ["peak_prominence", "event_width", "inter_event_interval"]))
    elif task["type"] == "artifact_cleanup":
        feature_names = list(dict.fromkeys(feature_names + ["hum_power", "drift_power", "clip_percent", "crest_factor"]))
    windowing = _pipeline_windowing(task, profile)
    label_protocol = _pipeline_label_protocol(task)
    model_candidates = _pipeline_model_candidates(task, feature_names)

    protocol_steps = [
        "Confirm electrode map, sample rate, and channel labels.",
        "Record 30 seconds of baseline/rest for noise and normalization.",
        "Collect 4 to 6 labeled examples for each requested state.",
        "Run artifact scan and reject spans with clipping, drift, or line noise.",
        "Fit a quick baseline model, then compare against foundation embeddings.",
        "Preview live predictions, confidence, latency, and per-channel evidence.",
    ]
    if source == "dataset":
        protocol_steps[1] = "Import dataset files, infer streams, and map labels to a common schema."
        protocol_steps[2] = "Slice windows around events or generate weak labels from annotations."
    if "5 minute" in q or "5-minute" in q or "five minute" in q:
        protocol_steps.append("Keep the first pass constrained to a 5 minute setup budget.")
    if output in {"software_control", "hardware_control"} or task["type"] == "realtime_control":
        protocol_steps.append("Bind high-confidence states to actions with cooldowns and an emergency stop.")

    models = [
        {
            "name": "Fast baseline",
            "kind": "LDA / logistic regression",
            "why": "Fits in seconds and gives a sanity-check confusion matrix.",
            "latency_target_ms": 15,
        },
        {
            "name": "Foundation embedding head",
            "kind": "NeuroRVQ-style embedding plus linear head",
            "why": "Uses live embedding vectors to improve small-data performance.",
            "latency_target_ms": 30,
        },
        {
            "name": "Temporal verifier",
            "kind": "TCN / shallow sequence model",
            "why": "Stabilizes state changes and rejects one-frame spikes.",
            "latency_target_ms": 45,
        },
    ]
    if task["type"] == "artifact_cleanup":
        models.append({
            "name": "Artifact segmenter",
            "kind": "per-channel quality classifier",
            "why": "Produces persistent colored regions for clean, hum, drift, motion, and clip.",
            "latency_target_ms": 20,
        })

    exports = [
        "Live model card with labels, channels, filters, features, and validation split.",
        "Windowed dataset manifest with subject/session metadata.",
        "CSV/JSON prediction stream for downstream apps.",
    ]
    if "bids" in q or output == "research_dataset":
        exports.append("BIDS-style folder scaffold where the selected signal type supports it.")
    if "matlab" in q:
        exports.append("MATLAB-friendly analysis bundle.")

    next_actions = [
        "Use this plan to create a runnable pipeline graph.",
        "Attach prompt, labels, filters, and guardrails to the session record.",
        "Start a guided acquisition timer and show live quality gates before training.",
    ]

    display = str(profile.get("display_name") or profile.get("key") or "Signal")
    title = f"{display} {task['type'].replace('_', ' ').title()} Pipeline"
    return {
        "title": title,
        "summary": (
            f"Build a {source.replace('_', ' ')} to {output.replace('_', ' ')} workflow for {display}. "
            "The plan is for research and engineering use; it is not a diagnosis pipeline."
        ),
        "source": source,
        "output": output,
        "signal": {
            "profile": str(profile.get("key") or ""),
            "display_name": display,
            "family": str(profile.get("family") or ""),
            "units": str(profile.get("units") or ""),
            "channels": channels,
        },
        "task": task,
        "filters": filters,
        "windowing": windowing,
        "features": feature_names,
        "label_protocol": label_protocol,
        "protocol": {
            "estimated_minutes": 5 if ("5 minute" in q or "5-minute" in q or "five minute" in q) else 12,
            "steps": protocol_steps,
        },
        "models": models,
        "model_candidates": model_candidates,
        "exports": exports,
        "guardrails": [
            "Require per-session calibration before live decisions.",
            "Keep raw data, predictions, and AI notes auditable.",
            "Flag medical interpretations as review-only unless a validated clinical workflow is added.",
        ],
        "next_actions": next_actions,
    }


def _pipeline_qa(req: PipelineQARequest) -> dict:
    window = app_state.last_vis_window
    if window is None or window.size == 0:
        raise ValueError("No live signal window is available yet.")
    arr = window.astype(np.float64, copy=False)
    diagnostics = app_state.diagnostics.status()
    noise = diagnostics.get("noise") or {}
    timing = diagnostics.get("timing") or {}
    rms = np.sqrt(np.mean(np.square(arr), axis=1))
    p2p = np.ptp(arr, axis=1)
    quality = app_state.pipeline.get_channel_quality(window)
    channel_labels = list(config.channel_labels)
    duplicate_pairs = _detect_duplicate_signal_channels(arr)
    duplicate_indexes = sorted({idx for pair in duplicate_pairs for idx in pair})
    channels = []
    for idx in range(arr.shape[0]):
        channels.append({
            "index": idx,
            "label": channel_labels[idx] if idx < len(channel_labels) else f"CH{idx + 1}",
            "rms": round(float(rms[idx]), 5),
            "peak_to_peak": round(float(p2p[idx]), 5),
            "quality": round(float(quality[idx]) if idx < len(quality) else 0.0, 3),
            "status": "duplicate" if idx in duplicate_indexes else "usable",
        })

    artifacts: List[Dict[str, Any]] = []
    clip_pct = float(noise.get("clip_pct") or 0.0)
    hum_50 = float(noise.get("hum_50_db") or -120.0)
    hum_60 = float(noise.get("hum_60_db") or -120.0)
    drift = float(noise.get("drift_db") or -120.0)
    crest = float(noise.get("crest_factor") or 0.0)
    if clip_pct > 1.0:
        artifacts.append({"kind": "clip", "severity": min(1.0, clip_pct / 10.0), "detail": f"Clip risk {clip_pct:.2f}%."})
    if max(hum_50, hum_60) > -20.0:
        artifacts.append({"kind": "hum", "severity": min(1.0, (max(hum_50, hum_60) + 40.0) / 40.0), "detail": "Line-noise band is elevated."})
    if drift > -20.0:
        artifacts.append({"kind": "drift", "severity": min(1.0, (drift + 40.0) / 40.0), "detail": "Low-frequency drift is elevated."})
    if crest > 10.0:
        artifacts.append({"kind": "motion", "severity": min(1.0, crest / 25.0), "detail": "Crest factor suggests impulsive motion."})
    if duplicate_pairs:
        labels = ", ".join(f"CH{a + 1}/CH{b + 1}" for a, b in duplicate_pairs[:4])
        artifacts.append({
            "kind": "duplicate_channel",
            "severity": 0.65,
            "detail": f"Channels are moving almost identically ({labels}). Unused or floating inputs may be mirroring the real electrode pair.",
        })

    local_ml: Dict[str, Any] = {}
    try:
        local_ml = app_state.ai.local_models.analyze(_live_ml_snapshot(window, diagnostics))
        artifact_head = (local_ml.get("outputs") or {}).get("artifact_classifier") if isinstance(local_ml.get("outputs"), dict) else None
        if not artifact_head:
            artifact_head = (local_ml.get("artifact_classifier") or {})
        label = str(artifact_head.get("label") or local_ml.get("artifact_label") or "").lower()
        confidence = float(artifact_head.get("confidence") or local_ml.get("artifact_confidence") or 0.0)
        if label and label not in {"clean", "none", "normal", "stable"} and confidence >= 0.35:
            artifacts.append({"kind": label, "severity": round(confidence, 3), "detail": "Local ML artifact head flagged this window."})
    except Exception as exc:
        local_ml = {"error": str(exc)}

    score = 100.0
    score -= min(35.0, clip_pct * 3.0)
    if max(hum_50, hum_60) > -40.0:
        score -= min(20.0, max(0.0, max(hum_50, hum_60) + 40.0) * 0.75)
    if drift > -40.0:
        score -= min(20.0, max(0.0, drift + 40.0) * 0.75)
    score -= min(15.0, max(0.0, crest - 8.0) * 1.5)
    if duplicate_pairs:
        score -= min(18.0, 6.0 * len(duplicate_pairs))
    if float(timing.get("signal_age_ms") or 0.0) > 1500.0:
        score -= 30.0
    score = round(max(0.0, min(100.0, score)), 1)
    ready = score >= 75.0 and len([a for a in artifacts if float(a.get("severity") or 0.0) >= 0.5]) == 0
    return {
        "ok": True,
        "qa": {
            "score": score,
            "ready_for_training": ready,
            "profile": config.signal_profile_name,
            "sample_rate_hz": config.sample_rate,
            "window_samples": int(arr.shape[1]),
            "channel_count": int(arr.shape[0]),
            "channels": channels,
            "artifacts": artifacts,
            "duplicate_channel_pairs": [[a + 1, b + 1] for a, b in duplicate_pairs],
            "diagnostics": diagnostics,
            "local_ml": local_ml,
            "recommendations": [
                "Proceed to baseline training." if ready else "Collect a cleaner window before fitting a model.",
                "Keep artifact spans linked to the session so failure cases are auditable.",
                "Use per-channel highlights only where QA or ML detected a reason.",
            ],
        },
    }


def _pipeline_train_baseline(req: PipelineBaselineTrainRequest) -> dict:
    classifier = str(req.classifier or "LDA").strip() or "LDA"
    if classifier != "LDA":
        classifier = "LDA"
    summary = app_state.pipeline.get_training_summary()
    per_label = {
        str(label): int(count)
        for label, count in (summary.get("per_gesture") or {}).items()
        if int(count or 0) > 0
    }
    total = int(summary.get("total_windows") or 0)
    ready = total >= 10 and len(per_label) >= 2
    if not ready:
        return {
            "ok": True,
            "trained": False,
            "ready": False,
            "classifier": classifier,
            "summary": summary,
            "requirements": {
                "minimum_windows_total": 10,
                "minimum_labeled_classes": 2,
                "current_windows_total": total,
                "current_labeled_classes": len(per_label),
            },
            "next_actions": [
                "Record at least two labels from the Training card or guided protocol.",
                "Target 25 or more clean windows per label for the first LDA baseline.",
                "Run signal QA again before fitting.",
            ],
        }

    result = app_state.pipeline.train(classifier_name=classifier)
    model_path = ""
    if result.get("success"):
        model_path = app_state.pipeline.save_model()
        result["model_path"] = model_path
        app_state.recorder.log_event("pipeline_baseline_trained", result)
        app_state.lsl.push_marker(
            "pipeline_baseline_trained",
            {
                "profile": config.signal_profile_name,
                "classifier": classifier,
                "path": model_path,
                "success": True,
            },
        )
    return {
        "ok": True,
        "trained": bool(result.get("success")),
        "ready": True,
        "classifier": classifier,
        "result": result,
        "model_path": model_path,
        "summary": app_state.pipeline.get_training_summary(),
        "next_actions": [
            "Compare validation accuracy against foundation embedding features.",
            "Inspect confusion or failure cases before exporting.",
            "Keep this baseline as the reproducible control model.",
        ],
    }


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        if isinstance(value, dict):
            return {str(key): _json_safe(val) for key, val in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [_json_safe(item) for item in value]
        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                pass
        return str(value)


def _write_pipeline_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")


def _embedding_snapshot_from_local(local_ml: dict, timestamp: Optional[float] = None) -> dict:
    embeddings = list((local_ml or {}).get("foundation_embeddings") or [])
    primary = dict(embeddings[0] or {}) if embeddings else {}
    return {
        "timestamp": float(timestamp or time.time()),
        "profile": config.signal_profile_name,
        "embedding_count": len(embeddings),
        "model_id": str(primary.get("model_id") or ""),
        "name": str(primary.get("name") or ""),
        "embedding_dim": int(primary.get("embedding_dim") or 0),
        "embedding_norm": round(float(primary.get("embedding_norm") or 0.0), 6),
        "mean": round(float(primary.get("mean") or 0.0), 6),
        "std": round(float(primary.get("std") or 0.0), 6),
        "preview": list(primary.get("preview") or [])[:8],
    }


def _append_pipeline_embedding_snapshot(local_ml: dict, timestamp: Optional[float] = None) -> None:
    snapshot = _embedding_snapshot_from_local(local_ml, timestamp)
    if not snapshot.get("embedding_count"):
        return
    app_state.pipeline_embedding_timeline.append(snapshot)
    if len(app_state.pipeline_embedding_timeline) > 240:
        app_state.pipeline_embedding_timeline = app_state.pipeline_embedding_timeline[-240:]


def _pipeline_dataset_readiness(qa: dict, train_result: dict) -> dict:
    qa_score = float((qa or {}).get("score") or 0.0)
    summary = dict((train_result or {}).get("summary") or app_state.pipeline.get_training_summary() or {})
    counts = {
        str(label): int(count)
        for label, count in dict(summary.get("per_gesture") or summary.get("per_label") or {}).items()
        if int(count or 0) > 0
    }
    total = int(summary.get("total_windows") or sum(counts.values()) or 0)
    label_count = len(counts)
    balance = 0.0
    if counts:
        vals = list(counts.values())
        balance = min(vals) / max(max(vals), 1)
    qa_component = min(40.0, qa_score * 0.4)
    label_component = min(25.0, label_count * 12.5)
    window_component = min(25.0, total * 1.0)
    balance_component = balance * 10.0
    score = round(min(100.0, qa_component + label_component + window_component + balance_component), 1)
    gaps = []
    if qa_score < 75.0:
        gaps.append("Run QA on a cleaner window.")
    if label_count < 2:
        gaps.append("Record at least two labels.")
    if total < 25:
        gaps.append("Collect at least 25 labeled windows for a first export.")
    if counts and balance < 0.5:
        gaps.append("Balance label counts before comparing models.")
    return {
        "score": score,
        "ready": score >= 70.0 and label_count >= 2 and total >= 10,
        "qa_score": round(qa_score, 1),
        "total_windows": total,
        "label_count": label_count,
        "label_balance": round(balance, 3),
        "per_label": counts,
        "gaps": gaps,
    }


def _pipeline_model_compare(req: PipelineActionRequest) -> dict:
    plan = dict(req.plan or {})
    qa = dict(req.qa or {})
    train_result = dict(req.train_result or {})
    if not train_result and app_state.pipeline_acquisition_model:
        train_result = {
            "trained": True,
            "classifier": app_state.pipeline_acquisition_model.get("classifier", "logistic_regression"),
            "model_path": app_state.pipeline_acquisition_model.get("model_path", ""),
            "result": dict(app_state.pipeline_acquisition_model),
            "summary": {
                "total_windows": app_state.pipeline_acquisition_model.get("n_windows", 0),
                "per_label": app_state.pipeline_acquisition_model.get("per_label", {}),
                "is_trained": True,
                "profile": config.signal_profile_name,
            },
        }
    readiness = _pipeline_dataset_readiness(qa, train_result)
    training_summary = app_state.pipeline.get_training_summary()
    trained = bool(training_summary.get("is_trained") or train_result.get("trained"))
    candidates = list(plan.get("model_candidates") or plan.get("models") or [])
    local_status = app_state.ai.status().get("local_models") or {}
    foundation_ready = [
        item for item in list(local_status.get("foundation_models") or [])
        if item.get("loaded") and item.get("inference_ready")
    ]
    comparisons = []
    for candidate in candidates:
        cid = str(candidate.get("id") or candidate.get("name") or "").lower()
        name = str(candidate.get("name") or candidate.get("id") or "Model")
        if "baseline" in cid or "lda" in cid.lower():
            status = "trained" if trained else "needs_data"
            score = 78 if trained else min(55, readiness["score"])
            note = "Fastest control model; use it as the first reproducible baseline."
        elif "foundation" in cid:
            status = "ready" if foundation_ready else "missing_encoder"
            score = 82 if foundation_ready else 35
            note = f"{len(foundation_ready)} local foundation encoder(s) available."
        elif "temporal" in cid or "sequence" in cid:
            status = "queued" if trained else "blocked"
            score = 65 if trained else 30
            note = "Train after baseline classes are separable."
        else:
            status = str(candidate.get("status") or "candidate")
            score = 50
            note = str(candidate.get("notes") or candidate.get("why") or "")
        comparisons.append({
            "id": str(candidate.get("id") or name.lower().replace(" ", "_")),
            "name": name,
            "status": status,
            "score": round(float(score), 1),
            "latency_target_ms": candidate.get("latency_target_ms"),
            "notes": note,
        })

    artifacts = list(qa.get("artifacts") or [])
    failure_cases = []
    if artifacts:
        failure_cases.extend([
            {
                "type": str(item.get("kind") or "artifact"),
                "detail": str(item.get("detail") or "Artifact flagged by QA."),
                "action": "Mark span, exclude from training, and keep it in failure-case export.",
            }
            for item in artifacts[:6]
        ])
    if readiness["label_count"] < 2:
        failure_cases.append({
            "type": "label_gap",
            "detail": "Not enough labeled classes for supervised comparison.",
            "action": "Record two or more labels before fitting.",
        })
    if not trained:
        failure_cases.append({
            "type": "baseline_missing",
            "detail": "No trained baseline is available yet.",
            "action": "Fit LDA after labels are collected.",
        })
    if not foundation_ready:
        failure_cases.append({
            "type": "foundation_missing",
            "detail": "No ready foundation embedding path for this signal.",
            "action": "Use the baseline path or install a matching encoder.",
        })
    return {
        "ok": True,
        "comparison": {
            "comparisons": comparisons,
            "failure_cases": failure_cases,
            "readiness": readiness,
            "recommendation": "Collect labels first." if not trained else "Export baseline and compare with foundation features.",
        },
    }


def _pipeline_browser_runtime(req: PipelineActionRequest) -> dict:
    context = dict(req.context or {})
    client_webgpu = bool(context.get("webgpu_available"))
    deploy = _pipeline_export_prompt_model_runtime(req)
    onnx_files = list(deploy.get("onnx_files") or [])
    browser_files = list(deploy.get("browser_files") or [])
    training_summary = app_state.pipeline.get_training_summary()
    prompt_summary = _pipeline_prompt_model_summary()
    trained = bool(
        training_summary.get("is_trained")
        or (req.train_result or {}).get("trained")
        or prompt_summary.get("loaded")
    )
    ready = bool(onnx_files or browser_files)
    status = "ready_onnx" if onnx_files else ("ready_browser_linear" if browser_files else ("needs_onnx_export" if trained else "needs_trained_small_model"))
    return {
        "ok": True,
        "runtime": {
            "ready": ready,
            "client_webgpu_available": client_webgpu,
            "onnx_model_count": len(onnx_files),
            "browser_model_count": len(browser_files),
            "trained_small_model_available": trained,
            "recommended_backend": "onnxruntime-web/webgpu" if client_webgpu else "onnxruntime-web/wasm",
            "status": status,
            "files": onnx_files,
            "browser_files": browser_files,
            "model_card": deploy.get("model_card") or prompt_summary.get("model_card_path") or "",
            "export_dir": deploy.get("export_dir") or "",
            "onnx_error": deploy.get("onnx_error") or "",
            "input_contract": {
                "layout": "channels_by_samples",
                "channels": config.n_channels,
                "sample_rate_hz": config.sample_rate,
                "window_samples": int(((req.plan or {}).get("windowing") or {}).get("window_samples") or config.window_size_samples),
                "feature_dim": int(prompt_summary.get("feature_dim") or 0),
                "dtype": "float32",
            },
            "loader": {
                "package": "onnxruntime-web",
                "entry_file": "browser_runtime.js",
                "cdn": "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js",
                "execution_providers": ["webgpu", "wasm"] if client_webgpu else ["wasm"],
            },
            "next_actions": [
                "Train the LDA baseline first." if not trained else "Export a small ONNX model for browser execution.",
                "Install skl2onnx and onnx for true ONNX export." if deploy.get("onnx_error") else "Serve ONNX assets with the exported pipeline package.",
                "Use browser_model.json today for the zero-dependency linear runtime." if browser_files else "Create browser_model.json from the active prompt model.",
                "Use WebGPU when available and fall back to WASM.",
            ],
        },
    }


def _pipeline_export_prompt_model_runtime(req: PipelineActionRequest) -> dict:
    runtime = app_state.pipeline_acquisition_runtime or {}
    model = runtime.get("model")
    encoder = runtime.get("label_encoder")
    model_path = Path(str(runtime.get("model_path") or ""))
    if model is None or encoder is None:
        return {"ok": False, "onnx_files": [], "browser_files": [], "onnx_error": "No active prompt model is loaded."}
    export_root = Path(config.data_dir) / "browser_models"
    export_root.mkdir(parents=True, exist_ok=True)
    stem = _safe_stem(model_path.stem if model_path.name else f"{config.signal_profile_name}_prompt_model", "prompt_model")
    browser_path = export_root / f"{stem}.browser.json"
    onnx_path = export_root / f"{stem}.onnx"
    card_path = str(runtime.get("model_card_path") or "")
    raw_labels = runtime.get("labels")
    if not raw_labels:
        raw_labels = list(getattr(encoder, "classes_", []) or [])
    labels = [str(label) for label in list(raw_labels or [])]
    feature_dim = int(runtime.get("feature_dim") or 0)
    browser_files: List[str] = []
    onnx_files: List[str] = []
    onnx_error = ""
    try:
        browser_payload = _pipeline_browser_linear_payload(runtime, req)
        _write_pipeline_json(browser_path, browser_payload)
        browser_files.append(browser_path.as_posix())
    except Exception as exc:
        onnx_error = f"Browser linear export failed: {type(exc).__name__}: {exc}"
    try:
        if not onnx_path.exists() or (model_path.exists() and onnx_path.stat().st_mtime < model_path.stat().st_mtime):
            from skl2onnx import convert_sklearn
            from skl2onnx.common.data_types import FloatTensorType

            initial_types = [("features", FloatTensorType([None, feature_dim or 1]))]
            onx = convert_sklearn(model, initial_types=initial_types, target_opset=15)
            onnx_path.write_bytes(onx.SerializeToString())
        onnx_files.append(onnx_path.as_posix())
    except Exception as exc:
        onnx_error = onnx_error or f"{type(exc).__name__}: {exc}"
    return {
        "ok": bool(browser_files or onnx_files),
        "export_dir": export_root.as_posix(),
        "onnx_files": onnx_files,
        "browser_files": browser_files,
        "model_card": card_path,
        "labels": labels,
        "feature_dim": feature_dim,
        "onnx_error": onnx_error,
    }


def _pipeline_browser_linear_payload(runtime: dict, req: PipelineActionRequest) -> dict:
    pipeline = runtime.get("model")
    encoder = runtime.get("label_encoder")
    named = getattr(pipeline, "named_steps", {}) or {}
    scaler = named.get("scaler")
    clf = named.get("clf")
    if scaler is None or clf is None:
        raise ValueError("Active prompt model is not a scaler + classifier pipeline.")
    raw_labels = runtime.get("labels")
    if not raw_labels:
        raw_labels = list(getattr(encoder, "classes_", []) or [])
    labels = [str(label) for label in list(raw_labels or [])]
    coef = np.asarray(getattr(clf, "coef_", []), dtype=np.float64)
    intercept = np.asarray(getattr(clf, "intercept_", []), dtype=np.float64)
    mean = np.asarray(getattr(scaler, "mean_", []), dtype=np.float64)
    scale = np.asarray(getattr(scaler, "scale_", []), dtype=np.float64)
    if coef.ndim != 2 or not coef.size:
        raise ValueError("Classifier coefficients are missing.")
    if len(labels) == 2 and coef.shape[0] == 1:
        coef = np.vstack([-coef[0], coef[0]])
        intercept = np.asarray([-intercept[0], intercept[0]], dtype=np.float64)
    return {
        "format": "kyma_browser_linear_v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "profile": config.signal_profile_name,
        "classifier": "logistic_regression",
        "labels": labels,
        "feature_dim": int(runtime.get("feature_dim") or coef.shape[1]),
        "input_contract": {
            "layout": "channels_by_samples",
            "channels": config.n_channels,
            "sample_rate_hz": config.sample_rate,
            "window_samples": int(((req.plan or {}).get("windowing") or {}).get("window_samples") or config.window_size_samples),
            "feature_order": [
                "rms_per_channel",
                "mav_per_channel",
                "mean_per_channel",
                "std_per_channel",
                "peak_to_peak_per_channel",
                "waveform_length_per_channel",
                "zero_crossings_per_channel",
                "rms_mean",
                "rms_std",
                "rms_max",
                "peak_to_peak_mean",
                "peak_to_peak_std",
            ],
        },
        "normalization": {
            "mean": [float(v) for v in mean.tolist()],
            "scale": [float(v if abs(v) > 1e-12 else 1.0) for v in scale.tolist()],
        },
        "linear": {
            "coef": [[float(v) for v in row] for row in coef.tolist()],
            "intercept": [float(v) for v in intercept.tolist()],
        },
        "source": {
            "model_path": str(runtime.get("model_path") or ""),
            "model_card_path": str(runtime.get("model_card_path") or ""),
        },
    }


def _pipeline_sample_window_for_deploy(req: PipelineActionRequest) -> np.ndarray:
    window = app_state.last_vis_window
    if window is not None and np.asarray(window).size:
        try:
            return _coerce_pipeline_window(np.asarray(window, dtype=np.float32))
        except Exception:
            pass
    records = list(app_state.pipeline_acquisition_windows or [])
    for record in reversed(records[-50:]):
        restored = _restore_acquisition_window(record)
        if restored is not None:
            return _coerce_pipeline_window(restored)
    samples = int(((req.plan or {}).get("windowing") or {}).get("window_samples") or config.window_size_samples)
    x = np.linspace(0.0, 1.0, max(samples, 8), dtype=np.float32)
    rows = []
    for ch in range(config.n_channels):
        rows.append((np.sin(2.0 * np.pi * (ch + 1) * x) * (0.2 + ch * 0.03)).astype(np.float32))
    return np.vstack(rows).astype(np.float32)


def _softmax_np(logits: np.ndarray) -> np.ndarray:
    arr = np.asarray(logits, dtype=np.float64).reshape(-1)
    arr = arr - float(np.max(arr)) if arr.size else arr
    exp = np.exp(arr)
    denom = float(np.sum(exp)) or 1.0
    return exp / denom


def _pipeline_browser_linear_smoke(browser_path: Path, window: np.ndarray) -> dict:
    payload = json.loads(browser_path.read_text(encoding="utf-8"))
    features = _pipeline_window_feature_vector(window).astype(np.float32)
    mean = np.asarray((payload.get("normalization") or {}).get("mean") or [], dtype=np.float32)
    scale = np.asarray((payload.get("normalization") or {}).get("scale") or [], dtype=np.float32)
    coef = np.asarray((payload.get("linear") or {}).get("coef") or [], dtype=np.float32)
    intercept = np.asarray((payload.get("linear") or {}).get("intercept") or [], dtype=np.float32)
    if mean.size != features.size or scale.size != features.size or coef.ndim != 2 or coef.shape[1] != features.size:
        raise ValueError("Browser model shape does not match extracted features.")
    x = (features - mean) / np.where(np.abs(scale) > 1e-12, scale, 1.0)
    logits = coef @ x + intercept
    probs = _softmax_np(logits)
    labels = [str(label) for label in payload.get("labels") or []]
    best = int(np.argmax(probs)) if probs.size else 0
    return {
        "ok": True,
        "file": browser_path.as_posix(),
        "feature_dim": int(features.size),
        "label": labels[best] if best < len(labels) else str(best),
        "confidence": round(float(probs[best]) if probs.size else 0.0, 4),
        "probabilities": [
            {"label": labels[idx] if idx < len(labels) else str(idx), "probability": round(float(prob), 4)}
            for idx, prob in enumerate(probs.tolist())
        ],
    }


def _pipeline_onnx_smoke(onnx_path: Path, window: np.ndarray) -> dict:
    features = _pipeline_window_feature_vector(window).astype(np.float32).reshape(1, -1)
    out: Dict[str, Any] = {"ok": False, "file": onnx_path.as_posix(), "feature_dim": int(features.shape[1])}
    import onnx

    model = onnx.load(onnx_path.as_posix())
    onnx.checker.check_model(model)
    out["model_checked"] = True
    try:
        import onnxruntime as ort

        session = ort.InferenceSession(onnx_path.as_posix(), providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: features})
        out.update({
            "ok": True,
            "input_name": input_name,
            "output_shapes": [list(np.asarray(item).shape) for item in outputs],
        })
    except Exception as exc:
        out["runtime_error"] = f"{type(exc).__name__}: {exc}"
        out["ok"] = True
    return out


def _pipeline_deploy_smoke(req: PipelineActionRequest) -> dict:
    runtime = _pipeline_browser_runtime(req)["runtime"]
    window = _pipeline_sample_window_for_deploy(req)
    browser_results = []
    onnx_results = []
    errors = []
    for raw_path in list(runtime.get("browser_files") or [])[:3]:
        try:
            browser_results.append(_pipeline_browser_linear_smoke(Path(str(raw_path)), window))
        except Exception as exc:
            errors.append(f"{raw_path}: {type(exc).__name__}: {exc}")
    for raw_path in list(runtime.get("files") or [])[:3]:
        try:
            onnx_results.append(_pipeline_onnx_smoke(Path(str(raw_path)), window))
        except Exception as exc:
            errors.append(f"{raw_path}: {type(exc).__name__}: {exc}")
    ready = bool(runtime.get("ready")) and (bool(browser_results) or bool(onnx_results)) and not errors
    return {
        "ok": True,
        "deploy": {
            "ready": ready,
            "status": "passed" if ready else "review",
            "runtime_status": runtime.get("status"),
            "sample_shape": list(window.shape),
            "browser_results": browser_results,
            "onnx_results": onnx_results,
            "errors": errors,
            "next_actions": [
                "Export package is deploy-smoke ready." if ready else "Review runtime export errors before deployment.",
                "Use the browser JSON runtime for zero-dependency demos.",
                "Use ONNX runtime when the exported .onnx file is present.",
            ],
        },
        "runtime": runtime,
    }


def _pipeline_browser_runtime_js(runtime: dict) -> str:
    contract = dict((runtime or {}).get("input_contract") or {})
    return f"""/* KYMA browser runtime for exported ONNX biosignal models.
   Load onnxruntime-web before this file:
   https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js
*/
export class KymaBrowserRuntime {{
  constructor(options = {{}}) {{
    this.options = options;
    this.session = null;
    this.inputName = '';
    this.outputName = '';
    this.linearModel = null;
    this.contract = {json.dumps(_json_safe(contract), indent=2)};
  }}

  async load({{ modelUrl, linearModelUrl, preferWebGPU = true, wasmBaseUrl = '' }} = {{}}) {{
    if (linearModelUrl) {{
      const response = await fetch(linearModelUrl);
      if (!response.ok) throw new Error(`Failed to load linear model: ${{response.status}}`);
      this.linearModel = await response.json();
      this.contract = this.linearModel.input_contract || this.contract;
      return {{ backend: 'browser_linear', labels: this.linearModel.labels || [] }};
    }}
    if (!globalThis.ort) throw new Error('onnxruntime-web is not loaded');
    if (!modelUrl) throw new Error('modelUrl or linearModelUrl is required');
    if (wasmBaseUrl && globalThis.ort.env?.wasm) {{
      globalThis.ort.env.wasm.wasmPaths = wasmBaseUrl;
    }}
    const providers = preferWebGPU && navigator.gpu ? ['webgpu', 'wasm'] : ['wasm'];
    this.session = await globalThis.ort.InferenceSession.create(modelUrl, {{
      executionProviders: providers,
      graphOptimizationLevel: 'all',
    }});
    this.inputName = this.session.inputNames[0];
    this.outputName = this.session.outputNames[0];
    return {{ inputName: this.inputName, outputName: this.outputName, providers }};
  }}

  async infer(channelsBySamples) {{
    if (this.linearModel) return this.inferLinear(channelsBySamples);
    if (!this.session) throw new Error('Call load() before infer()');
    const features = this.extractFeatures(channelsBySamples);
    const featureDim = this.contract.feature_dim || features.length;
    const data = new Float32Array(featureDim);
    for (let i = 0; i < featureDim; i += 1) data[i] = Number(features[i] || 0);
    const tensor = new globalThis.ort.Tensor('float32', data, [1, featureDim]);
    const outputs = await this.session.run({{ [this.inputName]: tensor }});
    return outputs[this.outputName];
  }}

  inferLinear(channelsBySamples) {{
    const model = this.linearModel;
    if (!model) throw new Error('Call load() before infer()');
    const features = this.extractFeatures(channelsBySamples);
    const mean = model.normalization?.mean || [];
    const scale = model.normalization?.scale || [];
    const x = features.map((v, i) => (v - Number(mean[i] || 0)) / (Number(scale[i] || 1) || 1));
    const coef = model.linear?.coef || [];
    const intercept = model.linear?.intercept || [];
    const logits = coef.map((row, c) => row.reduce((sum, w, i) => sum + Number(w || 0) * Number(x[i] || 0), Number(intercept[c] || 0)));
    const probs = this.softmax(logits);
    let best = 0;
    for (let i = 1; i < probs.length; i += 1) if (probs[i] > probs[best]) best = i;
    const labels = model.labels || [];
    return {{
      backend: 'browser_linear',
      label: labels[best] || String(best),
      confidence: probs[best] || 0,
      probabilities: labels.map((label, i) => ({{ label, probability: probs[i] || 0 }})),
      logits,
      features,
    }};
  }}

  extractFeatures(channelsBySamples) {{
    const rows = channelsBySamples || [];
    const channels = this.contract.channels || rows.length;
    const rms = [], mav = [], mean = [], std = [], p2p = [], wl = [], zc = [];
    for (let ch = 0; ch < channels; ch += 1) {{
      const row = Array.from(rows[ch] || []);
      const n = Math.max(row.length, 1);
      const avg = row.reduce((s, v) => s + Number(v || 0), 0) / n;
      let sq = 0, abs = 0, variance = 0, min = Infinity, max = -Infinity, length = 0, zeros = 0;
      for (let i = 0; i < row.length; i += 1) {{
        const v = Number(row[i] || 0);
        sq += v * v;
        abs += Math.abs(v);
        variance += (v - avg) * (v - avg);
        min = Math.min(min, v);
        max = Math.max(max, v);
        if (i > 0) {{
          const prev = Number(row[i - 1] || 0);
          length += Math.abs(v - prev);
          if (v * prev < 0) zeros += 1;
        }}
      }}
      rms.push(Math.sqrt(sq / n));
      mav.push(abs / n);
      mean.push(avg);
      std.push(Math.sqrt(variance / n));
      p2p.push(Number.isFinite(max - min) ? max - min : 0);
      wl.push(length);
      zc.push(zeros);
    }}
    const statsMean = arr => arr.reduce((s, v) => s + v, 0) / Math.max(arr.length, 1);
    const statsStd = arr => {{
      const avg = statsMean(arr);
      return Math.sqrt(arr.reduce((s, v) => s + (v - avg) * (v - avg), 0) / Math.max(arr.length, 1));
    }};
    return [
      ...rms, ...mav, ...mean, ...std, ...p2p, ...wl, ...zc,
      statsMean(rms), statsStd(rms), Math.max(...rms, 0), statsMean(p2p), statsStd(p2p),
    ];
  }}

  softmax(logits) {{
    const maxLogit = Math.max(...logits, 0);
    const exps = logits.map(v => Math.exp(Number(v || 0) - maxLogit));
    const denom = exps.reduce((s, v) => s + v, 0) || 1;
    return exps.map(v => v / denom);
  }}
}}

globalThis.KymaBrowserRuntime = KymaBrowserRuntime;
"""


def _pipeline_export_index_html(runtime: dict, deploy: dict, browser_model: dict, sample_window: np.ndarray) -> str:
    model_json = json.dumps(_json_safe(browser_model), separators=(",", ":"))
    sample_json = json.dumps(_json_safe(np.asarray(sample_window, dtype=np.float32).round(6).tolist()), separators=(",", ":"))
    deploy_json = json.dumps(_json_safe(deploy), separators=(",", ":"))
    labels = ", ".join(str(label) for label in list(browser_model.get("labels") or [])) or "labels"
    status = str((deploy or {}).get("status") or "ready")
    runtime_status = str((runtime or {}).get("status") or "")
    template = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>KYMA Deploy Self-Test</title>
  <link rel="icon" href="data:,">
  <style>
    :root { color-scheme: light; --bg:#f3eee7; --panel:#fffaf2; --line:#ddd2c4; --text:#252831; --muted:#687080; --ok:#327a57; --warn:#a06d16; --accent:#586fda; }
    * { box-sizing:border-box; }
    body { margin:0; min-height:100vh; background:var(--bg); color:var(--text); font:14px/1.45 Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif; }
    main { width:min(980px, calc(100vw - 32px)); margin:0 auto; padding:40px 0; }
    header { display:flex; align-items:flex-end; justify-content:space-between; gap:24px; padding-bottom:18px; border-bottom:1px solid var(--line); }
    h1 { margin:0; font-size:30px; letter-spacing:0; line-height:1.05; }
    p { margin:6px 0 0; color:var(--muted); }
    .status { display:inline-flex; align-items:center; gap:8px; min-height:32px; padding:6px 10px; border:1px solid var(--line); border-radius:8px; background:var(--panel); font-weight:700; }
    .dot { width:9px; height:9px; border-radius:999px; background:var(--warn); }
    .status.pass .dot { background:var(--ok); }
    .grid { display:grid; grid-template-columns:1fr 1fr; gap:14px; margin-top:18px; }
    section { border:1px solid var(--line); border-radius:8px; background:var(--panel); padding:16px; }
    h2 { margin:0 0 10px; font-size:13px; text-transform:uppercase; letter-spacing:.04em; color:var(--muted); }
    .metric { display:flex; justify-content:space-between; gap:16px; padding:7px 0; border-top:1px solid rgba(40,45,55,.08); }
    .metric:first-of-type { border-top:0; }
    .metric strong { font-size:13px; }
    .metric span { color:var(--muted); text-align:right; overflow-wrap:anywhere; }
    pre { margin:0; overflow:auto; white-space:pre-wrap; color:#394050; font-size:12px; }
    button { appearance:none; border:1px solid var(--accent); background:var(--accent); color:white; border-radius:8px; padding:8px 11px; font-weight:800; cursor:pointer; }
    @media (max-width:760px) { header { align-items:flex-start; flex-direction:column; } .grid { grid-template-columns:1fr; } }
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>KYMA Deploy Self-Test</h1>
        <p>No backend required. This page embeds the exported browser model and runs one sample inference locally.</p>
      </div>
      <div id="status" class="status"><span class="dot"></span><span>Not run</span></div>
    </header>
    <div class="grid">
      <section>
        <h2>Model</h2>
        <div class="metric"><strong>Profile</strong><span id="profile">--</span></div>
        <div class="metric"><strong>Labels</strong><span id="labels">--</span></div>
        <div class="metric"><strong>Feature Dim</strong><span id="feature-dim">--</span></div>
        <div class="metric"><strong>Runtime</strong><span id="runtime">--</span></div>
      </section>
      <section>
        <h2>Inference</h2>
        <div class="metric"><strong>Prediction</strong><span id="prediction">--</span></div>
        <div class="metric"><strong>Confidence</strong><span id="confidence">--</span></div>
        <div class="metric"><strong>Sample Shape</strong><span id="sample-shape">--</span></div>
        <button id="rerun" type="button">Run Again</button>
      </section>
      <section>
        <h2>Packaged Artifacts</h2>
        <div class="metric"><strong>ONNX</strong><span id="onnx">--</span></div>
        <div class="metric"><strong>Browser JSON</strong><span id="browser">embedded</span></div>
        <div class="metric"><strong>Smoke</strong><span id="smoke">--</span></div>
      </section>
      <section>
        <h2>Raw Result</h2>
        <pre id="raw">Waiting...</pre>
      </section>
    </div>
  </main>
  <script>
    const MODEL = __MODEL_JSON__;
    const SAMPLE = __SAMPLE_JSON__;
    const DEPLOY = __DEPLOY_JSON__;

    function softmax(logits) {
      const maxLogit = Math.max(...logits, 0);
      const exps = logits.map(v => Math.exp(Number(v || 0) - maxLogit));
      const denom = exps.reduce((s, v) => s + v, 0) || 1;
      return exps.map(v => v / denom);
    }

    function extractFeatures(rows) {
      const channels = MODEL.input_contract?.channels || rows.length;
      const rms = [], mav = [], mean = [], std = [], p2p = [], wl = [], zc = [];
      for (let ch = 0; ch < channels; ch += 1) {
        const row = Array.from(rows[ch] || []);
        const n = Math.max(row.length, 1);
        const avg = row.reduce((s, v) => s + Number(v || 0), 0) / n;
        let sq = 0, abs = 0, variance = 0, min = Infinity, max = -Infinity, length = 0, zeros = 0;
        for (let i = 0; i < row.length; i += 1) {
          const v = Number(row[i] || 0);
          sq += v * v;
          abs += Math.abs(v);
          variance += (v - avg) * (v - avg);
          min = Math.min(min, v);
          max = Math.max(max, v);
          if (i > 0) {
            const prev = Number(row[i - 1] || 0);
            length += Math.abs(v - prev);
            if (v * prev < 0) zeros += 1;
          }
        }
        rms.push(Math.sqrt(sq / n));
        mav.push(abs / n);
        mean.push(avg);
        std.push(Math.sqrt(variance / n));
        p2p.push(Number.isFinite(max - min) ? max - min : 0);
        wl.push(length);
        zc.push(zeros);
      }
      const statsMean = arr => arr.reduce((s, v) => s + v, 0) / Math.max(arr.length, 1);
      const statsStd = arr => {
        const avg = statsMean(arr);
        return Math.sqrt(arr.reduce((s, v) => s + (v - avg) * (v - avg), 0) / Math.max(arr.length, 1));
      };
      return [...rms, ...mav, ...mean, ...std, ...p2p, ...wl, ...zc, statsMean(rms), statsStd(rms), Math.max(...rms, 0), statsMean(p2p), statsStd(p2p)];
    }

    function infer(rows) {
      const features = extractFeatures(rows);
      const mean = MODEL.normalization?.mean || [];
      const scale = MODEL.normalization?.scale || [];
      const x = features.map((v, i) => (v - Number(mean[i] || 0)) / (Number(scale[i] || 1) || 1));
      const coef = MODEL.linear?.coef || [];
      const intercept = MODEL.linear?.intercept || [];
      const logits = coef.map((row, c) => row.reduce((sum, w, i) => sum + Number(w || 0) * Number(x[i] || 0), Number(intercept[c] || 0)));
      const probs = softmax(logits);
      let best = 0;
      for (let i = 1; i < probs.length; i += 1) if (probs[i] > probs[best]) best = i;
      const labels = MODEL.labels || [];
      return { label: labels[best] || String(best), confidence: probs[best] || 0, probabilities: labels.map((label, i) => ({ label, probability: probs[i] || 0 })), feature_dim: features.length };
    }

    function setText(id, value) { document.getElementById(id).textContent = String(value); }

    function run() {
      const result = infer(SAMPLE);
      const pass = result.feature_dim === Number(MODEL.feature_dim || result.feature_dim) && Number.isFinite(result.confidence);
      const status = document.getElementById('status');
      status.classList.toggle('pass', pass);
      status.querySelector('span:last-child').textContent = pass ? 'Passed' : 'Review';
      setText('profile', MODEL.profile || '--');
      setText('labels', (MODEL.labels || []).join(', '));
      setText('feature-dim', MODEL.feature_dim || result.feature_dim);
      setText('runtime', DEPLOY.runtime_status || '__RUNTIME_STATUS__');
      setText('prediction', result.label);
      setText('confidence', `${(result.confidence * 100).toFixed(1)}%`);
      setText('sample-shape', `${SAMPLE.length} x ${(SAMPLE[0] || []).length}`);
      setText('onnx', (DEPLOY.onnx_results || []).length ? 'packaged and checked' : 'not included');
      setText('smoke', DEPLOY.status || '__STATUS__');
      document.getElementById('raw').textContent = JSON.stringify(result, null, 2);
    }

    document.getElementById('rerun').addEventListener('click', run);
    run();
  </script>
</body>
</html>
"""
    return (
        template
        .replace("__MODEL_JSON__", model_json)
        .replace("__SAMPLE_JSON__", sample_json)
        .replace("__DEPLOY_JSON__", deploy_json)
        .replace("__RUNTIME_STATUS__", runtime_status.replace("\\", "\\\\").replace("'", "\\'"))
        .replace("__STATUS__", status.replace("\\", "\\\\").replace("'", "\\'"))
        .replace("KYMA Deploy Self-Test", f"KYMA Deploy Self-Test - {labels}", 1)
    )


def _pipeline_acquisition_protocol(req: PipelineActionRequest) -> dict:
    plan = dict(req.plan or {})
    qa = dict(req.qa or {})
    signal = dict(plan.get("signal") or {})
    task = dict(plan.get("task") or {})
    label_protocol = dict(plan.get("label_protocol") or {})
    labels = list(label_protocol.get("labels") or [])
    if not labels:
        labels = [{"name": str(label), "min_seconds": 8, "target_windows": 25} for label in list(task.get("labels") or config.class_labels)[:6]]
    baseline_s = 30 if config.signal_profile_name in {"emg", "eeg", "ecg", "eog"} else 45
    steps: List[Dict[str, Any]] = [
        {
            "id": "setup",
            "label": "Setup",
            "duration_s": 20,
            "action": "confirm_setup",
            "details": [
                f"Connect the {signal.get('display_name') or config.signal_profile.display_name} electrodes for the muscle or body signal you want to measure.",
                "Watch the channel bars. Connected channels should move a little at rest and more during the task.",
                "If several unused channels show the same shape, turn them off or connect/reference them before training.",
            ],
        },
        {
            "id": "baseline",
            "label": "Rest baseline",
            "duration_s": baseline_s,
            "action": "record_baseline",
            "details": [
                "Relax and stay still so KYMA can learn what quiet signal looks like.",
                "If the signal is noisy, KYMA will ask you to fix the electrodes and try again.",
            ],
        },
    ]
    for idx, item in enumerate(labels[:8], start=1):
        name = str(item.get("name") or f"label_{idx}").strip() or f"label_{idx}"
        seconds = max(5, int(item.get("min_seconds") or 8))
        target = max(10, int(item.get("target_windows") or 25))
        steps.append({
            "id": f"label_{idx}",
            "label": name,
            "duration_s": seconds,
            "target_windows": target,
            "action": "record_label",
            "gesture": name if name in config.class_labels else "",
            "details": [
                str(item.get("cue") or f"Do the {name} task and hold it steady."),
                "KYMA will accept clean signal and ignore noisy moments automatically.",
                "If too much data is rejected, adjust the electrodes and repeat this task.",
            ],
        })
    steps.extend([
        {
            "id": "qa_gate",
            "label": "QA gate",
            "duration_s": 10,
            "action": "run_qa",
            "details": [
                f"Current QA score is {float(qa.get('score') or 0):.1f} / 100.",
                "Require at least two labels and 10 clean windows before baseline training.",
            ],
        },
        {
            "id": "train_compare_export",
            "label": "Train and package",
            "duration_s": 20,
            "action": "train_compare_export",
            "details": [
                "Fit the baseline, compare against foundation embeddings, then export the package.",
                "Use the browser runtime bundle when ONNX files are available.",
            ],
        },
    ])
    total_s = sum(int(step.get("duration_s") or 0) for step in steps)
    return {
        "ok": True,
        "acquisition": {
            "mode": "guided",
            "profile": config.signal_profile_name,
            "estimated_seconds": total_s,
            "estimated_minutes": round(total_s / 60.0, 1),
            "steps": steps,
            "automation": [
                "Start/stop session recording for each label.",
                "Reject windows with QA below threshold while preserving failure cases.",
                "Send accepted windows to the active training buffer.",
                "Trigger train/compare/export when readiness gates pass.",
            ],
        },
    }


def _pipeline_label_suggestions(req: PipelineActionRequest) -> dict:
    window = app_state.last_vis_window
    if window is None or window.size == 0:
        raise ValueError("No live signal window is available yet.")
    arr = window.astype(np.float64, copy=False)
    rms = np.sqrt(np.mean(np.square(arr), axis=1))
    p2p = np.ptp(arr, axis=1)
    median_rms = max(float(np.median(rms)), 1e-9)
    dominant_idx = int(np.argmax(rms)) if rms.size else 0
    diagnostics = app_state.diagnostics.status()
    noise = diagnostics.get("noise") or {}
    qa = dict(req.qa or {})
    task = dict((req.plan or {}).get("task") or {})
    labels = [str(label) for label in list(task.get("labels") or config.class_labels)]
    suggestions: List[Dict[str, Any]] = []
    artifact_kind = ""
    if float(noise.get("clip_pct") or 0.0) > 1.0:
        artifact_kind = "clip"
    elif max(float(noise.get("hum_50_db") or -120.0), float(noise.get("hum_60_db") or -120.0)) > -24.0:
        artifact_kind = "hum"
    elif float(noise.get("drift_db") or -120.0) > -24.0:
        artifact_kind = "drift"
    elif float(noise.get("crest_factor") or 0.0) > 10.0:
        artifact_kind = "motion"
    if artifact_kind:
        suggestions.append({
            "label": f"artifact_{artifact_kind}",
            "confidence": 0.82,
            "kind": "reject_or_failure_case",
            "channel": config.channel_labels[dominant_idx] if dominant_idx < len(config.channel_labels) else f"CH{dominant_idx + 1}",
            "reason": f"{artifact_kind} threshold crossed; keep this as a failure span, not clean training data.",
        })
    activation_ratio = float(rms[dominant_idx] / median_rms) if rms.size else 0.0
    if labels:
        if config.signal_profile_name == "emg":
            label = "rest" if activation_ratio < 1.35 and float(np.mean(rms)) < median_rms * 1.2 else (labels[1] if len(labels) > 1 else labels[0])
            if "fatigue_tracking" == str(task.get("type") or ""):
                dom = float((diagnostics.get("frequency") or {}).get("dominant_hz") or 0.0)
                label = "fatiguing" if dom and dom < 45.0 and activation_ratio > 1.25 else "fresh"
        elif config.signal_profile_name == "eog":
            label = "blink" if float(np.max(p2p)) > median_rms * 6.0 else (labels[0] if labels else "rest")
        elif config.signal_profile_name == "ecg":
            label = "beat_candidate"
        else:
            label = labels[0]
        suggestions.append({
            "label": label,
            "confidence": round(min(0.92, 0.45 + max(0.0, activation_ratio - 1.0) * 0.18), 3),
            "kind": "weak_label",
            "channel": config.channel_labels[dominant_idx] if dominant_idx < len(config.channel_labels) else f"CH{dominant_idx + 1}",
            "reason": f"Dominant channel RMS is {activation_ratio:.2f}x the median channel RMS.",
        })
    suggestions.append({
        "label": "clean_window" if float(qa.get("score") or 0.0) >= 75.0 else "needs_review",
        "confidence": round(min(0.95, max(0.2, float(qa.get("score") or 50.0) / 100.0)), 3),
        "kind": "quality_gate",
        "channel": "all",
        "reason": "Quality gate derived from live QA score and artifact scan.",
    })
    return {
        "ok": True,
        "labels": {
            "suggestions": suggestions,
            "auditable": True,
            "policy": [
                "Treat suggestions as weak labels until a user accepts them.",
                "Never overwrite manual labels.",
                "Store rejected spans as failure cases for model comparison.",
            ],
        },
    }


def _estimate_labeled_window_counts(
    labels: List[str],
    sample_count: int,
    *,
    sample_rate: float,
    metadata: Optional[Dict[str, Any]] = None,
    max_windows: int = 240,
    max_windows_per_label: int = 80,
) -> tuple[int, Dict[str, int], List[str]]:
    segment_samples = max(8, int(round((float(((metadata or {}).get("window_size_ms") or config.window_size_ms)) / 1000.0) * float(sample_rate or config.sample_rate))))
    stride_samples = max(1, int(round((float(((metadata or {}).get("window_step_ms") or config.window_increment_ms)) / 1000.0) * float(sample_rate or config.sample_rate))))
    if sample_count < segment_samples:
        return 0, {}, [f"Need at least {segment_samples} samples for one training window."]
    per_label: Dict[str, int] = {}
    warnings: List[str] = []
    added = 0
    usable_labels = [str(v or "").strip() for v in labels[:sample_count]]
    for start in range(0, sample_count - segment_samples + 1, stride_samples):
        end = start + segment_samples
        label_slice = [label for label in usable_labels[start:end] if label]
        if not label_slice:
            continue
        counts: Dict[str, int] = {}
        for label in label_slice:
            counts[label] = counts.get(label, 0) + 1
        label, vote_count = max(counts.items(), key=lambda item: item[1])
        if vote_count / max(len(label_slice), 1) < 0.75:
            continue
        if per_label.get(label, 0) >= max_windows_per_label:
            continue
        per_label[label] = per_label.get(label, 0) + 1
        added += 1
        if added >= max_windows:
            break
    if not per_label:
        warnings.append("No stable labeled windows found.")
    return added, per_label, warnings


def _score_import_readiness(files: List[dict]) -> dict:
    label_counts: Dict[str, int] = {}
    estimated_windows = 0
    trainable_files = 0
    supported_count = 0
    gaps: List[str] = []
    for item in files:
        if item.get("supported"):
            supported_count += 1
        if item.get("trainable"):
            trainable_files += 1
        for label, count in dict(item.get("estimated_label_windows") or {}).items():
            label_counts[str(label)] = label_counts.get(str(label), 0) + int(count)
            estimated_windows += int(count)
    label_count = len([label for label, count in label_counts.items() if count > 0])
    min_label_windows = min(label_counts.values()) if label_counts else 0
    if not supported_count:
        gaps.append("No supported dataset files were found.")
    if label_count < 2:
        gaps.append("Need at least two labels/classes for supervised training.")
    if estimated_windows < 10:
        gaps.append("Need at least 10 usable labeled windows for a reliable quick model.")
    if label_count >= 2 and min_label_windows < 5:
        gaps.append("Each label should have at least 5 usable windows before training.")
    score = 0.0
    if supported_count:
        score += 20.0
    score += min(35.0, estimated_windows * 1.4)
    score += min(25.0, label_count * 12.5)
    if label_count >= 2:
        score += min(20.0, min_label_windows * 4.0)
    ready = label_count >= 2 and estimated_windows >= 10 and min_label_windows >= 5
    return {
        "ready": bool(ready),
        "score": round(min(100.0, score), 1),
        "supported_files": int(supported_count),
        "trainable_files": int(trainable_files),
        "estimated_windows": int(estimated_windows),
        "label_count": int(label_count),
        "min_label_windows": int(min_label_windows),
        "labels": dict(sorted(label_counts.items(), key=lambda item: (-item[1], item[0]))),
        "gaps": gaps[:6],
    }


def _normalize_schema_mapping(mapping: Optional[Dict[str, Any]]) -> dict:
    mapping = dict(mapping or {})
    roles = {str(k): str(v) for k, v in dict(mapping.get("roles") or {}).items()}
    channels = [str(v) for v in list(mapping.get("channels") or []) if str(v).strip()]
    label = str(mapping.get("label") or "").strip()
    time_col = str(mapping.get("time") or "").strip()
    for name, role in roles.items():
        if role == "channel" and name not in channels:
            channels.append(name)
        elif role == "label" and not label:
            label = name
        elif role == "time" and not time_col:
            time_col = name
    return {"roles": roles, "channels": channels, "label": label, "time": time_col}


def _csv_schema_preview(path: Path) -> dict:
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as fh:
        reader = csv.DictReader(fh)
        raw_names = [str(name or "").strip() for name in (reader.fieldnames or [])]
        rows = [row for _, row in zip(range(250), reader)]
    columns = []
    channel_idx = 1
    for raw_name in raw_names:
        clean = raw_name.lstrip("\ufeff")
        lower = clean.lower()
        values = [str(row.get(raw_name) or "").strip() for row in rows]
        non_empty = [value for value in values if value != ""]
        numeric = 0
        for value in non_empty:
            try:
                float(value)
                numeric += 1
            except Exception:
                pass
        numeric_coverage = numeric / max(len(non_empty), 1)
        unique_values = len(set(non_empty[:200]))
        if lower in {"timestamp", "timestamp_s", "time", "time_s"} or "time" in lower:
            role = "time"
        elif lower in {"label", "gesture", "class", "target", "state", "event"}:
            role = "label"
        elif numeric_coverage >= 0.9:
            role = "channel"
        elif non_empty and unique_values <= 32:
            role = "label"
        else:
            role = "metadata"
        item = {
            "name": clean,
            "raw_name": raw_name,
            "role": role,
            "numeric_coverage": round(float(numeric_coverage), 3),
            "unique_values": int(unique_values),
            "sample_values": non_empty[:4],
        }
        if role == "channel":
            item["channel_index"] = channel_idx
            channel_idx += 1
        columns.append(item)
    channels = [item["name"] for item in columns if item.get("role") == "channel"][: config.n_channels]
    label = next((item["name"] for item in columns if item.get("role") == "label"), "")
    time_col = next((item["name"] for item in columns if item.get("role") == "time"), "")
    return {
        "columns": columns,
        "detected": {
            "channels": channels,
            "label": label,
            "time": time_col,
        },
        "sample_rows": len(rows),
    }


def _dependency_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except Exception:
        return False


def _sample_rate_from_times(times: np.ndarray, fallback: float) -> float:
    arr = np.asarray(times, dtype=np.float64)
    if arr.size >= 3:
        diffs = np.diff(arr)
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        if diffs.size:
            return float(1.0 / np.median(diffs))
    return float(fallback or config.sample_rate)


def _labels_from_intervals(sample_count: int, sample_rate: float, intervals: List[dict]) -> Optional[List[str]]:
    labels = [""] * max(0, int(sample_count))
    for item in intervals:
        label = str(item.get("label") or item.get("description") or "").strip()
        if not label:
            continue
        start = max(0, int(round(float(item.get("start_s") or 0.0) * float(sample_rate or config.sample_rate))))
        end_raw = item.get("end_s") if item.get("end_s") is not None else item.get("stop_s")
        end = int(round(float(end_raw if end_raw is not None else 0.0) * float(sample_rate or config.sample_rate)))
        if end <= start:
            end = min(len(labels), start + max(1, int(round(float(sample_rate or config.sample_rate)))))
        end = min(len(labels), max(start, end))
        for idx in range(start, end):
            labels[idx] = label
    return labels if any(labels) else None


def _labels_from_marker_times(sample_count: int, sample_rate: float, markers: List[dict]) -> Optional[List[str]]:
    ordered = sorted(
        [item for item in markers if str(item.get("label") or "").strip()],
        key=lambda item: float(item.get("time_s") or 0.0),
    )
    if not ordered:
        return None
    intervals: List[dict] = []
    duration_s = float(sample_count) / max(float(sample_rate or config.sample_rate), 1.0)
    for idx, item in enumerate(ordered):
        start = max(0.0, float(item.get("time_s") or 0.0))
        next_start = float(ordered[idx + 1].get("time_s") or duration_s) if idx + 1 < len(ordered) else duration_s
        intervals.append({"label": str(item.get("label") or "").strip(), "start_s": start, "end_s": max(start, min(duration_s, next_start))})
    return _labels_from_intervals(sample_count, sample_rate, intervals)


def _coerce_signal_array(data: Any) -> np.ndarray:
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
    if arr.ndim != 2 or not arr.size:
        raise ValueError("Decoded signal must be a non-empty 2D array.")
    if arr.shape[0] <= config.n_channels and arr.shape[0] <= arr.shape[1]:
        return np.ascontiguousarray(arr[: config.n_channels, :])
    if arr.shape[1] <= config.n_channels and arr.shape[0] > arr.shape[1]:
        return np.ascontiguousarray(arr.T[: config.n_channels, :])
    if arr.shape[0] > config.n_channels and arr.shape[0] < arr.shape[1]:
        return np.ascontiguousarray(arr[: config.n_channels, :])
    return np.ascontiguousarray(arr.T[: config.n_channels, :])


def _read_pipeline_signal_table(
    fieldnames: List[str],
    records: List[Dict[str, Any]],
    schema_mapping: Optional[Dict[str, Any]] = None,
) -> tuple[np.ndarray, Optional[List[str]], float]:
    raw_names = [str(name or "").strip() for name in fieldnames]
    clean_by_raw = {name: name.lstrip("\ufeff") for name in raw_names}
    lower = {clean.lower(): raw for raw, clean in clean_by_raw.items()}
    normalized_mapping = _normalize_schema_mapping(schema_mapping)
    clean_to_raw = {clean: raw for raw, clean in clean_by_raw.items()}
    channel_names = []
    for name in normalized_mapping.get("channels") or []:
        raw = clean_to_raw.get(name) or lower.get(str(name).lower())
        if raw and raw not in channel_names:
            channel_names.append(raw)
    if not channel_names:
        for idx in range(config.n_channels):
            for candidate in (f"ch{idx + 1}", f"channel{idx + 1}", f"c{idx + 1}", f"emg{idx + 1}", f"eeg{idx + 1}"):
                if candidate in lower:
                    channel_names.append(lower[candidate])
                    break
    if not channel_names:
        excluded = {"timestamp", "timestamp_s", "time", "time_s", "label", "gesture", "class", "target", "state", "event"}
        channel_names = [raw for raw, clean in clean_by_raw.items() if clean.lower() not in excluded][: config.n_channels]
    label_name = clean_to_raw.get(normalized_mapping.get("label") or "") or lower.get(str(normalized_mapping.get("label") or "").lower())
    time_name = clean_to_raw.get(normalized_mapping.get("time") or "") or lower.get(str(normalized_mapping.get("time") or "").lower())
    if not label_name:
        label_name = next((lower[key] for key in ("label", "gesture", "class", "target", "state", "event") if key in lower), None)
    if not time_name:
        time_name = next((lower[key] for key in ("timestamp_s", "time_s", "timestamp", "time") if key in lower), None)
    rows: List[List[float]] = []
    labels: List[str] = []
    times: List[float] = []
    for row in records:
        try:
            rows.append([float(row.get(name) or 0.0) for name in channel_names])
        except Exception:
            continue
        if label_name:
            labels.append(str(row.get(label_name) or "").strip())
        if time_name:
            try:
                times.append(float(row.get(time_name) or 0.0))
            except Exception:
                pass
    if not rows:
        raise ValueError("Table did not contain numeric channel rows.")
    signal = np.asarray(rows, dtype=np.float32).T
    sample_rate = _sample_rate_from_times(np.asarray(times, dtype=np.float64), config.sample_rate) if len(times) >= 3 else float(config.sample_rate)
    return signal, (labels if label_name and len(labels) == len(rows) else None), sample_rate


def _read_pipeline_mat_file(path: Path) -> tuple[np.ndarray, Optional[List[str]], float, dict]:
    try:
        from scipy.io import loadmat
    except Exception as exc:
        raise RuntimeError("Install scipy to decode MATLAB .mat files.") from exc
    payload = loadmat(path.as_posix(), squeeze_me=True, struct_as_record=False)
    signal_key = next((key for key in ("signal", "data", "X", "samples", "emg", "eeg") if key in payload), "")
    if not signal_key:
        candidates = [(key, np.asarray(value)) for key, value in payload.items() if not key.startswith("__") and np.asarray(value).ndim in (1, 2) and np.issubdtype(np.asarray(value).dtype, np.number)]
        candidates.sort(key=lambda item: int(item[1].size), reverse=True)
        signal_key = candidates[0][0] if candidates else ""
    if not signal_key:
        raise ValueError("No numeric signal/data array found in MAT file.")
    signal = _coerce_signal_array(payload[signal_key])
    label_key = next((key for key in ("labels", "label", "y", "target", "classes", "events") if key in payload), "")
    labels = None
    if label_key:
        raw = np.asarray(payload[label_key]).reshape(-1).tolist()
        labels = [str(item.decode("utf-8") if isinstance(item, bytes) else item).strip() for item in raw]
        if len(labels) != signal.shape[1]:
            labels = None
    sample_rate = float(payload.get("sample_rate", payload.get("fs", payload.get("srate", config.sample_rate))) or config.sample_rate)
    return signal, labels, sample_rate, {"importer": "scipy_mat", "label_source": label_key, "notes": [f"signal key: {signal_key}"]}


def _read_pipeline_hdf5_file(path: Path) -> tuple[np.ndarray, Optional[List[str]], float, dict]:
    try:
        import h5py
    except Exception as exc:
        raise RuntimeError("Install h5py to decode HDF5 files.") from exc
    arrays: Dict[str, Any] = {}
    with h5py.File(path.as_posix(), "r") as fh:
        def visit(name: str, obj: Any) -> None:
            if hasattr(obj, "shape") and getattr(obj, "shape", None):
                arrays[name] = obj[()]
        fh.visititems(visit)
    signal_key = next((key for key in arrays if key.split("/")[-1].lower() in {"signal", "data", "x", "samples", "emg", "eeg"}), "")
    if not signal_key:
        candidates = [(key, np.asarray(value)) for key, value in arrays.items() if np.asarray(value).ndim in (1, 2) and np.issubdtype(np.asarray(value).dtype, np.number)]
        candidates.sort(key=lambda item: int(item[1].size), reverse=True)
        signal_key = candidates[0][0] if candidates else ""
    if not signal_key:
        raise ValueError("No numeric signal/data dataset found in HDF5 file.")
    signal = _coerce_signal_array(arrays[signal_key])
    label_key = next((key for key in arrays if key.split("/")[-1].lower() in {"labels", "label", "y", "target", "classes"}), "")
    labels = None
    if label_key:
        raw = np.asarray(arrays[label_key]).reshape(-1).tolist()
        labels = [str(item.decode("utf-8") if isinstance(item, bytes) else item).strip() for item in raw]
        if len(labels) != signal.shape[1]:
            labels = None
    sample_key = next((key for key in arrays if key.split("/")[-1].lower() in {"sample_rate", "fs", "srate"}), "")
    sample_rate = float(np.asarray(arrays[sample_key]).reshape(-1)[0]) if sample_key else float(config.sample_rate)
    return signal, labels, sample_rate, {"importer": "h5py", "label_source": label_key, "notes": [f"signal key: {signal_key}"]}


def _read_pipeline_parquet_file(path: Path, schema_mapping: Optional[Dict[str, Any]] = None) -> tuple[np.ndarray, Optional[List[str]], float, dict]:
    try:
        import pandas as pd
    except Exception as exc:
        raise RuntimeError("Install pandas plus pyarrow or fastparquet to decode Parquet files.") from exc
    df = pd.read_parquet(path.as_posix())
    records = df.head(1_000_000).to_dict(orient="records")
    signal, labels, sample_rate = _read_pipeline_signal_table([str(col) for col in df.columns], records, schema_mapping=schema_mapping)
    return signal, labels, sample_rate, {"importer": "pandas_parquet", "label_source": "table", "notes": [f"{len(records)} rows decoded"]}


def _read_pipeline_edf_bdf_file(path: Path) -> tuple[np.ndarray, Optional[List[str]], float, dict]:
    try:
        import mne
    except Exception as exc:
        raise RuntimeError("Install mne to decode EDF/BDF files.") from exc
    raw = mne.io.read_raw_edf(path.as_posix(), preload=False, verbose=False)
    sample_rate = float(raw.info.get("sfreq") or config.sample_rate)
    signal = _coerce_signal_array(raw.get_data(picks="data"))
    intervals = []
    for ann in raw.annotations:
        desc = str(ann["description"]).strip()
        if desc:
            intervals.append({"label": desc, "start_s": float(ann["onset"]), "end_s": float(ann["onset"]) + float(ann["duration"] or 0.0)})
    labels = _labels_from_intervals(signal.shape[1], sample_rate, intervals)
    return signal, labels, sample_rate, {"importer": "mne_edf", "label_source": "annotations", "notes": [f"{len(intervals)} annotations"]}


def _read_pipeline_wav_file(path: Path) -> tuple[np.ndarray, Optional[List[str]], float, dict]:
    try:
        from scipy.io import wavfile
    except Exception as exc:
        raise RuntimeError("Install scipy to decode WAV files.") from exc
    sample_rate, data = wavfile.read(path.as_posix())
    signal = _coerce_signal_array(np.asarray(data, dtype=np.float32).T if np.asarray(data).ndim == 2 else data)
    return signal, None, float(sample_rate), {"importer": "scipy_wav", "label_source": "", "notes": ["WAV files need an external label track for training."]}


def _read_pipeline_xdf_file(path: Path) -> tuple[np.ndarray, Optional[List[str]], float, dict]:
    try:
        import pyxdf
    except Exception as exc:
        raise RuntimeError("Install pyxdf to decode XDF files.") from exc
    streams, _ = pyxdf.load_xdf(path.as_posix())
    signal_stream = None
    marker_streams = []
    for stream in streams:
        arr = np.asarray(stream.get("time_series"))
        if arr.size and arr.ndim in (1, 2) and np.issubdtype(arr.dtype, np.number) and signal_stream is None:
            signal_stream = stream
        elif arr.size:
            marker_streams.append(stream)
    if signal_stream is None:
        raise ValueError("No numeric signal stream found in XDF file.")
    raw = np.asarray(signal_stream.get("time_series"), dtype=np.float32)
    if raw.ndim == 1:
        raw = raw[:, np.newaxis]
    times = np.asarray(signal_stream.get("time_stamps", []), dtype=np.float64)
    base = float(times[0]) if times.size else 0.0
    rel_times = times - base if times.size else np.arange(raw.shape[0], dtype=np.float64) / max(config.sample_rate, 1)
    sample_rate = _sample_rate_from_times(rel_times, config.sample_rate)
    signal = _coerce_signal_array(raw.T)
    markers: List[dict] = []
    for stream in marker_streams:
        samples = np.asarray(stream.get("time_series"), dtype=object)
        stamps = np.asarray(stream.get("time_stamps", []), dtype=np.float64)
        if samples.ndim == 0:
            samples = samples.reshape(1, 1)
        elif samples.ndim == 1:
            samples = samples[:, np.newaxis]
        for idx in range(min(samples.shape[0], stamps.size)):
            text = " ".join(str(item) for item in samples[idx].tolist() if str(item).strip()).strip()
            if text:
                markers.append({"label": text, "time_s": float(stamps[idx] - base)})
    labels = _labels_from_marker_times(signal.shape[1], sample_rate, markers)
    return signal, labels, sample_rate, {"importer": "pyxdf", "label_source": "marker_streams", "notes": [f"{len(markers)} markers"]}


def _read_pipeline_generic_signal_file(path: Path, schema_mapping: Optional[Dict[str, Any]] = None) -> tuple[np.ndarray, Optional[List[str]], float, dict]:
    suffix = path.suffix.lower()
    if suffix in {".edf", ".bdf"}:
        return _read_pipeline_edf_bdf_file(path)
    if suffix == ".xdf":
        return _read_pipeline_xdf_file(path)
    if suffix == ".mat":
        return _read_pipeline_mat_file(path)
    if suffix in {".h5", ".hdf5"}:
        return _read_pipeline_hdf5_file(path)
    if suffix == ".parquet":
        return _read_pipeline_parquet_file(path, schema_mapping=schema_mapping)
    if suffix == ".wav":
        return _read_pipeline_wav_file(path)
    raise ValueError(f"No generic signal importer for {suffix or path.name}.")


def _inspect_dataset_file(path: Path, schema_mapping: Optional[Dict[str, Any]] = None) -> dict:
    suffix = path.suffix.lower()
    info: Dict[str, Any] = {
        "path": path.as_posix(),
        "name": path.name,
        "extension": suffix,
        "bytes": int(path.stat().st_size) if path.exists() else 0,
        "supported": suffix in {".csv", ".json", ".npy", ".npz", ".xdf", ".edf", ".bdf", ".mat", ".h5", ".hdf5", ".parquet", ".wav", ".txt"},
        "notes": [],
    }
    if suffix in {".edf", ".bdf", ".xdf", ".mat", ".h5", ".hdf5", ".parquet", ".wav"}:
        info["needs_importer"] = True
        try:
            signal, labels, sample_rate, meta = _read_pipeline_generic_signal_file(path, schema_mapping=schema_mapping)
            info["channel_count"] = int(signal.shape[0])
            info["sample_count"] = int(signal.shape[1])
            info["sample_rate_hz"] = round(float(sample_rate or config.sample_rate), 3)
            info["importer"] = meta.get("importer") or suffix.lstrip(".")
            info["label_source"] = meta.get("label_source") or ""
            info["has_labels"] = bool(labels)
            info["notes"].extend(list(meta.get("notes") or [])[:3])
            if labels:
                estimated, label_counts, warnings = _estimate_labeled_window_counts(
                    labels,
                    int(signal.shape[1]),
                    sample_rate=float(sample_rate or config.sample_rate),
                )
                info["trainable"] = estimated > 0 and len(label_counts) >= 2
                info["estimated_windows"] = int(estimated)
                info["estimated_label_windows"] = label_counts
                if warnings:
                    info["notes"].extend(warnings[:2])
            else:
                info["trainable"] = False
                info["notes"].append("Signal decoded; add labels, annotations, or marker streams for supervised training.")
        except Exception as exc:
            info["trainable"] = False
            info["notes"].append(f"Importer preview failed: {type(exc).__name__}: {exc}")
        return info
    if suffix in {".csv", ".txt"}:
        with path.open("r", encoding="utf-8", errors="ignore", newline="") as fh:
            sample = "".join([fh.readline() for _ in range(8)])
        try:
            dialect = csv.Sniffer().sniff(sample) if sample.strip() else csv.excel
        except csv.Error:
            dialect = csv.excel
        rows = [row for row in csv.reader(sample.splitlines(), dialect=dialect) if row]
        info["preview_rows"] = rows[:5]
        info["columns"] = max((len(row) for row in rows), default=0)
        info["has_timestamp_candidate"] = bool(rows and any("time" in str(cell).lower() for cell in rows[0]))
        try:
            info["schema"] = _csv_schema_preview(path)
            signal, labels, sample_rate = _read_pipeline_signal_csv(path, schema_mapping=schema_mapping)
            info["channel_count"] = int(signal.shape[0])
            info["sample_count"] = int(signal.shape[1])
            info["sample_rate_hz"] = round(float(sample_rate or config.sample_rate), 3)
            info["has_labels"] = bool(labels)
            if labels:
                estimated, label_counts, warnings = _estimate_labeled_window_counts(
                    labels,
                    int(signal.shape[1]),
                    sample_rate=float(sample_rate or config.sample_rate),
                )
                info["trainable"] = estimated > 0 and len(label_counts) >= 2
                info["estimated_windows"] = int(estimated)
                info["estimated_label_windows"] = label_counts
                if warnings:
                    info["notes"].extend(warnings[:2])
            else:
                info["trainable"] = False
                info["notes"].append("No label/gesture/class column found for supervised training.")
        except Exception as exc:
            info["trainable"] = False
            info["notes"].append(f"Training preview failed: {type(exc).__name__}: {exc}")
    elif suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
        if isinstance(payload, dict):
            info["keys"] = list(payload.keys())[:24]
            info["record_count"] = len(payload.get("records") or payload.get("samples") or payload.get("channels") or [])
        elif isinstance(payload, list):
            info["record_count"] = len(payload)
            info["first_record_type"] = type(payload[0]).__name__ if payload else ""
    elif suffix in {".npy", ".npz"}:
        loaded = np.load(path, allow_pickle=False)
        if isinstance(loaded, np.lib.npyio.NpzFile):
            info["arrays"] = {name: list(loaded[name].shape) for name in loaded.files[:12]}
            signal_name = next((name for name in ("windows", "X", "signal", "data", "samples") if name in loaded.files), "")
            label_name = next((name for name in ("labels", "y", "target", "classes") if name in loaded.files), "")
            if signal_name and label_name:
                data = np.asarray(loaded[signal_name])
                labels_arr = loaded[label_name]
                labels = [str(v.decode("utf-8") if isinstance(v, bytes) else v).strip() for v in labels_arr.tolist()]
                if data.ndim == 3 and data.shape[0] == len(labels):
                    label_counts: Dict[str, int] = {}
                    for label in labels:
                        if label:
                            label_counts[label] = label_counts.get(label, 0) + 1
                    info["trainable"] = len(label_counts) >= 2
                    info["estimated_windows"] = int(sum(label_counts.values()))
                    info["estimated_label_windows"] = label_counts
                elif data.ndim == 2 and len(labels) == data.shape[-1]:
                    estimated, label_counts, warnings = _estimate_labeled_window_counts(
                        labels,
                        int(data.shape[-1]),
                        sample_rate=float(config.sample_rate),
                    )
                    info["trainable"] = estimated > 0 and len(label_counts) >= 2
                    info["estimated_windows"] = int(estimated)
                    info["estimated_label_windows"] = label_counts
                    if warnings:
                        info["notes"].extend(warnings[:2])
                else:
                    info["trainable"] = False
                    info["notes"].append(f"Signal/label arrays are present but shape alignment needs review.")
            else:
                info["trainable"] = False
                info["notes"].append("NPZ needs signal/data and labels/y arrays for direct training.")
        else:
            info["shape"] = list(loaded.shape)
            info["dtype"] = str(loaded.dtype)
    return info


def _pipeline_dataset_inspect(req: PipelineDatasetInspectRequest) -> dict:
    raw = str(req.path or (req.context or {}).get("dataset_path") or "").strip()
    root = Path(raw) if raw else Path(config.data_dir)
    if not root.exists():
        raise ValueError(f"Dataset path does not exist: {root}")
    files = []
    if root.is_dir():
        candidates = []
        for suffix in ("*.csv", "*.json", "*.npy", "*.npz", "*.xdf", "*.edf", "*.bdf", "*.mat", "*.h5", "*.hdf5", "*.parquet", "*.wav", "*.txt"):
            candidates.extend(root.rglob(suffix))
        for path in sorted(candidates, key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)[:24]:
            try:
                files.append(_inspect_dataset_file(path, schema_mapping=req.schema_mapping))
            except Exception as exc:
                files.append({"path": path.as_posix(), "name": path.name, "error": f"{type(exc).__name__}: {exc}"})
    else:
        files.append(_inspect_dataset_file(root, schema_mapping=req.schema_mapping))
    supported = [item for item in files if item.get("supported")]
    readiness = _score_import_readiness(files)
    return {
        "ok": True,
        "dataset": {
            "path": root.as_posix(),
            "mode": "directory" if root.is_dir() else "file",
            "file_count": len(files),
            "supported_count": len(supported),
            "readiness": readiness,
            "files": files,
            "pipeline": [
                "Detect stream table and timestamp column.",
                "Infer channel count, units, and sample rate.",
                "Map annotation/event columns into label spans.",
                "Window, QA, weak-label, then train baseline.",
            ],
        },
    }


def _coerce_pipeline_window(window: np.ndarray) -> np.ndarray:
    arr = np.asarray(window, dtype=np.float32)
    if arr.ndim != 2 or not arr.size:
        raise ValueError("Training window must be a 2D channel x sample array.")
    if arr.shape[0] == config.n_channels:
        return np.ascontiguousarray(arr)
    if arr.shape[1] == config.n_channels:
        return np.ascontiguousarray(arr.T)
    if arr.shape[0] > config.n_channels:
        return np.ascontiguousarray(arr[: config.n_channels, :])
    padded = np.zeros((config.n_channels, arr.shape[1]), dtype=np.float32)
    padded[: arr.shape[0], :] = arr
    return padded


def _pipeline_store_training_window(
    window: np.ndarray,
    label: str,
    *,
    timestamp: Optional[float] = None,
    qa_score: float = 100.0,
    source: str = "dataset",
    run_dir: str = "",
    metadata: Optional[Dict[str, Any]] = None,
) -> dict:
    clean_label = str(label or "").strip()
    if not clean_label:
        raise ValueError("Training window label is required.")
    arr = _coerce_pipeline_window(window)
    step_size = 0.02
    q = np.clip(np.rint(arr / step_size), -32768, 32767).astype(np.int16)
    encoded = zlib.compress(q.tobytes(order="C"), 3)
    payload_path = ""
    if run_dir:
        windows_dir = Path(run_dir) / "windows"
        windows_dir.mkdir(parents=True, exist_ok=True)
        next_idx = len(app_state.pipeline_acquisition_windows) + 1
        payload_path = (windows_dir / f"{next_idx:06d}_{_safe_stem(clean_label, 'label')}.qz").as_posix()
        Path(payload_path).write_bytes(encoded)
    record = {
        "label": clean_label,
        "timestamp": float(timestamp if timestamp is not None else time.time()),
        "qa_score": round(float(qa_score), 1),
        "shape": list(arr.shape),
        "dtype": "int16",
        "quantization_step": step_size,
        "bytes": len(encoded),
        "compression": "zlib",
        "source": source,
        "metadata": dict(metadata or {}),
        "payload_path": payload_path,
        "payload_b64": "" if payload_path else base64.b64encode(encoded).decode("ascii"),
        "rms": [round(float(v), 5) for v in np.sqrt(np.mean(np.square(arr.astype(np.float64)), axis=1))[: config.n_channels]],
    }
    app_state.pipeline_acquisition_windows.append(record)
    return record


def _read_pipeline_signal_csv(path: Path, schema_mapping: Optional[Dict[str, Any]] = None) -> tuple[np.ndarray, Optional[List[str]], float]:
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            raise ValueError("CSV has no header row.")
        raw_names = [str(name or "").strip() for name in reader.fieldnames]
        clean_by_raw = {name: name.lstrip("\ufeff") for name in raw_names}
        lower = {clean.lower(): raw for raw, clean in clean_by_raw.items()}
        normalized_mapping = _normalize_schema_mapping(schema_mapping)
        clean_to_raw = {clean: raw for raw, clean in clean_by_raw.items()}
        channel_names = []
        for name in normalized_mapping.get("channels") or []:
            raw = clean_to_raw.get(name) or lower.get(str(name).lower())
            if raw and raw not in channel_names:
                channel_names.append(raw)
        if not channel_names:
            for idx in range(config.n_channels):
                for candidate in (f"ch{idx + 1}", f"channel{idx + 1}", f"c{idx + 1}", f"emg{idx + 1}"):
                    if candidate in lower:
                        channel_names.append(lower[candidate])
                        break
        if not channel_names:
            numeric_names = [
                raw for raw, clean in clean_by_raw.items()
                if clean.lower() not in {"timestamp", "timestamp_s", "time", "time_s", "label", "gesture", "class", "target", "state"}
            ]
            channel_names = numeric_names[: config.n_channels]
        label_name = clean_to_raw.get(normalized_mapping.get("label") or "") or lower.get(str(normalized_mapping.get("label") or "").lower())
        time_name = clean_to_raw.get(normalized_mapping.get("time") or "") or lower.get(str(normalized_mapping.get("time") or "").lower())
        if not label_name:
            label_name = next((lower[key] for key in ("label", "gesture", "class", "target", "state", "event") if key in lower), None)
        if not time_name:
            time_name = next((lower[key] for key in ("timestamp_s", "time_s", "timestamp", "time") if key in lower), None)
        rows: List[List[float]] = []
        labels: List[str] = []
        times: List[float] = []
        for row in reader:
            try:
                rows.append([float(row.get(name) or 0.0) for name in channel_names])
            except Exception:
                continue
            if label_name:
                labels.append(str(row.get(label_name) or "").strip())
            if time_name:
                try:
                    times.append(float(row.get(time_name) or 0.0))
                except Exception:
                    pass
    if not rows:
        raise ValueError("CSV did not contain numeric channel rows.")
    signal = np.asarray(rows, dtype=np.float32).T
    sample_rate = float(config.sample_rate)
    if len(times) >= 3:
        diffs = np.diff(np.asarray(times, dtype=np.float64))
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        if diffs.size:
            sample_rate = float(1.0 / np.median(diffs))
    return signal, (labels if label_name and len(labels) == len(rows) else None), sample_rate


def _pipeline_window_records_from_signal(
    signal: np.ndarray,
    labels: List[str],
    *,
    sample_rate: float,
    source: str,
    metadata: Optional[Dict[str, Any]] = None,
    max_windows: int,
    max_windows_per_label: int,
) -> tuple[int, Dict[str, int], List[str]]:
    arr = _coerce_pipeline_window(signal)
    segment_samples = max(8, int(round((float(((metadata or {}).get("window_size_ms") or config.window_size_ms)) / 1000.0) * float(sample_rate or config.sample_rate))))
    stride_samples = max(1, int(round((float(((metadata or {}).get("window_step_ms") or config.window_increment_ms)) / 1000.0) * float(sample_rate or config.sample_rate))))
    if arr.shape[1] < segment_samples:
        return 0, {}, [f"{source}: not enough samples for one window."]
    per_label: Dict[str, int] = {}
    warnings: List[str] = []
    added = 0
    for start in range(0, arr.shape[1] - segment_samples + 1, stride_samples):
        end = start + segment_samples
        label_slice = [str(v or "").strip() for v in labels[start:end] if str(v or "").strip()]
        if not label_slice:
            continue
        counts: Dict[str, int] = {}
        for label in label_slice:
            counts[label] = counts.get(label, 0) + 1
        label, vote_count = max(counts.items(), key=lambda item: item[1])
        if vote_count / max(len(label_slice), 1) < 0.75:
            continue
        if per_label.get(label, 0) >= max_windows_per_label:
            continue
        _pipeline_store_training_window(
            arr[:, start:end],
            label,
            qa_score=100.0,
            source=source,
            run_dir=str((metadata or {}).get("run_dir") or ""),
            metadata={
                **dict(metadata or {}),
                "window_start_sample": int(start),
                "window_end_sample": int(end),
                "sample_rate_hz": round(float(sample_rate or config.sample_rate), 3),
            },
        )
        per_label[label] = per_label.get(label, 0) + 1
        added += 1
        if added >= max_windows:
            break
    if not added:
        warnings.append(f"{source}: no labeled windows found.")
    return added, per_label, warnings


def _pipeline_ingest_labeled_csv(path: Path, run_dir: str, req: PipelineDatasetIngestRequest) -> tuple[int, Dict[str, int], List[str]]:
    signal, labels, sample_rate = _read_pipeline_signal_csv(path, schema_mapping=req.schema_mapping)
    if not labels:
        return 0, {}, [f"{path.name}: no label/gesture/class column."]
    return _pipeline_window_records_from_signal(
        signal,
        labels,
        sample_rate=sample_rate,
        source="dataset_csv",
        metadata={"path": path.as_posix(), "run_dir": run_dir},
        max_windows=int(req.max_windows),
        max_windows_per_label=int(req.max_windows_per_label),
    )


def _pipeline_ingest_npz(path: Path, run_dir: str, req: PipelineDatasetIngestRequest) -> tuple[int, Dict[str, int], List[str]]:
    loaded = np.load(path, allow_pickle=False)
    if not isinstance(loaded, np.lib.npyio.NpzFile):
        return 0, {}, [f"{path.name}: use .npz with signal/data/X and labels/y arrays."]
    signal_name = next((name for name in ("windows", "X", "signal", "data", "samples") if name in loaded.files), "")
    label_name = next((name for name in ("labels", "y", "target", "classes") if name in loaded.files), "")
    if not signal_name or not label_name:
        return 0, {}, [f"{path.name}: missing signal/data and labels arrays."]
    data = np.asarray(loaded[signal_name])
    labels_arr = loaded[label_name]
    labels = [str(v.decode("utf-8") if isinstance(v, bytes) else v).strip() for v in labels_arr.tolist()]
    per_label: Dict[str, int] = {}
    warnings: List[str] = []
    added = 0
    if data.ndim == 3:
        if data.shape[0] != len(labels):
            return 0, {}, [f"{path.name}: windows and labels length mismatch."]
        for idx, label in enumerate(labels):
            if not label or per_label.get(label, 0) >= int(req.max_windows_per_label):
                continue
            _pipeline_store_training_window(
                data[idx],
                label,
                source="dataset_npz",
                run_dir=run_dir,
                metadata={"path": path.as_posix(), "window_index": idx},
            )
            per_label[label] = per_label.get(label, 0) + 1
            added += 1
            if added >= int(req.max_windows):
                break
    elif data.ndim == 2 and len(labels) == data.shape[-1]:
        added, per_label, warnings = _pipeline_window_records_from_signal(
            data,
            labels,
            sample_rate=float(config.sample_rate),
            source="dataset_npz",
            metadata={"path": path.as_posix(), "run_dir": run_dir},
            max_windows=int(req.max_windows),
            max_windows_per_label=int(req.max_windows_per_label),
        )
    else:
        warnings.append(f"{path.name}: unsupported NPZ array shape {list(data.shape)}.")
    return added, per_label, warnings


def _pipeline_ingest_generic_signal_file(path: Path, run_dir: str, req: PipelineDatasetIngestRequest) -> tuple[int, Dict[str, int], List[str]]:
    signal, labels, sample_rate, meta = _read_pipeline_generic_signal_file(path, schema_mapping=req.schema_mapping)
    if not labels:
        importer = str(meta.get("importer") or path.suffix.lower().lstrip(".") or "generic")
        return 0, {}, [f"{path.name}: {importer} decoded {signal.shape[0]} ch x {signal.shape[1]} samples, but no labels/annotations were found."]
    return _pipeline_window_records_from_signal(
        signal,
        labels,
        sample_rate=sample_rate,
        source=f"dataset_{str(meta.get('importer') or path.suffix.lower().lstrip('.'))}",
        metadata={
            "path": path.as_posix(),
            "run_dir": run_dir,
            "importer": meta.get("importer") or "",
            "label_source": meta.get("label_source") or "",
        },
        max_windows=int(req.max_windows),
        max_windows_per_label=int(req.max_windows_per_label),
    )


def _pipeline_ingest_session_dir(path: Path, run_dir: str, req: PipelineDatasetIngestRequest) -> tuple[int, Dict[str, int], List[str]]:
    meta_path = path / "meta.json"
    raw_path = path / "signal_raw.csv"
    if not raw_path.exists():
        raw_path = path / "emg_raw.csv"
    if not meta_path.exists() or not raw_path.exists():
        return 0, {}, [f"{path.name}: not a KYMA session directory."]
    meta = json.loads(meta_path.read_text(encoding="utf-8", errors="ignore"))
    profile_key = str(meta.get("signal_profile") or (meta.get("config") or {}).get("signal_profile") or "").lower()
    if profile_key and profile_key != config.signal_profile_name:
        return 0, {}, [f"{path.name}: profile {profile_key} does not match active {config.signal_profile_name}."]
    intervals = app_state.datasets._label_intervals(meta, tuple(config.class_labels))
    if not intervals:
        return 0, {}, [f"{path.name}: no compatible label intervals."]
    signal, _, sample_rate = _read_pipeline_signal_csv(raw_path)
    added_total = 0
    per_label_total: Dict[str, int] = {}
    warnings: List[str] = []
    labels = [""] * signal.shape[1]
    for interval in intervals:
        label = str(interval.get("label") or "").strip()
        start = max(0, int(round(float(interval.get("start_s") or 0.0) * sample_rate)))
        end = min(signal.shape[1], int(round(float(interval.get("end_s") or 0.0) * sample_rate)))
        for idx in range(start, end):
            labels[idx] = label
    added, per_label, local_warnings = _pipeline_window_records_from_signal(
        signal,
        labels,
        sample_rate=sample_rate,
        source="kyma_session",
        metadata={"path": path.as_posix(), "session_id": path.name, "run_dir": run_dir},
        max_windows=max(0, int(req.max_windows) - added_total),
        max_windows_per_label=int(req.max_windows_per_label),
    )
    added_total += added
    for label, count in per_label.items():
        per_label_total[label] = per_label_total.get(label, 0) + count
    warnings.extend(local_warnings)
    return added_total, per_label_total, warnings


def _pipeline_dataset_ingest(req: PipelineDatasetIngestRequest) -> dict:
    raw = str(req.path or (req.context or {}).get("dataset_path") or "").strip()
    root = Path(raw) if raw else Path(config.data_dir)
    if not root.exists():
        raise ValueError(f"Dataset path does not exist: {root}")
    stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    run_dir = Path(config.data_dir) / "pipeline_acquisitions" / f"{config.signal_profile_name}_dataset_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    if not req.append:
        app_state.pipeline_acquisition_windows = []
    app_state.pipeline_acquisition_run = {
        "id": f"{config.signal_profile_name}_dataset_{stamp}",
        "running": False,
        "profile": config.signal_profile_name,
        "started_at": time.time(),
        "completed_at": time.time(),
        "source": "dataset_ingest",
        "run_dir": run_dir.as_posix(),
    }
    candidates: List[Path] = []
    if root.is_dir():
        if (root / "meta.json").exists():
            candidates.append(root)
        else:
            candidates.extend([path for path in root.iterdir() if path.is_dir() and (path / "meta.json").exists()])
            default_root = Path(config.data_dir).resolve()
            search_roots = [root]
            try:
                if root.resolve() == default_root:
                    search_roots = [path for path in [root / "imports"] if path.exists()]
            except Exception:
                search_roots = [root]
            for search_root in search_roots:
                for suffix in ("*.csv", "*.txt", "*.npz", "*.xdf", "*.edf", "*.bdf", "*.mat", "*.h5", "*.hdf5", "*.parquet", "*.wav"):
                    candidates.extend(search_root.rglob(suffix))
    else:
        candidates.append(root)
    seen: Set[str] = set()
    files = []
    warnings: List[str] = []
    per_label_total: Dict[str, int] = {}
    added_total = 0
    for path in candidates:
        key = path.resolve().as_posix()
        if key in seen or added_total >= int(req.max_windows):
            continue
        seen.add(key)
        try:
            if path.is_dir():
                added, per_label, local_warnings = _pipeline_ingest_session_dir(path, run_dir.as_posix(), req)
            elif path.suffix.lower() in {".csv", ".txt"}:
                added, per_label, local_warnings = _pipeline_ingest_labeled_csv(path, run_dir.as_posix(), req)
            elif path.suffix.lower() == ".npz":
                added, per_label, local_warnings = _pipeline_ingest_npz(path, run_dir.as_posix(), req)
            elif path.suffix.lower() in {".xdf", ".edf", ".bdf", ".mat", ".h5", ".hdf5", ".parquet", ".wav"}:
                added, per_label, local_warnings = _pipeline_ingest_generic_signal_file(path, run_dir.as_posix(), req)
            else:
                added, per_label, local_warnings = 0, {}, [f"{path.name}: unsupported for direct ingest."]
        except Exception as exc:
            added, per_label, local_warnings = 0, {}, [f"{path.name}: {type(exc).__name__}: {exc}"]
        files.append({"path": path.as_posix(), "windows": int(added), "per_label": per_label})
        added_total += int(added)
        for label, count in per_label.items():
            per_label_total[label] = per_label_total.get(label, 0) + int(count)
        warnings.extend(local_warnings)
    _pipeline_write_acquisition_manifest()
    ingest = {
        "path": root.as_posix(),
        "run_dir": run_dir.as_posix(),
        "added_windows": int(added_total),
        "total_buffer_windows": len(app_state.pipeline_acquisition_windows),
        "per_label": per_label_total,
        "files": files,
        "warnings": warnings[:12],
        "ready": len(per_label_total) >= 2 and added_total >= 4,
    }
    out = {"ok": True, "ingest": ingest, "run": _pipeline_acquisition_status().get("run")}
    if req.train_after and ingest["ready"]:
        out["training"] = _pipeline_train_acquisition(req)
    return out


def _compressed_size_zlib(data: bytes, level: int = 3) -> tuple[int, float]:
    t0 = time.perf_counter()
    encoded = zlib.compress(data, level)
    decoded = zlib.decompress(encoded)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    if decoded != data:
        raise RuntimeError("zlib round trip mismatch")
    return len(encoded), elapsed_ms


def _pipeline_throughput_benchmark(req: PipelineThroughputRequest) -> dict:
    window = app_state.last_vis_window
    if window is None or window.size == 0:
        raise ValueError("No live signal window is available yet.")
    arr = np.asarray(window, dtype=np.float32)
    raw = arr.tobytes(order="C")
    raw_bytes = len(raw)
    results = []
    z_size, z_ms = _compressed_size_zlib(raw, level=3)
    results.append({
        "name": "lossless_zlib_float32",
        "bytes": z_size,
        "ratio": round(z_size / max(raw_bytes, 1), 4),
        "roundtrip_ms": round(z_ms, 3),
        "max_abs_error": 0.0,
        "notes": "Lossless float32, good default with no dependency.",
    })
    t0 = time.perf_counter()
    lz = lzma.compress(raw, preset=1)
    decoded_lz = lzma.decompress(lz)
    if decoded_lz != raw:
        raise RuntimeError("lzma round trip mismatch")
    results.append({
        "name": "lossless_lzma_float32",
        "bytes": len(lz),
        "ratio": round(len(lz) / max(raw_bytes, 1), 4),
        "roundtrip_ms": round((time.perf_counter() - t0) * 1000.0, 3),
        "max_abs_error": 0.0,
        "notes": "Higher compression candidate; usually too slow for live streaming.",
    })
    step = float(req.quantization_step or 0.02)
    q = np.clip(np.rint(arr / step), -32768, 32767).astype(np.int16)
    restored = q.astype(np.float32) * step
    q_raw = q.tobytes(order="C")
    q_size, q_ms = _compressed_size_zlib(q_raw, level=3)
    results.append({
        "name": "near_lossless_zlib_int16_quantized",
        "bytes": q_size,
        "ratio": round(q_size / max(raw_bytes, 1), 4),
        "roundtrip_ms": round(q_ms, 3),
        "max_abs_error": round(float(np.max(np.abs(restored - arr))) if arr.size else 0.0, 6),
        "quantization_step": step,
        "notes": "Fast stream/archive path when sub-step error is acceptable.",
    })
    chunks = [np.ascontiguousarray(q[idx:idx + 1]).tobytes(order="C") for idx in range(q.shape[0])]
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=int(req.max_workers or 4)) as pool:
        encoded = list(pool.map(lambda part: zlib.compress(part, 1), chunks))
        decoded = list(pool.map(zlib.decompress, encoded))
    if decoded != chunks:
        raise RuntimeError("parallel zlib round trip mismatch")
    p_bytes = sum(len(part) for part in encoded)
    results.append({
        "name": "parallel_channel_zlib_int16",
        "bytes": p_bytes,
        "ratio": round(p_bytes / max(raw_bytes, 1), 4),
        "roundtrip_ms": round((time.perf_counter() - t0) * 1000.0, 3),
        "max_abs_error": round(float(np.max(np.abs(restored - arr))) if arr.size else 0.0, 6),
        "workers": int(req.max_workers or 4),
        "notes": "Parallel per-channel chunks reduce latency for bigger windows and multi-stream batches.",
    })
    best = min(results, key=lambda item: (float(item["roundtrip_ms"]), float(item["ratio"])))
    return {
        "ok": True,
        "throughput": {
            "profile": config.signal_profile_name,
            "shape": list(arr.shape),
            "raw_bytes": raw_bytes,
            "sample_rate_hz": config.sample_rate,
            "results": results,
            "recommended": best,
            "architecture": [
                "Use a ring buffer for raw float32 windows.",
                "Quantize accepted training windows to int16 with stored scale for near-lossless archive.",
                "Compress independent channel/time chunks in a worker pool.",
                "Keep live inference uncompressed; compress only persistence, export, and network backfill.",
                "Use memory-mapped chunk files for long sessions so import/train can stream batches.",
            ],
        },
    }


def _pipeline_window_score(window: np.ndarray, diagnostics: dict) -> tuple[float, List[str]]:
    arr = np.asarray(window, dtype=np.float64)
    noise = diagnostics.get("noise") or {}
    timing = diagnostics.get("timing") or {}
    clip_pct = float(noise.get("clip_pct") or 0.0)
    hum = max(float(noise.get("hum_50_db") or -120.0), float(noise.get("hum_60_db") or -120.0))
    drift = float(noise.get("drift_db") or -120.0)
    crest = float(noise.get("crest_factor") or 0.0)
    score = 100.0
    reasons: List[str] = []
    if clip_pct > 0.2:
        reasons.append(f"clip {clip_pct:.2f}%")
    if hum > -28.0:
        reasons.append(f"hum {hum:.1f} dB")
    if drift > -28.0:
        reasons.append(f"drift {drift:.1f} dB")
    if crest > 10.0:
        reasons.append(f"crest {crest:.1f}")
    duplicate_pairs = _detect_duplicate_signal_channels(arr)
    if duplicate_pairs:
        labels = ", ".join(f"CH{a + 1}/CH{b + 1}" for a, b in duplicate_pairs[:3])
        reasons.append(f"duplicate channels {labels}")
    score -= min(35.0, clip_pct * 3.0)
    if hum > -40.0:
        score -= min(20.0, max(0.0, hum + 40.0) * 0.75)
    if drift > -40.0:
        score -= min(20.0, max(0.0, drift + 40.0) * 0.75)
    score -= min(15.0, max(0.0, crest - 8.0) * 1.5)
    if duplicate_pairs:
        score -= min(18.0, 6.0 * len(duplicate_pairs))
    if float(timing.get("signal_age_ms") or 0.0) > 1500.0:
        score -= 30.0
        reasons.append("stale signal")
    return round(max(0.0, min(100.0, score)), 1), reasons


def _pipeline_acquisition_counts() -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for item in app_state.pipeline_acquisition_windows:
        label = str(item.get("label") or "")
        if label:
            counts[label] = counts.get(label, 0) + 1
    return counts


def _safe_stem(value: Any, fallback: str = "item") -> str:
    stem = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value or ""))
    while "__" in stem:
        stem = stem.replace("__", "_")
    return stem.strip("_")[:80] or fallback


def _pipeline_acquisition_status() -> dict:
    run = dict(app_state.pipeline_acquisition_run or {})
    steps = list(run.get("steps") or [])
    idx = int(run.get("step_index") or 0)
    step = dict(steps[idx]) if 0 <= idx < len(steps) else {}
    counts = _pipeline_acquisition_counts()
    run["current_step"] = step
    run["accepted_windows"] = len(app_state.pipeline_acquisition_windows)
    run["per_label"] = counts
    run["buffer_bytes"] = int(sum(int(item.get("bytes") or 0) for item in app_state.pipeline_acquisition_windows))
    if run.get("running") and run.get("step_started_at"):
        elapsed = max(0.0, time.time() - float(run.get("step_started_at") or time.time()))
        run["step_elapsed_s"] = round(elapsed, 1)
        run["step_remaining_s"] = round(max(0.0, float(step.get("duration_s") or 0.0) - elapsed), 1)
    return {"ok": True, "run": run}


def _pipeline_write_acquisition_manifest() -> None:
    run = app_state.pipeline_acquisition_run
    if not run.get("run_dir"):
        return
    run_dir = Path(str(run["run_dir"]))
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run": _json_safe({k: v for k, v in run.items() if k not in {"current_step"}}),
        "accepted_windows": len(app_state.pipeline_acquisition_windows),
        "per_label": _pipeline_acquisition_counts(),
        "windows": _json_safe(app_state.pipeline_acquisition_windows[-500:]),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _write_pipeline_json(run_dir / "manifest.json", manifest)


def _pipeline_start_acquisition_run(req: PipelineAcquisitionRunRequest) -> dict:
    if not app_state.stream or not app_state.stream.is_running:
        raise ValueError("Start the stream before running guided acquisition.")
    protocol = dict(req.protocol or req.acquisition or {})
    if not protocol:
        protocol = _pipeline_acquisition_protocol(req).get("acquisition") or {}
    steps = list(protocol.get("steps") or [])
    if not steps:
        raise ValueError("No acquisition steps are available.")
    stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    append_run = bool((req.context or {}).get("append_acquisition"))
    existing_dir = str((app_state.pipeline_acquisition_run or {}).get("run_dir") or "")
    run_dir = Path(existing_dir) if append_run and existing_dir else Path(config.data_dir) / "pipeline_acquisitions" / f"{config.signal_profile_name}_{stamp}"
    if not append_run:
        app_state.pipeline_acquisition_windows = []
    app_state.pipeline_acquisition_last_accept_ts = 0.0
    app_state.pipeline_acquisition_run = {
        "id": f"{config.signal_profile_name}_{stamp}",
        "running": True,
        "paused": False,
        "profile": config.signal_profile_name,
        "started_at": time.time(),
        "step_started_at": time.time(),
        "step_index": 0,
        "steps": steps,
        "run_dir": run_dir.as_posix(),
        "qa_threshold": 75.0,
        "min_accept_interval_ms": max(100, int(((req.plan or {}).get("windowing") or {}).get("window_step_ms") or 250)),
        "rejected_windows": 0,
        "last_reject_reason": "",
        "last_accept": {},
    }
    app_state.recorder.log_event("pipeline_acquisition_started", {"run_dir": run_dir.as_posix(), "steps": len(steps)})
    app_state.lsl.push_marker("pipeline_acquisition_started", {"run_dir": run_dir.as_posix(), "profile": config.signal_profile_name})
    _pipeline_write_acquisition_manifest()
    return _pipeline_acquisition_status()


def _pipeline_stop_acquisition_run() -> dict:
    if app_state.pipeline_acquisition_run:
        app_state.pipeline_acquisition_run["running"] = False
        app_state.pipeline_acquisition_run["stopped_at"] = time.time()
        _pipeline_write_acquisition_manifest()
        app_state.recorder.log_event("pipeline_acquisition_stopped", _pipeline_acquisition_counts())
        app_state.lsl.push_marker("pipeline_acquisition_stopped", {"profile": config.signal_profile_name})
    return _pipeline_acquisition_status()


def _pipeline_advance_acquisition_step() -> None:
    run = app_state.pipeline_acquisition_run
    steps = list(run.get("steps") or [])
    idx = int(run.get("step_index") or 0) + 1
    if idx >= len(steps):
        run["running"] = False
        run["completed_at"] = time.time()
        run["step_index"] = max(0, len(steps) - 1)
        _pipeline_write_acquisition_manifest()
        return
    run["step_index"] = idx
    run["step_started_at"] = time.time()
    run["last_reject_reason"] = ""


def _pipeline_update_acquisition(window: np.ndarray, diagnostics: dict, now: float) -> None:
    run = app_state.pipeline_acquisition_run
    if not run or not run.get("running") or run.get("paused"):
        return
    steps = list(run.get("steps") or [])
    idx = int(run.get("step_index") or 0)
    if idx < 0 or idx >= len(steps):
        run["running"] = False
        return
    step = dict(steps[idx] or {})
    synthetic_scenario = str(step.get("synthetic_scenario") or "").strip()
    if synthetic_scenario and app_state.stream_source == "synthetic" and hasattr(app_state.stream, "set_scenario"):
        try:
            app_state.stream.set_scenario(synthetic_scenario)
            app_state.stream_details["scenario"] = synthetic_scenario
        except Exception:
            pass
    elapsed = now - float(run.get("step_started_at") or now)
    duration = float(step.get("duration_s") or 0.0)
    action = str(step.get("action") or "")
    if duration and elapsed >= duration:
        _pipeline_advance_acquisition_step()
        return
    if action != "record_label":
        return
    label = str(step.get("label") or step.get("id") or "").strip()
    if not label:
        return
    target = int(step.get("target_windows") or 0)
    counts = _pipeline_acquisition_counts()
    if target and counts.get(label, 0) >= target:
        _pipeline_advance_acquisition_step()
        return
    min_interval = max(0.05, float(run.get("min_accept_interval_ms") or 250.0) / 1000.0)
    if now - float(app_state.pipeline_acquisition_last_accept_ts or 0.0) < min_interval:
        return
    score, reasons = _pipeline_window_score(window, diagnostics)
    if score < float(run.get("qa_threshold") or 75.0):
        run["rejected_windows"] = int(run.get("rejected_windows") or 0) + 1
        run["last_reject_reason"] = ", ".join(reasons) or f"QA {score:.1f}"
        return
    arr = np.asarray(window, dtype=np.float32)
    step_size = 0.02
    q = np.clip(np.rint(arr / step_size), -32768, 32767).astype(np.int16)
    encoded = zlib.compress(q.tobytes(order="C"), 3)
    payload_path = ""
    run_dir = str(run.get("run_dir") or "")
    if run_dir:
        windows_dir = Path(run_dir) / "windows"
        windows_dir.mkdir(parents=True, exist_ok=True)
        next_idx = len(app_state.pipeline_acquisition_windows) + 1
        payload_path = (windows_dir / f"{next_idx:06d}_{_safe_stem(label, 'label')}.qz").as_posix()
        Path(payload_path).write_bytes(encoded)
    record = {
        "label": label,
        "timestamp": now,
        "step_index": idx,
        "qa_score": score,
        "shape": list(arr.shape),
        "dtype": "int16",
        "quantization_step": step_size,
        "bytes": len(encoded),
        "compression": "zlib",
        "payload_path": payload_path,
        "payload_b64": "" if payload_path else base64.b64encode(encoded).decode("ascii"),
        "rms": [round(float(v), 5) for v in np.sqrt(np.mean(np.square(arr.astype(np.float64)), axis=1))[: config.n_channels]],
    }
    app_state.pipeline_acquisition_windows.append(record)
    app_state.pipeline_acquisition_last_accept_ts = now
    run["last_accept"] = record
    if len(app_state.pipeline_acquisition_windows) % 10 == 0:
        _pipeline_write_acquisition_manifest()


def _pipeline_acquisition_control(req: PipelineAcquisitionRunRequest) -> dict:
    action = str(req.action or "start").strip().lower()
    if action == "start":
        return _pipeline_start_acquisition_run(req)
    if action == "stop":
        return _pipeline_stop_acquisition_run()
    if action == "pause":
        app_state.pipeline_acquisition_run["paused"] = True
        return _pipeline_acquisition_status()
    if action == "resume":
        app_state.pipeline_acquisition_run["paused"] = False
        app_state.pipeline_acquisition_run["step_started_at"] = time.time()
        return _pipeline_acquisition_status()
    if action == "next":
        if app_state.pipeline_acquisition_run:
            _pipeline_advance_acquisition_step()
        return _pipeline_acquisition_status()
    if action == "status":
        return _pipeline_acquisition_status()
    raise ValueError("Unknown acquisition action. Use start, stop, pause, resume, next, or status.")


def _restore_acquisition_window(record: dict) -> Optional[np.ndarray]:
    try:
        payload_path = str(record.get("payload_path") or "")
        if payload_path and Path(payload_path).exists():
            encoded = Path(payload_path).read_bytes()
        else:
            raw_b64 = str(record.get("payload_b64") or "")
            if not raw_b64:
                return None
            encoded = base64.b64decode(raw_b64.encode("ascii"))
        decoded = zlib.decompress(encoded)
        shape = tuple(int(v) for v in list(record.get("shape") or []))
        if len(shape) != 2 or shape[0] <= 0 or shape[1] <= 0:
            return None
        q = np.frombuffer(decoded, dtype=np.int16).reshape(shape)
        step = float(record.get("quantization_step") or 0.02)
        return q.astype(np.float32) * step
    except Exception:
        return None


def _pipeline_window_feature_vector(window: np.ndarray) -> np.ndarray:
    arr = np.asarray(window, dtype=np.float32)
    if arr.ndim != 2 or not arr.size:
        return np.zeros((1,), dtype=np.float32)
    x = arr.astype(np.float64, copy=False)
    dx = np.diff(x, axis=1) if x.shape[1] > 1 else np.zeros_like(x)
    rms = np.sqrt(np.mean(np.square(x), axis=1))
    mav = np.mean(np.abs(x), axis=1)
    mean = np.mean(x, axis=1)
    std = np.std(x, axis=1)
    p2p = np.ptp(x, axis=1)
    wl = np.sum(np.abs(dx), axis=1)
    zc = np.sum((x[:, 1:] * x[:, :-1]) < 0, axis=1) if x.shape[1] > 1 else np.zeros(x.shape[0])
    features = np.concatenate([
        rms,
        mav,
        mean,
        std,
        p2p,
        wl,
        zc,
        np.asarray([
            float(np.mean(rms)),
            float(np.std(rms)),
            float(np.max(rms)),
            float(np.mean(p2p)),
            float(np.std(p2p)),
        ]),
    ])
    return features.astype(np.float32, copy=False)


def _pipeline_train_acquisition(req: PipelineActionRequest) -> dict:
    records = list(app_state.pipeline_acquisition_windows or [])
    X_rows: List[np.ndarray] = []
    y_labels: List[str] = []
    skipped = 0
    for record in records:
        label = str(record.get("label") or "").strip()
        if not label:
            skipped += 1
            continue
        window = _restore_acquisition_window(record)
        if window is None:
            skipped += 1
            continue
        X_rows.append(_pipeline_window_feature_vector(window))
        y_labels.append(label)
    counts: Dict[str, int] = {}
    for label in y_labels:
        counts[label] = counts.get(label, 0) + 1
    if len(counts) < 2 or len(X_rows) < 4:
        return {
            "ok": True,
            "trained": False,
            "ready": False,
            "classifier": "logistic_regression",
            "summary": {
                "total_windows": len(X_rows),
                "per_label": counts,
                "skipped_windows": skipped,
            },
            "requirements": {
                "minimum_windows_total": 4,
                "minimum_labeled_classes": 2,
                "current_windows_total": len(X_rows),
                "current_labeled_classes": len(counts),
            },
            "next_actions": [
                "Run guided acquisition until at least two labels have accepted windows.",
                "Use shorter protocol steps for smoke tests or full target windows for a useful model.",
                "Keep QA threshold at 75+ for the first pass.",
            ],
        }
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, precision_recall_fscore_support
    from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    X = np.vstack(X_rows).astype(np.float32)
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_labels)
    min_count = min(counts.values()) if counts else 0
    val_accuracy = None
    cv_accuracy = None
    cv_balanced_accuracy = None
    macro_f1 = None
    confusion = None
    per_label_quality: Dict[str, Any] = {}
    quality_warnings: List[str] = []
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
    ])

    if min_count >= 2 and len(X_rows) >= 6:
        cv_folds = min(5, min_count)
        try:
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            cv_pred = cross_val_predict(clf, X, y, cv=cv)
            cv_accuracy = float(accuracy_score(y, cv_pred))
            cv_balanced_accuracy = float(balanced_accuracy_score(y, cv_pred))
            precision, recall, f1, support = precision_recall_fscore_support(
                y,
                cv_pred,
                labels=list(range(len(encoder.classes_))),
                zero_division=0,
            )
            macro_f1 = float(np.mean(f1)) if len(f1) else 0.0
            confusion = confusion_matrix(y, cv_pred, labels=list(range(len(encoder.classes_)))).tolist()
            for idx, label in enumerate(encoder.classes_.tolist()):
                row = confusion[idx] if idx < len(confusion) else []
                off_diag = [
                    (encoder.classes_[j], int(value))
                    for j, value in enumerate(row)
                    if j != idx and int(value) > 0
                ]
                off_diag.sort(key=lambda item: item[1], reverse=True)
                per_label_quality[str(label)] = {
                    "precision": round(float(precision[idx]), 4),
                    "recall": round(float(recall[idx]), 4),
                    "f1": round(float(f1[idx]), 4),
                    "support": int(support[idx]),
                    "top_confusion": {"label": str(off_diag[0][0]), "count": off_diag[0][1]} if off_diag else None,
                }
        except Exception as exc:
            logger.warning("Prompt model cross-validation failed: %s", exc)

    if min_count >= 2 and len(X_rows) >= 6:
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=min(0.35, max(0.2, len(counts) / max(len(X_rows), 1))),
            random_state=42,
            stratify=y,
        )
    else:
        X_train, y_train = X, y
        X_val = y_val = None
    split_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
    ])
    split_clf.fit(X_train, y_train)
    train_accuracy = float(accuracy_score(y_train, split_clf.predict(X_train)))
    if X_val is not None and y_val is not None and len(X_val):
        pred = split_clf.predict(X_val)
        val_accuracy = float(accuracy_score(y_val, pred))
    chance_floor = 1.0 / max(len(encoder.classes_), 1)
    ready_accuracy = max(0.55, chance_floor + 0.15)
    validation_metric = cv_balanced_accuracy if cv_balanced_accuracy is not None else val_accuracy
    validation_ready = validation_metric is None or validation_metric >= ready_accuracy
    label_balance = min(counts.values()) / max(max(counts.values()), 1) if counts else 0.0
    min_label_windows = min_count if counts else 0
    if min_label_windows < 10:
        quality_warnings.append("Each label should have at least 10 accepted windows before trusting live use.")
    if validation_metric is not None and validation_metric < ready_accuracy:
        quality_warnings.append(f"Cross-label validation is below the live-use gate ({validation_metric:.2f} < {ready_accuracy:.2f}).")
    if macro_f1 is not None and macro_f1 < ready_accuracy:
        quality_warnings.append(f"Macro F1 is low ({macro_f1:.2f}); at least one label is probably overlapping.")
    for label, metrics in per_label_quality.items():
        if float(metrics.get("recall") or 0.0) < 0.5:
            quality_warnings.append(f"{label} recall is below 50%; collect clearer examples for that label.")
    quality_score = round(min(100.0, (
        (float(validation_metric or 0.0) * 45.0)
        + (float(macro_f1 or validation_metric or 0.0) * 25.0)
        + (label_balance * 15.0)
        + min(15.0, min_label_windows * 1.5)
    )), 1)
    clf.fit(X, y)
    model_dir = Path(config.data_dir) / "pipeline_models"
    model_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    model_path = model_dir / f"{config.signal_profile_name}_prompt_model_{stamp}.pkl"
    model_card_path = model_dir / f"{config.signal_profile_name}_prompt_model_{stamp}.json"
    artifact = {
        "model": clf,
        "label_encoder": encoder,
        "labels": encoder.classes_.tolist(),
        "profile": config.signal_profile_name,
        "feature_dim": int(X.shape[1]),
        "input": {
            "source": "pipeline_acquisition_windows",
            "window_shape": list(_restore_acquisition_window(records[0]).shape) if records else [],
            "quantized": True,
        },
    }
    with model_path.open("wb") as fh:
        pickle.dump(artifact, fh)
    model_card = {
        "created_at": stamp,
        "profile": config.signal_profile_name,
        "classifier": "logistic_regression",
        "labels": encoder.classes_.tolist(),
        "n_windows": len(X_rows),
        "per_label": counts,
        "skipped_windows": skipped,
        "feature_dim": int(X.shape[1]),
        "train_accuracy": round(train_accuracy, 4),
        "val_accuracy": round(val_accuracy, 4) if val_accuracy is not None else None,
        "cv_accuracy": round(cv_accuracy, 4) if cv_accuracy is not None else None,
        "cv_balanced_accuracy": round(cv_balanced_accuracy, 4) if cv_balanced_accuracy is not None else None,
        "macro_f1": round(macro_f1, 4) if macro_f1 is not None else None,
        "ready": bool(validation_ready),
        "readiness_threshold": round(ready_accuracy, 4),
        "quality_score": quality_score,
        "quality_warnings": quality_warnings,
        "per_label_quality": per_label_quality,
        "confusion_matrix": confusion,
        "confusion_labels": encoder.classes_.tolist(),
        "model_path": model_path.as_posix(),
    }
    _write_pipeline_json(model_card_path, model_card)
    app_state.pipeline_acquisition_model = {
        **model_card,
        "model_card_path": model_card_path.as_posix(),
    }
    app_state.pipeline_acquisition_runtime = {
        **artifact,
        "model_path": model_path.as_posix(),
        "model_card_path": model_card_path.as_posix(),
        "classifier": "logistic_regression",
        "trained_at": stamp,
        "per_label": counts,
    }
    app_state.recorder.log_event("pipeline_acquisition_model_trained", model_card)
    app_state.lsl.push_marker("pipeline_acquisition_model_trained", {"profile": config.signal_profile_name, "path": model_path.as_posix()})
    return {
        "ok": True,
        "trained": True,
        "ready": bool(validation_ready),
        "classifier": "logistic_regression",
        "model_path": model_path.as_posix(),
        "model_card": model_card_path.as_posix(),
        "result": model_card,
        "summary": {
            "total_windows": len(X_rows),
            "per_label": counts,
            "is_trained": True,
            "profile": config.signal_profile_name,
            "supports_training": True,
        },
        "next_actions": [
            *([] if validation_ready else [
                "Collect more distinguishable examples; current validation is below the live-use gate.",
                "Use real label changes or varied synthetic scenarios so labels are separable.",
            ]),
            *quality_warnings[:3],
            "Compare this prompt model against the baseline and foundation embedding path.",
            "Export the model card with the pipeline package.",
            "Add ONNX conversion next for browser runtime deployment.",
        ],
    }


def _pipeline_prompt_model_summary() -> dict:
    model = dict(app_state.pipeline_acquisition_model or {})
    runtime = app_state.pipeline_acquisition_runtime or {}
    return {
        "loaded": bool(runtime.get("model") and runtime.get("label_encoder")),
        "profile": model.get("profile") or runtime.get("profile") or config.signal_profile_name,
        "classifier": model.get("classifier") or runtime.get("classifier") or "",
        "labels": list(model.get("labels") or runtime.get("labels") or []),
        "n_windows": int(model.get("n_windows") or 0),
        "per_label": dict(model.get("per_label") or runtime.get("per_label") or {}),
        "feature_dim": int(model.get("feature_dim") or runtime.get("feature_dim") or 0),
        "train_accuracy": model.get("train_accuracy"),
        "val_accuracy": model.get("val_accuracy"),
        "cv_balanced_accuracy": model.get("cv_balanced_accuracy"),
        "macro_f1": model.get("macro_f1"),
        "ready": model.get("ready"),
        "quality_score": model.get("quality_score"),
        "quality_warnings": list(model.get("quality_warnings") or []),
        "model_path": str(model.get("model_path") or runtime.get("model_path") or ""),
        "model_card_path": str(model.get("model_card_path") or runtime.get("model_card_path") or ""),
    }


def _pipeline_prompt_model_dir() -> Path:
    path = Path(config.data_dir) / "pipeline_models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _resolve_prompt_model_path(raw_path: str) -> Path:
    if not str(raw_path or "").strip():
        raise ValueError("Model path is required.")
    root = _pipeline_prompt_model_dir().resolve()
    candidate = Path(str(raw_path)).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    candidate = candidate.resolve()
    if candidate.suffix.lower() != ".pkl":
        raise ValueError("Prompt model path must point to a .pkl artifact.")
    if root not in candidate.parents and candidate != root:
        raise ValueError("Prompt model path is outside the managed model directory.")
    if not candidate.exists():
        raise ValueError("Prompt model artifact was not found.")
    return candidate


def _read_prompt_model_card(model_path: Path) -> dict:
    card_path = model_path.with_suffix(".json")
    card: Dict[str, Any] = {}
    if card_path.exists():
        try:
            card = json.loads(card_path.read_text(encoding="utf-8"))
        except Exception:
            card = {}
    stat = model_path.stat()
    active_path = str((app_state.pipeline_acquisition_model or {}).get("model_path") or "")
    active_resolved = ""
    if active_path:
        try:
            active_resolved = Path(active_path).resolve().as_posix()
        except Exception:
            active_resolved = active_path
    model_resolved = model_path.resolve().as_posix()
    return {
        "path": model_path.as_posix(),
        "model_path": model_path.as_posix(),
        "model_card_path": card_path.as_posix() if card_path.exists() else "",
        "active": model_resolved == active_resolved,
        "exists": True,
        "size_bytes": int(stat.st_size),
        "modified_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(stat.st_mtime)),
        "created_at": card.get("created_at") or "",
        "profile": card.get("profile") or "",
        "classifier": card.get("classifier") or "",
        "labels": list(card.get("labels") or []),
        "n_windows": int(card.get("n_windows") or 0),
        "per_label": dict(card.get("per_label") or {}),
        "train_accuracy": card.get("train_accuracy"),
        "val_accuracy": card.get("val_accuracy"),
        "cv_balanced_accuracy": card.get("cv_balanced_accuracy"),
        "macro_f1": card.get("macro_f1"),
        "ready": card.get("ready"),
        "quality_score": card.get("quality_score"),
        "quality_warnings": list(card.get("quality_warnings") or []),
        "confusion_matrix": card.get("confusion_matrix"),
        "confusion_labels": list(card.get("confusion_labels") or []),
        "per_label_quality": dict(card.get("per_label_quality") or {}),
    }


def _pipeline_list_prompt_models() -> dict:
    root = _pipeline_prompt_model_dir()
    models = [
        _read_prompt_model_card(path)
        for path in root.glob(f"{config.signal_profile_name}_prompt_model_*.pkl")
        if path.is_file()
    ]
    models.sort(key=lambda item: (item.get("active") is True, str(item.get("modified_at") or "")), reverse=True)
    return {
        "ok": True,
        "active": _pipeline_prompt_model_summary(),
        "models": models,
        "model_dir": root.as_posix(),
    }


def _pipeline_delete_prompt_model(req: PipelineModelRequest) -> dict:
    model_path = _resolve_prompt_model_path(req.path)
    active_before = _pipeline_prompt_model_summary()
    was_active = False
    try:
        was_active = Path(str(active_before.get("model_path") or "")).resolve() == model_path.resolve()
    except Exception:
        was_active = False
    card_path = model_path.with_suffix(".json")
    deleted = []
    for path in [model_path, card_path]:
        if path.exists():
            path.unlink()
            deleted.append(path.as_posix())
    if was_active:
        app_state.pipeline_acquisition_model = {}
        app_state.pipeline_acquisition_runtime = {}
    app_state.recorder.log_event("pipeline_prompt_model_deleted", {"path": model_path.as_posix(), "was_active": was_active})
    return {
        "ok": True,
        "deleted": deleted,
        "was_active": was_active,
        **_pipeline_list_prompt_models(),
    }


def _pipeline_load_prompt_model(model_path: Optional[Path] = None) -> dict:
    model_dir = Path(config.data_dir) / "pipeline_models"
    if model_path is None:
        candidates = sorted(
            model_dir.glob(f"{config.signal_profile_name}_prompt_model_*.pkl"),
            key=lambda path: path.stat().st_mtime if path.exists() else 0.0,
            reverse=True,
        )
        model_path = candidates[0] if candidates else None
    if not model_path or not Path(model_path).exists():
        app_state.pipeline_acquisition_model = {}
        app_state.pipeline_acquisition_runtime = {}
        return _pipeline_prompt_model_summary()
    try:
        with Path(model_path).open("rb") as fh:
            artifact = pickle.load(fh)
        if not isinstance(artifact, dict) or artifact.get("model") is None or artifact.get("label_encoder") is None:
            raise ValueError("Prompt model artifact is missing model or label encoder.")
        card_path = Path(model_path).with_suffix(".json")
        card = {}
        if card_path.exists():
            try:
                card = json.loads(card_path.read_text(encoding="utf-8"))
            except Exception:
                card = {}
        labels = list(artifact.get("labels") or card.get("labels") or [])
        profile = str(artifact.get("profile") or card.get("profile") or config.signal_profile_name)
        if profile != config.signal_profile_name:
            raise ValueError(f"Prompt model profile {profile} does not match active profile {config.signal_profile_name}.")
        app_state.pipeline_acquisition_model = {
            **card,
            "profile": profile,
            "classifier": card.get("classifier") or "logistic_regression",
            "labels": labels,
            "model_path": Path(model_path).as_posix(),
            "model_card_path": card_path.as_posix() if card_path.exists() else "",
        }
        app_state.pipeline_acquisition_runtime = {
            **artifact,
            "model_path": Path(model_path).as_posix(),
            "model_card_path": card_path.as_posix() if card_path.exists() else "",
            "classifier": card.get("classifier") or "logistic_regression",
            "per_label": dict(card.get("per_label") or {}),
        }
        logger.info("Loaded prompt acquisition model: %s", Path(model_path).as_posix())
    except Exception as exc:
        logger.warning("Prompt acquisition model load failed: %s", exc)
        app_state.pipeline_acquisition_model = {}
        app_state.pipeline_acquisition_runtime = {}
    return _pipeline_prompt_model_summary()


def _pipeline_load_prompt_model_request(req: PipelineModelRequest) -> dict:
    model_path = _resolve_prompt_model_path(req.path)
    loaded = _pipeline_load_prompt_model(model_path)
    if not loaded.get("loaded"):
        raise ValueError("Prompt model could not be loaded.")
    app_state.recorder.log_event("pipeline_prompt_model_loaded", {"path": model_path.as_posix(), "profile": config.signal_profile_name})
    app_state.lsl.push_marker("pipeline_prompt_model_loaded", {"path": model_path.as_posix(), "profile": config.signal_profile_name})
    return {
        "ok": True,
        "active": loaded,
        "models": _pipeline_list_prompt_models()["models"],
    }


def _pipeline_prompt_model_predict(window: np.ndarray, diagnostics: dict) -> Optional[dict]:
    runtime = app_state.pipeline_acquisition_runtime or {}
    model = runtime.get("model")
    encoder = runtime.get("label_encoder")
    labels = list(runtime.get("labels") or [])
    if model is None or encoder is None or not labels:
        return None
    try:
        started = time.perf_counter()
        features = _pipeline_window_feature_vector(window).reshape(1, -1)
        pred_idx = int(model.predict(features)[0])
        if hasattr(model, "predict_proba"):
            proba = np.asarray(model.predict_proba(features)[0], dtype=np.float64)
        else:
            proba = np.zeros((len(labels),), dtype=np.float64)
            proba[pred_idx] = 1.0
        label = str(encoder.inverse_transform([pred_idx])[0])
        confidence = float(proba[pred_idx]) if 0 <= pred_idx < len(proba) else 0.0
        x = np.asarray(window, dtype=np.float64)
        rms = np.sqrt(np.mean(np.square(x), axis=1)) if x.ndim == 2 else np.asarray([])
        p2p = np.ptp(x, axis=1) if x.ndim == 2 else np.asarray([])
        wl = np.sum(np.abs(np.diff(x, axis=1)), axis=1) if x.ndim == 2 and x.shape[1] > 1 else np.zeros_like(rms)
        energy = (rms / max(float(np.max(rms)) if rms.size else 1.0, 1e-9)) * 0.58
        range_score = (p2p / max(float(np.max(p2p)) if p2p.size else 1.0, 1e-9)) * 0.27
        motion_score = (wl / max(float(np.max(wl)) if wl.size else 1.0, 1e-9)) * 0.15
        channel_scores = energy + range_score + motion_score
        active_threshold = max(float(np.median(channel_scores) + np.std(channel_scores) * 0.35), 0.28) if channel_scores.size else 0.0
        evidence = []
        for idx, score in enumerate(channel_scores.tolist()):
            if score >= active_threshold or idx == int(np.argmax(channel_scores)):
                evidence.append(
                    {
                        "channel": idx,
                        "channel_label": config.channel_labels[idx] if idx < len(config.channel_labels) else f"CH{idx + 1}",
                        "score": round(float(score), 4),
                        "rms": round(float(rms[idx]), 5) if idx < len(rms) else 0.0,
                        "peak_to_peak": round(float(p2p[idx]), 5) if idx < len(p2p) else 0.0,
                    }
                )
        evidence = sorted(evidence, key=lambda item: item["score"], reverse=True)[: min(4, len(evidence))]
        return {
            "active": True,
            "source": "prompt_acquisition_model",
            "profile": config.signal_profile_name,
            "classifier": runtime.get("classifier") or "logistic_regression",
            "label": label,
            "confidence": round(confidence, 4),
            "probabilities": {
                str(labels[i]): round(float(proba[i]), 4)
                for i in range(min(len(labels), len(proba)))
            },
            "evidence_channels": evidence,
            "window_samples": int(window.shape[1]) if window.ndim == 2 else 0,
            "sample_rate_hz": config.sample_rate,
            "model_path": str(runtime.get("model_path") or ""),
            "model_card_path": str(runtime.get("model_card_path") or ""),
            "inference_ms": round((time.perf_counter() - started) * 1000.0, 3),
            "qa_score": diagnostics.get("quality", {}).get("score") if isinstance(diagnostics, dict) else None,
        }
    except Exception as exc:
        logger.warning("Prompt acquisition model inference failed: %s", exc)
        return None


def _pipeline_embedding_timeline(req: PipelineActionRequest) -> dict:
    if app_state.last_vis_window is not None:
        try:
            diagnostics = app_state.diagnostics.status()
            local_ml = app_state.ai.local_models.analyze(_live_ml_snapshot(app_state.last_vis_window, diagnostics))
            _append_pipeline_embedding_snapshot(local_ml, time.time())
        except Exception:
            pass
    timeline = list(app_state.pipeline_embedding_timeline[-120:])
    readiness = _pipeline_dataset_readiness(dict(req.qa or {}), dict(req.train_result or {}))
    if timeline:
        readiness["score"] = round(min(100.0, float(readiness["score"]) + 5.0), 1)
        readiness["embedding_count"] = len(timeline)
        readiness["latest_embedding"] = timeline[-1]
    else:
        readiness["embedding_count"] = 0
    return {
        "ok": True,
        "embedding": {
            "timeline": timeline,
            "readiness": readiness,
            "summary": f"{len(timeline)} embedding points captured for {config.signal_profile.display_name}.",
        },
    }


def _pipeline_export(req: PipelineActionRequest) -> dict:
    plan = dict(req.plan or {})
    qa = dict(req.qa or {})
    train_result = dict(req.train_result or {})
    acquisition = dict(req.acquisition or {})
    acquisition_run = dict(req.acquisition_run or app_state.pipeline_acquisition_run or {})
    labels = dict(req.labels or {})
    dataset = dict(req.dataset or {})
    throughput = dict(req.throughput or {})
    comparison = _pipeline_model_compare(req)["comparison"]
    runtime = _pipeline_browser_runtime(req)["runtime"]
    embedding = _pipeline_embedding_timeline(req)["embedding"]
    stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    export_dir = Path(config.data_dir) / "pipeline_exports" / f"{config.signal_profile_name}_{stamp}"
    export_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at": stamp,
        "profile": config.signal_profile_name,
        "title": plan.get("title") or "KYMA pipeline export",
        "files": {},
    }
    payloads = {
        "plan.json": plan,
        "signal_qa.json": qa,
        "training.json": train_result,
        "acquisition_protocol.json": acquisition,
        "acquisition_run.json": acquisition_run,
        "label_suggestions.json": labels,
        "dataset_inspect.json": dataset,
        "model_comparison.json": comparison,
        "browser_runtime.json": runtime,
        "embedding_timeline.json": embedding,
        "throughput_benchmark.json": throughput,
    }
    packaged_models = []
    packaged_browser_models = []
    models_dir = export_dir / "models"
    for model_file in list(runtime.get("files") or []):
        source = Path(str(model_file))
        if not source.exists() or source.suffix.lower() != ".onnx":
            continue
        models_dir.mkdir(parents=True, exist_ok=True)
        target = models_dir / source.name
        shutil.copy2(source, target)
        packaged_models.append(target.as_posix())
        manifest["files"][f"models/{source.name}"] = target.as_posix()
    for model_file in list(runtime.get("browser_files") or []):
        source = Path(str(model_file))
        if not source.exists() or source.suffix.lower() != ".json":
            continue
        models_dir.mkdir(parents=True, exist_ok=True)
        target = models_dir / source.name
        shutil.copy2(source, target)
        packaged_browser_models.append(target.as_posix())
        manifest["files"][f"models/{source.name}"] = target.as_posix()
    if packaged_models:
        runtime["packaged_model_files"] = packaged_models
    if packaged_browser_models:
        runtime["packaged_browser_model_files"] = packaged_browser_models
    for filename, payload in payloads.items():
        _write_pipeline_json(export_dir / filename, payload)
        manifest["files"][filename] = (export_dir / filename).as_posix()
    runtime_js_path = export_dir / "browser_runtime.js"
    runtime_js_path.write_text(_pipeline_browser_runtime_js(runtime), encoding="utf-8")
    manifest["files"]["browser_runtime.js"] = runtime_js_path.as_posix()
    deploy = _pipeline_deploy_smoke(req)["deploy"]
    if packaged_browser_models:
        try:
            browser_payload = json.loads(Path(packaged_browser_models[0]).read_text(encoding="utf-8"))
            sample_window = _pipeline_sample_window_for_deploy(req)
            index_path = export_dir / "index.html"
            index_path.write_text(_pipeline_export_index_html(runtime, deploy, browser_payload, sample_window), encoding="utf-8")
            manifest["files"]["index.html"] = index_path.as_posix()
        except Exception as exc:
            logger.warning("Pipeline export self-test page failed: %s", exc)
    report_lines = [
        f"# {manifest['title']}",
        "",
        f"Profile: {config.signal_profile.display_name}",
        f"Dataset readiness: {embedding.get('readiness', {}).get('score', 0)} / 100",
        f"QA: {qa.get('score', 0)} / 100",
        f"Baseline: {'trained' if train_result.get('trained') else 'not ready'}",
        f"Browser runtime: {runtime.get('status')}",
        f"Deploy smoke: {deploy.get('status')}",
        "",
        "## Failure Cases",
        *[f"- {item.get('type')}: {item.get('detail')}" for item in comparison.get("failure_cases", [])],
    ]
    (export_dir / "report.md").write_text("\n".join(report_lines), encoding="utf-8")
    manifest["files"]["report.md"] = (export_dir / "report.md").as_posix()
    _write_pipeline_json(export_dir / "manifest.json", manifest)
    return {
        "ok": True,
        "export": {
            "export_dir": export_dir.as_posix(),
            "manifest": (export_dir / "manifest.json").as_posix(),
            "files": manifest["files"],
            "readiness": embedding.get("readiness", {}),
        },
    }


def _pipeline_export_root() -> Path:
    return (Path(config.data_dir) / "pipeline_exports").resolve()


def _resolve_pipeline_export_dir(raw_path: str) -> Path:
    raw = str(raw_path or "").strip()
    if not raw:
        raise ValueError("Export package path is required.")
    path = Path(raw)
    if path.name in {"manifest.json", "index.html"}:
        path = path.parent
    resolved = path.resolve()
    root = _pipeline_export_root()
    if not resolved.exists() or not resolved.is_dir():
        raise ValueError(f"Export package does not exist: {raw}")
    if root != resolved and root not in resolved.parents:
        raise ValueError("Export package path is outside the pipeline export directory.")
    return resolved


def _summarize_pipeline_export_dir(path: Path) -> dict:
    manifest_path = path / "manifest.json"
    manifest: Dict[str, Any] = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8", errors="ignore"))
        except Exception as exc:
            manifest = {"error": f"{type(exc).__name__}: {exc}", "files": {}}
    files = dict(manifest.get("files") or {})
    actual_files = sorted([item for item in path.rglob("*") if item.is_file()])
    file_names = {item.relative_to(path).as_posix(): item for item in actual_files}
    has_index = "index.html" in files or (path / "index.html").exists()
    has_report = "report.md" in files or (path / "report.md").exists()
    onnx_files = [name for name in set(list(files.keys()) + list(file_names.keys())) if name.lower().endswith(".onnx")]
    browser_files = [name for name in set(list(files.keys()) + list(file_names.keys())) if name.lower().endswith(".browser.json")]
    runtime_path = path / "browser_runtime.json"
    training_path = path / "training.json"
    runtime: Dict[str, Any] = {}
    training: Dict[str, Any] = {}
    if runtime_path.exists():
        try:
            runtime = json.loads(runtime_path.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            runtime = {}
    if training_path.exists():
        try:
            training = json.loads(training_path.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            training = {}
    model_path = str(training.get("model_path") or ((training.get("result") or {}).get("model_path")) or "")
    ready = bool(has_index and (onnx_files or browser_files))
    modified = max((item.stat().st_mtime for item in actual_files), default=path.stat().st_mtime)
    return {
        "path": path.as_posix(),
        "name": path.name,
        "created_at": manifest.get("created_at") or path.name.replace(f"{config.signal_profile_name}_", ""),
        "modified_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(modified)),
        "profile": manifest.get("profile") or config.signal_profile_name,
        "title": manifest.get("title") or "KYMA pipeline export",
        "ready": ready,
        "has_manifest": manifest_path.exists(),
        "has_index": has_index,
        "has_report": has_report,
        "onnx_count": len(onnx_files),
        "browser_count": len(browser_files),
        "file_count": len(actual_files),
        "bytes": int(sum(item.stat().st_size for item in actual_files)),
        "runtime_status": runtime.get("status") or "",
        "model_path": model_path,
        "files": files,
        "missing_files": [
            name for name, raw in files.items()
            if raw and not Path(str(raw)).exists() and not (path / name).exists()
        ][:12],
    }


def _pipeline_list_exports() -> dict:
    root = _pipeline_export_root()
    root.mkdir(parents=True, exist_ok=True)
    packages = []
    for path in sorted([item for item in root.iterdir() if item.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)[:40]:
        try:
            packages.append(_summarize_pipeline_export_dir(path))
        except Exception as exc:
            packages.append({
                "path": path.as_posix(),
                "name": path.name,
                "ready": False,
                "error": f"{type(exc).__name__}: {exc}",
            })
    return {"ok": True, "exports": packages, "export_dir": root.as_posix()}


def _pipeline_verify_export_package(req: PipelineModelRequest) -> dict:
    export_dir = _resolve_pipeline_export_dir(req.path)
    package = _summarize_pipeline_export_dir(export_dir)
    errors = []
    browser_results = []
    onnx_results = []
    sample_window = _pipeline_sample_window_for_deploy(PipelineActionRequest())
    for name, raw_path in list((package.get("files") or {}).items()):
        path = Path(str(raw_path))
        if not path.exists():
            path = export_dir / str(name)
        if not path.exists():
            continue
        try:
            if path.suffix.lower() == ".json" and path.name.endswith(".browser.json"):
                browser_results.append(_pipeline_browser_linear_smoke(path, sample_window))
            elif path.suffix.lower() == ".onnx":
                onnx_results.append(_pipeline_onnx_smoke(path, sample_window))
        except Exception as exc:
            errors.append(f"{path.name}: {type(exc).__name__}: {exc}")
    missing = list(package.get("missing_files") or [])
    errors.extend([f"Missing: {name}" for name in missing])
    ready = bool(package.get("ready")) and not errors and bool(browser_results or onnx_results)
    return {
        "ok": True,
        "package": package,
        "deploy": {
            "ready": ready,
            "status": "passed" if ready else "review",
            "sample_shape": list(sample_window.shape),
            "browser_results": browser_results,
            "onnx_results": onnx_results,
            "errors": errors[:12],
        },
    }


def _pipeline_delete_export_package(req: PipelineModelRequest) -> dict:
    export_dir = _resolve_pipeline_export_dir(req.path)
    deleted = export_dir.as_posix()
    shutil.rmtree(export_dir)
    return {"ok": True, "deleted": deleted, **_pipeline_list_exports()}


def _pipeline_export_zip_root() -> Path:
    return (Path(config.data_dir) / "pipeline_export_zips").resolve()


def _pipeline_download_export_package(req: PipelineModelRequest) -> Path:
    export_dir = _resolve_pipeline_export_dir(req.path)
    root = _pipeline_export_zip_root()
    root.mkdir(parents=True, exist_ok=True)
    zip_base = root / _safe_stem(export_dir.name, "pipeline_export")
    zip_path = Path(shutil.make_archive(zip_base.as_posix(), "zip", root_dir=export_dir.as_posix()))
    return zip_path.resolve()


def _pipeline_project_root() -> Path:
    return (Path(config.data_dir) / "pipeline_projects").resolve()


def _pipeline_project_path(project_id: str, *, must_exist: bool = True) -> Path:
    raw = _safe_stem(project_id, "project")
    if not raw:
        raise ValueError("Project id is required.")
    root = _pipeline_project_root()
    path = (root / f"{raw}.json").resolve()
    if root != path.parent:
        raise ValueError("Project id resolves outside the project registry.")
    if must_exist and not path.exists():
        raise ValueError(f"Project does not exist: {project_id}")
    return path


def _pipeline_importer_status() -> List[dict]:
    pyxdf_ready = bool(getattr(app_state.xdf, "available", False)) or _dependency_available("pyxdf")
    scipy_ready = _dependency_available("scipy")
    mne_ready = _dependency_available("mne")
    h5_ready = _dependency_available("h5py")
    parquet_ready = _dependency_available("pandas") and (_dependency_available("pyarrow") or _dependency_available("fastparquet"))
    return [
        {
            "format": "CSV",
            "status": "trainable",
            "detail": "Schema mapper supports time, channel, label, and group columns.",
        },
        {
            "format": "NPZ",
            "status": "trainable",
            "detail": "Uses arrays from local acquisition or exported biosignal windows.",
        },
        {
            "format": "KYMA session folder",
            "status": "trainable",
            "detail": "Reads existing guided acquisition windows and manifest files.",
        },
        {
            "format": "EDF/BDF",
            "status": "trainable" if mne_ready else "needs mne",
            "detail": "MNE-backed importer reads signal channels and annotations as label spans.",
        },
        {
            "format": "XDF",
            "status": "trainable" if pyxdf_ready else "needs pyxdf",
            "detail": "pyxdf importer reads numeric LSL streams and marker streams as label spans.",
        },
        {
            "format": "MAT/WAV",
            "status": "trainable" if scipy_ready else "needs scipy",
            "detail": "SciPy importer reads MATLAB arrays and WAV signals; labels are required for training.",
        },
        {
            "format": "HDF5",
            "status": "trainable" if h5_ready else "needs h5py",
            "detail": "HDF5 importer searches signal/data arrays plus labels and sample-rate datasets.",
        },
        {
            "format": "Parquet",
            "status": "trainable" if parquet_ready else "needs pandas+pyarrow",
            "detail": "Table importer reuses the schema mapper for columnar biosignal archives.",
        },
    ]


def _pipeline_evaluation_summary(snapshot: dict) -> dict:
    train = dict(snapshot.get("train_result") or {})
    if "training" in train and isinstance(train.get("training"), dict):
        train = dict(train.get("training") or {})
    result = dict(train.get("result") or {})
    dataset = dict(snapshot.get("dataset") or {})
    ingest = dict(snapshot.get("ingest") or {})
    readiness = dict(dataset.get("readiness") or {})
    per_label = dict(train.get("per_label") or result.get("per_label") or ingest.get("per_label") or {})
    labels = list(train.get("confusion_labels") or result.get("confusion_labels") or per_label.keys())
    n_windows = int(train.get("n_windows") or result.get("n_windows") or sum(int(v or 0) for v in per_label.values()) or 0)
    quality_warnings = list(train.get("quality_warnings") or result.get("quality_warnings") or [])
    counts = [int(v or 0) for v in per_label.values()]
    min_label_windows = min(counts) if counts else 0
    max_label_windows = max(counts) if counts else 0
    label_balance = round(float(min_label_windows / max(max_label_windows, 1)), 4) if counts else 0.0
    readiness_threshold = train.get("readiness_threshold") or result.get("readiness_threshold")
    recommendations = []
    if len(labels) < 2:
        recommendations.append("Add at least two labeled classes before training.")
    if min_label_windows < 10:
        recommendations.append("Collect at least 10 windows per label for a credible first model.")
    if label_balance and label_balance < 0.55:
        recommendations.append("Balance the dataset; one class has far fewer windows than another.")
    if quality_warnings:
        recommendations.extend(quality_warnings[:4])
    if not recommendations and (train.get("ready") or result.get("ready")):
        recommendations.append("Model is ready for export smoke tests and live latency checks.")
    return {
        "trained": bool(train.get("trained") or result.get("trained")),
        "ready": train.get("ready") if train.get("ready") is not None else result.get("ready"),
        "n_windows": n_windows,
        "labels": labels,
        "per_label": per_label,
        "min_label_windows": min_label_windows,
        "max_label_windows": max_label_windows,
        "label_balance": label_balance,
        "cv_balanced_accuracy": train.get("cv_balanced_accuracy") or result.get("cv_balanced_accuracy"),
        "macro_f1": train.get("macro_f1") or result.get("macro_f1"),
        "readiness_threshold": readiness_threshold,
        "confusion_matrix": train.get("confusion_matrix") or result.get("confusion_matrix"),
        "confusion_labels": labels,
        "per_label_quality": train.get("per_label_quality") or result.get("per_label_quality") or {},
        "quality_warnings": quality_warnings,
        "recommendations": recommendations[:8],
        "dataset_readiness": readiness,
        "dataset_ready": readiness.get("ready"),
        "dataset_score": readiness.get("score"),
    }


def _pipeline_project_report(project: dict) -> str:
    snapshot = dict(project.get("snapshot") or {})
    evaluation = dict(project.get("evaluation") or _pipeline_evaluation_summary(snapshot))
    plan = dict(snapshot.get("plan") or {})
    active_recipe = dict(project.get("active_recipe") or snapshot.get("recipe") or {})
    importers = list(project.get("importer_status") or _pipeline_importer_status())
    dataset = dict(snapshot.get("dataset") or {})
    ingest = dict(snapshot.get("ingest") or {})
    confusion = list(evaluation.get("confusion_matrix") or [])
    confusion_labels = list(evaluation.get("confusion_labels") or evaluation.get("labels") or [])
    per_label_quality = dict(evaluation.get("per_label_quality") or {})
    per_label = dict(evaluation.get("per_label") or {})
    recommendations = list(evaluation.get("recommendations") or [])
    lines = [
        f"# {project.get('name') or plan.get('title') or 'Biosignal Project'}",
        "",
        f"- Project id: {project.get('project_id') or ''}",
        f"- Profile: {project.get('profile') or config.signal_profile_name}",
        f"- Updated: {project.get('updated_at') or ''}",
        f"- Source: {snapshot.get('source') or plan.get('source') or 'live'}",
        f"- Output: {snapshot.get('output') or plan.get('output') or 'live_model'}",
        f"- Dataset path: {snapshot.get('dataset_path') or 'none'}",
        "",
        "## Evaluation",
        f"- Trained: {'yes' if evaluation.get('trained') else 'no'}",
        f"- Ready: {evaluation.get('ready')}",
        f"- Windows: {evaluation.get('n_windows') or 0}",
        f"- Labels: {', '.join(map(str, evaluation.get('labels') or [])) or 'none'}",
        f"- Balanced accuracy: {evaluation.get('cv_balanced_accuracy')}",
        f"- Macro F1: {evaluation.get('macro_f1')}",
        f"- Readiness threshold: {evaluation.get('readiness_threshold')}",
        f"- Label balance: {evaluation.get('label_balance')}",
        f"- Min label windows: {evaluation.get('min_label_windows')}",
        "",
        "## Dataset",
        f"- Readiness score: {evaluation.get('dataset_score')}",
        f"- Dataset ready: {evaluation.get('dataset_ready')}",
        f"- Files scanned: {dataset.get('file_count') or 0}",
        f"- Supported files: {dataset.get('supported_count') or 0}",
        f"- Ingested windows: {ingest.get('added_windows') or 0}",
        f"- Ingest run: {ingest.get('run_dir') or 'none'}",
        "",
        "## Label Counts",
        *[f"- {label}: {count}" for label, count in per_label.items()],
        "",
        "## Per Label Quality",
        *[
            f"- {label}: P {metrics.get('precision')} | R {metrics.get('recall')} | F1 {metrics.get('f1')} | n {metrics.get('support')} | confusion {metrics.get('top_confusion') or 'none'}"
            for label, metrics in per_label_quality.items()
        ],
        "",
        "## Confusion Matrix",
        f"- Labels: {', '.join(map(str, confusion_labels)) or 'none'}",
        *[f"- {confusion_labels[idx] if idx < len(confusion_labels) else idx}: {row}" for idx, row in enumerate(confusion)],
        "",
        "## Recommendations",
        *[f"- {item}" for item in (recommendations or ["No recommendations yet."])],
        "",
        "## Active Recipe",
        f"- Window: {active_recipe.get('window_size_ms') or ((plan.get('windowing') or {}).get('window_size_ms')) or '--'} ms",
        f"- Step: {active_recipe.get('window_step_ms') or ((plan.get('windowing') or {}).get('window_step_ms')) or '--'} ms",
        f"- Model: {active_recipe.get('model_id') or 'auto'}",
        f"- Export: {active_recipe.get('export_target') or ', '.join(plan.get('exports') or []) or 'browser_onnx'}",
        "",
        "## Importers",
        *[f"- {item.get('format')}: {item.get('status')} - {item.get('detail')}" for item in importers],
    ]
    warnings = list(evaluation.get("quality_warnings") or [])
    if warnings:
        lines.extend(["", "## Warnings", *[f"- {item}" for item in warnings[:12]]])
    return "\n".join(lines).strip() + "\n"


def _summarize_pipeline_project(path: Path) -> dict:
    try:
        project = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception as exc:
        return {
            "project_id": path.stem,
            "name": path.stem,
            "ready": False,
            "error": f"{type(exc).__name__}: {exc}",
            "path": path.as_posix(),
        }
    snapshot = dict(project.get("snapshot") or {})
    evaluation = dict(project.get("evaluation") or _pipeline_evaluation_summary(snapshot))
    recipe_versions = list(project.get("recipe_versions") or [])
    export_result = dict(snapshot.get("export_result") or {})
    export_dir = str(export_result.get("export_dir") or ((snapshot.get("runtime") or {}).get("export_dir")) or "")
    return {
        "project_id": project.get("project_id") or path.stem,
        "name": project.get("name") or path.stem,
        "profile": project.get("profile") or "",
        "created_at": project.get("created_at") or "",
        "updated_at": project.get("updated_at") or "",
        "path": path.as_posix(),
        "source": snapshot.get("source") or "",
        "output": snapshot.get("output") or "",
        "dataset_path": snapshot.get("dataset_path") or "",
        "ready": bool(evaluation.get("ready")) if evaluation.get("ready") is not None else bool(evaluation.get("trained")),
        "trained": bool(evaluation.get("trained")),
        "n_windows": int(evaluation.get("n_windows") or 0),
        "labels": list(evaluation.get("labels") or []),
        "cv_balanced_accuracy": evaluation.get("cv_balanced_accuracy"),
        "macro_f1": evaluation.get("macro_f1"),
        "recipe_versions": len(recipe_versions),
        "report_path": project.get("report_path") or "",
        "export_dir": export_dir,
        "package_path": export_dir,
        "importer_status": project.get("importer_status") or _pipeline_importer_status(),
    }


def _pipeline_list_projects() -> dict:
    root = _pipeline_project_root()
    root.mkdir(parents=True, exist_ok=True)
    projects = [
        _summarize_pipeline_project(path)
        for path in sorted(root.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:80]
    ]
    return {
        "ok": True,
        "projects": projects,
        "project_dir": root.as_posix(),
        "rebuild_job_dir": _pipeline_rebuild_job_root().as_posix(),
        "rebuild_jobs": _pipeline_list_rebuild_jobs(limit=40),
        "importer_status": _pipeline_importer_status(),
    }


def _pipeline_save_project(req: PipelineProjectRequest) -> dict:
    snapshot = dict(req.snapshot or {})
    if not snapshot:
        raise ValueError("Project snapshot is required.")
    prompt = str(snapshot.get("prompt") or "").strip()
    plan = dict(snapshot.get("plan") or {})
    name = str(req.name or snapshot.get("name") or plan.get("title") or prompt[:80] or "Biosignal project").strip()
    project_id = _safe_stem(req.project_id or snapshot.get("project_id") or name, "project")
    root = _pipeline_project_root()
    root.mkdir(parents=True, exist_ok=True)
    path = _pipeline_project_path(project_id, must_exist=False)
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    previous: Dict[str, Any] = {}
    if path.exists():
        try:
            previous = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            previous = {}
    recipe = dict(snapshot.get("recipe") or snapshot.get("recipeDraft") or {})
    recipe_versions = list(previous.get("recipe_versions") or [])
    if recipe:
        recipe_versions.append({
            "version": len(recipe_versions) + 1,
            "saved_at": now,
            "recipe": recipe,
            "plan_title": plan.get("title") or "",
        })
        recipe_versions = recipe_versions[-30:]
    evaluation = _pipeline_evaluation_summary(snapshot)
    project = {
        "project_id": project_id,
        "name": name,
        "profile": snapshot.get("profile") or config.signal_profile_name,
        "created_at": previous.get("created_at") or now,
        "updated_at": now,
        "snapshot": _json_safe(snapshot),
        "active_recipe": _json_safe(recipe),
        "recipe_versions": _json_safe(recipe_versions),
        "evaluation": _json_safe(evaluation),
        "importer_status": _pipeline_importer_status(),
    }
    report = _pipeline_project_report(project)
    report_path = root / f"{project_id}.md"
    report_path.write_text(report, encoding="utf-8")
    project["report"] = report
    project["report_path"] = report_path.as_posix()
    _write_pipeline_json(path, project)
    return {"ok": True, "project": project, "summary": _summarize_pipeline_project(path), **_pipeline_list_projects()}


def _pipeline_load_project(req: PipelineProjectRequest) -> dict:
    path = _pipeline_project_path(req.project_id)
    project = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    return {"ok": True, "project": project, "summary": _summarize_pipeline_project(path)}


def _pipeline_project_report_file(project_id: str) -> Path:
    path = _pipeline_project_path(project_id, must_exist=True).with_suffix(".md").resolve()
    root = _pipeline_project_root()
    if not path.exists() or root not in path.parents:
        raise ValueError(f"Project report does not exist: {project_id}")
    return path


def _pipeline_rebuild_report_file(job_id: str) -> Path:
    job = _pipeline_get_project_rebuild(job_id).get("job") or {}
    raw = str((job.get("result") or {}).get("report_path") or "")
    if not raw:
        raise ValueError("Rebuild job has no report path.")
    path = Path(raw).resolve()
    root = _pipeline_project_root()
    if not path.exists() or root not in path.parents:
        raise ValueError("Rebuild report path is unavailable.")
    return path


def _pipeline_delete_project(req: PipelineProjectRequest) -> dict:
    path = _pipeline_project_path(req.project_id)
    deleted = path.as_posix()
    report = path.with_suffix(".md")
    path.unlink()
    if report.exists():
        report.unlink()
    return {"ok": True, "deleted": deleted, **_pipeline_list_projects()}


def _utc_job_ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _pipeline_rebuild_job_root() -> Path:
    return (Path(config.data_dir) / "pipeline_rebuild_jobs").resolve()


def _pipeline_rebuild_job_path(job_id: str, *, must_exist: bool = False) -> Path:
    safe_id = _safe_stem(job_id, "")
    if not safe_id:
        raise ValueError("Rebuild job id is required.")
    root = _pipeline_rebuild_job_root()
    path = (root / f"{safe_id}.json").resolve()
    if root != path.parent:
        raise ValueError("Rebuild job id resolves outside the job registry.")
    if must_exist and not path.exists():
        raise ValueError(f"Rebuild job does not exist: {job_id}")
    return path


def _pipeline_rebuild_job_snapshot(job: dict) -> dict:
    public = dict(job or {})
    public.pop("_thread", None)
    return _json_safe(public)


def _pipeline_persist_rebuild_job(job: dict) -> None:
    if not job or not job.get("job_id"):
        return
    root = _pipeline_rebuild_job_root()
    root.mkdir(parents=True, exist_ok=True)
    _write_pipeline_json(_pipeline_rebuild_job_path(str(job.get("job_id")), must_exist=False), _pipeline_rebuild_job_snapshot(job))


def _pipeline_load_rebuild_job(job_id: str) -> dict:
    path = _pipeline_rebuild_job_path(job_id, must_exist=True)
    job = json.loads(path.read_text(encoding="utf-8-sig", errors="ignore"))
    if str(job.get("status") or "") in {"queued", "running"}:
        job["status"] = "interrupted"
        job["error"] = job.get("error") or "Server restarted before this rebuild completed."
        job["updated_at"] = _utc_job_ts()
        job.setdefault("steps", []).append({
            "step": "Interrupted",
            "status": "interrupted",
            "detail": "Server restarted before this rebuild completed.",
            "at": job["updated_at"],
        })
        _pipeline_persist_rebuild_job(job)
    app_state.pipeline_rebuild_jobs[str(job.get("job_id") or path.stem)] = job
    return job


def _pipeline_list_rebuild_jobs(project_id: str = "", limit: int = 40) -> List[dict]:
    root = _pipeline_rebuild_job_root()
    root.mkdir(parents=True, exist_ok=True)
    out: List[dict] = []
    for path in sorted(root.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[: max(limit * 2, limit)]:
        try:
            job = _pipeline_load_rebuild_job(path.stem)
            if project_id and str(job.get("project_id") or "") != project_id:
                continue
            out.append(_pipeline_rebuild_job_snapshot(job))
            if len(out) >= limit:
                break
        except Exception:
            continue
    return out


def _pipeline_rebuild_job_update(job_id: str, *, status: Optional[str] = None, step: str = "", detail: str = "", data: Optional[dict] = None) -> None:
    job = app_state.pipeline_rebuild_jobs.get(job_id)
    if not job:
        return
    now = _utc_job_ts()
    if status:
        job["status"] = status
    job["updated_at"] = now
    if step:
        entry = {
            "step": step,
            "status": status or job.get("status") or "running",
            "detail": detail,
            "at": now,
        }
        if data:
            entry["data"] = _json_safe(data)
        job.setdefault("steps", []).append(entry)
        job["current_step"] = step
    _pipeline_persist_rebuild_job(job)


def _pipeline_action_from_snapshot(snapshot: dict, plan: dict, **overrides: Any) -> PipelineActionRequest:
    context = dict(snapshot.get("context") or {})
    context.update({
        "dataset_path": snapshot.get("dataset_path") or context.get("dataset_path") or "",
        "dataset_mode": str(snapshot.get("source") or "").lower() == "dataset",
        "rebuild_job": True,
    })
    payload = {
        "plan": plan or {},
        "qa": snapshot.get("qa") or {},
        "train_result": snapshot.get("train_result") or {},
        "acquisition": snapshot.get("acquisition") or {},
        "acquisition_run": snapshot.get("acquisition_run") or {},
        "labels": snapshot.get("labels") or {},
        "dataset": snapshot.get("dataset") or {},
        "throughput": snapshot.get("throughput") or {},
        "context": context,
    }
    payload.update(overrides)
    return PipelineActionRequest(**payload)


def _pipeline_run_optional_step(job_id: str, label: str, fn, *args, **kwargs) -> Optional[dict]:
    _pipeline_rebuild_job_update(job_id, status="running", step=label)
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        _pipeline_rebuild_job_update(
            job_id,
            status="running",
            step=f"{label} skipped",
            detail=f"{type(exc).__name__}: {exc}",
        )
        return None


def _pipeline_project_rebuild_worker(job_id: str, project_id: str) -> None:
    try:
        _pipeline_rebuild_job_update(job_id, status="running", step="Loading project")
        loaded = _pipeline_load_project(PipelineProjectRequest(project_id=project_id))
        project = dict(loaded.get("project") or {})
        snapshot = dict(project.get("snapshot") or {})
        if not snapshot:
            raise ValueError("Project snapshot is empty.")

        plan = dict(snapshot.get("plan") or {})
        if not plan:
            _pipeline_rebuild_job_update(job_id, status="running", step="Generating plan")
            plan = _pipeline_plan(PipelinePlanRequest(
                prompt=str(snapshot.get("prompt") or ""),
                source=str(snapshot.get("source") or "live"),
                output=str(snapshot.get("output") or "live_model"),
                signal_profile=str(snapshot.get("profile") or config.signal_profile_name),
                channel_labels=list(((snapshot.get("saved_from") or {}).get("channel_labels") or config.channel_labels)[: config.n_channels]),
                context={
                    "dataset_path": snapshot.get("dataset_path") or "",
                    "dataset_mode": str(snapshot.get("source") or "").lower() == "dataset",
                },
            ))
            snapshot["plan"] = plan

        source = str(snapshot.get("source") or plan.get("source") or "").lower()
        dataset_path = str(snapshot.get("dataset_path") or "").strip()
        schema_mapping = dict(snapshot.get("schema_mapping") or {})

        if source == "dataset" or dataset_path:
            if not dataset_path:
                raise ValueError("Dataset rebuild requires a dataset path.")
            _pipeline_rebuild_job_update(job_id, status="running", step="Scanning dataset")
            dataset_req = PipelineDatasetInspectRequest(
                **_pipeline_action_from_snapshot(snapshot, plan).dict(),
                path=dataset_path,
                schema_mapping=schema_mapping,
            )
            dataset = _pipeline_dataset_inspect(dataset_req).get("dataset") or {}
            snapshot["dataset"] = dataset
            readiness = dict(dataset.get("readiness") or {})
            if not readiness.get("ready"):
                gaps = list(readiness.get("gaps") or [])
                raise ValueError(gaps[0] if gaps else "Dataset is not ready for supervised rebuild.")

            _pipeline_rebuild_job_update(job_id, status="running", step="Ingesting dataset and training")
            ingest_req = PipelineDatasetIngestRequest(
                **_pipeline_action_from_snapshot(snapshot, plan).dict(),
                path=dataset_path,
                schema_mapping=schema_mapping,
                append=False,
                train_after=True,
                max_windows=int((snapshot.get("ingest_options") or {}).get("max_windows") or 240),
                max_windows_per_label=int((snapshot.get("ingest_options") or {}).get("max_windows_per_label") or 80),
            )
            ingest_out = _pipeline_dataset_ingest(ingest_req)
            snapshot["ingest"] = ingest_out.get("ingest") or {}
            snapshot["acquisition_run"] = ingest_out.get("run") or snapshot.get("acquisition_run") or {}
            if ingest_out.get("training"):
                snapshot["train_result"] = ingest_out.get("training")
            if not dict(snapshot.get("train_result") or {}).get("trained"):
                raise ValueError("Dataset rebuild did not produce a trained model.")
        else:
            _pipeline_rebuild_job_update(job_id, status="running", step="Running signal QA")
            qa = _pipeline_qa(PipelineQARequest(plan=plan, context=dict(snapshot.get("context") or {}))).get("qa") or {}
            snapshot["qa"] = qa

        action = _pipeline_action_from_snapshot(snapshot, plan)
        _pipeline_rebuild_job_update(job_id, status="running", step="Comparing models")
        snapshot["comparison"] = _pipeline_model_compare(action).get("comparison") or {}

        embedding = _pipeline_run_optional_step(job_id, "Refreshing embeddings", _pipeline_embedding_timeline, _pipeline_action_from_snapshot(snapshot, plan))
        if embedding:
            snapshot["embedding"] = embedding.get("embedding") or {}

        throughput = _pipeline_run_optional_step(
            job_id,
            "Benchmarking throughput",
            _pipeline_throughput_benchmark,
            PipelineThroughputRequest(**_pipeline_action_from_snapshot(snapshot, plan).dict(), quantization_step=0.02, max_workers=4),
        )
        if throughput:
            snapshot["throughput"] = throughput.get("throughput") or {}

        _pipeline_rebuild_job_update(job_id, status="running", step="Packaging export")
        export = _pipeline_export(_pipeline_action_from_snapshot(snapshot, plan))
        snapshot["export_result"] = export.get("export") or {}

        runtime = _pipeline_run_optional_step(job_id, "Checking browser runtime", _pipeline_browser_runtime, _pipeline_action_from_snapshot(snapshot, plan))
        if runtime:
            snapshot["runtime"] = runtime.get("runtime") or {}
        deploy = _pipeline_run_optional_step(job_id, "Running deploy smoke", _pipeline_deploy_smoke, _pipeline_action_from_snapshot(snapshot, plan))
        if deploy:
            snapshot["deploy"] = deploy.get("deploy") or {}
            if deploy.get("runtime"):
                snapshot["runtime"] = deploy.get("runtime")

        _pipeline_rebuild_job_update(job_id, status="running", step="Saving project report")
        saved = _pipeline_save_project(PipelineProjectRequest(
            project_id=project_id,
            name=str(project.get("name") or snapshot.get("name") or plan.get("title") or project_id),
            snapshot=snapshot,
        ))
        job = app_state.pipeline_rebuild_jobs[job_id]
        job["result"] = {
            "project": _summarize_pipeline_project(_pipeline_project_path(project_id)),
            "evaluation": (saved.get("project") or {}).get("evaluation") or {},
            "report_path": (saved.get("project") or {}).get("report_path") or "",
            "export_dir": str((snapshot.get("export_result") or {}).get("export_dir") or ""),
        }
        _pipeline_rebuild_job_update(job_id, status="completed", step="Rebuild completed")
    except Exception as exc:
        job = app_state.pipeline_rebuild_jobs.get(job_id)
        if job is not None:
            job["error"] = f"{type(exc).__name__}: {exc}"
        _pipeline_rebuild_job_update(job_id, status="failed", step="Rebuild failed", detail=f"{type(exc).__name__}: {exc}")


def _pipeline_start_project_rebuild(req: PipelineProjectRequest, background_tasks: BackgroundTasks) -> dict:
    project_id = _safe_stem(req.project_id, "")
    if not project_id:
        raise ValueError("Project id is required.")
    _pipeline_project_path(project_id)
    stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    job_id = f"{project_id}_{stamp}"
    suffix = 1
    while _pipeline_rebuild_job_path(job_id, must_exist=False).exists() or job_id in app_state.pipeline_rebuild_jobs:
        suffix += 1
        job_id = f"{project_id}_{stamp}_{suffix}"
    app_state.pipeline_rebuild_jobs[job_id] = {
        "job_id": job_id,
        "project_id": project_id,
        "status": "queued",
        "created_at": _utc_job_ts(),
        "updated_at": _utc_job_ts(),
        "current_step": "Queued",
        "steps": [],
        "result": {},
        "error": "",
    }
    _pipeline_persist_rebuild_job(app_state.pipeline_rebuild_jobs[job_id])
    background_tasks.add_task(_pipeline_project_rebuild_worker, job_id, project_id)
    return {"ok": True, "job": _pipeline_rebuild_job_snapshot(app_state.pipeline_rebuild_jobs[job_id])}


def _pipeline_get_project_rebuild(job_id: str) -> dict:
    job = app_state.pipeline_rebuild_jobs.get(str(job_id or ""))
    if not job:
        job = _pipeline_load_rebuild_job(job_id)
    return {"ok": True, "job": _pipeline_rebuild_job_snapshot(job)}


def _pipeline_delete_project_rebuild(req: PipelineProjectRequest) -> dict:
    job_id = str(req.project_id or "").strip()
    path = _pipeline_rebuild_job_path(job_id, must_exist=True)
    job = app_state.pipeline_rebuild_jobs.get(path.stem) or json.loads(path.read_text(encoding="utf-8-sig", errors="ignore"))
    if str(job.get("status") or "") in {"queued", "running"}:
        raise ValueError("Running rebuild jobs cannot be deleted yet.")
    path.unlink()
    app_state.pipeline_rebuild_jobs.pop(path.stem, None)
    return {"ok": True, "deleted": path.stem, "rebuild_jobs": _pipeline_list_rebuild_jobs(limit=40)}


def _rebuild_pipeline() -> None:
    app_state.pipeline = BiosignalPipeline()
    app_state.pipeline.add_prediction_callback(_on_prediction)
    app_state.last_prediction = None
    app_state.diagnostics.reset(full_scale=config.signal_profile.display_full_scale)
    _pipeline_load_prompt_model()


def _sync_lsl_outlets() -> None:
    if not app_state.lsl.is_active:
        return
    if not app_state.lsl.reconfigure(
        profile=config.signal_profile,
        channel_labels=config.channel_labels,
        sample_rate=config.sample_rate,
        stream_source=app_state.stream_source,
        ):
        logger.warning("LSL reconfigure failed: %s", app_state.lsl.last_error)


def _sync_stream_filter() -> None:
    if not app_state.stream:
        return
    app_state.stream.set_custom_filter(
        app_state.filter_lab.get_active_runtime_filter(config.signal_profile_name)
    )


def _arduino_connected() -> bool:
    return bool(app_state.arduino and app_state.arduino.is_connected)


def _control_output_ready() -> bool:
    return _arduino_connected() or app_state.osc.is_active


def _hardware_stream_preview_ready() -> bool:
    return bool(
        app_state.stream
        and app_state.stream.is_running
        and app_state.stream_source == "hardware"
    )


def _dispatch_move(joint_id: int, angle: int) -> bool:
    if app_state.system_state == SystemState.ESTOP:
        return False
    ok = False
    if _arduino_connected():
        ok = app_state.arduino.move(joint_id, angle) or ok
    if app_state.osc.is_active:
        ok = app_state.osc.send_move(joint_id, angle) or ok
    return ok


def _dispatch_gesture(name: str) -> bool:
    if app_state.system_state == SystemState.ESTOP:
        return False
    ok = False
    if _arduino_connected() and not app_state.arduino.estop_active:
        ok = app_state.arduino.execute_gesture(name) or ok
    if app_state.osc.is_active:
        ok = app_state.osc.send_gesture(name, profile=config.signal_profile_name) or ok
    return ok


def _dispatch_digital_write(pin: int, value: int) -> bool:
    if app_state.system_state == SystemState.ESTOP:
        return False
    ok = False
    if _arduino_connected():
        ok = app_state.arduino.digital_write(pin, value) or ok
    if app_state.osc.is_active:
        ok = app_state.osc.send_digital_write(pin, value) or ok
    return ok


def _dispatch_analog_write(pin: int, value: int) -> bool:
    if app_state.system_state == SystemState.ESTOP:
        return False
    ok = False
    if _arduino_connected():
        ok = app_state.arduino.analog_write(pin, value) or ok
    if app_state.osc.is_active:
        ok = app_state.osc.send_analog_write(pin, value) or ok
    return ok


def _resolve_stream_source(req: Optional[ConnectRequest]) -> str:
    if req and req.source:
        source = req.source.strip().lower()
    elif req and req.use_synthetic:
        source = "synthetic"
    else:
        source = "synthetic" if config.board_id == -1 else "hardware"

    if source not in {"hardware", "synthetic", "playback", "lsl"}:
        raise HTTPException(400, "Unknown stream source. Valid: hardware, synthetic, playback, lsl")
    return source


def _make_stream(
    source: str,
    playback_session_id: Optional[str] = None,
    playback_rate: float = 1.0,
    synthetic_scenario: Optional[str] = None,
    lsl_stream_name: Optional[str] = None,
    lsl_source_id: Optional[str] = None,
) -> CytonStream:
    if source == "synthetic":
        stream: CytonStream = SimulatedCytonStream(scenario=synthetic_scenario or "clean")
    elif source == "playback":
        stream = PlaybackCytonStream(
            session_id=playback_session_id,
            playback_rate=playback_rate,
        )
    elif source == "lsl":
        stream = LSLInletStream(
            stream_name=lsl_stream_name,
            source_id=lsl_source_id,
        )
    else:
        stream = CytonStream()
    stream.add_window_callback(_on_window_pipeline)
    stream.add_window_callback(_on_window_vis)
    stream.add_chunk_callback(app_state.lsl.push_chunk)
    app_state.stream_source = source
    app_state.playback_session_id = playback_session_id if source == "playback" else None
    stream.set_custom_filter(
        app_state.filter_lab.get_active_runtime_filter(config.signal_profile_name)
    )
    return stream


def _set_signal_profile(name: str) -> dict:
    profile = config.set_signal_profile(name)
    app_state.calibration.reset()
    _rebuild_pipeline()
    app_state.last_vis_window = None
    app_state.last_vis_timestamp = 0.0
    app_state.last_diagnostics_broadcast_ts = 0.0
    app_state.last_ml_broadcast_ts = 0.0
    if app_state.pipeline.load_model():
        logger.info("Loaded saved %s model for profile switch", profile.display_name)
    playback_rate = float(app_state.stream_details.get("playback_rate", 1.0))
    synthetic_scenario = str(app_state.stream_details.get("scenario") or "clean")
    app_state.stream = _make_stream(
        app_state.stream_source,
        playback_session_id=app_state.playback_session_id,
        playback_rate=playback_rate,
        synthetic_scenario=synthetic_scenario,
        lsl_stream_name=str(app_state.stream_details.get("stream_name") or app_state.stream_details.get("name") or ""),
        lsl_source_id=str(app_state.stream_details.get("source_id") or ""),
    )
    _sync_lsl_outlets()
    return profile.to_dict()


def _attach_recorder_callback() -> None:
    if app_state.stream:
        app_state.stream.add_chunk_callback(app_state.recorder_chunk_callback)


def _detach_recorder_callback() -> None:
    if app_state.stream:
        app_state.stream.remove_chunk_callback(app_state.recorder_chunk_callback)


async def _broadcast_worker() -> None:
    q = app_state._bcast_queue
    while True:
        msg = await q.get()
        dead: Set[WebSocket] = set()
        text = json.dumps(msg, separators=(",", ":"))
        for ws in list(app_state.websockets):
            try:
                await ws.send_text(text)
            except Exception:
                dead.add(ws)
        app_state.websockets -= dead
        q.task_done()


async def _watchdog_worker() -> None:
    while True:
        await asyncio.sleep(0.25)
        stream_running = bool(app_state.stream and app_state.stream.is_running)
        tripped = app_state.watchdog.check(stream_running=stream_running)
        if (
            tripped
            and app_state.watchdog.auto_estop_on_stale
            and _control_output_ready()
            and config.signal_profile.robotic_arm_supported
            and app_state.system_state != SystemState.ESTOP
        ):
            _watchdog_trip(app_state.watchdog.last_trip_reason or "signal_timeout")

        now = time.time()
        if tripped or (now - app_state.last_safety_broadcast_ts) >= 1.0:
            app_state.last_safety_broadcast_ts = now
            app_state.queue_broadcast(
                {
                    "type": "safety",
                    "data": app_state.watchdog.status(stream_running=stream_running),
                    "timestamp": now,
                }
            )


def _watchdog_trip(reason: str) -> None:
    payload = {
        "reason": reason,
        "profile": config.signal_profile_name,
        "source": app_state.stream_source,
    }
    if app_state.arduino:
        app_state.arduino.estop()
    app_state.osc.send_estop()
    app_state.set_state(SystemState.ESTOP)
    if app_state.recorder.is_recording:
        app_state.recorder.log_event("watchdog_trip", payload)
    app_state.lsl.push_marker("watchdog_trip", payload)
    app_state.queue_broadcast(
        {
            "type": "safety",
            "data": app_state.watchdog.status(
                stream_running=bool(app_state.stream and app_state.stream.is_running)
            ),
            "timestamp": time.time(),
        }
    )


def _on_window_pipeline(window: np.ndarray) -> None:
    started_at = app_state.diagnostics.mark_pipeline_start()
    app_state.pipeline.on_window(window)
    app_state.diagnostics.mark_pipeline_end(started_at)
    app_state.watchdog.note_signal()


def _on_prediction(result: dict, window: np.ndarray) -> None:
    label = result.get("label", "--")
    confidence = float(result.get("confidence", 0.0))
    class_idx = int(result.get("class_idx", 0))
    metrics = result.get("metrics", {})
    summary = result.get("summary", "")
    arousal = result.get("arousal")

    app_state.last_prediction = {
        "label": label,
        "gesture": label,  # compatibility with the current dashboard
        "class_idx": class_idx,
        "confidence": confidence,
        "summary": summary,
        "metrics": metrics,
        "arousal": arousal,
        "profile": config.signal_profile_name,
    }

    if (
        _control_output_ready()
        and config.signal_profile.robotic_arm_supported
        and app_state.system_state != SystemState.ESTOP
        and confidence >= config.prediction_confidence_threshold
    ):
        _dispatch_gesture(label)

    if app_state.recorder.is_recording:
        app_state.recorder.log_event(
            "prediction",
            {
                "label": label,
                "confidence": round(confidence, 3),
                "profile": config.signal_profile_name,
                "summary": summary,
                "metrics": metrics,
            },
        )

    app_state.lsl.push_prediction_marker(
        label=label,
        confidence=confidence,
        profile=config.signal_profile_name,
        summary=summary,
    )
    app_state.osc.send_prediction(
        label=label,
        confidence=confidence,
        profile=config.signal_profile_name,
        summary=summary,
    )

    app_state.queue_broadcast(
        {
            "type": "prediction",
            "data": {
                "label": label,
                "gesture": label,
                "class_idx": class_idx,
                "confidence": round(confidence, 3),
                "summary": summary,
                "metrics": metrics,
                "rms": [
                    round(float(np.sqrt(np.mean(window[i] ** 2))), 5)
                    for i in range(window.shape[0])
                ],
                "profile": config.signal_profile_name,
            },
            "timestamp": time.time(),
        }
    )


def _on_window_vis(window: np.ndarray) -> None:
    inc = config.window_increment_samples
    new_data = window[:, -inc:]
    app_state.last_vis_window = window.copy()
    app_state.last_vis_timestamp = time.time()
    app_state.watchdog.note_signal()
    recent = None
    if app_state.stream:
        recent = app_state.stream.get_latest_samples(min(config.sample_rate * 2, config.sample_rate))
    diagnostics = app_state.diagnostics.on_window(window, recent_samples=recent)
    _pipeline_update_acquisition(window, diagnostics, app_state.last_vis_timestamp)
    rms = [
        round(float(np.sqrt(np.mean(window[i] ** 2))), 5)
        for i in range(window.shape[0])
    ]
    quality = [
        round(float(v), 3)
        for v in app_state.pipeline.get_channel_quality(window)
    ]
    app_state.last_live_metrics = {
        "timestamp": app_state.last_vis_timestamp,
        "rms": rms,
        "quality": quality,
        "n_channels": int(window.shape[0]),
        "profile": config.signal_profile_name,
        "units": config.signal_profile.units,
    }
    app_state.queue_broadcast(
        {
            "type": "emg",  # kept for frontend compatibility
            "data": {
                "channels": new_data.tolist(),
                "rms": rms,
                "quality": quality,
                "n_channels": window.shape[0],
                "profile": config.signal_profile_name,
                "units": config.signal_profile.units,
            },
            "timestamp": time.time(),
        }
    )
    now = time.time()
    if now - app_state.pipeline_acquisition_last_predict_ts >= 0.25:
        prompt_prediction = _pipeline_prompt_model_predict(window, diagnostics)
        if prompt_prediction:
            app_state.pipeline_acquisition_last_predict_ts = now
            app_state.queue_broadcast(
                {
                    "type": "prompt_prediction",
                    "data": prompt_prediction,
                    "timestamp": now,
                }
            )
    if now - app_state.last_diagnostics_broadcast_ts >= 0.35:
        app_state.last_diagnostics_broadcast_ts = now
        app_state.queue_broadcast(
            {
                "type": "diagnostics",
                "data": {
                    **diagnostics,
                    "active_filter": app_state.filter_lab.status(config.signal_profile_name).get("active_filter"),
                },
                "timestamp": now,
            }
        )
    # ── Signal intelligence broadcasts (throttled 4 Hz) ───────────
    if now - app_state.last_confidence_broadcast_ts >= 0.25:
        app_state.last_confidence_broadcast_ts = now
        try:
            conf = app_state.signal_confidence.update(window)
            trust_scores = [m.trust_score for m in conf.channels]
            app_state.signal_recovery.process(window, trust_scores=trust_scores)
            app_state.queue_broadcast(
                {"type": "signal_confidence", "data": app_state.signal_confidence.to_dict(), "timestamp": now}
            )
        except Exception as exc:
            logger.debug("Signal confidence update failed: %s", exc)
    if now - app_state.last_recovery_broadcast_ts >= 0.5:
        app_state.last_recovery_broadcast_ts = now
        try:
            app_state.queue_broadcast(
                {"type": "signal_recovery", "data": app_state.signal_recovery.to_dict(), "timestamp": now}
            )
        except Exception as exc:
            logger.debug("Signal recovery broadcast failed: %s", exc)
    if now - app_state.last_ml_broadcast_ts >= 1.0:
        app_state.last_ml_broadcast_ts = now
        try:
            ml_start = time.perf_counter()
            local_ml = app_state.ai.local_models.analyze(_live_ml_snapshot(window, diagnostics))
            inference_ms = round((time.perf_counter() - ml_start) * 1000.0, 3)
            embedding_count = len(list(local_ml.get("foundation_embeddings") or [])) if isinstance(local_ml, dict) else 0
            output_count = len(list(local_ml.get("outputs") or [])) if isinstance(local_ml, dict) else 0
            _append_pipeline_embedding_snapshot(local_ml, now)
            app_state.queue_broadcast(
                {
                    "type": "ml_insights",
                    "data": {
                        "profile": config.signal_profile_name,
                        "sample_rate_hz": config.sample_rate,
                        "window_samples": int(window.shape[1]) if window.ndim == 2 else 0,
                        "channel_count": int(window.shape[0]) if window.ndim == 2 else 0,
                        "inference_ms": inference_ms,
                        "foundation_embedding_count": embedding_count,
                        "model_output_count": output_count,
                        "stats": _window_stats_for_ai(window),
                        "local_model_insights": local_ml,
                    },
                    "timestamp": now,
                }
            )
        except Exception as exc:
            logger.warning("Live ML insight pass failed: %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app_state._loop = asyncio.get_running_loop()
    app_state._bcast_queue = asyncio.Queue(maxsize=512)
    asyncio.create_task(_broadcast_worker())
    asyncio.create_task(_watchdog_worker())

    _rebuild_pipeline()
    app_state.stream = _make_stream("synthetic" if config.board_id == -1 else "hardware")
    app_state.arduino = ArduinoBridge()
    app_state.arduino.connect()

    app_state.calibration.set_status_callback(
        lambda data: app_state.queue_broadcast(
            {"type": "calibration", "data": data, "timestamp": time.time()}
        )
    )

    # ── Guided recording status callback ──────────────────────────
    app_state.guided_recording.set_status_callback(
        lambda data: app_state.queue_broadcast(
            {**data, "timestamp": time.time()}
        )
    )

    if app_state.pipeline.load_model():
        logger.info("Pre-trained model loaded")

    logger.info("KYMA Server ready - http://%s:%s", config.host, config.server_port)
    yield

    if app_state.recorder.is_recording:
        _detach_recorder_callback()
        app_state.recorder.stop_session()
    if app_state.stream and app_state.stream.is_running:
        app_state.stream.stop()
    app_state.lsl.stop()
    app_state.osc.stop()
    if app_state.arduino:
        app_state.arduino.disconnect()
    logger.info("Server stopped")


app = FastAPI(title="KYMA", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

dashboard_dir = os.path.join(os.path.dirname(__file__), "..", "dashboard")
if os.path.isdir(dashboard_dir):
    app.mount("/static", StaticFiles(directory=dashboard_dir), name="static")


@app.get("/", include_in_schema=False)
async def root():
    path = os.path.join(dashboard_dir, "index.html")
    if os.path.exists(path):
        return FileResponse(path)
    return {"message": "KYMA API. See /docs for endpoints."}


@app.get("/api/status")
async def get_status():
    return {
        "state": app_state.system_state.value,
        "stream_running": app_state.stream.is_running if app_state.stream else False,
        "stream_source": app_state.stream_source,
        "stream_details": app_state.stream_details,
        "playback_session_id": app_state.playback_session_id,
        "lsl": app_state.lsl.status(),
        "osc": app_state.osc.status(),
        "xdf_available": app_state.xdf.available,
        "arduino_connected": app_state.arduino.is_connected if app_state.arduino else False,
        "arduino_estop": app_state.arduino.estop_active if app_state.arduino else False,
        "model_trained": app_state.pipeline.is_trained,
        "prompt_model": _pipeline_prompt_model_summary(),
        "decoder_mode": app_state.pipeline.classifier_name,
        "live_decoder_ready": True,
        "is_recording": app_state.recorder.is_recording,
        "session_id": app_state.recorder.session_id,
        "last_prediction": app_state.last_prediction,
        "gestures": config.class_labels,
        "class_labels": config.class_labels,
        "signal_profile": config.signal_profile.to_dict(),
        "eeg_brain": app_state.eeg_brain.status(
            profile_key=config.signal_profile_name,
            has_window=app_state.last_vis_window is not None,
        ),
        "protocol_templates": _protocol_payload(),
        "eeg_experiments": _eeg_experiment_payload(),
        "calibration": app_state.calibration.status(),
        "diagnostics": app_state.diagnostics.status(),
        "safety": app_state.watchdog.status(
            stream_running=bool(app_state.stream and app_state.stream.is_running)
        ),
        "filter_lab": app_state.filter_lab.status(config.signal_profile_name),
        "workshop": app_state.workshop.status(),
        "firmware": app_state.firmware.status(),
        "ai": app_state.ai.status(),
    }


@app.get("/api/config")
async def get_config():
    return {
        "serial_port": config.serial_port,
        "arduino_port": config.arduino_port,
        "sample_rate": config.sample_rate,
        "n_channels": config.n_channels,
        "signal_channels": config.signal_channels,
        "emg_channels": config.signal_channels,
        "channel_labels": config.channel_labels,
        "class_labels": config.class_labels,
        "gestures": config.class_labels,
        "features": config.features,
        "window_size_ms": config.window_size_ms,
        "window_increment_ms": config.window_increment_ms,
        "confidence_threshold": config.prediction_confidence_threshold,
        "supports_synthetic": True,
        "available_stream_sources": ["hardware", "synthetic", "playback", "lsl"],
        "supports_lsl": app_state.lsl.available,
        "supports_osc": app_state.osc.available,
        "supports_xdf_import": app_state.xdf.available,
        "decoder_mode": app_state.pipeline.classifier_name,
        "prompt_model": _pipeline_prompt_model_summary(),
        "signal_profile": config.signal_profile.to_dict(),
        "osc": app_state.osc.status(),
        "eeg_brain": app_state.eeg_brain.status(
            profile_key=config.signal_profile_name,
            has_window=app_state.last_vis_window is not None,
        ),
        "available_profiles": list_profile_dicts(),
        "protocol_templates": _protocol_payload(),
        "eeg_experiments": _eeg_experiment_payload(),
        "calibration_protocol": app_state.calibration.describe_protocol(),
        "diagnostics": app_state.diagnostics.status(),
        "safety": app_state.watchdog.status(
            stream_running=bool(app_state.stream and app_state.stream.is_running)
        ),
        "filter_lab": app_state.filter_lab.status(config.signal_profile_name),
        "workshop": app_state.workshop.status(),
        "firmware": app_state.firmware.status(),
        "ai": app_state.ai.status(),
    }


@app.get("/api/debug/live_signal")
async def debug_live_signal():
    return {
        "stream_running": bool(app_state.stream and app_state.stream.is_running),
        "stream_source": app_state.stream_source,
        "stream_details": app_state.stream_details,
        "signal_age_ms": round(max(0.0, (time.time() - app_state.last_vis_timestamp) * 1000.0), 3)
        if app_state.last_vis_timestamp
        else None,
        "websocket_clients": len(app_state.websockets),
        "broadcast_queue_size": app_state._bcast_queue.qsize() if app_state._bcast_queue else 0,
        "broadcast_queue_max": app_state._bcast_queue.maxsize if app_state._bcast_queue else 0,
        "broadcast_dropped": app_state.broadcast_dropped,
        "live_metrics": dict(app_state.last_live_metrics or {}),
    }


@app.get("/api/eeg/experiments")
async def list_eeg_experiments():
    return _eeg_experiment_payload()


@app.get("/api/filterlab/status")
async def filterlab_status():
    return app_state.filter_lab.status(config.signal_profile_name)


@app.get("/api/workshop/status")
async def workshop_status():
    return app_state.workshop.status()


@app.post("/api/workshop/analyze")
async def workshop_analyze(req: WorkshopAnalyzeRequest):
    try:
        return app_state.workshop.analyze(req.dict())
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except RuntimeError as exc:
        raise HTTPException(503, str(exc))


@app.post("/api/workshop/save")
async def workshop_save(req: WorkshopSaveRequest):
    try:
        return app_state.workshop.save(req.dict())
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except RuntimeError as exc:
        raise HTTPException(503, str(exc))


@app.post("/api/workshop/export/matlab")
async def workshop_export_matlab(req: WorkshopSaveRequest):
    try:
        return app_state.workshop.export_matlab(req.dict())
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except RuntimeError as exc:
        raise HTTPException(503, str(exc))


@app.get("/api/ai/status")
async def ai_status():
    return app_state.ai.status()


@app.get("/api/lie_detector/status")
async def lie_detector_status():
    return app_state.lie_detector.status()


@app.post("/api/lie_detector/reset")
async def lie_detector_reset():
    return app_state.lie_detector.reset()


@app.post("/api/lie_detector/sample/{kind}")
async def lie_detector_sample(kind: str):
    try:
        return app_state.lie_detector.add_sample(kind, app_state.last_prediction)
    except ValueError as exc:
        raise HTTPException(400, str(exc))


@app.post("/api/lie_detector/question/start")
async def lie_detector_question_start(payload: Dict[str, Any] = None):
    question = str((payload or {}).get("question") or "")
    return app_state.lie_detector.start_question(question)


@app.post("/api/lie_detector/question/score")
async def lie_detector_question_score(payload: Dict[str, Any] = None):
    answer = str((payload or {}).get("answer") or "")
    try:
        return app_state.lie_detector.score_question(app_state.last_prediction, answer=answer)
    except ValueError as exc:
        raise HTTPException(400, str(exc))


@app.post("/api/ai/copilot")
async def ai_copilot(req: AICopilotRequest):
    try:
        return app_state.ai.summarize(_ai_snapshot(req))
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except RuntimeError as exc:
        raise HTTPException(503, str(exc))


@app.post("/api/pipeline/plan")
async def pipeline_plan(req: PipelinePlanRequest):
    try:
        return {
            "ok": True,
            "plan": _pipeline_plan(req),
            "ai": app_state.ai.status(),
        }
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except RuntimeError as exc:
        raise HTTPException(503, str(exc))


@app.post("/api/pipeline/qa")
async def pipeline_qa(req: PipelineQARequest):
    try:
        return _pipeline_qa(req)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except RuntimeError as exc:
        raise HTTPException(503, str(exc))


@app.post("/api/pipeline/train_baseline")
async def pipeline_train_baseline(req: PipelineBaselineTrainRequest):
    try:
        return _pipeline_train_baseline(req)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except RuntimeError as exc:
        raise HTTPException(503, str(exc))


@app.post("/api/pipeline/compare")
async def pipeline_compare(req: PipelineActionRequest):
    try:
        return _pipeline_model_compare(req)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except RuntimeError as exc:
        raise HTTPException(503, str(exc))


@app.post("/api/pipeline/export")
async def pipeline_export(req: PipelineActionRequest):
    try:
        return _pipeline_export(req)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except RuntimeError as exc:
        raise HTTPException(503, str(exc))


@app.get("/api/pipeline/exports")
async def pipeline_exports():
    try:
        return _pipeline_list_exports()
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except RuntimeError as exc:
        raise HTTPException(503, str(exc))


@app.get("/api/pipeline/projects")
async def pipeline_projects():
    try:
        return _pipeline_list_projects()
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except RuntimeError as exc:
        raise HTTPException(503, str(exc))


@app.post("/api/pipeline/projects/save")
async def pipeline_project_save(req: PipelineProjectRequest):
    try:
        return _pipeline_save_project(req)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except RuntimeError as exc:
        raise HTTPException(503, str(exc))


@app.post("/api/pipeline/projects/load")
async def pipeline_project_load(req: PipelineProjectRequest):
    try:
        return _pipeline_load_project(req)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except RuntimeError as exc:
        raise HTTPException(503, str(exc))


@app.get("/api/pipeline/projects/report/{project_id}")
async def pipeline_project_report(project_id: str):
    try:
        path = _pipeline_project_report_file(project_id)
        return FileResponse(path, filename=path.name, media_type="text/markdown", headers={"Cache-Control": "no-store"})
    except ValueError as exc:
        raise HTTPException(404, str(exc))
    except RuntimeError as exc:
        raise HTTPException(503, str(exc))


@app.post("/api/pipeline/projects/delete")
async def pipeline_project_delete(req: PipelineProjectRequest):
    try:
        return _pipeline_delete_project(req)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except RuntimeError as exc:
        raise HTTPException(503, str(exc))


@app.post("/api/pipeline/projects/rebuild")
async def pipeline_project_rebuild(req: PipelineProjectRequest, background_tasks: BackgroundTasks):
    try:
        return _pipeline_start_project_rebuild(req, background_tasks)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except RuntimeError as exc:
        raise HTTPException(503, str(exc))


@app.get("/api/pipeline/projects/rebuild/{job_id}")
async def pipeline_project_rebuild_status(job_id: str):
    try:
        return _pipeline_get_project_rebuild(job_id)
    except ValueError as exc:
        raise HTTPException(404, str(exc))
    except RuntimeError as exc:
        raise HTTPException(503, str(exc))


@app.get("/api/pipeline/projects/rebuild/{job_id}/report")
async def pipeline_project_rebuild_report(job_id: str):
    try:
        path = _pipeline_rebuild_report_file(job_id)
        return FileResponse(path, filename=path.name, media_type="text/markdown", headers={"Cache-Control": "no-store"})
    except ValueError as exc:
        raise HTTPException(404, str(exc))
    except RuntimeError as exc:
        raise HTTPException(503, str(exc))


@app.post("/api/pipeline/projects/rebuild/delete")
async def pipeline_project_rebuild_delete(req: PipelineProjectRequest):
    try:
        return _pipeline_delete_project_rebuild(req)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except RuntimeError as exc:
        raise HTTPException(503, str(exc))


@app.post("/api/pipeline/exports/verify")
async def pipeline_export_verify(req: PipelineModelRequest):
    try:
        return _pipeline_verify_export_package(req)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except RuntimeError as exc:
        raise HTTPException(503, str(exc))


@app.post("/api/pipeline/exports/delete")
async def pipeline_export_delete(req: PipelineModelRequest):
    try:
        return _pipeline_delete_export_package(req)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except RuntimeError as exc:
        raise HTTPException(503, str(exc))


@app.get("/api/pipeline/exports/download")
async def pipeline_export_download(path: str = ""):
    try:
        zip_path = _pipeline_download_export_package(PipelineModelRequest(path=path))
        return FileResponse(zip_path, filename=zip_path.name, media_type="application/zip", headers={"Cache-Control": "no-store"})
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except RuntimeError as exc:
        raise HTTPException(503, str(exc))


@app.get("/api/pipeline/exports/demo")
async def pipeline_export_demo(path: str = ""):
    try:
        export_dir = _resolve_pipeline_export_dir(path)
        index_path = export_dir / "index.html"
        if not index_path.exists():
            raise ValueError("Export package has no standalone index.html.")
        return FileResponse(index_path, headers={"Cache-Control": "no-store"})
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except RuntimeError as exc:
        raise HTTPException(503, str(exc))


@app.post("/api/pipeline/browser_runtime")
async def pipeline_browser_runtime(req: PipelineActionRequest):
    try:
        return _pipeline_browser_runtime(req)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except RuntimeError as exc:
        raise HTTPException(503, str(exc))


@app.post("/api/pipeline/deploy_smoke")
async def pipeline_deploy_smoke(req: PipelineActionRequest):
    try:
        return _pipeline_deploy_smoke(req)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except RuntimeError as exc:
        raise HTTPException(503, str(exc))


@app.post("/api/pipeline/acquisition_protocol")
async def pipeline_acquisition_protocol(req: PipelineActionRequest):
    try:
        return _pipeline_acquisition_protocol(req)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except RuntimeError as exc:
        raise HTTPException(503, str(exc))


@app.post("/api/pipeline/acquisition_run")
async def pipeline_acquisition_run(req: PipelineAcquisitionRunRequest):
    try:
        return _pipeline_acquisition_control(req)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except RuntimeError as exc:
        raise HTTPException(503, str(exc))


@app.post("/api/pipeline/train_acquisition")
async def pipeline_train_acquisition(req: PipelineActionRequest):
    try:
        return _pipeline_train_acquisition(req)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except RuntimeError as exc:
        raise HTTPException(503, str(exc))


@app.get("/api/pipeline/models")
async def pipeline_models():
    try:
        return _pipeline_list_prompt_models()
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except RuntimeError as exc:
        raise HTTPException(503, str(exc))


@app.post("/api/pipeline/models/load")
async def pipeline_model_load(req: PipelineModelRequest):
    try:
        return _pipeline_load_prompt_model_request(req)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except RuntimeError as exc:
        raise HTTPException(503, str(exc))


@app.post("/api/pipeline/models/delete")
async def pipeline_model_delete(req: PipelineModelRequest):
    try:
        return _pipeline_delete_prompt_model(req)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except RuntimeError as exc:
        raise HTTPException(503, str(exc))


@app.post("/api/pipeline/label_suggestions")
async def pipeline_label_suggestions(req: PipelineActionRequest):
    try:
        return _pipeline_label_suggestions(req)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except RuntimeError as exc:
        raise HTTPException(503, str(exc))


@app.post("/api/pipeline/dataset_inspect")
async def pipeline_dataset_inspect(req: PipelineDatasetInspectRequest):
    try:
        return _pipeline_dataset_inspect(req)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except RuntimeError as exc:
        raise HTTPException(503, str(exc))


@app.post("/api/pipeline/dataset_ingest")
async def pipeline_dataset_ingest(req: PipelineDatasetIngestRequest):
    try:
        return _pipeline_dataset_ingest(req)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except RuntimeError as exc:
        raise HTTPException(503, str(exc))


@app.post("/api/pipeline/throughput")
async def pipeline_throughput(req: PipelineThroughputRequest):
    try:
        return _pipeline_throughput_benchmark(req)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except RuntimeError as exc:
        raise HTTPException(503, str(exc))


@app.post("/api/pipeline/embedding_timeline")
async def pipeline_embedding_timeline(req: PipelineActionRequest):
    try:
        return _pipeline_embedding_timeline(req)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except RuntimeError as exc:
        raise HTTPException(503, str(exc))


@app.post("/api/ai/config")
async def ai_set_config(req: AIConfigRequest):
    try:
        return {"ok": True, "ai": app_state.ai.set_config(req.dict())}
    except ValueError as exc:
        raise HTTPException(400, str(exc))


@app.post("/api/ai/config/clear")
async def ai_clear_config():
    return {"ok": True, "ai": app_state.ai.clear_saved_config()}


@app.get("/api/firmware/status")
async def firmware_status():
    return app_state.firmware.status()


@app.get("/api/firmware/files")
async def firmware_files():
    return {
        **app_state.firmware.status(),
        **app_state.firmware.list_files(),
    }


@app.post("/api/firmware/file/read")
async def firmware_read_file(req: FirmwareFileRequest):
    try:
        return app_state.firmware.read_file(req.path)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))
    except ValueError as exc:
        raise HTTPException(400, str(exc))


@app.post("/api/firmware/file/save")
async def firmware_save_file(req: FirmwareSaveRequest):
    try:
        out = app_state.firmware.save_file(req.path, req.content)
        return {
            **out,
            **app_state.firmware.list_files(),
        }
    except ValueError as exc:
        raise HTTPException(400, str(exc))


@app.post("/api/firmware/generated")
async def firmware_generated(req: FirmwareGeneratedRequest):
    try:
        out = app_state.firmware.save_generated(
            name=req.name,
            content=req.content,
            filename=req.filename,
            target=req.target,
        )
        return {
            **out,
            **app_state.firmware.list_files(),
        }
    except ValueError as exc:
        raise HTTPException(400, str(exc))


@app.post("/api/firmware/compile")
async def firmware_compile(req: FirmwareCompileRequest):
    try:
        return app_state.firmware.compile_sketch(rel_path=req.path, fqbn=req.fqbn)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except RuntimeError as exc:
        raise HTTPException(503, str(exc))


@app.post("/api/firmware/upload")
async def firmware_upload(req: FirmwareUploadRequest):
    try:
        return app_state.firmware.upload_sketch(rel_path=req.path, fqbn=req.fqbn, port=req.port)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except RuntimeError as exc:
        raise HTTPException(503, str(exc))


@app.post("/api/filterlab/design")
async def filterlab_design(req: FilterDesignRequest):
    try:
        preview = app_state.filter_lab.design(req.dict())
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except RuntimeError as exc:
        raise HTTPException(503, str(exc))
    return {"ok": True, "preview": preview}


@app.post("/api/filterlab/save")
async def filterlab_save(req: FilterDesignRequest):
    try:
        item = app_state.filter_lab.save(req.dict())
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except RuntimeError as exc:
        raise HTTPException(503, str(exc))
    return {"ok": True, "filter": item, "filter_lab": app_state.filter_lab.status(config.signal_profile_name)}


@app.get("/api/filterlab/{filter_id}")
async def filterlab_get(filter_id: str):
    item = app_state.filter_lab.get_filter(filter_id)
    if not item:
        raise HTTPException(404, "Filter not found")
    return item


@app.post("/api/filterlab/activate")
async def filterlab_activate(req: FilterActivateRequest):
    try:
        status = app_state.filter_lab.activate(req.filter_id, req.profile or config.signal_profile_name)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    _sync_stream_filter()
    app_state.queue_broadcast(
        {
            "type": "diagnostics",
            "data": {
                **app_state.diagnostics.status(),
                "active_filter": status.get("active_filter"),
            },
            "timestamp": time.time(),
        }
    )
    return {"ok": True, "filter_lab": status}


@app.post("/api/filterlab/clear")
async def filterlab_clear(req: Optional[ProfileRequest] = None):
    profile_key = (req.profile if req else "") or config.signal_profile_name
    status = app_state.filter_lab.clear_active(profile_key)
    _sync_stream_filter()
    app_state.queue_broadcast(
        {
            "type": "diagnostics",
            "data": {
                **app_state.diagnostics.status(),
                "active_filter": status.get("active_filter"),
            },
            "timestamp": time.time(),
        }
    )
    return {"ok": True, "filter_lab": status}


@app.post("/api/filterlab/delete")
async def filterlab_delete(req: FilterActivateRequest):
    try:
        app_state.filter_lab.delete(req.filter_id)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))
    if str(req.profile or config.signal_profile_name) == config.signal_profile_name:
        _sync_stream_filter()
    return {"ok": True, "filter_lab": app_state.filter_lab.status(config.signal_profile_name)}


@app.post("/api/profile")
async def set_profile(req: ProfileRequest):
    if app_state.stream and app_state.stream.is_running:
        raise HTTPException(409, "Stop stream before changing biosignal profile")
    if app_state.recorder.is_recording:
        raise HTTPException(409, "Stop the active session before changing biosignal profile")

    profile = _set_signal_profile(req.profile)
    app_state.lsl.push_marker(
        "profile_changed",
        {"profile": config.signal_profile_name, "source": app_state.stream_source},
    )
    return {"ok": True, "signal_profile": profile}


@app.get("/api/eeg/brain-view")
async def eeg_brain_view():
    return app_state.eeg_brain.render(
        profile_key=config.signal_profile_name,
        window=app_state.last_vis_window,
        source_ts=app_state.last_vis_timestamp,
    )


@app.get("/api/eeg/brain-view/topomap.png")
async def eeg_brain_topomap():
    if not app_state.eeg_brain.topomap_path.exists() and app_state.last_vis_window is not None:
        app_state.eeg_brain.render(
            profile_key=config.signal_profile_name,
            window=app_state.last_vis_window,
            source_ts=app_state.last_vis_timestamp,
        )
    if not app_state.eeg_brain.topomap_path.exists():
        raise HTTPException(404, "EEG topomap not generated yet")
    return FileResponse(app_state.eeg_brain.topomap_path, headers={"Cache-Control": "no-store"})


@app.get("/api/eeg/brain-view/sensors.png")
async def eeg_brain_sensors():
    if not app_state.eeg_brain.sensors_path.exists() and app_state.last_vis_window is not None:
        app_state.eeg_brain.render(
            profile_key=config.signal_profile_name,
            window=app_state.last_vis_window,
            source_ts=app_state.last_vis_timestamp,
        )
    if not app_state.eeg_brain.sensors_path.exists():
        raise HTTPException(404, "EEG sensor map not generated yet")
    return FileResponse(app_state.eeg_brain.sensors_path, headers={"Cache-Control": "no-store"})


@app.get("/api/eeg/brain-view/markers.html")
async def eeg_brain_markers():
    if not app_state.eeg_brain.markers_path.exists() and app_state.last_vis_window is not None:
        app_state.eeg_brain.render(
            profile_key=config.signal_profile_name,
            window=app_state.last_vis_window,
            source_ts=app_state.last_vis_timestamp,
        )
    if not app_state.eeg_brain.markers_path.exists():
        raise HTTPException(404, "EEG Nilearn marker view not generated yet")
    return FileResponse(app_state.eeg_brain.markers_path, headers={"Cache-Control": "no-store"})


@app.get("/api/ports")
async def list_serial_ports():
    import serial.tools.list_ports

    ports = []
    for port in serial.tools.list_ports.comports():
        ports.append(
            {
                "device": port.device,
                "description": port.description,
                "hwid": port.hwid,
            }
        )
    return {"ports": ports}


@app.post("/api/stream/start")
async def stream_start(req: ConnectRequest = None):
    if app_state.recorder.is_recording:
        raise HTTPException(409, "Stop the active recording session before changing the stream source")
    if app_state.stream and app_state.stream.is_running:
        app_state.stream.stop()
    app_state.watchdog.last_signal_monotonic = 0.0
    app_state.watchdog.stale = False
    app_state.last_diagnostics_broadcast_ts = 0.0
    app_state.last_ml_broadcast_ts = 0.0

    source = _resolve_stream_source(req)
    app_state.stream_details = {}

    if req and req.arduino_port and app_state.arduino:
        app_state.arduino.connect(port=req.arduino_port)

    if source == "synthetic":
        synthetic_scenario = str((req.synthetic_scenario if req else None) or app_state.stream_details.get("scenario") or "clean").strip() or "clean"
        app_state.stream = _make_stream("synthetic", synthetic_scenario=synthetic_scenario)
        logger.info("Connecting synthetic %s stream (%s)", config.signal_profile.display_name, synthetic_scenario)
        if not app_state.stream.connect():
            raise HTTPException(500, "Failed to initialize synthetic biosignal stream")
        app_state.stream_details = {"mode": "synthetic", "scenario": synthetic_scenario}
    elif source == "hardware":
        app_state.stream = _make_stream("hardware")
        if not config.signal_profile.hardware_supported:
            raise HTTPException(
                400,
                f"{config.signal_profile.display_name} does not have live hardware support in the current stack",
            )
        port = (req.cyton_port if req else None) or config.serial_port
        logger.info(
            "Connecting Cyton stream on %s for %s",
            port,
            config.signal_profile.display_name,
        )
        try:
            connected = app_state.stream.connect(serial_port=port)
        except ModuleNotFoundError as exc:
            missing = exc.name or "brainflow"
            logger.error("Hardware runtime dependency missing while connecting Cyton: %s", missing)
            raise HTTPException(
                503,
                f"Hardware streaming is unavailable because the active Python runtime is missing '{missing}'",
            ) from exc
        if not connected:
            raise HTTPException(500, f"Failed to connect to Cyton board on {port}")
        app_state.stream_details = {"mode": "hardware", "cyton_port": port}
    elif source == "lsl":
        lsl_stream_name = (req.lsl_stream_name if req else None) or str(app_state.stream_details.get("stream_name") or app_state.stream_details.get("name") or "")
        lsl_source_id = (req.lsl_source_id if req else None) or str(app_state.stream_details.get("source_id") or "")
        app_state.stream = _make_stream(
            "lsl",
            lsl_stream_name=lsl_stream_name,
            lsl_source_id=lsl_source_id,
        )
        logger.info(
            "Connecting LSL inlet for %s (name=%s, source_id=%s)",
            config.signal_profile.display_name,
            lsl_stream_name or "--",
            lsl_source_id or "--",
        )
        if not app_state.stream.connect(stream_name=lsl_stream_name, source_id=lsl_source_id):
            raise HTTPException(500, "Failed to connect to the selected LSL stream")
        app_state.stream_details = {"mode": "lsl", **getattr(app_state.stream, "stream_details", {})}
    else:
        playback_session_id = (req.playback_session_id if req else None) or app_state.playback_session_id
        if not playback_session_id:
            raise HTTPException(400, "Select a recorded session for playback")

        session_meta = app_state.recorder.get_session_meta(playback_session_id)
        if not session_meta:
            raise HTTPException(404, f"Playback session not found: {playback_session_id}")

        session_cfg = session_meta.get("config", {})
        session_profile = session_cfg.get("signal_profile")
        if not session_profile:
            legacy_features = {str(name).upper() for name in session_cfg.get("features", [])}
            if session_cfg.get("gestures") or {"MAV", "RMS", "WL", "ZC", "SSC"} & legacy_features:
                session_profile = "emg"
        if session_profile and session_profile != config.signal_profile_name:
            logger.info(
                "Switching active profile from %s to %s for playback session %s",
                config.signal_profile_name,
                session_profile,
                playback_session_id,
            )
            _set_signal_profile(session_profile)

        session_rate = int(session_cfg.get("sample_rate") or config.sample_rate)
        if session_rate != config.sample_rate:
            raise HTTPException(
                400,
                f"Playback session uses {session_rate} Hz, but this runtime expects {config.sample_rate} Hz",
            )

        playback_rate = req.playback_rate if req else 1.0
        app_state.stream = _make_stream(
            "playback",
            playback_session_id=playback_session_id,
            playback_rate=playback_rate,
        )
        logger.info(
            "Connecting playback stream for session %s at %.2fx",
            playback_session_id,
            playback_rate,
        )
        if not app_state.stream.connect(session_id=playback_session_id):
            raise HTTPException(500, f"Failed to load playback session {playback_session_id}")
        app_state.stream_details = {
            "mode": "playback",
            "session_id": playback_session_id,
            "playback_rate": playback_rate,
        }

    _sync_lsl_outlets()
    if not app_state.stream.start():
        raise HTTPException(500, "Failed to start stream")

    app_state.set_state(SystemState.STREAMING)
    app_state.lsl.push_marker(
        "stream_started",
        {
            "source": app_state.stream_source,
            "profile": config.signal_profile_name,
            "details": app_state.stream_details,
        },
    )
    return {
        "ok": True,
        "stream_source": app_state.stream_source,
        "stream_details": app_state.stream_details,
        "playback_session_id": app_state.playback_session_id,
        "signal_profile": config.signal_profile.to_dict(),
    }


@app.post("/api/stream/stop")
async def stream_stop():
    saved_to = None
    if app_state.recorder.is_recording:
        _detach_recorder_callback()
        saved_to = app_state.recorder.stop_session()
    if app_state.stream:
        app_state.stream.stop()
    app_state.watchdog.last_signal_monotonic = 0.0
    app_state.watchdog.stale = False
    app_state.lsl.push_marker(
        "stream_stopped",
        {"source": app_state.stream_source, "profile": config.signal_profile_name},
    )
    app_state.set_state(SystemState.IDLE)
    return {"ok": True, "saved_to": saved_to}


@app.post("/api/calibrate")
async def calibrate(background_tasks: BackgroundTasks):
    if not app_state.stream or not app_state.stream.is_running:
        raise HTTPException(400, "Stream not running - start stream first")
    app_state.set_state(SystemState.CALIBRATING)

    async def _run():
        result = await app_state.calibration.run_calibration(app_state.stream)
        new_state = SystemState.STREAMING if result.get("success") else SystemState.IDLE
        app_state.set_state(new_state)

    background_tasks.add_task(_run)
    return {
        "ok": True,
        "message": "Calibration started",
        "protocol": app_state.calibration.describe_protocol(),
    }


@app.post("/api/train/start")
async def train_start(req: TrainRequest):
    if not app_state.stream or not app_state.stream.is_running:
        raise HTTPException(400, "Stream not running")
    if not app_state.pipeline.supports_training:
        raise HTTPException(
            400,
            f"{config.signal_profile.display_name} training is not implemented yet",
        )
    if req.gesture not in config.class_labels:
        raise HTTPException(400, f"Unknown label. Valid: {config.class_labels}")

    app_state.set_state(SystemState.TRAINING)
    app_state.recorder.log_event(
        "train_start",
        {"label": req.gesture, "profile": config.signal_profile_name},
    )
    app_state.lsl.push_marker(
        "train_start",
        {"label": req.gesture, "profile": config.signal_profile_name},
    )
    app_state.pipeline.start_recording(req.gesture)
    return {"ok": True, "gesture": req.gesture, "label": req.gesture}


@app.post("/api/train/stop")
async def train_stop():
    app_state.pipeline.stop_recording()
    app_state.lsl.push_marker("train_stop", {"profile": config.signal_profile_name})
    if app_state.stream and app_state.stream.is_running:
        app_state.set_state(SystemState.STREAMING)
    return {"ok": True, "summary": app_state.pipeline.get_training_summary()}


@app.post("/api/train/fit")
async def train_fit(classifier: str = "LDA"):
    if classifier not in ("LDA", "TCN", "Mamba"):
        raise HTTPException(400, "Unknown classifier. Options: LDA, TCN, Mamba")

    result = app_state.pipeline.train(classifier_name=classifier)
    if result.get("success"):
        path = app_state.pipeline.save_model()
        result["model_path"] = path
        result["accuracy"] = result.get("val_accuracy", 0)
        app_state.recorder.log_event("model_trained", result)
        app_state.lsl.push_marker(
            "model_trained",
            {
                "profile": config.signal_profile_name,
                "classifier": classifier,
                "path": path,
                "success": True,
            },
        )
    return result


@app.post("/api/train/auto")
async def train_auto():
    """Try LDA, then TCN and Mamba if enough data. Pick the best accuracy model.
    
    This makes training completely hands-off: one API call → best model.
    """
    summary = app_state.pipeline.get_training_summary()
    n_windows = summary.get("total_windows", 0)
    if n_windows < 10:
        return {"success": False, "error": f"Need >= 10 windows, have {n_windows}"}

    best_result = None
    best_acc = -1.0
    best_clf = None
    tried = []

    # Always try LDA (fast, works with any data size)
    try:
        r = app_state.pipeline.train(classifier_name="LDA")
        acc = r.get("val_accuracy", 0) or 0
        tried.append({"classifier": "LDA", "accuracy": acc, "success": r.get("success", False)})
        if r.get("success") and acc > best_acc:
            best_acc = acc
            best_result = r
            best_clf = "LDA"
    except Exception as e:
        tried.append({"classifier": "LDA", "accuracy": 0, "error": str(e)})

    # Try TCN if enough data (>= 30 windows)
    if n_windows >= 30:
        try:
            r = app_state.pipeline.train(classifier_name="TCN")
            acc = r.get("val_accuracy", 0) or 0
            tried.append({"classifier": "TCN", "accuracy": acc, "success": r.get("success", False)})
            if r.get("success") and acc > best_acc:
                best_acc = acc
                best_result = r
                best_clf = "TCN"
        except Exception as e:
            tried.append({"classifier": "TCN", "accuracy": 0, "error": str(e)})

    # Try Mamba if enough data (>= 50 windows)
    if n_windows >= 50:
        try:
            r = app_state.pipeline.train(classifier_name="Mamba")
            acc = r.get("val_accuracy", 0) or 0
            tried.append({"classifier": "Mamba", "accuracy": acc, "success": r.get("success", False)})
            if r.get("success") and acc > best_acc:
                best_acc = acc
                best_result = r
                best_clf = "Mamba"
        except Exception as e:
            tried.append({"classifier": "Mamba", "accuracy": 0, "error": str(e)})

    if best_result and best_clf:
        # Re-train the best classifier so it's the active one
        if best_clf != app_state.pipeline._impl._classifier_name if hasattr(app_state.pipeline, '_impl') and app_state.pipeline._impl else True:
            app_state.pipeline.train(classifier_name=best_clf)
        path = app_state.pipeline.save_model()
        best_result["model_path"] = path
        best_result["accuracy"] = best_acc
        best_result["best_classifier"] = best_clf
        best_result["tried"] = tried
        app_state.recorder.log_event("auto_train_complete", {
            "best_classifier": best_clf,
            "accuracy": best_acc,
            "n_windows": n_windows,
            "path": path,
        })
        return best_result

    return {"success": False, "error": "All classifiers failed", "tried": tried}


@app.post("/api/train/clear")
async def train_clear():
    app_state.pipeline.clear_training_data()
    return {"ok": True}


@app.get("/api/train/summary")
async def train_summary():
    return app_state.pipeline.get_training_summary()


@app.post("/api/estop")
async def estop():
    if app_state.arduino:
        app_state.arduino.estop()
    app_state.osc.send_estop()
    app_state.set_state(SystemState.ESTOP)
    app_state.recorder.log_event("estop", {"profile": config.signal_profile_name})
    app_state.lsl.push_marker("estop", {"profile": config.signal_profile_name})
    return {"ok": True, "message": "E-STOP activated"}


@app.post("/api/home")
async def home():
    if app_state.arduino:
        app_state.arduino.home()
    app_state.osc.send_home()
    if app_state.system_state == SystemState.ESTOP or (
        app_state.stream and app_state.stream.is_running
    ):
        app_state.set_state(
            SystemState.STREAMING
            if app_state.stream and app_state.stream.is_running
            else SystemState.IDLE
        )
    app_state.recorder.log_event("home", {"profile": config.signal_profile_name})
    app_state.lsl.push_marker("home", {"profile": config.signal_profile_name})
    return {"ok": True}


@app.post("/api/move")
async def manual_move(cmd: MoveCommand):
    if not _control_output_ready():
        raise HTTPException(503, "No control output is active. Connect Arduino or start OSC.")
    ok = _dispatch_move(cmd.joint_id, cmd.angle)
    return {"ok": ok}


@app.post("/api/gesture/{name}")
async def execute_gesture(name: str):
    if not config.signal_profile.robotic_arm_supported:
        raise HTTPException(
            400,
            f"{config.signal_profile.display_name} does not use the robotic arm gesture path",
        )
    control_ready = _control_output_ready()
    preview_only = not control_ready and _hardware_stream_preview_ready()
    if not control_ready and not preview_only:
        raise HTTPException(
            503,
            "No control output is active. Connect Arduino, start OSC, or run the hardware board stream for shortcut preview.",
        )
    if name not in config.class_labels:
        raise HTTPException(400, f"Unknown gesture. Valid: {config.class_labels}")
    if preview_only:
        return {"ok": True, "preview_only": True}
    ok = _dispatch_gesture(name)
    return {"ok": ok, "preview_only": False}


@app.post("/api/control/sensitivity")
async def set_control_sensitivity(req: ControlSensitivityRequest):
    threshold = max(0.30, min(0.95, float(req.confidence_threshold)))
    config.prediction_confidence_threshold = threshold
    return {
        "ok": True,
        "confidence_threshold": config.prediction_confidence_threshold,
        "sensitivity": round(1.0 - config.prediction_confidence_threshold, 3),
    }


@app.post("/api/digital_write")
async def digital_write(cmd: DigitalWriteCommand):
    if not _control_output_ready():
        raise HTTPException(503, "No control output is active. Connect Arduino or start OSC.")
    ok = _dispatch_digital_write(cmd.pin, cmd.value)
    return {"ok": ok}


@app.post("/api/analog_write")
async def analog_write(cmd: AnalogWriteCommand):
    if not _control_output_ready():
        raise HTTPException(503, "No control output is active. Connect Arduino or start OSC.")
    ok = _dispatch_analog_write(cmd.pin, cmd.value)
    return {"ok": ok}


@app.post("/api/session/start")
async def session_start(req: SessionStartRequest = None):
    if not app_state.stream or not app_state.stream.is_running:
        raise HTTPException(400, "Stream not running - start stream first")
    if app_state.stream_source == "playback":
        raise HTTPException(400, "Playback sessions are already recorded - start hardware or synthetic streaming to record a new session")
    if app_state.recorder.is_recording:
        raise HTTPException(409, "A session is already being recorded")
    label = req.label if req else ""
    session_metadata = {
        "subject_id": req.subject_id if req else "",
        "condition": req.condition if req else "",
        "notes": req.notes if req else "",
        "protocol_key": req.protocol_key if req else "",
        "protocol_title": req.protocol_title if req else "",
        "session_group_id": req.session_group_id if req else "",
        "trial_index": req.trial_index if req else None,
        "repetition_index": req.repetition_index if req else None,
    }
    if session_metadata["subject_id"]:
        app_state.subjects.touch_subject(session_metadata["subject_id"])
    session_id = app_state.recorder.start_session(
        label=label,
        stream_source=app_state.stream_source,
        source_details=app_state.stream_details,
        session_metadata=session_metadata,
    )
    _attach_recorder_callback()
    app_state.lsl.push_marker(
        "session_start",
        {
            "session_id": session_id,
            "profile": config.signal_profile_name,
            "source": app_state.stream_source,
            "label": label,
            "subject_id": session_metadata["subject_id"],
            "condition": session_metadata["condition"],
            "session_group_id": session_metadata["session_group_id"],
            "protocol_key": session_metadata["protocol_key"],
            "trial_index": session_metadata["trial_index"],
            "repetition_index": session_metadata["repetition_index"],
        },
    )
    return {"ok": True, "session_id": session_id}


@app.get("/api/subjects")
async def list_subjects():
    return app_state.subjects.list_subjects()


@app.get("/api/subjects/{subject_id}")
async def get_subject(subject_id: str):
    subject = app_state.subjects.get_subject(subject_id)
    if not subject:
        raise HTTPException(404, "Subject not found")
    return subject


@app.post("/api/subjects")
async def upsert_subject(req: SubjectUpsertRequest):
    try:
        subject = app_state.subjects.upsert_subject(req.dict())
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return {"ok": True, "subject": subject}


@app.post("/api/session/stop")
async def session_stop():
    _detach_recorder_callback()
    path = app_state.recorder.stop_session()
    app_state.lsl.push_marker(
        "session_stop",
        {"profile": config.signal_profile_name, "saved_to": path},
    )
    return {"ok": True, "saved_to": path}


@app.get("/api/sessions")
async def list_sessions():
    return app_state.recorder.list_sessions()


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    meta = app_state.recorder.get_session_meta(session_id)
    if not meta:
        raise HTTPException(404, "Session not found")
    return meta


@app.post("/api/sessions/{session_id}/export/bids")
async def export_session_bids(session_id: str):
    try:
        export_info = app_state.exporter.export_bids(session_id)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        logger.exception("BIDS export failed for %s", session_id)
        raise HTTPException(500, f"BIDS export failed: {exc}")
    return {"ok": True, **export_info}


@app.get("/api/datasets")
async def list_datasets():
    return app_state.datasets.list_datasets()


@app.get("/api/datasets/{dataset_id}")
async def get_dataset(dataset_id: str):
    dataset = app_state.datasets.get_dataset(dataset_id)
    if not dataset:
        raise HTTPException(404, "Dataset not found")
    return dataset


@app.post("/api/datasets")
async def create_dataset(req: DatasetCreateRequest):
    try:
        dataset = app_state.datasets.create_dataset(
            session_ids=req.session_ids,
            name=req.name,
            profile_key=config.signal_profile_name,
        )
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        logger.exception("Dataset creation failed")
        raise HTTPException(500, f"Dataset creation failed: {exc}")
    return {"ok": True, "dataset": dataset}


@app.get("/api/experiments")
async def list_experiments():
    return app_state.experiments.list_experiments()


@app.get("/api/experiments/{experiment_id}")
async def get_experiment(experiment_id: str):
    report = app_state.experiments.get_experiment(experiment_id)
    if not report:
        raise HTTPException(404, "Experiment not found")
    return report


@app.post("/api/experiments/run")
async def run_experiment(req: ExperimentRunRequest):
    if app_state.stream and app_state.stream.is_running:
        raise HTTPException(409, "Stop the live stream before running offline experiments")
    if app_state.recorder.is_recording:
        raise HTTPException(409, "Stop the active session recording before running offline experiments")
    try:
        report = app_state.experiments.run_experiment(
            dataset_id=req.dataset_id,
            classifier=req.classifier,
            notes=req.notes,
            split_strategy=req.split_strategy,
            holdout_fraction=req.holdout_fraction,
            holdout_gap_s=req.holdout_gap_s,
        )
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        logger.exception("Offline experiment failed")
        raise HTTPException(500, f"Offline experiment failed: {exc}")
    return {"ok": True, "report": report}


@app.post("/api/xdf/inspect")
async def inspect_xdf(req: XDFInspectRequest):
    try:
        return app_state.xdf.inspect(req.path)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))
    except Exception as exc:
        logger.exception("XDF inspect failed for %s", req.path)
        raise HTTPException(400, f"XDF inspect failed: {exc}")


@app.post("/api/xdf/import")
async def import_xdf(req: XDFImportRequest):
    try:
        result = app_state.xdf.import_session(
            path=req.path,
            stream_name=req.stream_name,
            stream_id=req.stream_id,
            signal_profile=req.signal_profile,
            label=req.label,
        )
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))
    except Exception as exc:
        logger.exception("XDF import failed for %s", req.path)
        raise HTTPException(400, f"XDF import failed: {exc}")
    return result


@app.get("/api/lsl/status")
async def lsl_status():
    return app_state.lsl.status()


@app.get("/api/lsl/inputs")
async def lsl_inputs():
    return list_lsl_streams()


@app.post("/api/lsl/start")
async def lsl_start(req: Optional[LSLStartRequest] = None):
    ok = app_state.lsl.start(
        profile=config.signal_profile,
        channel_labels=config.channel_labels,
        sample_rate=config.sample_rate,
        stream_source=app_state.stream_source,
        stream_name=req.stream_name if req else None,
        include_markers=req.include_markers if req else True,
    )
    if not ok:
        raise HTTPException(
            503,
            app_state.lsl.last_error
            or "Failed to start LSL outlets. Install requirements-research.txt if pylsl is missing.",
        )
    app_state.lsl.push_marker(
        "lsl_started",
        {
            "profile": config.signal_profile_name,
            "source": app_state.stream_source,
            "stream_name": app_state.lsl.stream_name,
        },
    )
    return {"ok": True, "lsl": app_state.lsl.status()}


@app.post("/api/lsl/stop")
async def lsl_stop():
    was_active = app_state.lsl.is_active
    app_state.lsl.stop()
    return {"ok": True, "was_active": was_active}


@app.post("/api/lsl/marker")
async def lsl_marker(req: LSLMarkerRequest):
    if not app_state.lsl.is_active or not app_state.lsl.include_markers:
        raise HTTPException(503, "LSL markers are not active. Start LSL output with markers enabled first.")

    event = str(req.event or "").strip()
    if not event:
        raise HTTPException(400, "Marker event is required")

    payload = dict(req.payload or {})
    payload.setdefault("profile", config.signal_profile_name)
    payload.setdefault("source", app_state.stream_source)
    app_state.lsl.push_marker(event, payload)
    if app_state.recorder.is_recording:
        app_state.recorder.log_event(
            "manual_lsl_marker",
            {"event": event, **payload},
        )
    return {
        "ok": True,
        "event": event,
        "payload": payload,
        "marker_stream_name": app_state.lsl.marker_stream_name,
    }


@app.post("/api/review/marker")
async def review_marker(req: ReviewMarkerRequest):
    if not app_state.stream or not app_state.stream.is_running:
        raise HTTPException(400, "Stream not running")

    event = str(req.event or "").strip()
    if not event:
        raise HTTPException(400, "Marker event is required")

    selection = None
    if (
        req.selection_start_s is not None
        or req.selection_end_s is not None
        or req.selection_start_sample is not None
        or req.selection_end_sample is not None
    ):
        selection = {
            "start_s": round(float(req.selection_start_s or 0.0), 4),
            "end_s": round(float(req.selection_end_s or req.selection_start_s or 0.0), 4),
            "start_sample": int(req.selection_start_sample or 0),
            "end_sample": int(req.selection_end_sample or req.selection_start_sample or 0),
        }
        if selection["end_s"] < selection["start_s"]:
            selection["start_s"], selection["end_s"] = selection["end_s"], selection["start_s"]
        if selection["end_sample"] < selection["start_sample"]:
            selection["start_sample"], selection["end_sample"] = selection["end_sample"], selection["start_sample"]

    payload = {
        "event": event,
        "note": str(req.note or "").strip(),
        "profile": config.signal_profile_name,
        "source": app_state.stream_source,
        "sample_rate": config.sample_rate,
        "selection": selection,
        "sample_time_s": round(float(req.sample_time_s or 0.0), 4) if req.sample_time_s is not None else None,
        "sample_index": int(req.sample_index) if req.sample_index is not None else None,
        "metrics": dict(req.metrics or {}),
    }

    if app_state.recorder.is_recording:
        app_state.recorder.log_event("review_marker", payload)
    app_state.lsl.push_marker(event, payload)
    app_state.queue_broadcast(
        {
            "type": "review_marker",
            "data": payload,
            "timestamp": time.time(),
        }
    )
    return {"ok": True, "marker": payload}


@app.get("/api/osc/status")
async def osc_status():
    return app_state.osc.status()


@app.post("/api/osc/start")
async def osc_start(req: Optional[OSCStartRequest] = None):
    ok = app_state.osc.start(
        host=req.host if req else "127.0.0.1",
        port=req.port if req else 9000,
        prefix=req.prefix if req else "/kyma",
        mirror_events=req.mirror_events if req else True,
    )
    if not ok:
        raise HTTPException(
            503,
            app_state.osc.last_error
            or "Failed to start OSC output. Install requirements.txt if python-osc is missing.",
        )
    return {"ok": True, "osc": app_state.osc.status()}


@app.post("/api/osc/stop")
async def osc_stop():
    was_active = app_state.osc.is_active
    app_state.osc.stop()
    return {"ok": True, "was_active": was_active, "osc": app_state.osc.status()}


@app.post("/api/osc/send")
async def osc_send(req: OSCSendRequest):
    if not app_state.osc.is_active:
        raise HTTPException(503, "OSC output is not active.")
    ok = app_state.osc.send_custom(req.address, req.value)
    if not ok:
        raise HTTPException(503, app_state.osc.last_error or "OSC send failed.")
    return {"ok": True, "osc": app_state.osc.status(), "address": req.address}


@app.post("/api/arduino/connect")
async def arduino_connect(req: ConnectRequest = None):
    port = (req.arduino_port if req else None) or config.arduino_port
    ok = app_state.arduino.connect(port=port)
    return {"ok": ok}


@app.post("/api/arduino/serial_write")
async def arduino_serial_write(req: ArduinoSerialWriteRequest):
    if not app_state.arduino or not app_state.arduino.is_connected:
        raise HTTPException(503, "Arduino is not connected.")
    ok = app_state.arduino.serial_write(req.text)
    if not ok:
        raise HTTPException(503, "Serial write failed.")
    return {"ok": True, "bytes": len(req.text.encode('utf-8'))}


@app.post("/api/integrations/webhook")
async def trigger_webhook(req: WebhookRequest):
    method = str(req.method or "POST").upper()
    if method not in {"POST", "PUT", "PATCH"}:
        raise HTTPException(400, "Webhook method must be POST, PUT, or PATCH.")
    payload = json.dumps(req.body or {}).encode("utf-8")
    request = urllib.request.Request(
        req.url,
        data=payload,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=float(req.timeout_s or 3.0)) as response:
            status_code = int(getattr(response, "status", response.getcode()))
            response_body = response.read(512).decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        body = exc.read(512).decode("utf-8", errors="replace")
        raise HTTPException(exc.code, f"Webhook failed: {body or exc.reason}")
    except Exception as exc:
        raise HTTPException(503, f"Webhook failed: {exc}")
    return {"ok": True, "status_code": status_code, "preview": response_body}


# ══════════════════════════════════════════════════════════════════════════
#  INTELLIGENCE LAYER — Signal Confidence, Recovery, Intent, Guided Recording
# ══════════════════════════════════════════════════════════════════════════


@app.get("/api/signal/confidence")
async def get_signal_confidence():
    return app_state.signal_confidence.to_dict()


@app.get("/api/signal/recovery")
async def get_signal_recovery():
    return app_state.signal_recovery.to_dict()


@app.get("/api/model/viability")
async def get_model_viability():
    conf = app_state.signal_confidence.to_dict()
    return {
        "viability": conf["model_viability"],
        "improvement_potential": conf["improvement_potential"],
        "usable_channels": conf["usable_channels"],
        "total_channels": conf["total_channels"],
        "overall_confidence": conf["overall_confidence"],
        "channel_weights": app_state.signal_confidence.get_channel_weights().tolist(),
    }


@app.post("/api/intent")
async def parse_intent(req: IntentRequest):
    if req.api_key:
        app_state.intent_engine.set_api_key(req.api_key, req.provider, req.base_url, req.model)
    try:
        plan = app_state.intent_engine.parse(req.prompt)
        # Sync gestures if new plan has them
        if plan.suggested_gestures:
            config.class_labels = list(plan.suggested_gestures)
        app_state.set_state(SystemState.PLAN_GENERATED)
        return {"ok": True, "plan": plan.to_dict()}
    except Exception as exc:
        raise HTTPException(400, str(exc))


@app.post("/api/intent/api_key")
async def set_intent_api_key(req: ApiKeyConfig):
    app_state.intent_engine.set_api_key(req.api_key, req.provider, req.base_url, req.model)
    return {"ok": True}


@app.post("/api/intent/api_key/test")
async def test_intent_api_key(req: ApiKeyConfig):
    if not req.api_key:
        raise HTTPException(400, "Paste an API key first.")
    provider = str(req.provider or "").strip() or "openai"
    if provider == "anthropic":
        raise HTTPException(400, "The built-in key test is for OpenAI-compatible providers.")
    base = str(req.base_url or "").strip().rstrip("/")
    if not base:
        base = "https://integrate.api.nvidia.com/v1" if provider == "nvidia" else "https://api.openai.com/v1"
    url = base if base.endswith("/chat/completions") else f"{base}/chat/completions"
    model = str(req.model or "").strip() or ("deepseek-ai/deepseek-v4-pro" if provider == "nvidia" else "gpt-4o-mini")
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with OK."}],
        "temperature": 0,
        "top_p": 0.95,
        "max_tokens": 1,
        "stream": provider == "nvidia",
    }
    if provider == "nvidia":
        lowered_model = model.lower()
        payload["chat_template_kwargs"] = (
            {"enable_thinking": False}
            if ("glm" in lowered_model or "qwen" in lowered_model)
            else {"thinking": False}
        )
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {req.api_key}",
            "Content-Type": "application/json",
            **({"Accept": "text/event-stream"} if provider == "nvidia" else {}),
        },
        method="POST",
    )
    try:
        start = time.time()
        test_timeout = 300 if provider == "nvidia" else (180 if provider in {"ollama", "local"} else 45)
        with urllib.request.urlopen(request, timeout=test_timeout) as response:
            if provider == "nvidia":
                text = ""
                data = {"model": model}
                for raw in response:
                    line = raw.decode("utf-8", errors="replace").strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data_text = line[5:].strip()
                    if data_text == "[DONE]":
                        break
                    event = json.loads(data_text)
                    data["model"] = event.get("model") or data["model"]
                    delta = ((event.get("choices") or [{}])[0].get("delta") or {})
                    text = delta.get("content") or ""
                    break
            else:
                body = response.read().decode("utf-8", errors="replace")
                data = json.loads(body)
                text = ((data.get("choices") or [{}])[0].get("message") or {}).get("content", "")
        return {
            "ok": True,
            "provider": provider,
            "model": data.get("model") or model,
            "elapsed_s": round(time.time() - start, 1),
            "text": text,
        }
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise HTTPException(exc.code, body[:1000] or str(exc))
    except Exception as exc:
        raise HTTPException(400, f"{type(exc).__name__}: {exc}")


@app.post("/api/recording/guided/start")
async def start_guided_recording(req: GuidedRecordingStartRequest):
    if not app_state.stream or not app_state.stream.is_running:
        raise HTTPException(400, "Start the stream first.")
    if app_state._guided_task and not app_state._guided_task.done():
        app_state.guided_recording.cancel()
        app_state._guided_task.cancel()
        with suppress(asyncio.CancelledError):
            await app_state._guided_task
    protocol = req.protocol
    if not protocol:
        # Use default protocol from last intent plan or simple default
        protocol = [
            {"step": 1, "instruction": "Relax completely", "gesture": "rest", "duration_s": 5, "type": "record", "reps": 3},
            {"step": 2, "instruction": "Relax briefly", "duration_s": 3, "type": "rest"},
        ]
        for gesture in config.class_labels:
            if gesture != "rest":
                protocol.append({"step": len(protocol)+1, "instruction": f"Perform '{gesture}' — hold steady", "gesture": gesture, "duration_s": 5, "type": "record", "reps": 3})
                protocol.append({"step": len(protocol)+1, "instruction": "Relax briefly", "duration_s": 3, "type": "rest"})
    app_state.guided_recording.create_session(protocol, req.target_windows)
    app_state.set_state(SystemState.GUIDED_RECORDING)

    async def _run_guided():
        try:
            result = await app_state.guided_recording.run(app_state.stream, app_state.pipeline)
            if result.get("success"):
                app_state.set_state(SystemState.STREAMING)
                app_state.queue_broadcast({"type": "guided_step", "data": {"phase": "complete", "message": "Guided recording complete", **result}, "timestamp": time.time()})
        except Exception as exc:
            logger.error("Guided recording error: %s", exc)
            app_state.set_state(SystemState.STREAMING)

    app_state._guided_task = asyncio.create_task(_run_guided())
    return {"ok": True, "session": app_state.guided_recording.get_state()}


@app.get("/api/recording/guided/status")
async def get_guided_recording_status():
    return app_state.guided_recording.get_state()


@app.post("/api/recording/guided/cancel")
async def cancel_guided_recording():
    app_state.guided_recording.cancel()
    if app_state._guided_task and not app_state._guided_task.done():
        app_state._guided_task.cancel()
    app_state.set_state(SystemState.STREAMING if (app_state.stream and app_state.stream.is_running) else SystemState.IDLE)
    return {"ok": True}


@app.post("/api/recording/guided/confirm")
async def confirm_guided_step():
    """User clicks 'I'm Ready' or 'Next' during guided recording."""
    app_state.guided_recording.confirm_step()
    return {"ok": True, "session": app_state.guided_recording.get_state()}


@app.post("/api/recording/guided/redo")
async def redo_guided_step():
    """User clicks 'Redo' after seeing quality results."""
    app_state.guided_recording.redo_step()
    return {"ok": True, "session": app_state.guided_recording.get_state()}


@app.post("/api/codegen")
async def generate_code(req: IntentRequest):
    """Generate full-stack project code from prompt + training context."""
    if not hasattr(app_state, 'code_generator'):
        app_state.code_generator = CodeGenerator()
    if req.api_key:
        app_state.code_generator.set_api_key(req.api_key, req.provider, req.base_url, req.model)
    elif app_state.intent_engine.api_key:
        app_state.code_generator.set_api_key(
            app_state.intent_engine.api_key,
            app_state.intent_engine.provider,
            app_state.intent_engine.base_url,
            app_state.intent_engine.model)
    app_state.code_generator.set_status_callback(app_state.queue_broadcast)

    context = _codegen_context()
    project = app_state.code_generator.generate(req.prompt, context)
    app_state.queue_broadcast({"type": "codegen_complete", "data": project.to_dict(), "timestamp": time.time()})
    return {"ok": True, "project": project.to_dict()}


def _codegen_context() -> dict:
    summary = app_state.pipeline.get_training_summary() if app_state.pipeline else {}
    prompt_model = _pipeline_prompt_model_summary()
    use_prompt_model = bool(prompt_model.get("loaded") and prompt_model.get("model_path"))
    labels = prompt_model.get("labels") if use_prompt_model else config.class_labels
    accuracy = (
        prompt_model.get("val_accuracy")
        or prompt_model.get("cv_balanced_accuracy")
        or summary.get("accuracy")
        or "N/A"
    )
    model_path = prompt_model.get("model_path") if use_prompt_model else summary.get("model_path", "models/model.pkl")
    return {
        "signal_modality": config.signal_profile_name.upper(),
        "gestures": list(labels or config.class_labels),
        "classifier": prompt_model.get("classifier") if use_prompt_model else summary.get("classifier", "LDA"),
        "accuracy": accuracy,
        "n_channels": config.n_channels,
        "sample_rate": config.sample_rate,
        "window_size_ms": config.window_size_ms,
        "window_size_samples": config.window_size_samples,
        "window_increment_ms": config.window_increment_ms,
        "features": list(config.features),
        "feature_dim": int(prompt_model.get("feature_dim") or 0) if use_prompt_model else config.n_channels * max(len(config.features), 1),
        "model_path": model_path or "models/model.pkl",
    }


@app.post("/api/codegen/refine")
async def refine_code(req: CodegenRefineRequest):
    """Revise the currently generated project from a user change request."""
    if not hasattr(app_state, 'code_generator'):
        app_state.code_generator = CodeGenerator()
    if req.api_key:
        app_state.code_generator.set_api_key(req.api_key, req.provider, req.base_url, req.model)
    elif app_state.intent_engine.api_key:
        app_state.code_generator.set_api_key(
            app_state.intent_engine.api_key,
            app_state.intent_engine.provider,
            app_state.intent_engine.base_url,
            app_state.intent_engine.model)
    app_state.code_generator.set_status_callback(app_state.queue_broadcast)
    project = app_state.code_generator.refine(req.project or {}, req.prompt, _codegen_context())
    app_state.queue_broadcast({"type": "codegen_complete", "data": project.to_dict(), "timestamp": time.time()})
    return {"ok": True, "project": project.to_dict()}


def _safe_codegen_relpath(raw_path: str) -> Path:
    clean = str(raw_path or "unknown.txt").replace("\\", "/").strip().lstrip("/")
    path = Path(clean)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe generated file path: {raw_path}")
    return path


@app.post("/api/codegen/check")
async def check_codegen_project(req: CodegenProjectRequest):
    """Run safe static checks on generated project files."""
    files = (req.project or {}).get("files") or []
    results: List[Dict[str, Any]] = []
    ok = True
    with tempfile.TemporaryDirectory(prefix="kyma_codegen_check_") as tmp:
        root = Path(tmp)
        for fd in files:
            path = str(fd.get("path") or "unknown.txt")
            content = str(fd.get("content") or "")
            try:
                rel = _safe_codegen_relpath(path)
                target = root / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")
                suffix = target.suffix.lower()
                if suffix == ".py":
                    py_compile.compile(str(target), doraise=True)
                    results.append({"path": path, "status": "pass", "message": "Python syntax OK"})
                elif suffix == ".json":
                    json.loads(content)
                    results.append({"path": path, "status": "pass", "message": "JSON parse OK"})
                elif suffix in {".js", ".mjs", ".cjs"} and shutil.which("node"):
                    proc = subprocess.run(
                        ["node", "--check", str(target)],
                        capture_output=True,
                        text=True,
                        timeout=15,
                    )
                    if proc.returncode:
                        ok = False
                        results.append({"path": path, "status": "fail", "message": (proc.stderr or proc.stdout).strip()})
                    else:
                        results.append({"path": path, "status": "pass", "message": "JavaScript syntax OK"})
                else:
                    results.append({"path": path, "status": "skip", "message": "No safe static checker for this file type"})
            except Exception as exc:
                ok = False
                results.append({"path": path, "status": "fail", "message": str(exc)})
    return {"ok": ok, "results": results}


@app.post("/api/codegen/save")
async def save_codegen_project(req: CodegenProjectRequest):
    """Persist generated project files into sessions/generated_code."""
    files = (req.project or {}).get("files") or []
    stamp = time.strftime("%Y%m%d_%H%M%S")
    root = Path(config.data_dir) / "generated_code" / f"project_{stamp}"
    root.mkdir(parents=True, exist_ok=True)
    saved = []
    for fd in files:
        rel = _safe_codegen_relpath(str(fd.get("path") or "unknown.txt"))
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(str(fd.get("content") or ""), encoding="utf-8")
        saved.append(target.as_posix())
    manifest = {
        "saved_at": time.time(),
        "summary": (req.project or {}).get("summary") or "",
        "files": saved,
    }
    manifest_path = root / "kyma_generated_project.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"ok": True, "root": root.as_posix(), "files": saved, "manifest": manifest_path.as_posix()}


@app.get("/api/codegen/last")
async def get_last_project():
    if hasattr(app_state, 'code_generator') and app_state.code_generator.last_project:
        return {"ok": True, "project": app_state.code_generator.last_project.to_dict()}
    return {"ok": False, "project": None}


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    app_state.websockets.add(ws)
    logger.info("WS connected (total=%s)", len(app_state.websockets))

    await ws.send_text(
        json.dumps(
            {
                "type": "state",
                "data": {"state": app_state.system_state.value},
                "timestamp": time.time(),
            }
        )
    )

    try:
        while True:
            try:
                raw = await asyncio.wait_for(ws.receive_text(), timeout=25.0)
                msg = json.loads(raw)
                await _handle_ws_message(msg)
            except asyncio.TimeoutError:
                await ws.send_text(json.dumps({"type": "ping", "timestamp": time.time()}))
    except WebSocketDisconnect:
        pass
    finally:
        app_state.websockets.discard(ws)
        logger.info("WS disconnected (total=%s)", len(app_state.websockets))


async def _handle_ws_message(msg: dict) -> None:
    msg_type = msg.get("type")
    if msg_type == "pong":
        return
    if msg_type == "estop":
        await estop()
        return
    if msg_type == "home":
        await home()
        return
    if msg_type == "move":
        cmd = MoveCommand(**msg["data"])
        await manual_move(cmd)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=config.host,
        port=config.server_port,
        reload=False,
        log_level="info",
    )
