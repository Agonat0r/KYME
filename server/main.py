"""
KYMA - Biosignal Control Server
===============================
FastAPI + WebSocket backend.
"""

import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, Set

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
from live_diagnostics import LiveDiagnostics, SafetyWatchdog
from lsl_bridge import LSLBridge
from lsl_input import LSLInletStream, list_lsl_streams
from osc_bridge import OSCBridge
from signal_workshop import SignalWorkshop
from models import (
    AnalogWriteCommand,
    ConnectRequest,
    DatasetCreateRequest,
    DigitalWriteCommand,
    ExperimentRunRequest,
    FilterActivateRequest,
    FilterDesignRequest,
    LSLMarkerRequest,
    LSLStartRequest,
    MoveCommand,
    OSCStartRequest,
    ProfileRequest,
    ReviewMarkerRequest,
    SessionStartRequest,
    SubjectUpsertRequest,
    SystemState,
    TrainRequest,
    WorkshopAnalyzeRequest,
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
        self.diagnostics = LiveDiagnostics(
            sample_rate=config.sample_rate,
            window_increment_ms=config.window_increment_ms,
            full_scale=config.signal_profile.display_full_scale,
        )
        self.watchdog = SafetyWatchdog()
        self.workshop = SignalWorkshop()
        self.eeg_brain = EEGBrainVisualizer(
            sample_rate=config.sample_rate,
            channel_labels=list(get_profile("eeg").channel_labels),
            root_dir=Path(__file__).resolve().parents[1],
        )
        self.websockets: Set[WebSocket] = set()
        self.last_prediction: Optional[dict] = None
        self.last_vis_window: Optional[np.ndarray] = None
        self.last_vis_timestamp: float = 0.0
        self.last_diagnostics_broadcast_ts: float = 0.0
        self.last_safety_broadcast_ts: float = 0.0

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._bcast_queue: Optional[asyncio.Queue] = None

    def queue_broadcast(self, msg: dict) -> None:
        if self._loop and self._bcast_queue:
            self._loop.call_soon_threadsafe(self._bcast_queue.put_nowait, msg)

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


def _rebuild_pipeline() -> None:
    app_state.pipeline = BiosignalPipeline()
    app_state.pipeline.add_prediction_callback(_on_prediction)
    app_state.last_prediction = None
    app_state.diagnostics.reset(full_scale=config.signal_profile.display_full_scale)


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
    lsl_stream_name: Optional[str] = None,
    lsl_source_id: Optional[str] = None,
) -> CytonStream:
    if source == "synthetic":
        stream: CytonStream = SimulatedCytonStream()
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
    if app_state.pipeline.load_model():
        logger.info("Loaded saved %s model for profile switch", profile.display_name)
    playback_rate = float(app_state.stream_details.get("playback_rate", 1.0))
    app_state.stream = _make_stream(
        app_state.stream_source,
        playback_session_id=app_state.playback_session_id,
        playback_rate=playback_rate,
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

    app_state.last_prediction = {
        "label": label,
        "gesture": label,  # compatibility with the current dashboard
        "class_idx": class_idx,
        "confidence": confidence,
        "summary": summary,
        "metrics": metrics,
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
    app_state.queue_broadcast(
        {
            "type": "emg",  # kept for frontend compatibility
            "data": {
                "channels": new_data.tolist(),
                "rms": [
                    round(float(np.sqrt(np.mean(window[i] ** 2))), 5)
                    for i in range(window.shape[0])
                ],
                "quality": [
                    round(float(v), 3)
                    for v in app_state.pipeline.get_channel_quality(window)
                ],
                "n_channels": window.shape[0],
                "profile": config.signal_profile_name,
                "units": config.signal_profile.units,
            },
            "timestamp": time.time(),
        }
    )
    now = time.time()
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    app_state._loop = asyncio.get_running_loop()
    app_state._bcast_queue = asyncio.Queue()
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

    source = _resolve_stream_source(req)
    app_state.stream_details = {}

    if req and req.arduino_port and app_state.arduino:
        app_state.arduino.connect(port=req.arduino_port)

    if source == "synthetic":
        app_state.stream = _make_stream("synthetic")
        logger.info("Connecting synthetic %s stream", config.signal_profile.display_name)
        if not app_state.stream.connect():
            raise HTTPException(500, "Failed to initialize synthetic biosignal stream")
        app_state.stream_details = {"mode": "synthetic"}
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
        if not app_state.stream.connect(serial_port=port):
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
    if not _control_output_ready():
        raise HTTPException(503, "No control output is active. Connect Arduino or start OSC.")
    if name not in config.class_labels:
        raise HTTPException(400, f"Unknown gesture. Valid: {config.class_labels}")
    ok = _dispatch_gesture(name)
    return {"ok": ok}


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


@app.post("/api/arduino/connect")
async def arduino_connect(req: ConnectRequest = None):
    port = (req.arduino_port if req else None) or config.arduino_port
    ok = app_state.arduino.connect(port=port)
    return {"ok": ok}


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
