"""
KYMA — Biosignal Control Server
======================
FastAPI + WebSocket backend.

Environment variables
─────────────────────
EMG_MOCK=1          use simulated Cyton + mock Arduino (no hardware needed)
CYTON_PORT=COMx     override config serial port
ARDUINO_PORT=COMx   override config arduino port
PORT=8000           HTTP port

Run
───
  cd server
  python main.py              (real hardware)
  EMG_MOCK=1 python main.py   (simulation)
"""

import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Optional, Set

import numpy as np
from fastapi import BackgroundTasks, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from arduino_bridge import ArduinoBridge, MockArduinoBridge
from brainflow_stream import CytonStream, SimulatedCytonStream
from calibration import CalibrationManager
from config import config
from emg_pipeline import EMGPipeline
from models import (
    AnalogWriteCommand, ConnectRequest, DigitalWriteCommand,
    MoveCommand, SessionStartRequest, SystemState, TrainRequest,
)
from session_recorder import SessionRecorder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

USE_MOCK = os.getenv("EMG_MOCK", "0") == "1"

# ── Application state ─────────────────────────────────────────────────────────

class AppState:
    def __init__(self):
        self.system_state: SystemState = SystemState.IDLE
        self.stream: Optional[CytonStream] = None
        self.pipeline = EMGPipeline()
        self.arduino: Optional[ArduinoBridge] = None
        self.recorder = SessionRecorder()
        self.calibration = CalibrationManager()
        self.websockets: Set[WebSocket] = set()
        self.last_prediction: Optional[dict] = None

        # Thread-safe broadcast queue (filled from BrainFlow thread)
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._bcast_queue: Optional[asyncio.Queue] = None

    def queue_broadcast(self, msg: dict) -> None:
        """Can be called from any thread."""
        if self._loop and self._bcast_queue:
            self._loop.call_soon_threadsafe(self._bcast_queue.put_nowait, msg)

    def set_state(self, s: SystemState) -> None:
        self.system_state = s
        self.queue_broadcast({
            "type": "state",
            "data": {"state": s.value},
            "timestamp": time.time(),
        })


app_state = AppState()


# ── WebSocket broadcast worker ────────────────────────────────────────────────

async def _broadcast_worker() -> None:
    """
    drains the broadcast queue and fans out to all connected websockets.

    this is the only async path that touches the websocket set, so we
    don't need a lock. if a client is slow or dead we just drop it —
    we'd rather skip a frame than block the whole pipeline waiting for
    one laggy browser tab.
    """
    q = app_state._bcast_queue
    while True:
        msg = await q.get()
        dead: Set[WebSocket] = set()
        # pre-serialize once instead of per-client (saves ~0.5ms per broadcast)
        text = json.dumps(msg, separators=(',', ':'))
        for ws in list(app_state.websockets):
            try:
                await ws.send_text(text)
            except Exception:
                dead.add(ws)
        app_state.websockets -= dead
        q.task_done()


# ── EMG callbacks (called from BrainFlow background thread) ──────────────────

def _on_prediction(class_idx: int, confidence: float, window: np.ndarray) -> None:
    gesture = config.gestures[class_idx]
    app_state.last_prediction = {
        "gesture": gesture,
        "class_idx": class_idx,
        "confidence": confidence,
    }

    # Drive servos when confidence is high enough and ESTOP is not active
    if (
        app_state.arduino
        and not app_state.arduino.estop_active
        and confidence >= config.prediction_confidence_threshold
    ):
        app_state.arduino.execute_gesture(gesture)

    # Record event if session active
    if app_state.recorder.is_recording:
        app_state.recorder.log_event("prediction", {
            "gesture": gesture,
            "confidence": round(confidence, 3),
        })

    app_state.queue_broadcast({
        "type": "prediction",
        "data": {
            "gesture": gesture,
            "class_idx": class_idx,
            "confidence": round(confidence, 3),
            "rms": [
                round(float(np.sqrt(np.mean(window[i] ** 2))), 5)
                for i in range(window.shape[0])
            ],
        },
        "timestamp": time.time(),
    })


def _on_window_vis(window: np.ndarray) -> None:
    """Broadcast only the NEW samples (the increment) for smooth visualization."""
    inc = config.window_increment_samples          # ~12 samples of new data
    new_data = window[:, -inc:]                     # only the non-overlapping tail
    app_state.queue_broadcast({
        "type": "emg",
        "data": {
            "channels": new_data.tolist(),
            "rms": [
                round(float(np.sqrt(np.mean(window[i] ** 2))), 5)
                for i in range(window.shape[0])
            ],
            "n_channels": window.shape[0],
        },
        "timestamp": time.time(),
    })


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Capture event loop for thread-safe queue access
    app_state._loop = asyncio.get_running_loop()
    app_state._bcast_queue = asyncio.Queue()
    asyncio.create_task(_broadcast_worker())

    # Create stream and Arduino
    app_state.stream = SimulatedCytonStream() if USE_MOCK else CytonStream()
    app_state.arduino = MockArduinoBridge() if USE_MOCK else ArduinoBridge()

    # Arduino connect (best-effort; user can reconnect via API)
    app_state.arduino.connect()

    # Register pipeline + visualization callbacks
    app_state.pipeline.add_prediction_callback(_on_prediction)
    app_state.stream.add_window_callback(app_state.pipeline.on_window)
    app_state.stream.add_window_callback(_on_window_vis)

    # Calibration broadcasts
    app_state.calibration.set_status_callback(
        lambda d: app_state.queue_broadcast({
            "type": "calibration",
            "data": d,
            "timestamp": time.time(),
        })
    )

    # Load pre-trained model if available
    if app_state.pipeline.load_model():
        logger.info("Pre-trained model loaded")

    mode = "MOCK" if USE_MOCK else "REAL"
    logger.info(f"KYMA Server ready [{mode}] — http://{config.host}:{config.server_port}")
    yield

    # Shutdown
    if app_state.stream and app_state.stream.is_running:
        app_state.stream.stop()
    if app_state.arduino:
        app_state.arduino.disconnect()
    logger.info("Server stopped")


# ── App factory ───────────────────────────────────────────────────────────────

app = FastAPI(title="KYMA", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the dashboard from ../dashboard/
dashboard_dir = os.path.join(os.path.dirname(__file__), "..", "dashboard")
if os.path.isdir(dashboard_dir):
    app.mount("/static", StaticFiles(directory=dashboard_dir), name="static")


# ── HTTP routes ───────────────────────────────────────────────────────────────

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
        "arduino_connected": app_state.arduino.is_connected if app_state.arduino else False,
        "arduino_estop": app_state.arduino.estop_active if app_state.arduino else False,
        "model_trained": app_state.pipeline.is_trained,
        "is_recording": app_state.recorder.is_recording,
        "session_id": app_state.recorder.session_id,
        "last_prediction": app_state.last_prediction,
        "gestures": config.gestures,
        "mock_mode": USE_MOCK,
    }


@app.get("/api/config")
async def get_config():
    return {
        "serial_port": config.serial_port,
        "arduino_port": config.arduino_port,
        "sample_rate": config.sample_rate,
        "n_channels": config.n_channels,
        "emg_channels": config.emg_channels,
        "gestures": config.gestures,
        "features": config.features,
        "window_size_ms": config.window_size_ms,
        "window_increment_ms": config.window_increment_ms,
        "confidence_threshold": config.prediction_confidence_threshold,
        "mock": USE_MOCK,
    }


@app.get("/api/ports")
async def list_serial_ports():
    """List available serial/COM ports for the frontend port selector."""
    import serial.tools.list_ports
    ports = []
    for p in serial.tools.list_ports.comports():
        ports.append({
            "device": p.device,
            "description": p.description,
            "hwid": p.hwid,
        })
    return {"ports": ports}


# ── Stream ────────────────────────────────────────────────────────────────────

@app.post("/api/stream/start")
async def stream_start(req: ConnectRequest = None):
    if app_state.stream.is_running:
        return {"ok": True, "message": "Already streaming"}

    # Switch stream backend if mode was specified
    mode = (req.mode if req else None)
    if mode == "real" and isinstance(app_state.stream, SimulatedCytonStream):
        new_stream = CytonStream()
        new_stream.add_window_callback(app_state.pipeline.on_window)
        new_stream.add_window_callback(_on_window_vis)
        app_state.stream = new_stream
    elif mode == "mock" and not isinstance(app_state.stream, SimulatedCytonStream):
        new_stream = SimulatedCytonStream()
        new_stream.add_window_callback(app_state.pipeline.on_window)
        new_stream.add_window_callback(_on_window_vis)
        app_state.stream = new_stream

    port = (req.cyton_port if req else None) or config.serial_port
    if not app_state.stream.connect(serial_port=port):
        raise HTTPException(500, "Failed to connect to Cyton board")
    if not app_state.stream.start():
        raise HTTPException(500, "Failed to start stream")
    app_state.set_state(SystemState.STREAMING)
    return {"ok": True}


@app.post("/api/stream/stop")
async def stream_stop():
    if app_state.stream:
        app_state.stream.stop()
    app_state.set_state(SystemState.IDLE)
    return {"ok": True}


# ── Calibration ───────────────────────────────────────────────────────────────

@app.post("/api/calibrate")
async def calibrate(background_tasks: BackgroundTasks):
    if not app_state.stream.is_running:
        raise HTTPException(400, "Stream not running — start stream first")
    app_state.set_state(SystemState.CALIBRATING)

    async def _run():
        result = await app_state.calibration.run_calibration(app_state.stream)
        new_state = SystemState.STREAMING if result.get("success") else SystemState.IDLE
        app_state.set_state(new_state)

    background_tasks.add_task(_run)
    return {"ok": True, "message": "Calibration started"}


# ── Training ──────────────────────────────────────────────────────────────────

@app.post("/api/train/start")
async def train_start(req: TrainRequest):
    if not app_state.stream.is_running:
        raise HTTPException(400, "Stream not running")
    if req.gesture not in config.gestures:
        raise HTTPException(400, f"Unknown gesture. Valid: {config.gestures}")
    app_state.set_state(SystemState.TRAINING)
    app_state.recorder.log_event("train_start", {"gesture": req.gesture})
    app_state.pipeline.start_recording(req.gesture)
    return {"ok": True, "gesture": req.gesture}


@app.post("/api/train/stop")
async def train_stop():
    app_state.pipeline.stop_recording()
    if app_state.stream.is_running:
        app_state.set_state(SystemState.STREAMING)
    return {"ok": True, "summary": app_state.pipeline.get_training_summary()}


@app.post("/api/train/fit")
async def train_fit(classifier: str = "LDA"):
    if classifier not in ("LDA", "TCN", "Mamba"):
        raise HTTPException(400, f"Unknown classifier. Options: LDA, TCN, Mamba")
    result = app_state.pipeline.train(classifier_name=classifier)
    if result.get("success"):
        path = app_state.pipeline.save_model()
        result["model_path"] = path
        app_state.recorder.log_event("model_trained", result)
    return result


@app.post("/api/train/clear")
async def train_clear():
    app_state.pipeline.clear_training_data()
    return {"ok": True}


@app.get("/api/train/summary")
async def train_summary():
    return app_state.pipeline.get_training_summary()


# ── Safety ────────────────────────────────────────────────────────────────────

@app.post("/api/estop")
async def estop():
    if app_state.arduino:
        app_state.arduino.estop()
    app_state.set_state(SystemState.ESTOP)
    app_state.recorder.log_event("estop")
    return {"ok": True, "message": "E-STOP activated"}


@app.post("/api/home")
async def home():
    if app_state.arduino:
        app_state.arduino.home()
    prev_estop = app_state.system_state == SystemState.ESTOP
    if prev_estop or app_state.stream.is_running:
        app_state.set_state(
            SystemState.STREAMING if app_state.stream.is_running else SystemState.IDLE
        )
    app_state.recorder.log_event("home")
    return {"ok": True}


# ── Manual control ────────────────────────────────────────────────────────────

@app.post("/api/move")
async def manual_move(cmd: MoveCommand):
    if not app_state.arduino:
        raise HTTPException(503, "Arduino not connected")
    ok = app_state.arduino.move(cmd.joint_id, cmd.angle)
    return {"ok": ok}


@app.post("/api/gesture/{name}")
async def execute_gesture(name: str):
    if not app_state.arduino:
        raise HTTPException(503, "Arduino not connected")
    if name not in config.gestures:
        raise HTTPException(400, f"Unknown gesture. Valid: {config.gestures}")
    ok = app_state.arduino.execute_gesture(name)
    return {"ok": ok}


@app.post("/api/digital_write")
async def digital_write(cmd: DigitalWriteCommand):
    if not app_state.arduino:
        raise HTTPException(503, "Arduino not connected")
    ok = app_state.arduino.digital_write(cmd.pin, cmd.value)
    return {"ok": ok}


@app.post("/api/analog_write")
async def analog_write(cmd: AnalogWriteCommand):
    if not app_state.arduino:
        raise HTTPException(503, "Arduino not connected")
    ok = app_state.arduino.analog_write(cmd.pin, cmd.value)
    return {"ok": ok}


# ── Session recording ─────────────────────────────────────────────────────────

@app.post("/api/session/start")
async def session_start(req: SessionStartRequest = None):
    label = req.label if req else ""
    session_id = app_state.recorder.start_session(label)
    # Hook recorder into stream
    app_state.stream.add_window_callback(app_state.recorder.record_window)
    return {"ok": True, "session_id": session_id}


@app.post("/api/session/stop")
async def session_stop():
    path = app_state.recorder.stop_session()
    return {"ok": True, "saved_to": path}


@app.get("/api/sessions")
async def list_sessions():
    return app_state.recorder.list_sessions()


# ── Utility ───────────────────────────────────────────────────────────────────

@app.get("/api/ports")
async def list_ports():
    return {"ports": ArduinoBridge.list_ports()}


@app.post("/api/arduino/connect")
async def arduino_connect(req: ConnectRequest = None):
    port = (req.arduino_port if req else None) or config.arduino_port
    ok = app_state.arduino.connect(port=port)
    return {"ok": ok}


# ── WebSocket ─────────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    app_state.websockets.add(ws)
    logger.info(f"WS connected  (total={len(app_state.websockets)})")

    # Push current state immediately on connect
    await ws.send_text(json.dumps({
        "type": "state",
        "data": {"state": app_state.system_state.value},
        "timestamp": time.time(),
    }))

    try:
        while True:
            try:
                raw = await asyncio.wait_for(ws.receive_text(), timeout=25.0)
                msg = json.loads(raw)
                await _handle_ws_message(ws, msg)
            except asyncio.TimeoutError:
                # Keep-alive ping
                await ws.send_text(json.dumps({"type": "ping", "timestamp": time.time()}))
    except WebSocketDisconnect:
        pass
    finally:
        app_state.websockets.discard(ws)
        logger.info(f"WS disconnected (total={len(app_state.websockets)})")


async def _handle_ws_message(ws: WebSocket, msg: dict) -> None:
    t = msg.get("type")
    if t == "pong":
        return
    elif t == "estop":
        await estop()
    elif t == "home":
        await home()
    elif t == "move":
        cmd = MoveCommand(**msg["data"])
        await manual_move(cmd)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=config.host,
        port=config.server_port,
        reload=False,
        log_level="info",
    )
