"""
Session recorder — persists raw EMG windows and event log to disk.

Each session creates:
  sessions/<id>/emg_raw.csv   — one row per sample, columns = ch1…ch8
  sessions/<id>/events.json   — labelled events (training, predictions, estop…)
  sessions/<id>/meta.json     — session metadata
"""

import csv
import json
import logging
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from config import config

logger = logging.getLogger(__name__)


class SessionRecorder:
    def __init__(self):
        self._recording: bool = False
        self._session_id: Optional[str] = None
        self._session_dir: Optional[str] = None
        self._start_time: Optional[float] = None

        self._raw_fh = None
        self._raw_writer = None
        self._events: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

        Path(config.data_dir).mkdir(exist_ok=True)

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def session_id(self) -> Optional[str]:
        return self._session_id

    # ── Session lifecycle ─────────────────────────────────────────────────

    def start_session(self, label: str = "") -> str:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._session_id = f"{ts}_{label}" if label else ts
        self._session_dir = os.path.join(config.data_dir, self._session_id)
        Path(self._session_dir).mkdir(parents=True, exist_ok=True)

        # Open CSV
        raw_path = os.path.join(self._session_dir, "emg_raw.csv")
        ch_headers = [f"ch{i + 1}" for i in range(config.n_channels)]
        self._raw_fh = open(raw_path, "w", newline="", buffering=1)
        self._raw_writer = csv.writer(self._raw_fh)
        self._raw_writer.writerow(["timestamp_s"] + ch_headers)

        self._events = []
        self._start_time = time.monotonic()
        self._recording = True
        logger.info(f"Session started: {self._session_id}")
        return self._session_id

    def stop_session(self) -> Optional[str]:
        if not self._recording:
            return None
        self._recording = False

        if self._raw_fh:
            self._raw_fh.flush()
            self._raw_fh.close()
            self._raw_fh = None
            self._raw_writer = None

        if self._session_dir:
            self._write_events()
            self._write_meta()

        path = self._session_dir
        logger.info(f"Session saved → {path}")
        return path

    # ── Data recording ────────────────────────────────────────────────────

    def record_window(self, window: np.ndarray) -> None:
        """Called with each EMG window: shape (n_ch, window_size)."""
        if not self._recording or self._raw_writer is None:
            return
        elapsed = time.monotonic() - (self._start_time or 0.0)
        with self._lock:
            # One row per sample column (avoids large per-window writes)
            for i in range(window.shape[1]):
                t = elapsed + i / config.sample_rate
                row = [f"{t:.5f}"] + [f"{v:.6f}" for v in window[:, i]]
                self._raw_writer.writerow(row)

    def log_event(self, event_type: str, data: Dict[str, Any] = None) -> None:
        elapsed = time.monotonic() - (self._start_time or 0.0)
        with self._lock:
            self._events.append({
                "type": event_type,
                "timestamp_s": round(elapsed, 4),
                "data": data or {},
            })

    # ── Session listing ───────────────────────────────────────────────────

    def list_sessions(self) -> List[Dict]:
        result = []
        base = Path(config.data_dir)
        if not base.exists():
            return result
        for d in sorted(base.iterdir(), reverse=True):
            meta_path = d / "meta.json"
            if meta_path.exists():
                try:
                    with open(meta_path) as fh:
                        result.append(json.load(fh))
                except Exception:
                    result.append({"session_id": d.name})
        return result

    # ── Internal ──────────────────────────────────────────────────────────

    def _write_events(self) -> None:
        path = os.path.join(self._session_dir, "events.json")
        with open(path, "w") as fh:
            json.dump(self._events, fh, indent=2)

    def _write_meta(self) -> None:
        elapsed = time.monotonic() - (self._start_time or 0.0)
        meta = {
            "session_id": self._session_id,
            "duration_s": round(elapsed, 2),
            "n_events": len(self._events),
            "config": {
                "sample_rate": config.sample_rate,
                "channels": config.emg_channels,
                "features": config.features,
                "gestures": config.gestures,
            },
        }
        path = os.path.join(self._session_dir, "meta.json")
        with open(path, "w") as fh:
            json.dump(meta, fh, indent=2)
