"""
Session recorder and archive helpers for KYMA.

Each session creates:
  sessions/<id>/signal_raw.csv   - one row per sample, columns = ch1..ch8
  sessions/<id>/events.json      - labelled events (training, predictions, estop..)
  sessions/<id>/meta.json        - session metadata and artifact manifest
"""

from __future__ import annotations

import csv
import json
import logging
import os
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from biosignal_profiles import get_profile
from config import config

logger = logging.getLogger(__name__)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _safe_label(label: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9._-]+", "_", (label or "").strip())
    return clean.strip("._-")[:40]


class SessionRecorder:
    def __init__(self):
        self._recording: bool = False
        self._session_id: Optional[str] = None
        self._session_label: str = ""
        self._session_dir: Optional[str] = None
        self._start_time: Optional[float] = None
        self._started_at_utc: Optional[str] = None
        self._stream_source: str = "hardware"
        self._source_details: Dict[str, Any] = {}
        self._session_metadata: Dict[str, Any] = {}
        self._raw_samples_written: int = 0

        self._raw_fh = None
        self._raw_writer = None
        self._events: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

        Path(config.data_dir).mkdir(parents=True, exist_ok=True)

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def session_id(self) -> Optional[str]:
        return self._session_id

    def start_session(
        self,
        label: str = "",
        stream_source: str = "hardware",
        source_details: Optional[Dict[str, Any]] = None,
        session_metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        if self._recording:
            raise RuntimeError("A session is already being recorded")

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_label = _safe_label(label)
        self._session_label = label.strip()
        self._session_id = f"{ts}_{safe_label}" if safe_label else ts
        self._session_dir = os.path.join(config.data_dir, self._session_id)
        Path(self._session_dir).mkdir(parents=True, exist_ok=True)

        raw_path = os.path.join(self._session_dir, "signal_raw.csv")
        ch_headers = [f"ch{i + 1}" for i in range(config.n_channels)]
        self._raw_fh = open(raw_path, "w", newline="", buffering=1)
        self._raw_writer = csv.writer(self._raw_fh)
        self._raw_writer.writerow(["timestamp_s"] + ch_headers)

        self._events = []
        self._start_time = time.monotonic()
        self._started_at_utc = _utc_now_iso()
        self._stream_source = stream_source or "hardware"
        self._source_details = dict(source_details or {})
        self._session_metadata = dict(session_metadata or {})
        self._raw_samples_written = 0
        self._recording = True
        logger.info("Session started: %s", self._session_id)
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
        logger.info("Session saved -> %s", path)
        return path

    def record_chunk(self, samples: np.ndarray) -> None:
        """Record non-overlapping samples with shape (n_ch, n_samples)."""
        if not self._recording or self._raw_writer is None or samples.size == 0:
            return

        if samples.ndim != 2:
            raise ValueError(f"Expected chunk shape (n_ch, n_samples), got {samples.shape}")

        with self._lock:
            for i in range(samples.shape[1]):
                t = self._raw_samples_written / config.sample_rate
                row = [f"{t:.5f}"] + [f"{float(v):.6f}" for v in samples[:, i]]
                self._raw_writer.writerow(row)
                self._raw_samples_written += 1

    def record_window(self, window: np.ndarray) -> None:
        """
        Backward-compatible fallback for callers still sending full windows.

        The first callback writes the full window, then later callbacks only
        persist the newly advanced increment so overlapping windows do not
        duplicate samples on disk.
        """
        if window.ndim != 2 or window.size == 0:
            return
        if self._raw_samples_written == 0:
            self.record_chunk(window)
            return
        inc = min(window.shape[1], config.window_increment_samples)
        self.record_chunk(window[:, -inc:])

    def log_event(self, event_type: str, data: Dict[str, Any] = None) -> None:
        elapsed = time.monotonic() - (self._start_time or 0.0)
        with self._lock:
            self._events.append(
                {
                    "type": event_type,
                    "timestamp_s": round(elapsed, 4),
                    "data": data or {},
                }
            )

    def list_sessions(self) -> List[Dict[str, Any]]:
        result: List[Dict[str, Any]] = []
        base = Path(config.data_dir)
        if not base.exists():
            return result
        for session_dir in sorted(base.iterdir(), reverse=True):
            if not session_dir.is_dir():
                continue
            if session_dir.name in {"datasets", "experiments", "exports"}:
                continue
            meta = self.get_session_meta(session_dir.name)
            if meta:
                result.append(self._build_summary(meta, session_dir))
            else:
                result.append(
                    {
                        "session_id": session_dir.name,
                        "stream_source": "unknown",
                        "signal_profile": "unknown",
                        "playable": False,
                    }
                )
        return result

    def get_session_meta(self, session_id: str) -> Optional[Dict[str, Any]]:
        meta_path = self._meta_path(session_id)
        if not meta_path.exists():
            return None
        try:
            with open(meta_path, encoding="utf-8") as fh:
                return json.load(fh)
        except Exception as exc:
            logger.warning("Failed to read session metadata for %s: %s", session_id, exc)
            return None

    def get_session_path(self, session_id: str) -> Optional[str]:
        session_dir = self._session_path(session_id)
        return str(session_dir) if session_dir.exists() else None

    def get_signal_path(self, session_id: str) -> Optional[str]:
        raw_path = self._resolve_signal_path(self._session_path(session_id))
        return str(raw_path) if raw_path.exists() else None

    def get_session_summary(self, session_id: str) -> Optional[Dict[str, Any]]:
        session_dir = self._session_path(session_id)
        if not session_dir.exists():
            return None
        meta = self.get_session_meta(session_id)
        if not meta:
            return None
        return self._build_summary(meta, session_dir)

    def register_export(self, session_id: str, export_name: str, export_info: Dict[str, Any]) -> None:
        meta = self.get_session_meta(session_id)
        if not meta:
            raise FileNotFoundError(f"Session not found: {session_id}")
        exports = meta.setdefault("exports", {})
        exports[export_name] = export_info
        with open(self._meta_path(session_id), "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)

    def _session_path(self, session_id: str) -> Path:
        return Path(config.data_dir) / session_id

    def _meta_path(self, session_id: str) -> Path:
        return self._session_path(session_id) / "meta.json"

    def _resolve_signal_path(self, session_dir: Path) -> Path:
        current = session_dir / "signal_raw.csv"
        legacy = session_dir / "emg_raw.csv"
        return current if current.exists() else legacy

    def _signal_file_has_samples(self, raw_path: Path) -> bool:
        if not raw_path.exists():
            return False
        try:
            with open(raw_path, encoding="utf-8") as fh:
                next(fh, None)
                return next(fh, None) is not None
        except Exception:
            return False

    def _count_signal_samples(self, raw_path: Path) -> int:
        if not raw_path.exists():
            return 0
        try:
            with open(raw_path, encoding="utf-8") as fh:
                next(fh, None)
                return sum(1 for _ in fh)
        except Exception:
            return 0

    def _infer_profile_key(self, meta: Dict[str, Any]) -> str:
        cfg = meta.get("config", {})
        explicit = meta.get("signal_profile") or cfg.get("signal_profile")
        if explicit:
            return str(explicit)

        legacy_gestures = cfg.get("gestures") or cfg.get("class_labels") or []
        legacy_features = {str(name).upper() for name in cfg.get("features", [])}
        if legacy_gestures or {"MAV", "RMS", "WL", "ZC", "SSC"} & legacy_features:
            return "emg"
        return "unknown"

    def _write_events(self) -> None:
        path = os.path.join(self._session_dir, "events.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self._events, fh, indent=2)

    def _write_meta(self) -> None:
        elapsed = time.monotonic() - (self._start_time or 0.0)
        profile = config.signal_profile
        meta = {
            "schema_version": 3,
            "session_id": self._session_id,
            "label": self._session_label,
            "created_at_utc": self._started_at_utc,
            "ended_at_utc": _utc_now_iso(),
            "duration_s": round(elapsed, 2),
            "n_events": len(self._events),
            "n_samples": self._raw_samples_written,
            "stream_source": self._stream_source,
            "source_details": self._source_details,
            "subject_id": str(self._session_metadata.get("subject_id") or "").strip(),
            "condition": str(self._session_metadata.get("condition") or "").strip(),
            "notes": str(self._session_metadata.get("notes") or "").strip(),
            "protocol": {
                "key": str(self._session_metadata.get("protocol_key") or "").strip(),
                "title": str(self._session_metadata.get("protocol_title") or "").strip(),
                "session_group_id": str(self._session_metadata.get("session_group_id") or "").strip(),
                "trial_index": self._session_metadata.get("trial_index"),
                "repetition_index": self._session_metadata.get("repetition_index"),
            },
            "artifacts": {
                "signal_raw_csv": "signal_raw.csv",
                "events_json": "events.json",
                "meta_json": "meta.json",
            },
            "config": {
                "sample_rate": config.sample_rate,
                "window_size_ms": config.window_size_ms,
                "window_increment_ms": config.window_increment_ms,
                "signal_profile": config.signal_profile_name,
                "profile": profile.to_dict(),
                "channels": config.signal_channels,
                "channel_labels": config.channel_labels,
                "units": profile.units,
                "features": config.features,
                "class_labels": config.class_labels,
            },
        }
        path = os.path.join(self._session_dir, "meta.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)

    def _build_summary(self, meta: Dict[str, Any], session_dir: Path) -> Dict[str, Any]:
        cfg = meta.get("config", {})
        profile_key = self._infer_profile_key(meta)
        sample_rate = int(meta.get("sample_rate") or cfg.get("sample_rate") or 0)
        channels = meta.get("n_channels") or len(cfg.get("channels", [])) or config.n_channels
        stream_source = meta.get("stream_source") or meta.get("source") or "unknown"
        label = meta.get("label") or ""
        created_at = meta.get("created_at_utc") or meta.get("started_at_utc")
        raw_path = self._resolve_signal_path(session_dir)
        n_samples = int(meta.get("n_samples") or 0)
        protocol_meta = meta.get("protocol") or {}
        if n_samples <= 0 and raw_path.exists():
            n_samples = self._count_signal_samples(raw_path)
        playable = (
            self._signal_file_has_samples(raw_path)
            and sample_rate == config.sample_rate
            and channels == config.n_channels
        )

        try:
            profile = get_profile(profile_key)
            profile_name = profile.display_name
            units = cfg.get("units") or profile.units
        except Exception:
            profile_name = str(profile_key).upper()
            units = cfg.get("units") or "a.u."

        return {
            "session_id": meta.get("session_id") or session_dir.name,
            "label": label,
            "created_at_utc": created_at,
            "duration_s": float(meta.get("duration_s") or 0.0),
            "n_events": int(meta.get("n_events") or 0),
            "n_samples": n_samples,
            "sample_rate": sample_rate,
            "stream_source": stream_source,
            "source_details": meta.get("source_details") or {},
            "subject_id": str(meta.get("subject_id") or ""),
            "condition": str(meta.get("condition") or ""),
            "notes": str(meta.get("notes") or ""),
            "session_group_id": str(protocol_meta.get("session_group_id") or ""),
            "protocol_key": str(protocol_meta.get("key") or ""),
            "protocol_title": str(protocol_meta.get("title") or ""),
            "trial_index": protocol_meta.get("trial_index"),
            "repetition_index": protocol_meta.get("repetition_index"),
            "signal_profile": profile_key,
            "signal_profile_name": profile_name,
            "units": units,
            "playable": playable,
        }
