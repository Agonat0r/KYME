"""Import XDF recordings into normal KYMA sessions for playback and export."""

from __future__ import annotations

import csv
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from biosignal_profiles import get_profile
from config import config

logger = logging.getLogger(__name__)

try:
    import pyxdf

    _PYXDF_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - optional dependency
    pyxdf = None  # type: ignore[assignment]
    _PYXDF_ERROR = exc


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _safe_label(label: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9._-]+", "_", (label or "").strip())
    return clean.strip("._-")[:40]


def _flatten_first(value: Any, default: str = "") -> str:
    if isinstance(value, list):
        if not value:
            return default
        return _flatten_first(value[0], default=default)
    if value is None:
        return default
    return str(value)


def _channel_labels_from_info(info: Dict[str, Any], count: int) -> List[str]:
    desc = info.get("desc")
    if isinstance(desc, list) and desc:
        desc = desc[0]
    channels = []
    if isinstance(desc, dict):
        channel_group = desc.get("channels")
        if isinstance(channel_group, list) and channel_group:
            channel_group = channel_group[0]
        if isinstance(channel_group, dict):
            channels = channel_group.get("channel") or []

    labels: List[str] = []
    if isinstance(channels, list):
        for idx, channel in enumerate(channels[:count], start=1):
            if isinstance(channel, dict):
                labels.append(_flatten_first(channel.get("label"), f"CH{idx}"))

    while len(labels) < count:
        labels.append(f"CH{len(labels) + 1}")
    return labels


def _sample_rate_from_info(info: Dict[str, Any], timestamps: np.ndarray) -> int:
    nominal = float(_flatten_first(info.get("nominal_srate"), "0") or 0.0)
    if nominal > 0:
        return int(round(nominal))

    if timestamps.size < 2:
        return config.sample_rate

    diffs = np.diff(timestamps.astype(np.float64))
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size == 0:
        return config.sample_rate
    return int(round(1.0 / np.median(diffs)))


def _infer_profile(explicit: Optional[str], stream_name: str, stream_type: str) -> str:
    if explicit:
        return get_profile(explicit).key

    haystack = f"{stream_name} {stream_type}".strip().lower()
    for key in ("emg", "eeg", "ecg", "eog", "eda", "ppg", "resp", "temp"):
        if key in haystack:
            return key
        if key == "resp" and "respiration" in haystack:
            return "resp"
    return config.signal_profile_name


def _stream_id(stream: Dict[str, Any]) -> str:
    info = stream.get("info", {})
    return _flatten_first(info.get("stream_id"), "")


def _stream_name(stream: Dict[str, Any]) -> str:
    info = stream.get("info", {})
    return _flatten_first(info.get("name"), "")


def _stream_type(stream: Dict[str, Any]) -> str:
    info = stream.get("info", {})
    return _flatten_first(info.get("type"), "")


def _is_numeric_signal_stream(stream: Dict[str, Any]) -> bool:
    try:
        arr = np.asarray(stream.get("time_series"))
    except Exception:
        return False
    return arr.size > 0 and arr.ndim in (1, 2) and np.issubdtype(arr.dtype, np.number)


class XDFImporter:
    """Create session folders from XDF recordings so they plug into playback/export."""

    def __init__(self) -> None:
        self.session_root = Path(config.data_dir)
        self.session_root.mkdir(parents=True, exist_ok=True)

    @property
    def available(self) -> bool:
        return pyxdf is not None

    @property
    def last_error(self) -> str:
        if _PYXDF_ERROR is None:
            return ""
        return f"{type(_PYXDF_ERROR).__name__}: {_PYXDF_ERROR}"

    def inspect(self, path: str) -> Dict[str, Any]:
        if pyxdf is None:
            raise RuntimeError(self.last_error or "pyxdf is not installed")

        file_path = Path(path).expanduser()
        if not file_path.exists():
            raise FileNotFoundError(f"XDF file not found: {file_path}")

        streams, _ = pyxdf.load_xdf(str(file_path))
        items: List[Dict[str, Any]] = []
        for stream in streams:
            info = stream.get("info", {})
            timestamps = np.asarray(stream.get("time_stamps", []), dtype=np.float64)
            samples = np.asarray(stream.get("time_series", []))
            channel_count = int(_flatten_first(info.get("channel_count"), "0") or 0)
            if channel_count <= 0 and samples.ndim == 2:
                channel_count = int(samples.shape[1])

            items.append(
                {
                    "stream_id": _stream_id(stream),
                    "name": _stream_name(stream),
                    "type": _stream_type(stream),
                    "sample_count": int(samples.shape[0]) if samples.ndim >= 1 else 0,
                    "channel_count": channel_count,
                    "sample_rate": _sample_rate_from_info(info, timestamps),
                    "is_numeric": _is_numeric_signal_stream(stream),
                    "suggested_profile": _infer_profile(None, _stream_name(stream), _stream_type(stream)),
                }
            )

        return {
            "ok": True,
            "path": str(file_path),
            "streams": items,
        }

    def import_session(
        self,
        *,
        path: str,
        stream_name: Optional[str] = None,
        stream_id: Optional[str] = None,
        signal_profile: Optional[str] = None,
        label: str = "",
    ) -> Dict[str, Any]:
        if pyxdf is None:
            raise RuntimeError(self.last_error or "pyxdf is not installed")

        file_path = Path(path).expanduser()
        if not file_path.exists():
            raise FileNotFoundError(f"XDF file not found: {file_path}")

        streams, _ = pyxdf.load_xdf(str(file_path))
        signal_stream = self._select_signal_stream(streams, stream_name=stream_name, stream_id=stream_id)
        if signal_stream is None:
            raise ValueError("No numeric signal stream found in the XDF file")

        info = signal_stream.get("info", {})
        stream_name = _stream_name(signal_stream)
        stream_id = _stream_id(signal_stream)
        stream_type = _stream_type(signal_stream)

        raw = np.asarray(signal_stream.get("time_series"), dtype=np.float64)
        if raw.ndim == 1:
            raw = raw[:, np.newaxis]
        if raw.ndim != 2 or raw.shape[0] == 0:
            raise ValueError("Selected XDF stream does not contain a usable 2D signal array")

        timestamps = np.asarray(signal_stream.get("time_stamps", []), dtype=np.float64)
        if timestamps.size != raw.shape[0]:
            timestamps = np.arange(raw.shape[0], dtype=np.float64) / max(config.sample_rate, 1)
        timestamps = timestamps - float(timestamps[0])

        sample_rate = _sample_rate_from_info(info, timestamps)
        profile_key = _infer_profile(signal_profile, stream_name, stream_type)
        profile = get_profile(profile_key)
        channel_labels = _channel_labels_from_info(info, raw.shape[1])

        session_id = self._make_session_id(file_path.stem, label or stream_name or stream_type or "xdf")
        session_dir = self.session_root / session_id
        session_dir.mkdir(parents=True, exist_ok=False)

        signal_path = session_dir / "signal_raw.csv"
        self._write_signal_csv(signal_path, timestamps, raw)
        raw_timestamps = np.asarray(signal_stream.get("time_stamps", []), dtype=np.float64)
        start_time = float(raw_timestamps[0]) if raw_timestamps.size else 0.0
        events = self._collect_marker_events(streams, start_time=start_time)
        self._write_events(session_dir / "events.json", events)
        self._write_meta(
            session_dir=session_dir,
            session_id=session_id,
            label=label or stream_name or file_path.stem,
            sample_rate=sample_rate,
            n_samples=raw.shape[0],
            duration_s=float(timestamps[-1]) if timestamps.size else 0.0,
            stream_name=stream_name,
            stream_id=stream_id,
            stream_type=stream_type,
            channel_labels=channel_labels,
            signal_profile=profile.key,
            units=profile.units,
            xdf_path=str(file_path),
            events=events,
        )

        logger.info("Imported XDF session %s from %s", session_id, file_path)
        return {
            "ok": True,
            "session_id": session_id,
            "session_path": str(session_dir),
            "signal_profile": profile.key,
            "stream_name": stream_name,
            "stream_id": stream_id,
            "sample_rate": sample_rate,
            "n_channels": int(raw.shape[1]),
            "n_samples": int(raw.shape[0]),
        }

    def _select_signal_stream(
        self,
        streams: Sequence[Dict[str, Any]],
        *,
        stream_name: Optional[str],
        stream_id: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        matches: List[Dict[str, Any]] = []
        for stream in streams:
            if not _is_numeric_signal_stream(stream):
                continue
            if stream_id and _stream_id(stream) == stream_id:
                return stream
            if stream_name and _stream_name(stream) == stream_name:
                matches.append(stream)
            elif not stream_name and not stream_id:
                matches.append(stream)
        return matches[0] if matches else None

    def _collect_marker_events(self, streams: Sequence[Dict[str, Any]], start_time: float) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        for stream in streams:
            if _is_numeric_signal_stream(stream):
                continue

            name = _stream_name(stream) or "marker"
            stype = _stream_type(stream) or "marker"
            samples = np.asarray(stream.get("time_series"), dtype=object)
            timestamps = np.asarray(stream.get("time_stamps", []), dtype=np.float64)
            if samples.ndim == 0:
                samples = samples.reshape(1, 1)
            elif samples.ndim == 1:
                samples = samples[:, np.newaxis]

            for idx in range(min(samples.shape[0], timestamps.size)):
                payload = samples[idx].tolist()
                text = " ".join(str(item) for item in payload if str(item).strip()).strip()
                events.append(
                    {
                        "type": "xdf_marker",
                        "timestamp_s": round(float(timestamps[idx] - start_time), 4),
                        "data": {
                            "stream": name,
                            "stream_type": stype,
                            "value": text or name,
                        },
                    }
                )

        events.sort(key=lambda item: item.get("timestamp_s", 0.0))
        return events

    def _make_session_id(self, stem: str, label: str) -> str:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe = _safe_label(label or stem)
        base = f"{ts}_{safe}" if safe else ts
        candidate = self.session_root / base
        suffix = 1
        while candidate.exists():
            candidate = self.session_root / f"{base}_{suffix}"
            suffix += 1
        return candidate.name

    def _write_signal_csv(self, path: Path, timestamps: np.ndarray, raw: np.ndarray) -> None:
        with open(path, "w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["timestamp_s"] + [f"ch{i + 1}" for i in range(raw.shape[1])])
            for idx in range(raw.shape[0]):
                row = [f"{float(timestamps[idx]):.5f}"] + [f"{float(value):.6f}" for value in raw[idx]]
                writer.writerow(row)

    def _write_events(self, path: Path, events: List[Dict[str, Any]]) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(events, fh, indent=2)

    def _write_meta(
        self,
        *,
        session_dir: Path,
        session_id: str,
        label: str,
        sample_rate: int,
        n_samples: int,
        duration_s: float,
        stream_name: str,
        stream_id: str,
        stream_type: str,
        channel_labels: List[str],
        signal_profile: str,
        units: str,
        xdf_path: str,
        events: List[Dict[str, Any]],
    ) -> None:
        meta = {
            "schema_version": 2,
            "session_id": session_id,
            "label": label,
            "created_at_utc": _utc_now_iso(),
            "ended_at_utc": _utc_now_iso(),
            "duration_s": round(float(duration_s), 2),
            "n_events": len(events),
            "n_samples": int(n_samples),
            "stream_source": "xdf_import",
            "source_details": {
                "xdf_path": xdf_path,
                "stream_name": stream_name,
                "stream_id": stream_id,
                "stream_type": stream_type,
            },
            "artifacts": {
                "signal_raw_csv": "signal_raw.csv",
                "events_json": "events.json",
                "meta_json": "meta.json",
            },
            "config": {
                "sample_rate": int(sample_rate),
                "window_size_ms": config.window_size_ms,
                "window_increment_ms": config.window_increment_ms,
                "signal_profile": signal_profile,
                "profile": get_profile(signal_profile).to_dict(),
                "channels": list(range(1, len(channel_labels) + 1)),
                "channel_labels": channel_labels,
                "units": units,
                "features": list(get_profile(signal_profile).default_features),
                "class_labels": list(get_profile(signal_profile).class_labels),
            },
        }
        with open(session_dir / "meta.json", "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)
