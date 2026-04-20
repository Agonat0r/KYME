"""LSL discovery helpers and an inlet-backed stream source for KYMA."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from brainflow_stream import CytonStream
from config import config

logger = logging.getLogger(__name__)

try:
    from mne_lsl.lsl import resolve_streams as resolve_streams_mne

    _MNE_LSL_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - optional dependency
    resolve_streams_mne = None  # type: ignore[assignment]
    _MNE_LSL_ERROR = exc

try:
    from pylsl import (
        StreamInlet,
        StreamInfo,
        proc_clocksync,
        proc_dejitter,
        resolve_streams as resolve_streams_pylsl,
    )

    _PYLSL_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - optional dependency
    StreamInlet = None  # type: ignore[assignment]
    StreamInfo = None  # type: ignore[assignment]
    proc_clocksync = 0  # type: ignore[assignment]
    proc_dejitter = 0  # type: ignore[assignment]
    resolve_streams_pylsl = None  # type: ignore[assignment]
    _PYLSL_ERROR = exc


def _value(obj: Any, attr: str, default: Any = None) -> Any:
    raw = getattr(obj, attr, default)
    if callable(raw):
        try:
            return raw()
        except TypeError:
            return default
    return raw


def _coerce_list(raw: Optional[Iterable[Any]]) -> List[str]:
    if raw is None:
        return []
    try:
        return [str(item) for item in list(raw)]
    except Exception:
        return []


def _mne_info_to_dict(info: Any) -> Dict[str, Any]:
    return {
        "name": str(_value(info, "name", "") or ""),
        "type": str(_value(info, "stype", "") or ""),
        "source_id": str(_value(info, "source_id", "") or ""),
        "uid": str(_value(info, "uid", "") or ""),
        "hostname": str(_value(info, "hostname", "") or ""),
        "n_channels": int(_value(info, "n_channels", 0) or 0),
        "sample_rate": float(_value(info, "sfreq", 0.0) or 0.0),
        "dtype": str(_value(info, "dtype", "") or ""),
        "channel_labels": _coerce_list(_value(info, "get_channel_names", [])),
        "channel_types": _coerce_list(_value(info, "get_channel_types", [])),
        "channel_units": _coerce_list(_value(info, "get_channel_units", [])),
        "backend": "mne-lsl",
    }


def _pylsl_info_to_dict(info: Any) -> Dict[str, Any]:
    return {
        "name": str(_value(info, "name", "") or ""),
        "type": str(_value(info, "type", "") or ""),
        "source_id": str(_value(info, "source_id", "") or ""),
        "uid": str(_value(info, "uid", "") or ""),
        "hostname": str(_value(info, "hostname", "") or ""),
        "n_channels": int(_value(info, "channel_count", 0) or 0),
        "sample_rate": float(_value(info, "nominal_srate", 0.0) or 0.0),
        "dtype": str(_value(info, "channel_format", "") or ""),
        "channel_labels": _coerce_list(_value(info, "get_channel_labels", [])),
        "channel_types": _coerce_list(_value(info, "get_channel_types", [])),
        "channel_units": _coerce_list(_value(info, "get_channel_units", [])),
        "backend": "pylsl",
    }


def list_lsl_streams(timeout: float = 0.35) -> Dict[str, Any]:
    """Return currently discoverable LSL streams."""

    streams: List[Dict[str, Any]] = []
    error = ""

    if resolve_streams_mne is not None:
        try:
            streams = [_mne_info_to_dict(info) for info in resolve_streams_mne(timeout=timeout, minimum=0)]
        except Exception as exc:  # pragma: no cover - depends on runtime network stack
            error = f"{type(exc).__name__}: {exc}"
            logger.warning("mne-lsl discovery failed: %s", exc)

    if not streams and resolve_streams_pylsl is not None:
        try:
            streams = [_pylsl_info_to_dict(info) for info in resolve_streams_pylsl(wait_time=timeout)]
        except Exception as exc:  # pragma: no cover - depends on runtime network stack
            error = f"{type(exc).__name__}: {exc}"
            logger.warning("pylsl discovery failed: %s", exc)

    if not streams and not error:
        missing = _MNE_LSL_ERROR or _PYLSL_ERROR
        error = f"{type(missing).__name__}: {missing}" if missing else "LSL discovery unavailable"

    # Deduplicate by source_id/name so the fallback path does not double-report streams.
    deduped: Dict[str, Dict[str, Any]] = {}
    for stream in streams:
        key = stream.get("source_id") or stream.get("uid") or stream.get("name") or repr(stream)
        deduped[key] = stream

    return {
        "available": resolve_streams_mne is not None or resolve_streams_pylsl is not None,
        "streams": sorted(deduped.values(), key=lambda item: (item.get("name", ""), item.get("source_id", ""))),
        "last_error": error,
    }


class LSLInletStream(CytonStream):
    """Read an external LSL signal stream into the normal KYMA callbacks."""

    def __init__(
        self,
        *,
        stream_name: Optional[str] = None,
        source_id: Optional[str] = None,
        timeout: float = 2.0,
    ) -> None:
        super().__init__()
        self._requested_name = (stream_name or "").strip() or None
        self._requested_source_id = (source_id or "").strip() or None
        self._timeout = float(timeout)

        self._resolved_info: Optional[StreamInfo] = None
        self._inlet: Optional[StreamInlet] = None
        self._nominal_rate: float = 0.0
        self._source_channels: int = 0
        self._thread: Optional[threading.Thread] = None
        self.stream_details: Dict[str, Any] = {}

    def connect(self, serial_port: str = None, stream_name: str = None, source_id: str = None) -> bool:
        if StreamInlet is None or resolve_streams_pylsl is None:
            logger.error("LSL inlet unavailable: %s", _PYLSL_ERROR)
            return False

        self._requested_name = (stream_name or self._requested_name or "").strip() or None
        self._requested_source_id = (source_id or self._requested_source_id or "").strip() or None

        try:
            infos = list(resolve_streams_pylsl(wait_time=self._timeout))
        except Exception as exc:  # pragma: no cover - depends on runtime network stack
            logger.error("LSL resolve failed: %s", exc)
            return False

        if self._requested_source_id:
            infos = [info for info in infos if info.source_id() == self._requested_source_id]
        if self._requested_name:
            infos = [info for info in infos if info.name() == self._requested_name]
        if not infos:
            logger.error("No matching LSL stream found (name=%s, source_id=%s)", self._requested_name, self._requested_source_id)
            return False

        info = infos[0]
        nominal_rate = float(info.nominal_srate() or 0.0)
        source_channels = int(info.channel_count() or 0)

        if nominal_rate > 0 and abs(nominal_rate - config.sample_rate) > 1e-3:
            logger.error(
                "LSL stream %s uses %.3f Hz, but KYMA expects %s Hz",
                info.name(),
                nominal_rate,
                config.sample_rate,
            )
            return False

        if source_channels <= 0:
            logger.error("LSL stream %s has no numeric channels", info.name())
            return False

        self._resolved_info = info
        self._nominal_rate = nominal_rate or float(config.sample_rate)
        self._source_channels = source_channels
        self.stream_details = _pylsl_info_to_dict(info)
        logger.info(
            "LSL inlet ready: %s (%s ch @ %.3f Hz)",
            self.stream_details.get("name") or "stream",
            source_channels,
            self._nominal_rate,
        )
        return True

    def start(self) -> bool:
        if self._resolved_info is None or StreamInlet is None:
            return False

        try:
            self._inlet = StreamInlet(
                self._resolved_info,
                max_buflen=60,
                processing_flags=proc_clocksync | proc_dejitter,
            )
            self._inlet.open_stream()
        except Exception as exc:  # pragma: no cover - depends on runtime network stack
            logger.error("Failed to open LSL inlet: %s", exc)
            self._inlet = None
            return False

        self.is_running = True
        self._thread = threading.Thread(target=self._inlet_loop, daemon=True, name="lsl-inlet")
        self._thread.start()
        logger.info("LSL inlet started")
        return True

    def stop(self) -> None:
        self.is_running = False
        if self._thread:
            self._thread.join(timeout=3.0)
        if self._inlet is not None:
            try:
                self._inlet.close_stream()
            except Exception:
                pass
        self._inlet = None
        logger.info("LSL inlet stopped")

    def _inlet_loop(self) -> None:
        assert self._inlet is not None
        max_samples = max(config.window_increment_samples * 2, config.sample_rate // 10)

        while self.is_running:
            try:
                samples, _ = self._inlet.pull_chunk(timeout=0.2, max_samples=max_samples)
                if not samples:
                    time.sleep(0.01)
                    continue

                arr = np.asarray(samples, dtype=np.float64)
                if arr.ndim == 1:
                    arr = arr[:, np.newaxis]
                if arr.shape[1] != self._source_channels:
                    arr = np.asarray(samples, dtype=np.float64).reshape(-1, self._source_channels)

                self._append(self._adapt_channels(arr.T))
            except Exception as exc:  # pragma: no cover - depends on runtime network stack
                logger.error("LSL inlet read failed: %s", exc)
                time.sleep(0.05)

    def _adapt_channels(self, chunk: np.ndarray) -> np.ndarray:
        if chunk.shape[0] == config.n_channels:
            return chunk
        if chunk.shape[0] > config.n_channels:
            return chunk[: config.n_channels, :]

        padded = np.zeros((config.n_channels, chunk.shape[1]), dtype=np.float64)
        padded[: chunk.shape[0], :] = chunk
        return padded
