"""Optional Lab Streaming Layer bridge for KYMA."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Iterable, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    from pylsl import StreamInfo, StreamOutlet, cf_float32, cf_string, local_clock

    _PYLSL_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - depends on optional runtime dependency
    StreamInfo = None  # type: ignore[assignment]
    StreamOutlet = None  # type: ignore[assignment]
    cf_float32 = None  # type: ignore[assignment]
    cf_string = None  # type: ignore[assignment]
    local_clock = None  # type: ignore[assignment]
    _PYLSL_ERROR = exc


class LSLBridge:
    """Publish live biosignal chunks and markers over LSL when available."""

    def __init__(self) -> None:
        self.available: bool = _PYLSL_ERROR is None
        self.last_error: str = self._format_error(_PYLSL_ERROR)
        self.is_active: bool = False
        self.include_markers: bool = True

        self.stream_name: str = ""
        self.marker_stream_name: str = ""
        self.stream_source: str = "hardware"
        self.profile_key: str = "emg"
        self.profile_name: str = "EMG"
        self.sample_rate: int = 0

        self._requested_name: Optional[str] = None
        self._signal_outlet = None
        self._marker_outlet = None
        self._last_prediction_label: Optional[str] = None

    def status(self) -> Dict[str, Any]:
        return {
            "available": self.available,
            "active": self.is_active,
            "include_markers": self.include_markers,
            "stream_name": self.stream_name,
            "marker_stream_name": self.marker_stream_name,
            "stream_source": self.stream_source,
            "profile": self.profile_key,
            "profile_name": self.profile_name,
            "sample_rate": self.sample_rate,
            "last_error": self.last_error,
        }

    def start(
        self,
        *,
        profile,
        channel_labels: Iterable[str],
        sample_rate: int,
        stream_source: str,
        stream_name: Optional[str] = None,
        include_markers: bool = True,
    ) -> bool:
        if not self.available:
            self.last_error = (
                self._format_error(_PYLSL_ERROR)
                or "pylsl is not available. Install requirements-research.txt first."
            )
            logger.warning("LSL start failed: %s", self.last_error)
            return False

        self.stop()

        labels = list(channel_labels)
        self.include_markers = include_markers
        self.stream_source = stream_source
        self.profile_key = profile.key
        self.profile_name = profile.display_name
        self.sample_rate = sample_rate
        self._requested_name = (stream_name or "").strip() or None
        self.stream_name = self._requested_name or f"KYMA_{profile.display_name.upper()}"
        self.marker_stream_name = (
            f"{self.stream_name}_Markers" if include_markers else ""
        )

        source_id_base = (
            f"kyma.{profile.key}.{stream_source}.{self.stream_name.lower()}"
            .replace(" ", "_")
        )

        try:
            signal_info = StreamInfo(
                self.stream_name,
                profile.display_name.upper(),
                len(labels),
                float(sample_rate),
                cf_float32,
                source_id_base,
            )
            desc = signal_info.desc()
            desc.append_child_value("manufacturer", "KYMA")
            desc.append_child_value("stream_source", stream_source)
            desc.append_child_value("signal_profile", profile.key)
            desc.append_child_value("signal_name", profile.display_name)
            desc.append_child_value("units", profile.units)

            channels = desc.append_child("channels")
            for idx, label in enumerate(labels, start=1):
                channel = channels.append_child("channel")
                channel.append_child_value("label", str(label))
                channel.append_child_value("unit", profile.units)
                channel.append_child_value("type", profile.display_name.upper())
                channel.append_child_value("index", str(idx))

            self._signal_outlet = StreamOutlet(signal_info, chunk_size=1, max_buffered=360)

            if include_markers:
                marker_info = StreamInfo(
                    self.marker_stream_name,
                    "Markers",
                    1,
                    0.0,
                    cf_string,
                    f"{source_id_base}.markers",
                )
                marker_desc = marker_info.desc()
                marker_desc.append_child_value("manufacturer", "KYMA")
                marker_desc.append_child_value("stream_source", stream_source)
                marker_desc.append_child_value("signal_profile", profile.key)
                self._marker_outlet = StreamOutlet(
                    marker_info,
                    chunk_size=1,
                    max_buffered=360,
                )

            self.is_active = True
            self.last_error = ""
            self._last_prediction_label = None
            logger.info(
                "LSL active: signal=%s markers=%s",
                self.stream_name,
                self.marker_stream_name or "disabled",
            )
            return True
        except Exception as exc:  # pragma: no cover - optional runtime dependency
            self.last_error = self._format_error(exc)
            self.is_active = False
            self._signal_outlet = None
            self._marker_outlet = None
            logger.error("Failed to start LSL bridge: %s", exc)
            return False

    def reconfigure(self, *, profile, channel_labels: Iterable[str], sample_rate: int, stream_source: str) -> bool:
        if not self.is_active:
            return False
        return self.start(
            profile=profile,
            channel_labels=channel_labels,
            sample_rate=sample_rate,
            stream_source=stream_source,
            stream_name=self._requested_name,
            include_markers=self.include_markers,
        )

    def stop(self) -> None:
        self.is_active = False
        self._signal_outlet = None
        self._marker_outlet = None
        self._last_prediction_label = None

    def push_chunk(self, samples: np.ndarray) -> None:
        if not self.is_active or self._signal_outlet is None:
            return
        if samples.ndim != 2 or samples.size == 0 or self.sample_rate <= 0:
            return

        try:
            n_samples = samples.shape[1]
            first_ts = local_clock() - ((n_samples - 1) / float(self.sample_rate))
            chunk = samples.astype(np.float32, copy=False)
            for i in range(n_samples):
                self._signal_outlet.push_sample(
                    chunk[:, i].tolist(),
                    timestamp=first_ts + (i / float(self.sample_rate)),
                )
        except Exception as exc:  # pragma: no cover - optional runtime dependency
            self.last_error = self._format_error(exc)
            logger.error("Failed to push LSL biosignal chunk: %s", exc)

    def push_prediction_marker(
        self,
        *,
        label: str,
        confidence: float,
        profile: str,
        summary: str = "",
    ) -> None:
        if label == self._last_prediction_label:
            return
        self._last_prediction_label = label
        self.push_marker(
            "prediction",
            {
                "label": label,
                "confidence": round(float(confidence), 4),
                "profile": profile,
                "summary": summary,
            },
        )

    def push_marker(self, event: str, payload: Optional[Dict[str, Any]] = None) -> None:
        if not self.is_active or self._marker_outlet is None:
            return
        marker = {"event": event, **(payload or {})}
        try:
            self._marker_outlet.push_sample(
                [json.dumps(marker, separators=(",", ":"))],
                timestamp=local_clock(),
            )
        except Exception as exc:  # pragma: no cover - optional runtime dependency
            self.last_error = self._format_error(exc)
            logger.error("Failed to push LSL marker: %s", exc)

    @staticmethod
    def _format_error(exc: Optional[Exception]) -> str:
        if exc is None:
            return ""
        return f"{type(exc).__name__}: {exc}"
