"""Optional OSC bridge for KYMA."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

try:
    from pythonosc.udp_client import SimpleUDPClient

    _OSC_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - depends on optional runtime dependency
    SimpleUDPClient = None  # type: ignore[assignment]
    _OSC_ERROR = exc


class OSCBridge:
    """Mirror decoded labels, state, and control commands over OSC."""

    def __init__(self) -> None:
        self.available: bool = _OSC_ERROR is None
        self.last_error: str = self._format_error(_OSC_ERROR)
        self.is_active: bool = False

        self.host: str = ""
        self.port: int = 9000
        self.prefix: str = "/kyma"
        self.mirror_events: bool = True

        self._client = None
        self._last_prediction_label: Optional[str] = None

    def status(self) -> Dict[str, Any]:
        return {
            "available": self.available,
            "active": self.is_active,
            "host": self.host,
            "port": self.port,
            "prefix": self.prefix,
            "mirror_events": self.mirror_events,
            "last_error": self.last_error,
        }

    def start(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 9000,
        prefix: str = "/kyma",
        mirror_events: bool = True,
    ) -> bool:
        if not self.available:
            self.last_error = (
                self._format_error(_OSC_ERROR)
                or "python-osc is not available. Install requirements.txt first."
            )
            logger.warning("OSC start failed: %s", self.last_error)
            return False

        self.stop()

        clean_host = str(host or "127.0.0.1").strip() or "127.0.0.1"
        clean_port = int(port or 9000)
        clean_prefix = self._normalize_prefix(prefix)

        try:
            self._client = SimpleUDPClient(clean_host, clean_port)
            self.host = clean_host
            self.port = clean_port
            self.prefix = clean_prefix
            self.mirror_events = bool(mirror_events)
            self.is_active = True
            self.last_error = ""
            self._last_prediction_label = None
            logger.info(
                "OSC active: %s:%s prefix=%s mirror_events=%s",
                self.host,
                self.port,
                self.prefix,
                self.mirror_events,
            )
            return True
        except Exception as exc:  # pragma: no cover - optional runtime dependency
            self.last_error = self._format_error(exc)
            self._client = None
            self.is_active = False
            logger.error("Failed to start OSC bridge: %s", exc)
            return False

    def stop(self) -> None:
        self.is_active = False
        self._client = None
        self._last_prediction_label = None

    def send_state(self, *, state: str, profile: str, source: str) -> bool:
        if not self.is_active or not self.mirror_events:
            return False
        payload = {"state": state, "profile": profile, "source": source}
        ok = self._send("/state", [state, profile, source])
        self._send_event("state", payload)
        return ok

    def send_prediction(
        self,
        *,
        label: str,
        confidence: float,
        profile: str,
        summary: str = "",
    ) -> bool:
        if not self.is_active or not self.mirror_events:
            return False
        if label == self._last_prediction_label:
            return False
        self._last_prediction_label = label
        payload = {
            "label": label,
            "confidence": round(float(confidence), 4),
            "profile": profile,
            "summary": summary,
        }
        ok = self._send(
            "/prediction",
            [label, round(float(confidence), 4), profile, summary],
        )
        self._send_event("prediction", payload)
        return ok

    def send_gesture(self, name: str, *, profile: str = "") -> bool:
        if not self.is_active:
            return False
        payload = {"gesture": name, "profile": profile}
        ok = self._send("/gesture", [name, profile])
        self._send_event("gesture", payload)
        return ok

    def send_move(self, joint_id: int, angle: int) -> bool:
        if not self.is_active:
            return False
        payload = {"joint_id": int(joint_id), "angle": int(angle)}
        ok = self._send("/move", [payload["joint_id"], payload["angle"]])
        self._send_event("move", payload)
        return ok

    def send_estop(self) -> bool:
        if not self.is_active:
            return False
        ok = self._send("/estop", 1)
        self._send_event("estop", {"active": True})
        return ok

    def send_home(self) -> bool:
        if not self.is_active:
            return False
        ok = self._send("/home", 1)
        self._send_event("home", {"active": True})
        return ok

    def send_digital_write(self, pin: int, value: int) -> bool:
        if not self.is_active:
            return False
        payload = {"pin": int(pin), "value": int(value)}
        ok = self._send("/digital_write", [payload["pin"], payload["value"]])
        self._send_event("digital_write", payload)
        return ok

    def send_analog_write(self, pin: int, value: int) -> bool:
        if not self.is_active:
            return False
        payload = {"pin": int(pin), "value": int(value)}
        ok = self._send("/analog_write", [payload["pin"], payload["value"]])
        self._send_event("analog_write", payload)
        return ok

    def _send_event(self, event: str, payload: Optional[Dict[str, Any]] = None) -> bool:
        if not self.is_active or not self.mirror_events:
            return False
        message = json.dumps({"event": event, **(payload or {})}, separators=(",", ":"))
        ok_generic = self._send("/event", message)
        ok_typed = self._send(f"/event/{event}", message)
        return ok_generic or ok_typed

    def _send(self, suffix: str, value: Any) -> bool:
        if not self.is_active or self._client is None:
            return False
        try:
            self._client.send_message(self._addr(suffix), value)
            return True
        except Exception as exc:  # pragma: no cover - optional runtime dependency
            self.last_error = self._format_error(exc)
            logger.error("Failed to send OSC %s: %s", suffix, exc)
            return False

    def _addr(self, suffix: str) -> str:
        clean_suffix = "/" + str(suffix or "").strip().lstrip("/")
        return f"{self.prefix}{clean_suffix}" if self.prefix != "/" else clean_suffix

    @staticmethod
    def _normalize_prefix(prefix: str) -> str:
        text = str(prefix or "/kyma").strip() or "/kyma"
        if not text.startswith("/"):
            text = f"/{text}"
        text = text.rstrip("/")
        return text or "/"

    @staticmethod
    def _format_error(exc: Optional[Exception]) -> str:
        if exc is None:
            return ""
        return f"{type(exc).__name__}: {exc}"
