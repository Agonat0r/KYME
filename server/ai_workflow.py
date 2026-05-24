"""AI-assisted workflow summaries for KYMA."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ai_modality_router import AIModalityRouter
from biosignal_ml import BiosignalModelRuntime


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _clamp(value: float, low: float, high: float) -> float:
    return max(float(low), min(float(high), float(value)))


def _clean_text(value: Any, fallback: str = "") -> str:
    text = str(value or "").strip()
    return text or fallback


def _snake_case(value: Any, fallback: str = "marker") -> str:
    raw = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value or "").strip())
    while "__" in raw:
        raw = raw.replace("__", "_")
    raw = raw.strip("_")
    return raw[:48] or fallback


class AIWorkflow:
    """Workflow-focused AI helper with a deterministic fallback path."""

    DEFAULT_MODEL = ""

    def __init__(self) -> None:
        self.api_key = ""
        self.base_url = ""
        self.model = self.DEFAULT_MODEL
        self.credential_source = "none"
        self.config_source = "none"
        self.key_hint = ""
        self.last_error = ""
        self._root_dir = Path("sessions")
        self._runtime_path = self._root_dir / "ai_runtime.json"
        self._saved_api_key = ""
        self._saved_base_url = ""
        self._saved_model = ""
        self._root_dir.mkdir(parents=True, exist_ok=True)
        self.router = AIModalityRouter(Path(__file__).resolve().parents[1])
        self.local_models = BiosignalModelRuntime(
            [
                Path(__file__).resolve().parents[1] / "server" / "models",
                self._root_dir / "ai_models",
            ]
        )
        self._load_runtime()
        self._refresh_env()

    def _backend_label(self, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            return "local heuristics"
        trimmed = text.replace("https://", "").replace("http://", "").strip("/")
        return trimmed or "remote api"

    def _mask_key(self, value: str) -> str:
        text = str(value or "").strip()
        if len(text) < 8:
            return "configured" if text else ""
        return f"{text[:3]}...{text[-4:]}"

    def _load_runtime(self) -> None:
        self._saved_api_key = ""
        self._saved_base_url = ""
        self._saved_model = ""
        if not self._runtime_path.exists():
            return
        try:
            with open(self._runtime_path, encoding="utf-8") as fh:
                payload = json.load(fh)
            self._saved_api_key = str(payload.get("api_key") or "").strip()
            self._saved_base_url = str(payload.get("base_url") or "").strip().rstrip("/")
            self._saved_model = str(payload.get("model") or "").strip()
        except Exception:
            self._saved_api_key = ""
            self._saved_base_url = ""
            self._saved_model = ""

    def _save_runtime(self) -> None:
        payload = {
            "api_key": self._saved_api_key,
            "base_url": self._saved_base_url,
            "model": self._saved_model,
            "updated_at": _utc_now_iso(),
        }
        with open(self._runtime_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

    def _refresh_env(self) -> None:
        self._load_runtime()
        env_api_key = str(os.getenv("KYMA_AI_API_KEY") or "").strip()
        env_base_url = str(os.getenv("KYMA_AI_BASE_URL") or "").strip().rstrip("/")
        env_model = str(
            os.getenv("KYMA_AI_MODEL")
            or self.DEFAULT_MODEL
        ).strip()
        self.api_key = self._saved_api_key or env_api_key
        self.base_url = self._saved_base_url or env_base_url
        self.model = self._saved_model or env_model
        saved_present = bool(self._saved_api_key or self._saved_base_url or self._saved_model)
        env_present = bool(env_api_key or env_base_url or env_model)
        self.credential_source = "saved" if self._saved_api_key else ("env" if env_api_key else "none")
        self.config_source = "saved" if saved_present else ("env" if env_present else "none")
        self.key_hint = self._mask_key(self.api_key)

    @property
    def configured(self) -> bool:
        self._refresh_env()
        return bool(self.base_url and self.model)

    def status(self) -> Dict[str, Any]:
        self._refresh_env()
        local_status = self.local_models.status()
        local_any = bool(local_status.get("model_count") or local_status.get("foundation_model_count"))
        mode = "hybrid" if local_any else ("remote" if self.configured else "heuristic")
        provider = "hybrid" if local_any else ("api" if self.configured else "local")
        return {
            "available": True,
            "configured": self.configured,
            "mode": mode,
            "provider": provider,
            "model": self.model,
            "base_url": self.base_url,
            "backend_label": self._backend_label(self.base_url),
            "credential_source": self.credential_source,
            "config_source": self.config_source,
            "config_saved": self.config_source == "saved",
            "key_hint": self.key_hint,
            "storage_path": self._runtime_path.as_posix(),
            "last_error": self.last_error,
            "local_models": local_status,
        }

    def set_config(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        api_key = str(payload.get("api_key") or "").strip()
        base_url = str(payload.get("base_url") or "").strip().rstrip("/")
        model = str(payload.get("model") or "").strip()
        self._saved_api_key = api_key
        self._saved_base_url = base_url
        self._saved_model = model
        if not (self._saved_api_key or self._saved_base_url or self._saved_model):
            raise ValueError("Enter at least one AI config value")
        self._save_runtime()
        self._refresh_env()
        return self.status()

    def clear_saved_config(self) -> Dict[str, Any]:
        self._saved_api_key = ""
        self._saved_base_url = ""
        self._saved_model = ""
        if self._runtime_path.exists():
            try:
                self._runtime_path.unlink()
            except OSError:
                self._save_runtime()
        self._refresh_env()
        return self.status()

    def summarize(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        snapshot = self._normalize_snapshot(payload)
        routing_plan = self.router.plan(snapshot, self.local_models.status())
        result = self._heuristic_result(snapshot)
        result["routing"] = routing_plan
        local_result = self.local_models.analyze(snapshot)
        result = self._merge_local_model_result(result, local_result)
        result["measurements"] = self._measurement_summary(snapshot)
        heuristic_screens = self._heuristic_condition_screens(snapshot, local_result)
        result["condition_screen"] = heuristic_screens
        if heuristic_screens and not list(local_result.get("condition_screen") or []):
            primary_screen = dict(heuristic_screens[0] or {})
            result["next_actions"] = self._prepend_action(
                result.get("next_actions") or [],
                {
                    "title": f"Review {str(primary_screen.get('display_label') or primary_screen.get('label') or 'screening cue')}",
                    "detail": f"{str(primary_screen.get('detail') or 'Research-only screening cue')} Confirm it with manual review and export if it matters.",
                    "priority": "medium",
                },
            )
            screen_label = str(primary_screen.get("display_label") or primary_screen.get("label") or "").strip()
            screen_score = int(round(_safe_float(primary_screen.get("score"))))
            if screen_label:
                result["summary"] = f"{result.get('summary') or ''} Research screen highlights {screen_label.lower()} at {screen_score} percent.".strip()
        result["top_action"] = dict((result.get("next_actions") or [None])[0] or {}) or None
        result["research_watchlist"] = self._research_watchlist(snapshot, routing_plan, local_result, heuristic_screens)
        source = "heuristic"

        if self.configured:
            try:
                enhanced = self._remote_enhancement(self._remote_snapshot(snapshot), result, routing_plan)
                result = self._merge_result(result, enhanced)
                source = "remote"
                self.last_error = ""
            except Exception as exc:  # pragma: no cover - network path depends on env
                self.last_error = f"{type(exc).__name__}: {exc}"
                source = "remote_fallback"

        result["top_action"] = dict((result.get("next_actions") or [None])[0] or {}) or None
        result["source"] = source
        result["generated_at"] = _utc_now_iso()
        return {
            "ok": True,
            "ai": self.status(),
            "result": result,
        }

    def _normalize_snapshot(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        profile = dict(payload.get("profile") or {})
        diagnostics = dict(payload.get("diagnostics") or {})
        noise = dict(diagnostics.get("noise") or {})
        timing = dict(diagnostics.get("timing") or {})
        spectrum = dict(diagnostics.get("spectrum") or {})
        review = dict(payload.get("review") or {})
        stats = dict(review.get("stats") or {})
        selection = dict(review.get("selection") or {})
        window = dict(review.get("window") or {})
        workshop = dict(payload.get("workshop") or {})
        protocol = dict(payload.get("protocol") or {})
        last_prediction = dict(payload.get("last_prediction") or {})
        filter_chain = dict(payload.get("filter_chain") or {})
        session = dict(payload.get("session") or {})

        window_channels: List[List[float]] = []
        for series in list(window.get("channels") or [])[:16]:
            if isinstance(series, list):
                window_channels.append([round(_safe_float(value), 6) for value in series[:1024]])

        artifacts: List[Dict[str, Any]] = []
        for item in list(review.get("artifacts") or []):
            start_ms = _safe_float(item.get("start_ms"))
            end_ms = _safe_float(item.get("end_ms"), start_ms)
            artifacts.append(
                {
                    "kind": _clean_text(item.get("kind"), "artifact"),
                    "label": _clean_text(item.get("label"), "Artifact"),
                    "channel_label": _clean_text(item.get("channel_label"), "CH1"),
                    "detail": _clean_text(item.get("detail"), "Review artifact candidate."),
                    "score": round(_safe_float(item.get("score"), 0.0), 3),
                    "start_ms": round(start_ms, 3),
                    "end_ms": round(max(start_ms, end_ms), 3),
                    "duration_ms": round(max(0.0, end_ms - start_ms), 3),
                }
            )

        dominant_frequency_hz, dominant_magnitude_db = self._dominant_spectrum(spectrum)

        return {
            "captured_at": payload.get("captured_at") or _utc_now_iso(),
            "state": _clean_text(payload.get("state"), "idle"),
            "stream_running": bool(payload.get("stream_running")),
            "is_recording": bool(payload.get("is_recording")),
            "profile": {
                "key": _clean_text(profile.get("key"), "signal"),
                "name": _clean_text(profile.get("display_name") or profile.get("name"), "Signal"),
                "units": _clean_text(profile.get("units"), "a.u."),
            },
            "channel_labels": [str(label) for label in list(payload.get("channel_labels") or [])][:16],
            "diagnostics": {
                "spectrum": {
                    "dominant_frequency_hz": round(dominant_frequency_hz, 3),
                    "dominant_magnitude_db": round(dominant_magnitude_db, 3),
                    "segment_ms": round(_safe_float(spectrum.get("segment_ms")), 3),
                },
                "noise": {
                    "hum_50_db": round(_safe_float(noise.get("hum_50_db")), 3),
                    "hum_60_db": round(_safe_float(noise.get("hum_60_db")), 3),
                    "drift_db": round(_safe_float(noise.get("drift_db")), 3),
                    "clip_pct": round(_safe_float(noise.get("clip_pct")), 3),
                    "crest_factor": round(_safe_float(noise.get("crest_factor")), 3),
                },
                "timing": {
                    "window_count": _safe_int(timing.get("window_count")),
                    "interval_jitter_ms": round(_safe_float(timing.get("interval_jitter_ms")), 3),
                    "signal_age_ms": round(_safe_float(timing.get("signal_age_ms")), 3),
                    "dropped_windows": _safe_int(timing.get("dropped_windows")),
                },
            },
            "review": {
                "paused": bool(review.get("paused")),
                "cursor_label": _clean_text(review.get("cursor_label")),
                "stats": {
                    "samples": _safe_int(stats.get("samples")),
                    "duration_ms": round(_safe_float(stats.get("duration_ms")), 3),
                    "mean": round(_safe_float(stats.get("mean")), 6),
                    "rms": round(_safe_float(stats.get("rms")), 6),
                    "peak_to_peak": round(_safe_float(stats.get("peak_to_peak")), 6),
                    "focus_channel": _safe_int(stats.get("focus_channel")),
                    "focus_label": _clean_text(stats.get("focus_label"), "CH1"),
                    "focus_rms": round(_safe_float(stats.get("focus_rms")), 6),
                },
                "selection": {
                    "start_ms": round(_safe_float(selection.get("start_ms")), 3),
                    "end_ms": round(_safe_float(selection.get("end_ms")), 3),
                    "duration_ms": round(_safe_float(selection.get("duration_ms")), 3),
                },
                "window": {
                    "sample_rate_hz": round(_safe_float(window.get("sample_rate_hz")), 3),
                    "start_sample": _safe_int(window.get("start_sample")),
                    "end_sample": _safe_int(window.get("end_sample")),
                    "samples": _safe_int(window.get("samples")),
                    "step_samples": max(_safe_int(window.get("step_samples"), 1), 1),
                    "focus_channel": _safe_int(window.get("focus_channel")),
                    "focus_label": _clean_text(window.get("focus_label"), _clean_text(stats.get("focus_label"), "CH1")),
                    "channels": window_channels,
                },
                "artifacts": artifacts,
            },
            "protocol": {
                "active": bool(protocol.get("active")),
                "template_key": _clean_text(protocol.get("template_key")),
                "template_title": _clean_text(protocol.get("template_title")),
                "phase": _clean_text(protocol.get("phase"), "idle"),
                "step_label": _clean_text(protocol.get("step_label")),
                "remaining_ms": round(_safe_float(protocol.get("remaining_ms")), 3),
            },
            "workshop": {
                "has_result": bool(workshop.get("has_result")),
                "view": _clean_text(workshop.get("view")),
                "summary": dict(workshop.get("summary") or {}),
                "selection_label": _clean_text(workshop.get("selection_label")),
            },
            "last_prediction": {
                "label": _clean_text(last_prediction.get("label") or last_prediction.get("gesture"), "--"),
                "confidence": round(_safe_float(last_prediction.get("confidence")), 4),
            },
            "filter_chain": {
                "label": _clean_text(filter_chain.get("label")),
            },
            "session": {
                "session_id": _clean_text(session.get("session_id")),
                "label": _clean_text(session.get("label")),
                "subject_id": _clean_text(session.get("subject_id")),
                "condition": _clean_text(session.get("condition")),
                "notes": _clean_text(session.get("notes")),
                "workspace": _clean_text(session.get("workspace"), "live"),
            },
        }

    def _heuristic_result(self, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        review = snapshot["review"]
        diagnostics = snapshot["diagnostics"]
        noise = diagnostics["noise"]
        timing = diagnostics["timing"]
        protocol = snapshot["protocol"]
        workshop = snapshot["workshop"]
        prediction = snapshot["last_prediction"]
        artifacts = list(review["artifacts"])
        stats = dict(review["stats"])

        hum_max = max(_safe_float(noise.get("hum_50_db")), _safe_float(noise.get("hum_60_db")))
        drift_db = _safe_float(noise.get("drift_db"))
        clip_pct = _safe_float(noise.get("clip_pct"))
        jitter_ms = _safe_float(timing.get("interval_jitter_ms"))
        dropped = _safe_int(timing.get("dropped_windows"))
        signal_age_ms = _safe_float(timing.get("signal_age_ms"))
        artifact_ms = sum(_safe_float(item.get("duration_ms")) for item in artifacts)
        artifact_types = [str(item.get("kind") or "artifact") for item in artifacts]
        stats_duration_ms = max(_safe_float(stats.get("duration_ms")), 1.0)

        channel_stability = 100.0
        if "spike" in artifact_types:
            channel_stability -= 18.0
        if "flatline" in artifact_types:
            channel_stability -= 30.0
        if "drift" in artifact_types:
            channel_stability -= 22.0
        if jitter_ms > 8.0:
            channel_stability -= 10.0
        if dropped > 0:
            channel_stability -= min(18.0, dropped * 3.0)
        channel_stability = _clamp(channel_stability, 0.0, 100.0)

        noise_score = 100.0
        if hum_max >= -12.0:
            noise_score -= 45.0
        elif hum_max >= -20.0:
            noise_score -= 25.0
        elif hum_max >= -28.0:
            noise_score -= 12.0
        if drift_db >= -12.0:
            noise_score -= 28.0
        elif drift_db >= -20.0:
            noise_score -= 14.0
        if signal_age_ms > 1200.0:
            noise_score -= 10.0
        noise_score = _clamp(noise_score, 0.0, 100.0)

        clipping_score = _clamp(100.0 - min(100.0, clip_pct * 15.0), 0.0, 100.0)

        artifact_score = 100.0
        artifact_score -= min(60.0, len(artifacts) * 13.0)
        artifact_score -= min(20.0, (artifact_ms / stats_duration_ms) * 40.0)
        artifact_score = _clamp(artifact_score, 0.0, 100.0)

        protocol_score = 70.0
        if protocol["template_key"]:
            protocol_score = 82.0
            if protocol["active"]:
                protocol_score = 88.0
            if protocol["phase"] == "complete":
                protocol_score = 96.0
            elif protocol["phase"] == "rest":
                protocol_score = 90.0
            elif protocol["phase"] == "trial":
                protocol_score = 92.0

        training_score = 62.0
        focus_rms = _safe_float(stats.get("focus_rms"))
        confidence = _safe_float(prediction.get("confidence"))
        if _safe_int(stats.get("samples")) > 1:
            training_score += 10.0
        if focus_rms >= 5.0:
            training_score += 12.0
        elif focus_rms >= 1.5:
            training_score += 6.0
        if confidence >= 0.8:
            training_score += 10.0
        elif confidence >= 0.6:
            training_score += 6.0
        if len(artifacts) >= 3:
            training_score -= 20.0
        if clip_pct > 1.0:
            training_score -= 18.0
        training_score = _clamp(training_score, 0.0, 100.0)

        weighted = (
            (channel_stability * 0.22)
            + (noise_score * 0.22)
            + (clipping_score * 0.16)
            + (artifact_score * 0.18)
            + (protocol_score * 0.10)
            + (training_score * 0.12)
        )
        overall = int(round(_clamp(weighted, 0.0, 100.0)))

        dimensions = [
            self._dimension("Channel stability", channel_stability),
            self._dimension("Noise floor", noise_score),
            self._dimension("Clipping", clipping_score),
            self._dimension("Artifact burden", artifact_score),
            self._dimension("Protocol", protocol_score),
            self._dimension("Training readiness", training_score),
        ]

        top_issue = "clean_window"
        if artifacts:
            top_issue = _snake_case(artifacts[0].get("kind"), "artifact")
        elif clip_pct > 1.0:
            top_issue = "clipping"
        elif hum_max >= -20.0:
            top_issue = "hum"
        elif drift_db >= -20.0:
            top_issue = "baseline_drift"

        return {
            "summary": self._build_summary(snapshot, overall, top_issue),
            "qa_score": {
                "overall": overall,
                "grade": self._grade(overall),
                "dimensions": dimensions,
            },
            "artifact_summary": {
                "count": len(artifacts),
                "burden": self._artifact_burden(len(artifacts), artifact_ms, stats_duration_ms),
                "top_issue": top_issue,
            },
            "flagged_channels": self._flagged_channels(snapshot),
            "next_actions": self._next_actions(snapshot, overall, top_issue),
            "suggested_markers": self._marker_suggestions(snapshot, top_issue),
            "export_recommendations": self._export_recommendations(snapshot, workshop, overall),
        }

    def _dimension(self, label: str, score: float) -> Dict[str, Any]:
        rounded = int(round(_clamp(score, 0.0, 100.0)))
        return {
            "label": label,
            "score": rounded,
            "grade": self._grade(rounded),
        }

    def _grade(self, score: float) -> str:
        numeric = _safe_float(score)
        if numeric >= 90.0:
            return "excellent"
        if numeric >= 75.0:
            return "good"
        if numeric >= 55.0:
            return "watch"
        return "poor"

    def _artifact_burden(self, count: int, artifact_ms: float, total_ms: float) -> str:
        ratio = artifact_ms / max(total_ms, 1.0)
        if count >= 4 or ratio >= 0.35:
            return "high"
        if count >= 2 or ratio >= 0.18:
            return "moderate"
        if count >= 1:
            return "low"
        return "minimal"

    def _dominant_spectrum(self, spectrum: Dict[str, Any]) -> tuple[float, float]:
        freq = [float(item) for item in list(spectrum.get("freq_hz") or [])[:2048]]
        mag = [float(item) for item in list(spectrum.get("mag_db") or [])[:2048]]
        if not freq or not mag:
            return 0.0, 0.0
        best_freq = 0.0
        best_mag = float("-inf")
        for hz, db in zip(freq, mag):
            if hz <= 0.5:
                continue
            if db > best_mag:
                best_mag = db
                best_freq = hz
        if best_mag == float("-inf"):
            return 0.0, 0.0
        return best_freq, best_mag

    def _measurement_summary(self, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        stats = dict((snapshot.get("review") or {}).get("stats") or {})
        selection = dict((snapshot.get("review") or {}).get("selection") or {})
        window = dict((snapshot.get("review") or {}).get("window") or {})
        spectrum = dict((snapshot.get("diagnostics") or {}).get("spectrum") or {})
        sample_rate_hz = _safe_float(window.get("sample_rate_hz"))
        samples = _safe_int(window.get("samples"))
        visible_span_ms = _safe_float(selection.get("duration_ms")) or _safe_float(stats.get("duration_ms"))
        if visible_span_ms <= 0.0 and sample_rate_hz > 0 and samples > 0:
            visible_span_ms = (samples / sample_rate_hz) * 1000.0
        return {
            "units": _clean_text(((snapshot.get("profile") or {}).get("units")), "a.u."),
            "sample_rate_hz": round(sample_rate_hz, 3),
            "visible_span_ms": round(visible_span_ms, 3),
            "mean": round(_safe_float(stats.get("mean")), 6),
            "rms": round(_safe_float(stats.get("rms")), 6),
            "peak_to_peak": round(_safe_float(stats.get("peak_to_peak")), 6),
            "focus_label": _clean_text(stats.get("focus_label"), _clean_text(window.get("focus_label"), "CH1")),
            "focus_rms": round(_safe_float(stats.get("focus_rms")), 6),
            "dominant_frequency_hz": round(_safe_float(spectrum.get("dominant_frequency_hz")), 3),
            "dominant_magnitude_db": round(_safe_float(spectrum.get("dominant_magnitude_db")), 3),
        }

    def _screening_item(
        self,
        label: str,
        score: float,
        detail: str,
        source: str = "heuristic",
    ) -> Dict[str, Any]:
        numeric = int(round(_clamp(score, 0.0, 100.0)))
        clean = _snake_case(label, "screen")
        return {
            "kind": "screening_result",
            "label": clean,
            "display_label": clean.replace("_", " "),
            "status": "active",
            "score": numeric,
            "confidence": round(_clamp(numeric / 100.0, 0.0, 1.0), 4),
            "source": source,
            "detail": _clean_text(detail),
            "research_only": True,
        }

    def _heuristic_condition_screens(self, snapshot: Dict[str, Any], local: Dict[str, Any]) -> List[Dict[str, Any]]:
        profile = dict(snapshot.get("profile") or {})
        review = dict(snapshot.get("review") or {})
        diagnostics = dict(snapshot.get("diagnostics") or {})
        noise = dict(diagnostics.get("noise") or {})
        spectrum = dict(diagnostics.get("spectrum") or {})
        stats = dict(review.get("stats") or {})
        artifacts = list(review.get("artifacts") or [])
        modality = _clean_text(profile.get("key"), "signal").lower()
        screens: List[Dict[str, Any]] = []

        artifact_count = len(artifacts)
        hum = max(_safe_float(noise.get("hum_50_db")), _safe_float(noise.get("hum_60_db")))
        clip_pct = _safe_float(noise.get("clip_pct"))
        drift_db = _safe_float(noise.get("drift_db"))
        dominant_hz = _safe_float(spectrum.get("dominant_frequency_hz"))
        rms = _safe_float(stats.get("rms"))
        peak_to_peak = _safe_float(stats.get("peak_to_peak"))
        mean = abs(_safe_float(stats.get("mean")))
        readiness = _safe_float((local.get("fatigue_readiness") or {}).get("score"))
        clean_window = artifact_count == 0 and clip_pct < 1.0 and hum < -18.0 and drift_db < -18.0

        if modality == "ecg":
            bpm = dominant_hz * 60.0 if dominant_hz > 0 else 0.0
            if bpm >= 105.0:
                screens.append(
                    self._screening_item(
                        "tachycardic_range_candidate",
                        min(94.0, 62.0 + (bpm - 105.0) * 0.7),
                        f"Dominant rhythm energy is around {int(round(bpm))} bpm. Treat this as a rhythm-screening cue, not a diagnosis.",
                    )
                )
            elif 0.0 < bpm <= 50.0:
                screens.append(
                    self._screening_item(
                        "bradycardic_range_candidate",
                        min(92.0, 62.0 + (50.0 - bpm) * 0.9),
                        f"Dominant rhythm energy is around {int(round(bpm))} bpm. Review the chunk offline before making any claim.",
                    )
                )
            if clean_window and peak_to_peak > max(0.4, abs(mean) * 2.0) and 0.8 <= dominant_hz <= 2.2:
                screens.append(
                    self._screening_item(
                        "stable_sinus_range_pattern",
                        64.0,
                        f"The visible ECG chunk sits in a roughly sinus-range band near {int(round(bpm))} bpm with low artifact pressure.",
                    )
                )

        elif modality == "eeg":
            spike_like = any(str(item.get("kind") or "") == "spike" for item in artifacts)
            if spike_like and clip_pct < 1.0:
                screens.append(
                    self._screening_item(
                        "spike_like_event_candidate",
                        68.0,
                        "Sharp transient activity is visible in the review region. This is a research review cue only.",
                    )
                )
            if clean_window and 8.0 <= dominant_hz <= 12.5:
                screens.append(
                    self._screening_item(
                        "alpha_dominant_pattern",
                        72.0,
                        f"Dominant spectral energy sits near {dominant_hz:.1f} Hz, which is consistent with an alpha-dominant resting pattern.",
                    )
                )
            elif clean_window and 1.5 <= dominant_hz <= 7.5:
                screens.append(
                    self._screening_item(
                        "slowing_candidate",
                        min(88.0, 60.0 + (7.5 - dominant_hz) * 4.0),
                        f"Dominant spectral energy is shifted toward {dominant_hz:.1f} Hz. Review for slowing versus drowsiness or confounds.",
                    )
                )

        elif modality == "emg":
            if readiness and readiness < 45.0 and artifact_count <= 2:
                screens.append(
                    self._screening_item(
                        "fatigue_like_activation_drop",
                        min(93.0, 58.0 + (45.0 - readiness) * 0.9),
                        f"Local readiness is low at {int(round(readiness))} percent, which can be compatible with fatigue-like decline.",
                        source="local+heuristic",
                    )
                )
            if clean_window and rms > 0 and mean >= rms * 0.65:
                screens.append(
                    self._screening_item(
                        "sustained_tonic_activation_pattern",
                        min(84.0, 60.0 + min(mean / max(rms, 1e-6), 1.4) * 12.0),
                        "Baseline offset stays elevated relative to RMS, which can reflect sustained tonic activation or tension.",
                    )
                )

        elif modality == "eog":
            spike_count = sum(1 for item in artifacts if str(item.get("kind") or "") == "spike")
            if spike_count >= 2 or peak_to_peak >= max(0.8, abs(mean) * 4.0):
                screens.append(
                    self._screening_item(
                        "blink_burst_candidate",
                        min(90.0, 60.0 + spike_count * 9.0 + min(peak_to_peak, 2.0) * 8.0),
                        "Large transient eye-potential excursions are present, which is consistent with blink-burst or strong eye-movement activity.",
                    )
                )
            elif clean_window and 0.2 <= dominant_hz <= 2.5 and peak_to_peak > 0.2:
                screens.append(
                    self._screening_item(
                        "saccade_activity_pattern",
                        67.0,
                        f"Slow eye-movement energy is centered near {dominant_hz:.2f} Hz with visible excursion amplitude.",
                    )
                )

        elif modality == "resp":
            breaths_per_min = dominant_hz * 60.0 if dominant_hz > 0 else 0.0
            if breaths_per_min >= 24.0:
                screens.append(
                    self._screening_item(
                        "tachypnea_candidate",
                        min(90.0, 62.0 + (breaths_per_min - 24.0) * 1.2),
                        f"Dominant respiratory energy suggests about {int(round(breaths_per_min))} breaths per minute.",
                    )
                )
            elif 0.0 < breaths_per_min <= 8.0:
                screens.append(
                    self._screening_item(
                        "bradypnea_candidate",
                        min(90.0, 62.0 + (8.0 - breaths_per_min) * 2.5),
                        f"Dominant respiratory energy suggests about {int(round(breaths_per_min))} breaths per minute.",
                    )
                )

        elif modality == "ppg":
            bpm = dominant_hz * 60.0 if dominant_hz > 0 else 0.0
            if bpm >= 105.0:
                screens.append(
                    self._screening_item(
                        "high_pulse_rate_candidate",
                        min(90.0, 60.0 + (bpm - 105.0) * 0.8),
                        f"Dominant pulse energy sits near {int(round(bpm))} bpm in the visible PPG chunk.",
                    )
                )
            elif 0.0 < bpm <= 50.0:
                screens.append(
                    self._screening_item(
                        "low_pulse_rate_candidate",
                        min(90.0, 60.0 + (50.0 - bpm) * 0.8),
                        f"Dominant pulse energy sits near {int(round(bpm))} bpm in the visible PPG chunk.",
                    )
                )

        ordered = sorted(
            screens,
            key=lambda item: (-(int(item.get("score") or 0)), str(item.get("label") or "")),
        )
        return ordered[:3]

    def _build_summary(self, snapshot: Dict[str, Any], overall: int, top_issue: str) -> str:
        profile_name = snapshot["profile"]["name"]
        profile_key = _clean_text(snapshot["profile"].get("key"), profile_name).upper()
        focus_label = snapshot["review"]["stats"].get("focus_label") or "CH1"
        prediction = snapshot["last_prediction"]["label"]
        artifacts = snapshot["review"]["artifacts"]
        paused = snapshot["review"]["paused"]
        stats = snapshot["review"]["stats"]
        measurements = self._measurement_summary(snapshot)
        focus_rms = _safe_float(stats.get("focus_rms"))
        dominant_hz = _safe_float(measurements.get("dominant_frequency_hz"))
        peak_to_peak = _safe_float(measurements.get("peak_to_peak"))
        visible_span_ms = _safe_float(measurements.get("visible_span_ms"))
        emg_low_activity = profile_key == "EMG" and 0.0 < focus_rms < 1.5
        emg_stronger_activity = profile_key == "EMG" and focus_rms >= 1.5

        if top_issue == "hum":
            lead = f"Line noise is the main issue in this {profile_name} capture."
        elif top_issue == "baseline_drift":
            lead = f"Baseline drift is the main issue in this {profile_name} capture."
        elif top_issue == "clipping":
            lead = f"This {profile_name} capture is clipping, so amplitude is not trustworthy yet."
        elif artifacts:
            lead = f"Artifact activity is present in this {profile_name} capture and should be reviewed before export."
        elif overall >= 85:
            if emg_low_activity:
                lead = "This EMG capture is stable, but the current selection is still low-amplitude."
            elif emg_stronger_activity:
                lead = "This EMG capture is stable and the current selection shows stronger muscle activity."
            else:
                lead = f"No strong artifact or stability issue is standing out in this {profile_name} capture."
        elif overall >= 65:
            if emg_low_activity:
                lead = "This EMG capture is stable enough to review, but muscle activation is still light in the current selection."
            else:
                lead = f"This {profile_name} capture has no major failure signal, but one targeted check is still worth doing before export."
        else:
            lead = f"This {profile_name} capture needs cleanup before you trust it for training or export."

        detail_parts = []
        if _safe_int(snapshot["review"]["stats"].get("samples")) > 1:
            detail_parts.append(f"Current focus is {focus_label}.")
        if prediction and prediction != "--":
            detail_parts.append(f"Latest decoded label is {prediction}.")
        if visible_span_ms > 0.0:
            detail_parts.append(f"Visible span is {int(round(visible_span_ms))} ms.")
        if focus_rms > 0.0:
            detail_parts.append(f"Focus RMS is {round(focus_rms, 3)} {snapshot['profile']['units']}.")
        if peak_to_peak > 0.0:
            detail_parts.append(f"Peak-to-peak is {round(peak_to_peak, 3)} {snapshot['profile']['units']}.")
        if dominant_hz > 0.0:
            detail_parts.append(f"Dominant activity is around {round(dominant_hz, 2)} Hz.")
        if emg_low_activity:
            detail_parts.append("This reads more like rest or light recruitment than a strong contraction.")
        elif emg_stronger_activity:
            detail_parts.append("This chunk carries stronger muscle recruitment and is a better training candidate.")
        if artifacts:
            detail_parts.append(f"{len(artifacts)} artifact candidate(s) are visible in the current {'frozen' if paused else 'live'} window.")
        elif paused:
            detail_parts.append("The frozen review window is paused for inspection.")
        if top_issue == "hum":
            detail_parts.append("Check grounding and cable placement before collecting more data.")
        elif top_issue == "baseline_drift":
            detail_parts.append("Re-seat contact before collecting more data.")
        elif top_issue == "clipping":
            detail_parts.append("Back off gain or reposition electrodes before using this chunk.")
        elif top_issue not in {"clean_window", ""}:
            detail_parts.append(f"The top review issue is {top_issue.replace('_', ' ')}.")
        return " ".join([lead] + detail_parts).strip()

    def _flagged_channels(self, snapshot: Dict[str, Any]) -> List[Dict[str, Any]]:
        grouped: Dict[str, Dict[str, Any]] = {}
        for item in snapshot["review"]["artifacts"]:
            label = _clean_text(item.get("channel_label"), "CH1")
            record = grouped.setdefault(
                label,
                {
                    "label": label,
                    "reason": _clean_text(item.get("detail"), "Review artifact candidate."),
                    "severity": "medium",
                    "count": 0,
                },
            )
            record["count"] += 1
            if str(item.get("kind") or "") in {"clip", "flatline"}:
                record["severity"] = "high"
        ordered = sorted(
            grouped.values(),
            key=lambda item: ({"high": 0, "medium": 1, "low": 2}.get(str(item.get("severity")), 3), -int(item.get("count") or 0), str(item.get("label"))),
        )
        return [
            {
                "label": str(item["label"]),
                "reason": str(item["reason"]),
                "severity": str(item["severity"]),
            }
            for item in ordered[:5]
        ]

    def _next_actions(self, snapshot: Dict[str, Any], overall: int, top_issue: str) -> List[Dict[str, Any]]:
        actions: List[Dict[str, Any]] = []
        noise = snapshot["diagnostics"]["noise"]
        review = snapshot["review"]
        stats = review["stats"]
        profile_name = _clean_text(((snapshot.get("profile") or {}).get("name")), "Signal")
        focus_rms = _safe_float(stats.get("focus_rms"))
        if not review["paused"]:
            actions.append(
                {
                    "title": "Freeze a review window",
                    "detail": "Pause the live trace and drag a representative chunk before you label or export anything.",
                    "priority": "high",
                }
            )
        if top_issue == "hum":
            actions.append(
                {
                    "title": "Clean up line noise",
                    "detail": "Check ground and reference contact, then reroute cables away from power bricks and motors.",
                    "priority": "high",
                }
            )
        if top_issue == "baseline_drift":
            actions.append(
                {
                    "title": "Re-seat the electrodes",
                    "detail": "The current window shows baseline drift, so reset contact before collecting more training data.",
                    "priority": "high",
                }
            )
        if _safe_float(noise.get("clip_pct")) > 1.0:
            actions.append(
                {
                    "title": "Reduce clipping",
                    "detail": "Back off gain or reposition the electrodes before you trust this chunk for training.",
                    "priority": "high",
                }
            )
        if snapshot["review"]["artifacts"]:
            actions.append(
                {
                    "title": "Stamp artifact markers",
                    "detail": "Save markers on the flagged regions so they can be excluded later in Workshop or export.",
                    "priority": "medium",
                }
            )
        if profile_name.upper() == "EMG" and focus_rms < 1.5:
            actions.append(
                {
                    "title": "Capture a stronger activation window",
                    "detail": "The current EMG chunk is low amplitude, so collect a clearer contraction if you want a training-quality segment.",
                    "priority": "medium",
                }
            )
        if review["paused"] and _safe_int(review["stats"].get("samples")) > 1:
            actions.append(
                {
                    "title": "Analyze the selection",
                    "detail": "Send the frozen chunk to Signal Workshop and confirm the spectrum before export.",
                    "priority": "medium",
                }
            )
        if overall >= 75 and review["paused"]:
            actions.append(
                {
                    "title": "Save this reviewed training chunk",
                    "detail": "This window is stable enough to keep as a reviewed marker, so save it and keep the protocol balanced.",
                    "priority": "low",
                }
            )
        if not actions and overall >= 75:
            actions.append(
                {
                    "title": "Capture a reviewed labeled window",
                    "detail": "Freeze a representative window, label it, and send it to Workshop or export.",
                    "priority": "low",
                }
            )
        return actions[:4]

    def _marker_suggestions(self, snapshot: Dict[str, Any], top_issue: str) -> List[Dict[str, Any]]:
        suggestions: List[Dict[str, Any]] = []
        protocol = snapshot["protocol"]
        review = snapshot["review"]
        prediction = snapshot["last_prediction"]

        if top_issue == "hum":
            suggestions.append(self._marker("artifact_hum", "Line noise is dominating this window.", "Hum is elevated in diagnostics.", "range"))
        if top_issue == "baseline_drift":
            suggestions.append(self._marker("baseline_drift", "Baseline offset is drifting across the review chunk.", "Review artifacts include drift.", "range"))
        for item in review["artifacts"]:
            kind = str(item.get("kind") or "artifact")
            if kind == "clip":
                suggestions.append(self._marker("artifact_clip", "Signal is saturating in the marked region.", "Review artifacts include clipping.", "range"))
            elif kind == "flatline":
                suggestions.append(self._marker("bad_contact", "Signal drops toward a flat segment in this window.", "Review artifacts include a flatline segment.", "range"))
            elif kind == "spike":
                suggestions.append(self._marker("artifact_motion", "Sudden step change suggests motion or cable disturbance.", "Review artifacts include a spike.", "range"))
            if len(suggestions) >= 3:
                break

        phase = str(protocol.get("phase") or "")
        if phase == "rest":
            suggestions.append(self._marker("rest_window", "Protocol runner is currently in a rest phase.", "Protocol phase is rest.", "range"))
        elif phase == "trial":
            suggestions.append(self._marker("trial_window", "Protocol runner is currently in an active trial phase.", "Protocol phase is trial.", "range"))

        confidence = _safe_float(prediction.get("confidence"))
        if _safe_int(review["stats"].get("samples")) > 1 and not review["artifacts"]:
            if confidence >= 0.7:
                suggestions.append(
                    self._marker(
                        "reviewed_training_chunk",
                        f"Reviewed chunk near {prediction.get('label') or 'current'} activity.",
                        "The review chunk is selected, stable, and the decoder confidence is steady.",
                        "range",
                    )
                )
            else:
                suggestions.append(self._marker("review_candidate", "Selected chunk is worth checking in Workshop.", "There is a manual selection with low artifact burden.", "range"))

        unique: List[Dict[str, Any]] = []
        seen = set()
        for item in suggestions:
            key = str(item.get("event") or "")
            if key in seen:
                continue
            seen.add(key)
            unique.append(item)
            if len(unique) >= 5:
                break
        return unique

    def _marker(self, event: str, note: str, reason: str, kind: str) -> Dict[str, Any]:
        return {
            "event": _snake_case(event, "marker"),
            "note": _clean_text(note),
            "reason": _clean_text(reason),
            "kind": "point" if kind == "point" else "range",
        }

    def _export_recommendations(self, snapshot: Dict[str, Any], workshop: Dict[str, Any], overall: int) -> List[str]:
        out: List[str] = []
        review = snapshot["review"]
        if review["paused"] and _safe_int(review["stats"].get("samples")) > 1:
            out.append("Save the current review range as a marker so the chunk stays traceable.")
        if review["paused"] and not workshop["has_result"]:
            out.append("Send this frozen chunk to Signal Workshop before you export it.")
        if workshop["has_result"]:
            out.append("Export the analyzed chunk to MATLAB for offline review or handoff.")
        if overall >= 80:
            out.append("Keep this session in your training set candidate list.")
        return out[:4]

    def _merge_result(self, base: Dict[str, Any], enhanced: Dict[str, Any]) -> Dict[str, Any]:
        merged = dict(base)
        for key in ("summary", "next_actions", "suggested_markers", "flagged_channels", "export_recommendations"):
            value = enhanced.get(key)
            if value:
                merged[key] = value
        return merged

    def _remote_snapshot(self, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        review = dict(snapshot.get("review") or {})
        window = dict(review.get("window") or {})
        if window:
            review["window"] = {
                "sample_rate_hz": round(_safe_float(window.get("sample_rate_hz")), 3),
                "start_sample": _safe_int(window.get("start_sample")),
                "end_sample": _safe_int(window.get("end_sample")),
                "samples": _safe_int(window.get("samples")),
                "step_samples": max(_safe_int(window.get("step_samples"), 1), 1),
                "focus_channel": _safe_int(window.get("focus_channel")),
                "focus_label": _clean_text(window.get("focus_label"), "CH1"),
                "channel_count": len(list(window.get("channels") or [])),
            }
        safe = dict(snapshot)
        safe["review"] = review
        return safe

    def _merge_local_model_result(self, base: Dict[str, Any], local: Dict[str, Any]) -> Dict[str, Any]:
        merged = dict(base)
        merged["local_model_insights"] = local
        if not local.get("available"):
            return merged

        qa = dict(merged.get("qa_score") or {})
        dimensions = list(qa.get("dimensions") or [])
        overall = _safe_float(qa.get("overall"), 0.0)

        signal_qa = dict(local.get("signal_qa") or {})
        if signal_qa:
            model_score = int(round(_clamp(_safe_float(signal_qa.get("score")), 0.0, 100.0)))
            dimensions.append(self._dimension("Local signal QA", model_score))
            overall = _clamp((overall * 0.78) + (model_score * 0.22), 0.0, 100.0)

        readiness = dict(local.get("fatigue_readiness") or {})
        if readiness:
            readiness_score = int(round(_clamp(_safe_float(readiness.get("score")), 0.0, 100.0)))
            dimensions.append(self._dimension("Readiness", readiness_score))
            if readiness_score < 45:
                merged["next_actions"] = self._prepend_action(
                    merged.get("next_actions") or [],
                    {
                        "title": "Capture a fresh readiness window",
                        "detail": "The local readiness model is reading low, so take a short rest baseline or collect a cleaner contraction window before export.",
                        "priority": "medium",
                    },
                )

        foundation_embeddings = list(local.get("foundation_embeddings") or [])
        if foundation_embeddings:
            primary_embedding = dict(foundation_embeddings[0] or {})
            foundation_name = str(primary_embedding.get("name") or primary_embedding.get("model_id") or "").strip()
            foundation_dim = int(round(_safe_float(primary_embedding.get("embedding_dim"))))
            if foundation_name:
                merged["next_actions"] = self._prepend_action(
                    merged.get("next_actions") or [],
                    {
                        "title": f"Review {foundation_name} encoder pass",
                        "detail": f"The local foundation encoder produced a {foundation_dim}-d embedding for this window. Use it as a feature pass, not a diagnosis.",
                        "priority": "low",
                    },
                )

        condition_screens = list(local.get("condition_screen") or [])
        if condition_screens:
            top_screen = dict(condition_screens[0] or {})
            label = str(top_screen.get("label") or "").replace("_", " ").strip()
            score = int(round(_safe_float(top_screen.get("score"))))
            if label:
                merged["next_actions"] = self._prepend_action(
                    merged.get("next_actions") or [],
                    {
                        "title": f"Review {label} screening output",
                        "detail": f"A research-only condition head flagged {label} at about {score} percent. Treat it as a screening cue for offline review, not a diagnosis.",
                        "priority": "medium",
                    },
                )

        artifact = dict(local.get("artifact_classifier") or {})
        artifact_label = _snake_case(artifact.get("label"), "")
        artifact_conf = _safe_float(artifact.get("confidence"))
        artifact_detail = _clean_text(artifact.get("detail"))
        if artifact_label and artifact_label not in {"", "clean", "none", "normal", "stable", "ok"}:
            artifact_summary = dict(merged.get("artifact_summary") or {})
            artifact_summary["model_hint"] = artifact_label
            if str(artifact_summary.get("top_issue") or "") in {"", "clean_window"}:
                artifact_summary["top_issue"] = artifact_label
            merged["artifact_summary"] = artifact_summary
            merged["next_actions"] = self._prepend_action(
                merged.get("next_actions") or [],
                {
                    "title": f"Check {artifact_label.replace('_', ' ')} risk",
                    "detail": artifact_detail or f"The local artifact model is leaning toward {artifact_label.replace('_', ' ')} at about {int(round(artifact_conf * 100.0))} percent confidence.",
                    "priority": "medium" if artifact_conf < 0.8 else "high",
                },
            )
            merged["suggested_markers"] = self._prepend_marker(
                merged.get("suggested_markers") or [],
                self._marker(
                    f"model_{artifact_label}",
                    f"Model hint: {artifact_label.replace('_', ' ')} risk in this window.",
                    "A local biosignal model flagged this review span.",
                    "range",
                ),
            )

        qa["overall"] = int(round(_clamp(overall, 0.0, 100.0)))
        qa["grade"] = self._grade(qa["overall"])
        qa["dimensions"] = dimensions[:8]
        merged["qa_score"] = qa
        merged["summary"] = self._augment_summary_with_local_models(
            str(merged.get("summary") or ""),
            signal_qa,
            artifact,
            readiness,
            condition_screens,
            foundation_embeddings,
        )
        return merged

    def _augment_summary_with_local_models(
        self,
        summary: str,
        signal_qa: Dict[str, Any],
        artifact: Dict[str, Any],
        readiness: Dict[str, Any],
        condition_screens: List[Dict[str, Any]],
        foundation_embeddings: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        parts: List[str] = [summary.strip()]
        if signal_qa:
            label = str(signal_qa.get("label") or "").replace("_", " ").strip()
            score = int(round(_safe_float(signal_qa.get("score"))))
            if label:
                parts.append(f"Local QA reads {label} at {score} percent.")
        artifact_label = str(artifact.get("label") or "").replace("_", " ").strip()
        if artifact_label and artifact_label.lower() not in {"", "clean", "none", "normal", "stable", "ok"}:
            parts.append(f"Local artifact model leans toward {artifact_label}.")
        if readiness:
            parts.append(f"Readiness is {int(round(_safe_float(readiness.get('score'))))} percent.")
        if condition_screens:
            top = dict(condition_screens[0] or {})
            label = str(top.get("label") or "").replace("_", " ").strip()
            score = int(round(_safe_float(top.get("score"))))
            if label:
                parts.append(f"Research screen highlights {label} at {score} percent.")
        if foundation_embeddings:
            primary = dict(foundation_embeddings[0] or {})
            name = str(primary.get("name") or primary.get("model_id") or "").strip()
            dim = int(round(_safe_float(primary.get("embedding_dim"))))
            if name and dim > 0:
                parts.append(f"{name} produced a {dim}-d feature embedding for this window.")
        return " ".join(part for part in parts if part).strip()

    def _research_watchlist(
        self,
        snapshot: Dict[str, Any],
        routing_plan: Dict[str, Any],
        local: Dict[str, Any],
        heuristic_screens: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for item in list(local.get("condition_screen") or [])[:3]:
            top = dict(item or {})
            label = _snake_case(top.get("label"), "")
            if not label:
                continue
            out.append(
                {
                    "kind": "screening_result",
                    "label": label,
                    "display_label": label.replace("_", " "),
                    "status": "active",
                    "score": int(round(_safe_float(top.get("score")))),
                    "confidence": round(_safe_float(top.get("confidence")), 4),
                    "source": _clean_text(top.get("model_id"), "local head"),
                    "detail": "Research-only screening output. Do not interpret as a diagnosis.",
                    "research_only": True,
                }
            )
        if out:
            return out[:3]

        for item in list(heuristic_screens or [])[:3]:
            out.append(dict(item))
        if out:
            return out[:3]

        for item in list(routing_plan.get("research_heads") or [])[:3]:
            if str(item.get("status") or "") != "ready":
                continue
            out.append(
                {
                    "kind": "available_head",
                    "label": _snake_case(item.get("id"), "research_head"),
                    "display_label": _clean_text(item.get("name"), "Research head"),
                    "status": "ready",
                    "score": 0,
                    "confidence": 0.0,
                    "source": _clean_text(item.get("id"), "registry"),
                    "detail": _clean_text(item.get("notes"), "Research-only screening head."),
                    "research_only": True,
                }
            )

        if out:
            return out[:3]

        modality = _clean_text(((snapshot.get("profile") or {}).get("name")), "signal")
        return [
            {
                "kind": "status",
                "label": "no_condition_head",
                "display_label": f"No {modality} condition head active",
                "status": "idle",
                "score": 0,
                "confidence": 0.0,
                "source": "router",
                "detail": "The current modality has no installed research screening head. KYMA is using measurements, QA, artifacts, and routing only.",
                "research_only": True,
            }
        ]

    def _prepend_action(self, actions: List[Dict[str, Any]], action: Dict[str, Any]) -> List[Dict[str, Any]]:
        title = str(action.get("title") or "")
        if not title:
            return actions[:4]
        deduped = [item for item in actions if str(item.get("title") or "") != title]
        return [action, *deduped][:4]

    def _prepend_marker(self, markers: List[Dict[str, Any]], marker: Dict[str, Any]) -> List[Dict[str, Any]]:
        event = str(marker.get("event") or "")
        if not event:
            return markers[:5]
        deduped = [item for item in markers if str(item.get("event") or "") != event]
        return [marker, *deduped][:5]

    def _remote_enhancement(self, snapshot: Dict[str, Any], baseline: Dict[str, Any], routing_plan: Dict[str, Any]) -> Dict[str, Any]:
        routing_brief = self.router.prompt(routing_plan)
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "summary": {"type": "string"},
                "next_actions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "title": {"type": "string"},
                            "detail": {"type": "string"},
                            "priority": {"type": "string", "enum": ["high", "medium", "low"]},
                        },
                        "required": ["title", "detail", "priority"],
                    },
                    "maxItems": 4,
                },
                "suggested_markers": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "event": {"type": "string"},
                            "note": {"type": "string"},
                            "reason": {"type": "string"},
                            "kind": {"type": "string", "enum": ["point", "range"]},
                        },
                        "required": ["event", "note", "reason", "kind"],
                    },
                    "maxItems": 5,
                },
                "flagged_channels": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "label": {"type": "string"},
                            "reason": {"type": "string"},
                            "severity": {"type": "string", "enum": ["high", "medium", "low"]},
                        },
                        "required": ["label", "reason", "severity"],
                    },
                    "maxItems": 5,
                },
                "export_recommendations": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": 4,
                },
            },
            "required": [
                "summary",
                "next_actions",
                "suggested_markers",
                "flagged_channels",
                "export_recommendations",
            ],
        }

        request_body = {
            "model": self.model,
            "store": False,
            "temperature": 0.2,
            "max_output_tokens": 700,
            "instructions": (
                "You are KYMA's biosignal workflow copilot. "
                "Use only the provided snapshot and heuristic baseline. "
                f"{routing_brief} "
                "Do not diagnose, infer disease, or make clinical claims. "
                "Keep outputs short, operational, and grounded in the provided fields. "
                "Suggested marker names must be lowercase snake_case."
            ),
            "input": [
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "snapshot": snapshot,
                            "baseline": baseline,
                            "routing": routing_plan,
                            "task": "Refine the summary, next actions, marker suggestions, flagged channels, and export recommendations.",
                        },
                        separators=(",", ":"),
                    ),
                }
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "kyma_copilot_v1",
                    "strict": True,
                    "schema": schema,
                }
            },
        }

        req = urllib.request.Request(
            f"{self.base_url}/responses",
            data=json.dumps(request_body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        if self.api_key:
            req.add_header("Authorization", f"Bearer {self.api_key}")
        try:
            with urllib.request.urlopen(req, timeout=20) as response:  # nosec B310
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:  # pragma: no cover - depends on external API
            body = exc.read().decode("utf-8", errors="replace")
            try:
                data = json.loads(body)
                message = (
                    data.get("error", {}).get("message")
                    or data.get("message")
                    or body
                )
            except json.JSONDecodeError:
                message = body
            raise RuntimeError(message.strip() or f"Remote AI request failed with {exc.code}") from exc
        except urllib.error.URLError as exc:  # pragma: no cover - depends on external API
            raise RuntimeError(str(exc.reason or exc)) from exc

        text = self._extract_output_text(payload)
        if not text:
            raise RuntimeError("Remote AI response did not include structured output text")
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Remote AI response was not valid JSON") from exc

    def _extract_output_text(self, payload: Dict[str, Any]) -> str:
        for item in list(payload.get("output") or []):
            if str(item.get("type") or "") != "message":
                continue
            for content in list(item.get("content") or []):
                if str(content.get("type") or "") == "output_text":
                    return str(content.get("text") or "")
        return ""
