"""Optional local PyTorch runtime for compact biosignal model packs."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from classifiers import _import_torch
from foundation_models import load_foundation_adapter


VALID_TASKS = {"signal_qa", "artifact_classifier", "fatigue_readiness", "condition_screen"}
FOUNDATION_TASK = "foundation_encoder"
BUILTIN_ARTIFACT_HEAD_ID = "builtin_artifact_head"
POSITIVE_QA_LABELS = {"clean", "good", "stable", "ready", "pass"}
NEGATIVE_QA_LABELS = {"artifact", "bad", "noisy", "reject", "poor", "fail"}
READY_LABELS = {"ready", "fresh", "rested", "stable", "good", "pass"}
FATIGUE_LABELS = {"fatigue", "tired", "stressed", "poor", "fail"}
CLEAN_ARTIFACT_LABELS = {"clean", "none", "normal", "stable", "ok"}


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


def _snake_case(value: Any, fallback: str = "model") -> str:
    raw = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value or "").strip())
    while "__" in raw:
        raw = raw.replace("__", "_")
    raw = raw.strip("_")
    return raw[:64] or fallback


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp = np.exp(shifted)
    total = np.sum(exp)
    if total <= 0:
        return np.full_like(values, 1.0 / max(len(values), 1), dtype=np.float32)
    return exp / total


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + float(np.exp(-value)))


def _ensure_local_pydeps_on_path() -> None:
    base_dir = Path(__file__).resolve().parents[1] / "sessions"
    for local_pydeps in (base_dir / "pydeps_rt", base_dir / "pydeps"):
        if not local_pydeps.exists():
            continue
        local_path = str(local_pydeps)
        if local_path not in sys.path:
            sys.path.insert(0, local_path)


def _import_safetensors():
    _ensure_local_pydeps_on_path()
    from safetensors import safe_open

    return safe_open


class BiosignalModelRuntime:
    """Loads small TorchScript model packs for non-diagnostic biosignal tasks."""

    def __init__(self, roots: List[Path]) -> None:
        self.roots = [Path(root) for root in roots]
        self.models: List[Dict[str, Any]] = []
        self.foundation_models: List[Dict[str, Any]] = []
        self.last_error = ""
        self.torch_available = False
        self.torch_error = ""
        self.reload()

    def reload(self) -> None:
        self.models = []
        self.foundation_models = []
        self.last_error = ""
        self.torch_available = False
        self.torch_error = ""
        try:
            torch, _, _ = _import_torch()
            self.torch_available = torch is not None
        except Exception as exc:
            self.torch_error = f"{type(exc).__name__}: {exc}"
            self.last_error = self.torch_error

        for manifest_path in self._discover_manifest_paths():
            record = self._read_manifest_record(manifest_path)
            if not record:
                continue
            if str(record.get("task") or "") == FOUNDATION_TASK:
                self._load_foundation_record(record)
                continue
            if not self.torch_available:
                record["error"] = self.torch_error or "PyTorch runtime is unavailable."
                self.models.append(record)
                continue
            if str(record.get("format") or "") != "torchscript":
                record["error"] = "Unsupported format. Use TorchScript packs."
                self.models.append(record)
                continue
            if str(record.get("error") or "").strip():
                self.models.append(record)
                continue
            try:
                torch, _, _ = _import_torch()
                module = torch.jit.load(str(record["weights_path"]), map_location="cpu")
                module.eval()
                record["loaded"] = True
                record["module"] = module
            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
                self.last_error = record["error"]
            self.models.append(record)

    def status(self) -> Dict[str, Any]:
        loaded = [item for item in self.models if item.get("loaded")]
        foundation_loaded = [item for item in self.foundation_models if item.get("loaded")]
        return {
            "available": True,
            "torch_available": self.torch_available,
            "model_count": len(loaded),
            "foundation_model_count": len(foundation_loaded),
            "builtin_heads": [
                {
                    "id": BUILTIN_ARTIFACT_HEAD_ID,
                    "task": "artifact_classifier",
                    "name": "Built-in Artifact Head",
                    "loaded": True,
                    "modalities": ["emg", "eeg", "ecg", "eog", "eda", "ppg", "resp", "temp"],
                }
            ],
            "tasks": sorted({str(item.get("task") or "") for item in loaded if item.get("task")}),
            "models": [
                {
                    "id": str(item.get("id") or ""),
                    "name": str(item.get("name") or ""),
                    "task": str(item.get("task") or ""),
                    "loaded": bool(item.get("loaded")),
                    "window_samples": _safe_int(item.get("window_samples"), 0),
                    "channels": _safe_int(item.get("channels"), 0),
                    "error": str(item.get("error") or ""),
                }
                for item in self.models
            ],
            "foundation_models": [
                {
                    "id": str(item.get("id") or ""),
                    "name": str(item.get("name") or ""),
                    "task": str(item.get("task") or ""),
                    "format": str(item.get("format") or ""),
                    "adapter": str(item.get("adapter") or ""),
                    "loaded": bool(item.get("loaded")),
                    "validated": bool(item.get("validated")),
                    "inference_ready": bool(item.get("inference_ready")),
                    "modalities": [str(mod) for mod in list(item.get("modalities") or [])],
                    "embedding_dim": _safe_int(item.get("embedding_dim"), 0),
                    "outputs": [str(out) for out in list(item.get("outputs") or [])],
                    "summary": dict(item.get("checkpoint_summary") or {}),
                    "error": str(item.get("error") or ""),
                }
                for item in self.foundation_models
            ],
            "last_error": self.last_error or self.torch_error,
            "search_roots": [root.as_posix() for root in self.roots],
        }

    def analyze(self, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        status = self.status()
        window = dict(((snapshot.get("review") or {}).get("window") or {}))
        channels = window.get("channels") or []
        if not status["model_count"] and not status.get("foundation_model_count") and not status.get("builtin_heads"):
            return {"available": False, "reason": "no_models", **status}
        if not channels:
            return {"available": False, "reason": "no_window", **status}

        array = np.asarray(channels, dtype=np.float32)
        if array.ndim != 2 or not array.size:
            return {"available": False, "reason": "bad_window", **status}

        outputs: List[Dict[str, Any]] = []
        foundation_outputs: List[Dict[str, Any]] = []
        errors: List[str] = []
        for record in self.models:
            if not record.get("loaded"):
                continue
            try:
                prepared = self._prepare_window(array, record)
                result = self._run_model(prepared, record)
                if result:
                    outputs.append(result)
            except Exception as exc:
                errors.append(f"{record.get('name') or record.get('id')}: {type(exc).__name__}: {exc}")
        modality = str(((snapshot.get("profile") or {}).get("key")) or "").strip().lower()
        for record in self.foundation_models:
            if not record.get("loaded") or not record.get("inference_ready"):
                continue
            modalities = [str(item).strip().lower() for item in list(record.get("modalities") or [])]
            if modalities and modality and modality not in modalities:
                continue
            try:
                prepared = self._prepare_window(array, record)
                result = self._run_foundation_model(prepared, record)
                if result:
                    foundation_outputs.append(result)
            except Exception as exc:
                errors.append(f"{record.get('name') or record.get('id')}: {type(exc).__name__}: {exc}")

        builtin_artifact = self._builtin_artifact_classifier(snapshot, array)
        if builtin_artifact and not self._first_task(outputs, "artifact_classifier"):
            outputs.append(builtin_artifact)

        bundled = {
            "available": bool(outputs or foundation_outputs),
            "reason": "" if (outputs or foundation_outputs) else ("no_task_heads" if status.get("foundation_model_count") else "no_outputs"),
            "signal_qa": self._first_task(outputs, "signal_qa"),
            "artifact_classifier": self._first_task(outputs, "artifact_classifier"),
            "fatigue_readiness": self._first_task(outputs, "fatigue_readiness"),
            "condition_screen": self._all_tasks(outputs, "condition_screen"),
            "foundation_embeddings": foundation_outputs,
            "outputs": outputs,
            "errors": errors[:4],
            **status,
        }
        if errors and not bundled["available"]:
            bundled["last_error"] = errors[0]
        return bundled

    def _builtin_artifact_classifier(self, snapshot: Dict[str, Any], channels: np.ndarray) -> Dict[str, Any]:
        diagnostics = dict((snapshot.get("diagnostics") or {}).get("noise") or {})
        review = dict(snapshot.get("review") or {})
        review_artifacts = list(review.get("artifacts") or [])

        hum_db = max(_safe_float(diagnostics.get("hum_50_db"), -120.0), _safe_float(diagnostics.get("hum_60_db"), -120.0))
        drift_db = _safe_float(diagnostics.get("drift_db"), -120.0)
        clip_pct = _safe_float(diagnostics.get("clip_pct"), 0.0)
        crest_factor = _safe_float(diagnostics.get("crest_factor"), 0.0)
        peak_to_peak = np.ptp(channels.astype(np.float64), axis=1) if channels.size else np.asarray([], dtype=np.float64)
        rms = np.sqrt(np.mean(np.square(channels.astype(np.float64)), axis=1)) if channels.size else np.asarray([], dtype=np.float64)
        diff_rms = np.sqrt(np.mean(np.square(np.diff(channels.astype(np.float64), axis=1)), axis=1)) if channels.shape[1] > 1 else np.asarray([], dtype=np.float64)
        spike_ratio = float(np.median(diff_rms / np.maximum(rms, 1e-6))) if diff_rms.size and rms.size else 0.0
        flatline_ratio = float(np.mean(peak_to_peak <= np.maximum(np.median(peak_to_peak) * 0.08, 1e-4))) if peak_to_peak.size else 0.0
        low_signal = float(np.median(rms)) if rms.size else 0.0

        ranked: List[tuple[str, float, str]] = []
        if clip_pct > 0.2 or any(str(item.get("kind") or "").lower() == "clip" for item in review_artifacts):
            ranked.append((
                "clip",
                _clamp(0.62 + min(clip_pct, 10.0) / 12.0, 0.0, 0.99),
                f"Clip percentage is {clip_pct:.2f} percent, so the window is saturating.",
            ))
        if hum_db >= -28.0:
            ranked.append((
                "hum",
                _clamp(0.48 + max(0.0, hum_db + 28.0) / 20.0, 0.0, 0.97),
                f"Line-noise energy is elevated at {hum_db:.1f} dB relative power.",
            ))
        if drift_db >= -26.0 or any(str(item.get("kind") or "").lower() == "drift" for item in review_artifacts):
            ranked.append((
                "drift",
                _clamp(0.46 + max(0.0, drift_db + 26.0) / 18.0, 0.0, 0.96),
                f"Baseline drift energy is elevated at {drift_db:.1f} dB relative power.",
            ))
        if flatline_ratio >= 0.25 or any(str(item.get("kind") or "").lower() == "flatline" for item in review_artifacts):
            ranked.append((
                "flatline",
                _clamp(0.34 + flatline_ratio * 0.28, 0.0, 0.82),
                "One or more channels are near-flat compared with the rest of the window.",
            ))
        if spike_ratio >= 1.85 or crest_factor >= 5.5 or any(str(item.get("kind") or "").lower() == "spike" for item in review_artifacts):
            ranked.append((
                "motion_artifact",
                _clamp(0.44 + max(spike_ratio - 1.5, 0.0) * 0.28 + max(crest_factor - 4.0, 0.0) * 0.04, 0.0, 0.95),
                f"Fast transient energy is elevated with spike ratio {spike_ratio:.2f} and crest factor {crest_factor:.2f}.",
            ))
        if not ranked and low_signal <= 0.015:
            ranked.append((
                "low_signal",
                0.58,
                "Window amplitude is very small, so the signal may be closer to rest or weak contact than strong activity.",
            ))

        if ranked:
            ranked.sort(key=lambda item: item[1], reverse=True)
            label, confidence, detail = ranked[0]
        else:
            label, confidence, detail = (
                "clean",
                0.82,
                "No clip, hum, drift, or motion threshold crossed in this pass.",
            )

        return {
            "task": "artifact_classifier",
            "model_id": BUILTIN_ARTIFACT_HEAD_ID,
            "label": str(label),
            "confidence": round(float(confidence), 4),
            "clean": str(label).lower() in CLEAN_ARTIFACT_LABELS,
            "detail": detail,
            "ranked": [
                {
                    "label": item[0],
                    "confidence": round(float(item[1]), 4),
                    "detail": item[2],
                }
                for item in ranked[:4]
            ],
        }

    def _discover_manifest_paths(self) -> List[Path]:
        manifests: List[Path] = []
        for root in self.roots:
            if not root.exists():
                continue
            manifests.extend(root.rglob("manifest.json"))
            manifests.extend(root.glob("manifest.json"))
        unique = sorted({path.resolve() for path in manifests})
        return unique

    def _read_manifest_record(self, manifest_path: Path) -> Optional[Dict[str, Any]]:
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            self.last_error = f"{manifest_path.name}: {type(exc).__name__}: {exc}"
            return None

        task = str(payload.get("task") or "").strip()
        if task not in VALID_TASKS and task != FOUNDATION_TASK:
            return None
        weights_rel = str(payload.get("weights") or "").strip()
        if not weights_rel:
            return None
        weights_path = (manifest_path.parent / weights_rel).resolve()
        record = {
            "id": _snake_case(payload.get("id") or payload.get("name") or manifest_path.parent.name, manifest_path.parent.name),
            "name": str(payload.get("name") or manifest_path.parent.name),
            "task": task,
            "format": str(payload.get("format") or "torchscript"),
            "weights_path": weights_path,
            "layout": str(payload.get("layout") or "bct").lower(),
            "normalize": str(payload.get("normalize") or "zscore").lower(),
            "window_samples": _safe_int(payload.get("window_samples"), 0),
            "channels": _safe_int(payload.get("channels"), 0),
            "sample_rate_hz": _safe_float(payload.get("sample_rate_hz"), 0.0),
            "labels": [str(item) for item in list(payload.get("labels") or [])],
            "modalities": [str(item).strip().lower() for item in list(payload.get("modalities") or []) if str(item).strip()],
            "outputs": [str(item) for item in list(payload.get("outputs") or [])],
            "embedding_dim": _safe_int(payload.get("embedding_dim"), 0),
            "adapter": str(payload.get("adapter") or "").strip(),
            "validated": False,
            "inference_ready": bool(payload.get("inference_ready")),
            "trusted": bool(payload.get("trusted")),
            "loaded": False,
            "checkpoint_summary": {},
            "notes": str(payload.get("notes") or ""),
            "error": "" if weights_path.exists() else f"Missing weights: {weights_path.name}",
        }
        return record

    def _prepare_window(self, channels: np.ndarray, record: Dict[str, Any]) -> np.ndarray:
        window = np.asarray(channels, dtype=np.float32)
        target_channels = _safe_int(record.get("channels"), 0)
        target_samples = _safe_int(record.get("window_samples"), 0)
        if target_channels > 0 and window.shape[0] != target_channels:
            padded = np.zeros((target_channels, window.shape[1]), dtype=np.float32)
            copy = min(target_channels, window.shape[0])
            padded[:copy] = window[:copy]
            window = padded
        if target_samples > 0 and window.shape[1] != target_samples:
            window = self._resample_window(window, target_samples)
        normalize = str(record.get("normalize") or "zscore").lower()
        if normalize == "zscore":
            mean = window.mean(axis=1, keepdims=True)
            std = window.std(axis=1, keepdims=True)
            window = (window - mean) / np.maximum(std, 1e-6)
        elif normalize == "rms":
            rms = np.sqrt(np.mean(np.square(window), axis=1, keepdims=True))
            window = window / np.maximum(rms, 1e-6)
        elif normalize == "peak":
            peak = np.max(np.abs(window), axis=1, keepdims=True)
            window = window / np.maximum(peak, 1e-6)
        return window.astype(np.float32, copy=False)

    def _resample_window(self, channels: np.ndarray, target_samples: int) -> np.ndarray:
        if channels.shape[1] == target_samples:
            return channels
        if channels.shape[1] <= 1:
            return np.repeat(channels, target_samples, axis=1)
        old_x = np.linspace(0.0, 1.0, channels.shape[1], dtype=np.float32)
        new_x = np.linspace(0.0, 1.0, target_samples, dtype=np.float32)
        out = np.zeros((channels.shape[0], target_samples), dtype=np.float32)
        for idx in range(channels.shape[0]):
            out[idx] = np.interp(new_x, old_x, channels[idx]).astype(np.float32)
        return out

    def _load_foundation_record(self, record: Dict[str, Any]) -> None:
        if str(record.get("error") or "").strip():
            self.foundation_models.append(record)
            return

        fmt = str(record.get("format") or "").strip().lower()
        try:
            if fmt == "safetensors_checkpoint":
                summary = self._summarize_safetensors_checkpoint(record["weights_path"], record)
            elif fmt == "pytorch_checkpoint_bundle":
                if not self.torch_available:
                    raise RuntimeError(self.torch_error or "PyTorch runtime is unavailable.")
                summary = self._summarize_pytorch_checkpoint(record["weights_path"], record)
            else:
                raise RuntimeError(f"Unsupported foundation checkpoint format: {fmt or 'unknown'}")
            record["loaded"] = True
            record["validated"] = True
            record["checkpoint_summary"] = summary
            try:
                module = load_foundation_adapter(record)
                record["module"] = module
                record["inference_ready"] = True
            except Exception:
                record["inference_ready"] = False
            if not record.get("outputs") and _safe_int(record.get("embedding_dim"), 0) > 0:
                record["outputs"] = [f"{_safe_int(record.get('embedding_dim'), 0)}-d embedding"]
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            self.last_error = record["error"]
        self.foundation_models.append(record)

    def _summarize_safetensors_checkpoint(self, weights_path: Path, record: Dict[str, Any]) -> Dict[str, Any]:
        safe_open = _import_safetensors()
        with safe_open(str(weights_path), framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
            sample_shapes: Dict[str, List[int]] = {}
            for key in keys[:5]:
                tensor = handle.get_tensor(key)
                shape = getattr(tensor, "shape", None)
                sample_shapes[key] = list(shape) if shape is not None else []
            embedding_dim = _safe_int(record.get("embedding_dim"), 0)
            if not embedding_dim and "norm.weight" in keys:
                tensor = handle.get_tensor("norm.weight")
                shape = getattr(tensor, "shape", None)
                if shape:
                    embedding_dim = int(shape[0])
            return {
                "type": "safetensors",
                "tensor_count": len(keys),
                "top_keys": keys[:8],
                "sample_shapes": sample_shapes,
                "embedding_dim": embedding_dim,
            }

    def _summarize_pytorch_checkpoint(self, weights_path: Path, record: Dict[str, Any]) -> Dict[str, Any]:
        torch, _, _ = _import_torch()
        payload = torch.load(
            weights_path,
            map_location="cpu",
            weights_only=False if bool(record.get("trusted")) else True,
        )
        top_keys = list(payload.keys())[:20] if isinstance(payload, dict) else []
        state = None
        if isinstance(payload, dict):
            for candidate in ("model_state_dict", "state_dict"):
                if isinstance(payload.get(candidate), dict):
                    state = payload.get(candidate)
                    break
            if state is None and all(isinstance(key, str) for key in payload.keys()):
                state = payload
        sample_shapes: Dict[str, Any] = {}
        state_keys: List[str] = []
        tensor_count = 0
        if isinstance(state, dict):
            state_keys = list(state.keys())
            tensor_count = len(state_keys)
            for key in state_keys[:5]:
                tensor = state[key]
                shape = getattr(tensor, "shape", None)
                sample_shapes[key] = list(shape) if shape is not None else type(tensor).__name__
        embedding_dim = _safe_int(record.get("embedding_dim"), 0)
        return {
            "type": "pytorch_bundle",
            "top_keys": top_keys,
            "tensor_count": tensor_count,
            "state_key_count": tensor_count,
            "state_top_keys": state_keys[:8],
            "sample_shapes": sample_shapes,
            "embedding_dim": embedding_dim,
        }

    def _run_model(self, channels: np.ndarray, record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        torch, _, _ = _import_torch()
        layout = str(record.get("layout") or "bct").lower()
        tensor = torch.tensor(channels, dtype=torch.float32)
        if layout == "btc":
            tensor = tensor.transpose(0, 1).unsqueeze(0)
        elif layout == "ct":
            pass
        else:
            tensor = tensor.unsqueeze(0)
        with torch.no_grad():
            output = record["module"](tensor)
        if isinstance(output, (list, tuple)):
            output = output[0]
        if hasattr(output, "detach"):
            array = output.detach().cpu().numpy().astype(np.float32).reshape(-1)
        else:
            array = np.asarray(output, dtype=np.float32).reshape(-1)
        return self._interpret_output(array, record)

    def _run_foundation_model(self, channels: np.ndarray, record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        torch, _, _ = _import_torch()
        tensor = torch.tensor(channels, dtype=torch.float32).unsqueeze(0)
        module = record.get("module")
        if module is None:
            return None
        with torch.no_grad():
            output = module(tensor)
        if hasattr(output, "detach"):
            array = output.detach().cpu().numpy().astype(np.float32).reshape(-1)
        else:
            array = np.asarray(output, dtype=np.float32).reshape(-1)
        if not len(array):
            return None
        preview = [round(float(item), 6) for item in array[:8].tolist()]
        norm = float(np.linalg.norm(array))
        return {
            "task": "foundation_encoder",
            "model_id": str(record.get("id") or ""),
            "name": str(record.get("name") or ""),
            "embedding_dim": int(array.shape[0]),
            "embedding_norm": round(norm, 6),
            "mean": round(float(np.mean(array)), 6),
            "std": round(float(np.std(array)), 6),
            "preview": preview,
        }

    def _interpret_output(self, values: np.ndarray, record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        task = str(record.get("task") or "")
        labels = [str(item) for item in list(record.get("labels") or [])]
        if not len(values):
            return None
        if task == "signal_qa":
            return self._interpret_signal_qa(values, labels, record)
        if task == "artifact_classifier":
            return self._interpret_artifact(values, labels, record)
        if task == "fatigue_readiness":
            return self._interpret_readiness(values, labels, record)
        if task == "condition_screen":
            return self._interpret_condition_screen(values, labels, record)
        return None

    def _interpret_signal_qa(self, values: np.ndarray, labels: List[str], record: Dict[str, Any]) -> Dict[str, Any]:
        if len(values) == 1:
            quality_score = _sigmoid(float(values[0]))
            label = "stable" if quality_score >= 0.5 else "watch"
            confidence = quality_score if quality_score >= 0.5 else 1.0 - quality_score
        else:
            probs = _softmax(values)
            top = int(np.argmax(probs))
            label = labels[top] if top < len(labels) else f"class_{top}"
            confidence = float(probs[top])
            positives = [probs[idx] for idx, item in enumerate(labels) if item.lower() in POSITIVE_QA_LABELS]
            negatives = [probs[idx] for idx, item in enumerate(labels) if item.lower() in NEGATIVE_QA_LABELS]
            if positives:
                quality_score = float(np.sum(positives))
            elif negatives:
                quality_score = float(1.0 - np.sum(negatives))
            else:
                quality_score = confidence if label.lower() in POSITIVE_QA_LABELS else 1.0 - confidence
        score = int(round(_clamp(quality_score * 100.0, 0.0, 100.0)))
        return {
            "task": "signal_qa",
            "model_id": str(record.get("id") or ""),
            "label": str(label),
            "score": score,
            "confidence": round(float(confidence), 4),
        }

    def _interpret_artifact(self, values: np.ndarray, labels: List[str], record: Dict[str, Any]) -> Dict[str, Any]:
        probs = _softmax(values if len(values) > 1 else np.asarray([0.0, float(values[0])], dtype=np.float32))
        top = int(np.argmax(probs))
        label = labels[top] if top < len(labels) else ("clean" if top == 0 else "artifact")
        return {
            "task": "artifact_classifier",
            "model_id": str(record.get("id") or ""),
            "label": str(label),
            "confidence": round(float(probs[top]), 4),
            "clean": str(label).lower() in CLEAN_ARTIFACT_LABELS,
        }

    def _interpret_readiness(self, values: np.ndarray, labels: List[str], record: Dict[str, Any]) -> Dict[str, Any]:
        if len(values) == 1:
            ready = _sigmoid(float(values[0]))
            label = "ready" if ready >= 0.5 else "fatigue"
            confidence = ready if ready >= 0.5 else 1.0 - ready
        else:
            probs = _softmax(values)
            top = int(np.argmax(probs))
            label = labels[top] if top < len(labels) else f"class_{top}"
            confidence = float(probs[top])
            positives = [probs[idx] for idx, item in enumerate(labels) if item.lower() in READY_LABELS]
            negatives = [probs[idx] for idx, item in enumerate(labels) if item.lower() in FATIGUE_LABELS]
            if positives:
                ready = float(np.sum(positives))
            elif negatives:
                ready = float(1.0 - np.sum(negatives))
            else:
                ready = confidence if label.lower() in READY_LABELS else 1.0 - confidence
        score = int(round(_clamp(ready * 100.0, 0.0, 100.0)))
        return {
            "task": "fatigue_readiness",
            "model_id": str(record.get("id") or ""),
            "label": str(label),
            "score": score,
            "confidence": round(float(confidence), 4),
        }

    def _interpret_condition_screen(self, values: np.ndarray, labels: List[str], record: Dict[str, Any]) -> Dict[str, Any]:
        if len(values) == 1:
            risk = _sigmoid(float(values[0]))
            label = labels[0] if labels else "screen"
            ranked = [
                {
                    "label": str(label),
                    "score": int(round(_clamp(risk * 100.0, 0.0, 100.0))),
                    "confidence": round(float(max(risk, 1.0 - risk)), 4),
                }
            ]
            top_label = str(label)
            top_confidence = float(ranked[0]["confidence"])
            top_score = int(ranked[0]["score"])
        else:
            probs = _softmax(values)
            ranked_pairs = sorted(
                [
                    (
                        labels[idx] if idx < len(labels) else f"class_{idx}",
                        float(probs[idx]),
                    )
                    for idx in range(len(probs))
                ],
                key=lambda item: item[1],
                reverse=True,
            )
            ranked = [
                {
                    "label": str(label),
                    "score": int(round(_clamp(confidence * 100.0, 0.0, 100.0))),
                    "confidence": round(float(confidence), 4),
                }
                for label, confidence in ranked_pairs[:3]
            ]
            top_label = str(ranked[0]["label"]) if ranked else "screen"
            top_confidence = float(ranked[0]["confidence"]) if ranked else 0.0
            top_score = int(ranked[0]["score"]) if ranked else 0
        return {
            "task": "condition_screen",
            "model_id": str(record.get("id") or ""),
            "label": top_label,
            "score": top_score,
            "confidence": round(top_confidence, 4),
            "screenings": ranked,
            "research_only": True,
        }

    def _first_task(self, outputs: List[Dict[str, Any]], task: str) -> Optional[Dict[str, Any]]:
        for item in outputs:
            if str(item.get("task") or "") == task:
                return item
        return None

    def _all_tasks(self, outputs: List[Dict[str, Any]], task: str) -> List[Dict[str, Any]]:
        return [item for item in outputs if str(item.get("task") or "") == task]
