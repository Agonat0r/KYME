"""Saved filter design registry for KYMA."""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

import numpy as np

from config import config

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency at runtime
    from scipy import signal as scipy_signal

    _SCIPY_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - optional dependency at runtime
    scipy_signal = None  # type: ignore[assignment]
    _SCIPY_ERROR = exc


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


class FilterLab:
    SUPPORTED_METHODS = ("butter", "cheby1", "cheby2", "ellip")
    SUPPORTED_RESPONSES = ("lowpass", "highpass", "bandpass", "bandstop")
    SUPPORTED_APPLY_MODES = ("append", "replace_defaults")
    SUPPORTED_EXPORTS = (
        "kyma_host",
        "iir1_cpp",
        "arduino_filters",
        "bode_csv",
        "pole_zero_json",
        "fixed_point_header",
    )

    def __init__(self) -> None:
        self._dir = Path(config.data_dir) / "filters"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._registry_path = self._dir / "registry.json"
        self._state = self._load_state()

    @property
    def available(self) -> bool:
        return scipy_signal is not None

    def status(self, profile_key: str) -> Dict[str, Any]:
        active_id = str(self._state.get("active_by_profile", {}).get(profile_key) or "")
        active = self._find_filter(active_id)
        visible_filters = [
            item for item in self._state.get("filters", [])
            if str(item.get("profile_key") or "") == str(profile_key)
        ]
        return {
            "available": self.available,
            "last_error": f"{type(_SCIPY_ERROR).__name__}: {_SCIPY_ERROR}" if _SCIPY_ERROR else "",
            "methods": list(self.SUPPORTED_METHODS),
            "responses": list(self.SUPPORTED_RESPONSES),
            "apply_modes": list(self.SUPPORTED_APPLY_MODES),
            "exports": list(self.SUPPORTED_EXPORTS),
            "active_filter_id": active_id,
            "active_filter": self._summary(active, active_id) if active else None,
            "filters": [self._summary(item, active_id) for item in visible_filters],
        }

    def get_filter(self, filter_id: str) -> Optional[Dict[str, Any]]:
        item = self._find_filter(filter_id)
        if not item:
            return None
        profile_key = str(item.get("profile_key") or config.signal_profile_name)
        return {
            **item,
            "response": item.get("response") or self._response_payload(np.asarray(item.get("sos") or [], dtype=np.float64), float(item.get("sample_rate") or config.sample_rate)),
            "exports": self._build_exports(item),
            "active": str(self._state.get("active_by_profile", {}).get(profile_key) or "") == str(filter_id),
        }

    def design(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        normalized = self._normalize_spec(spec)
        design = self._design_from_spec(normalized)
        return {
            "available": self.available,
            "spec": normalized,
            "summary": self._summary(design, ""),
            "response": design["response"],
            "sos": design["sos"],
            "exports": self._build_exports(design),
        }

    def save(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        normalized = self._normalize_spec(spec)
        if not normalized["name"]:
            raise ValueError("Filter name is required")
        design = self._design_from_spec(normalized)
        record = {
            **design,
            "id": f"flt_{uuid4().hex[:10]}",
            "created_at": _utc_now_iso(),
        }
        self._state.setdefault("filters", []).append(record)
        self._save_state()
        return self.get_filter(record["id"]) or record

    def activate(self, filter_id: str, profile_key: Optional[str] = None) -> Dict[str, Any]:
        item = self._find_filter(filter_id)
        if not item:
            raise FileNotFoundError(f"Filter not found: {filter_id}")
        key = profile_key or str(item.get("profile_key") or config.signal_profile_name)
        if str(item.get("profile_key") or "") != str(key):
            raise ValueError(
                f"Filter {filter_id} belongs to {item.get('profile_key')} and cannot be activated on {key}"
            )
        self._state.setdefault("active_by_profile", {})[key] = filter_id
        self._save_state()
        return self.status(key)

    def clear_active(self, profile_key: str) -> Dict[str, Any]:
        self._state.setdefault("active_by_profile", {}).pop(profile_key, None)
        self._save_state()
        return self.status(profile_key)

    def delete(self, filter_id: str) -> None:
        filters = list(self._state.get("filters", []))
        next_filters = [item for item in filters if str(item.get("id")) != str(filter_id)]
        if len(next_filters) == len(filters):
            raise FileNotFoundError(f"Filter not found: {filter_id}")
        self._state["filters"] = next_filters
        active = self._state.setdefault("active_by_profile", {})
        for key, value in list(active.items()):
            if str(value) == str(filter_id):
                active.pop(key, None)
        self._save_state()

    def get_active_runtime_filter(self, profile_key: str) -> Optional[Dict[str, Any]]:
        active_id = str(self._state.get("active_by_profile", {}).get(profile_key) or "")
        item = self._find_filter(active_id)
        if not item:
            return None
        return {
            "id": item.get("id"),
            "name": item.get("name"),
            "profile_key": item.get("profile_key"),
            "apply_mode": item.get("apply_mode", "append"),
            "sample_rate": float(item.get("sample_rate") or config.sample_rate),
            "sos": np.asarray(item.get("sos") or [], dtype=np.float64),
            "summary": self._summary(item, active_id),
        }

    def _load_state(self) -> Dict[str, Any]:
        if not self._registry_path.exists():
            return {"filters": [], "active_by_profile": {}}
        try:
            with open(self._registry_path, encoding="utf-8") as fh:
                raw = json.load(fh)
            raw.setdefault("filters", [])
            raw.setdefault("active_by_profile", {})
            return raw
        except Exception as exc:
            logger.warning("Failed to load filter registry: %s", exc)
            return {"filters": [], "active_by_profile": {}}

    def _save_state(self) -> None:
        with open(self._registry_path, "w", encoding="utf-8") as fh:
            json.dump(self._state, fh, indent=2)

    def _find_filter(self, filter_id: str) -> Optional[Dict[str, Any]]:
        if not filter_id:
            return None
        for item in self._state.get("filters", []):
            if str(item.get("id")) == str(filter_id):
                return item
        return None

    def _normalize_spec(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        if scipy_signal is None:
            raise RuntimeError(
                f"SciPy is not available for filter design: {type(_SCIPY_ERROR).__name__}: {_SCIPY_ERROR}"
            )

        profile_key = str(spec.get("profile") or config.signal_profile_name).strip().lower() or config.signal_profile_name
        method = str(spec.get("method") or "butter").strip().lower()
        response_type = str(spec.get("response_type") or "bandpass").strip().lower()
        apply_mode = str(spec.get("apply_mode") or "append").strip().lower()
        order = int(spec.get("order") or 2)
        sample_rate = float(spec.get("sample_rate") or config.sample_rate)
        name = str(spec.get("name") or "").strip()

        if method not in self.SUPPORTED_METHODS:
            raise ValueError(f"Unsupported filter method: {method}")
        if response_type not in self.SUPPORTED_RESPONSES:
            raise ValueError(f"Unsupported filter response type: {response_type}")
        if apply_mode not in self.SUPPORTED_APPLY_MODES:
            raise ValueError(f"Unsupported apply mode: {apply_mode}")
        if order < 1 or order > 10:
            raise ValueError("Filter order must be between 1 and 10")
        if sample_rate <= 1:
            raise ValueError("Sample rate must be greater than 1 Hz")

        nyquist = sample_rate / 2.0
        cutoff_hz = float(spec.get("cutoff_hz") or 0.0) or None
        low_hz = float(spec.get("low_hz") or 0.0) or None
        high_hz = float(spec.get("high_hz") or 0.0) or None

        if response_type in {"lowpass", "highpass"}:
            if cutoff_hz is None:
                raise ValueError("cutoff_hz is required for lowpass/highpass filters")
            if not 0 < cutoff_hz < nyquist:
                raise ValueError(f"cutoff_hz must be between 0 and Nyquist ({nyquist:.2f} Hz)")
        else:
            if low_hz is None or high_hz is None:
                raise ValueError("low_hz and high_hz are required for bandpass/bandstop filters")
            if not 0 < low_hz < high_hz < nyquist:
                raise ValueError(f"Band edges must satisfy 0 < low_hz < high_hz < Nyquist ({nyquist:.2f} Hz)")

        rp_db = float(spec.get("rp_db") or 1.0)
        rs_db = float(spec.get("rs_db") or 40.0)
        if method in {"cheby1", "ellip"} and rp_db <= 0:
            raise ValueError("rp_db must be greater than 0 for cheby1/ellip filters")
        if method in {"cheby2", "ellip"} and rs_db <= 0:
            raise ValueError("rs_db must be greater than 0 for cheby2/ellip filters")

        return {
            "name": name,
            "profile_key": profile_key,
            "method": method,
            "response_type": response_type,
            "apply_mode": apply_mode,
            "order": order,
            "sample_rate": round(sample_rate, 6),
            "cutoff_hz": round(cutoff_hz, 6) if cutoff_hz is not None else None,
            "low_hz": round(low_hz, 6) if low_hz is not None else None,
            "high_hz": round(high_hz, 6) if high_hz is not None else None,
            "rp_db": round(rp_db, 6),
            "rs_db": round(rs_db, 6),
        }

    def _design_from_spec(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        assert scipy_signal is not None

        response_type = str(spec["response_type"])
        cutoff = (
            float(spec["cutoff_hz"])
            if response_type in {"lowpass", "highpass"}
            else (float(spec["low_hz"]), float(spec["high_hz"]))
        )
        kwargs: Dict[str, Any] = {
            "N": int(spec["order"]),
            "Wn": cutoff,
            "btype": response_type,
            "fs": float(spec["sample_rate"]),
            "output": "sos",
        }
        method = str(spec["method"])
        if method == "butter":
            sos = scipy_signal.butter(**kwargs)
        elif method == "cheby1":
            sos = scipy_signal.cheby1(rp=float(spec["rp_db"]), **kwargs)
        elif method == "cheby2":
            sos = scipy_signal.cheby2(rs=float(spec["rs_db"]), **kwargs)
        else:
            sos = scipy_signal.ellip(rp=float(spec["rp_db"]), rs=float(spec["rs_db"]), **kwargs)

        response = self._response_payload(sos, float(spec["sample_rate"]))
        return {
            **spec,
            "sos": [[round(float(v), 12) for v in row] for row in np.asarray(sos, dtype=np.float64).tolist()],
            "response": response,
        }

    def _response_payload(self, sos: np.ndarray, sample_rate: float) -> Dict[str, Any]:
        assert scipy_signal is not None
        sos = np.asarray(sos, dtype=np.float64)
        if sos.size == 0:
            return {
                "freq_hz": [],
                "mag_db": [],
                "phase_rad": [],
                "phase_deg": [],
                "stable": False,
                "sections": 0,
                "peak_gain_db": 0.0,
                "min_gain_db": 0.0,
                "gain": 0.0,
                "zeros": [],
                "poles": [],
                "quantization": self._quantization_payload(sos),
            }

        freq, resp = scipy_signal.freqz_sos(sos, worN=512, fs=sample_rate)
        mag_db = 20.0 * np.log10(np.maximum(np.abs(resp), 1e-9))
        phase = np.unwrap(np.angle(resp))
        phase_deg = np.rad2deg(phase)
        zeros, poles, gain = scipy_signal.sos2zpk(sos)
        stable = bool(np.all(np.abs(poles) < 1.0))
        return {
            "freq_hz": [round(float(v), 6) for v in freq.tolist()],
            "mag_db": [round(float(v), 6) for v in mag_db.tolist()],
            "phase_rad": [round(float(v), 6) for v in phase.tolist()],
            "phase_deg": [round(float(v), 6) for v in phase_deg.tolist()],
            "stable": stable,
            "sections": int(sos.shape[0]),
            "peak_gain_db": round(float(np.max(mag_db)), 6),
            "min_gain_db": round(float(np.min(mag_db)), 6),
            "gain": round(float(gain), 12),
            "zeros": self._complex_payload(zeros),
            "poles": self._complex_payload(poles),
            "quantization": self._quantization_payload(sos),
        }

    def _complex_payload(self, values: np.ndarray) -> List[Dict[str, float]]:
        arr = np.asarray(values, dtype=np.complex128).ravel()
        payload: List[Dict[str, float]] = []
        for value in arr:
            payload.append(
                {
                    "re": round(float(np.real(value)), 9),
                    "im": round(float(np.imag(value)), 9),
                    "radius": round(float(abs(value)), 9),
                }
            )
        return payload

    def _recommended_frac_bits(self, max_abs: float, total_bits: int) -> int:
        if max_abs <= 0:
            return total_bits - 1
        integer_bits = max(0, int(math.ceil(math.log2(max_abs + 1e-12))))
        return max(0, total_bits - 1 - integer_bits)

    def _quantize_sos(
        self,
        sos: np.ndarray,
        total_bits: int,
        frac_bits: int,
        label: str,
    ) -> Dict[str, Any]:
        assert scipy_signal is not None

        sos = np.asarray(sos, dtype=np.float64)
        scale = float(2 ** max(frac_bits, 0))
        qmin = -(2 ** (total_bits - 1))
        qmax = (2 ** (total_bits - 1)) - 1
        raw = np.round(sos * scale)
        overflow_mask = np.logical_or(raw < qmin, raw > qmax)
        raw_clipped = np.clip(raw, qmin, qmax)
        quantized = raw_clipped / scale
        error = quantized - sos

        stable = False
        try:
            _, poles, _ = scipy_signal.sos2zpk(quantized)
            stable = bool(np.all(np.abs(poles) < 1.0))
        except Exception:
            stable = False

        return {
            "label": label,
            "total_bits": int(total_bits),
            "integer_bits": int(max(0, total_bits - 1 - frac_bits)),
            "frac_bits": int(max(frac_bits, 0)),
            "lsb": round(float(1.0 / scale), 12),
            "fits": bool(not np.any(overflow_mask)),
            "overflow_coefficients": int(np.count_nonzero(overflow_mask)),
            "max_abs_error": round(float(np.max(np.abs(error))) if error.size else 0.0, 12),
            "rms_error": round(float(np.sqrt(np.mean(np.square(error)))) if error.size else 0.0, 12),
            "stable": stable,
            "int_rows": raw_clipped.astype(int).tolist(),
            "float_rows": [[round(float(v), 12) for v in row] for row in quantized.tolist()],
        }

    def _quantization_payload(self, sos: np.ndarray) -> Dict[str, Any]:
        sos = np.asarray(sos, dtype=np.float64)
        sections = int(sos.shape[0]) if sos.ndim == 2 else 0
        coefficient_count = int(sos.size)
        state_count = int(sections * 2)
        max_abs_coeff = float(np.max(np.abs(sos))) if sos.size else 0.0

        q15_direct = self._quantize_sos(sos, total_bits=16, frac_bits=15, label="Q1.15 direct")
        q31_direct = self._quantize_sos(sos, total_bits=32, frac_bits=31, label="Q1.31 direct")
        s16_frac = self._recommended_frac_bits(max_abs_coeff, total_bits=16)
        s32_frac = self._recommended_frac_bits(max_abs_coeff, total_bits=32)
        s16 = self._quantize_sos(
            sos,
            total_bits=16,
            frac_bits=s16_frac,
            label=f"s16 ({16 - 1 - s16_frac} int / {s16_frac} frac)",
        )
        s32 = self._quantize_sos(
            sos,
            total_bits=32,
            frac_bits=s32_frac,
            label=f"s32 ({32 - 1 - s32_frac} int / {s32_frac} frac)",
        )

        return {
            "coefficient_count": coefficient_count,
            "state_count": state_count,
            "max_abs_coeff": round(max_abs_coeff, 12),
            "memory_bytes": {
                "float32": int((coefficient_count + state_count) * 4),
                "float64": int((coefficient_count + state_count) * 8),
                "int16": int((coefficient_count + state_count) * 2),
                "int32": int((coefficient_count + state_count) * 4),
            },
            "direct_q15": q15_direct,
            "direct_q31": q31_direct,
            "recommended_s16": s16,
            "recommended_s32": s32,
        }

    def _summary(self, item: Optional[Dict[str, Any]], active_id: str) -> Dict[str, Any]:
        if not item:
            return {}
        response = item.get("response") or {}
        return {
            "id": item.get("id", ""),
            "name": item.get("name", ""),
            "profile_key": item.get("profile_key", ""),
            "method": item.get("method", ""),
            "response_type": item.get("response_type", ""),
            "apply_mode": item.get("apply_mode", "append"),
            "order": int(item.get("order") or 0),
            "sample_rate": float(item.get("sample_rate") or config.sample_rate),
            "cutoff_hz": item.get("cutoff_hz"),
            "low_hz": item.get("low_hz"),
            "high_hz": item.get("high_hz"),
            "rp_db": item.get("rp_db"),
            "rs_db": item.get("rs_db"),
            "created_at": item.get("created_at", ""),
            "sections": int(response.get("sections") or len(item.get("sos") or [])),
            "stable": bool(response.get("stable", False)),
            "active": str(item.get("id", "")) == str(active_id),
        }

    def _build_exports(self, item: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        exports: Dict[str, Dict[str, Any]] = {}
        exports["kyma_host"] = {
            "label": "KYMA Host (SciPy SOS)",
            "code": self._export_kyma_host(item),
            "filename": f"{self._safe_name(item)}_scipy.py",
            "available": True,
        }
        exports["iir1_cpp"] = {
            "label": "iir1 C++",
            "code": self._export_iir1(item),
            "filename": f"{self._safe_name(item)}_iir1.hpp",
            "available": True,
        }
        arduino_available = str(item.get("method")) == "butter" and str(item.get("response_type")) in {"lowpass", "highpass"}
        exports["arduino_filters"] = {
            "label": "Arduino-Filters",
            "code": self._export_arduino_filters(item) if arduino_available else self._unsupported_arduino_filters(item),
            "filename": f"{self._safe_name(item)}_arduino_filters.ino",
            "available": arduino_available,
        }
        exports["bode_csv"] = {
            "label": "Bode CSV",
            "code": self._export_bode_csv(item),
            "filename": f"{self._safe_name(item)}_bode.csv",
            "available": True,
        }
        exports["pole_zero_json"] = {
            "label": "Pole / Zero JSON",
            "code": self._export_pole_zero_json(item),
            "filename": f"{self._safe_name(item)}_pole_zero.json",
            "available": True,
        }
        exports["fixed_point_header"] = {
            "label": "Fixed-Point C Header",
            "code": self._export_fixed_point_header(item),
            "filename": f"{self._safe_name(item)}_fixed_point.h",
            "available": True,
        }
        return exports

    def _export_kyma_host(self, item: Dict[str, Any]) -> str:
        lines = [
            "# SciPy SOS export from KYMA Filter Lab",
            "import numpy as np",
            "from scipy import signal",
            "",
            f"sample_rate = {float(item.get('sample_rate') or config.sample_rate)}",
            "sos = np.array([",
        ]
        for row in item.get("sos") or []:
            lines.append("    [" + ", ".join(f"{float(v):.12g}" for v in row) + "],")
        lines += [
            "], dtype=float)",
            "",
            "def apply_filter(x):",
            "    return signal.sosfilt(sos, x)",
            "",
        ]
        return "\n".join(lines)

    def _export_iir1(self, item: Dict[str, Any]) -> str:
        sections = item.get("sos") or []
        n_sections = len(sections)
        lines = [
            "// iir1 export from KYMA Filter Lab",
            "// https://berndporr.github.io/iir1/",
            "#include <Iir.h>",
            "",
            f"constexpr double kSampleRate = {float(item.get('sample_rate') or config.sample_rate):.6f};",
            f"constexpr size_t kSections = {n_sections};",
            "const double kSOS[kSections][6] = {",
        ]
        for row in sections:
            lines.append("    {" + ", ".join(f"{float(v):.12g}" for v in row) + "},")
        lines += [
            "};",
            "",
            f"Iir::Custom::SOSCascade<kSections> filter(kSOS);",
            "",
            "double applyFilter(double x) {",
            "    return filter.filter(x);",
            "}",
            "",
        ]
        return "\n".join(lines)

    def _export_arduino_filters(self, item: Dict[str, Any]) -> str:
        sample_rate = float(item.get("sample_rate") or config.sample_rate)
        cutoff = float(item.get("cutoff_hz") or 0.0)
        normalized = 2.0 * cutoff / max(sample_rate, 1e-9)
        response_type = str(item.get("response_type"))
        order = int(item.get("order") or 2)
        header = "ButterworthLowPass" if response_type == "lowpass" else "ButterworthHighPass"
        factory = "butter" if response_type == "lowpass" else "butter"
        return "\n".join(
            [
                "// Arduino-Filters export from KYMA Filter Lab",
                "// https://github.com/tttapa/Arduino-Filters",
                "#include <Filters.h>",
                "#include <AH/Timing/MillisMicrosTimer.hpp>",
                "#include <Filters/Butterworth.hpp>",
                "",
                f"constexpr double kSampleRate = {sample_rate:.6f};",
                f"constexpr double kCutoffHz = {cutoff:.6f};",
                f"constexpr double kNormalizedCutoff = {normalized:.12g};",
                f"Timer<micros> timer = std::round(1e6 / kSampleRate);",
                f"auto filter = {factory}<{order}>(kNormalizedCutoff);",
                "",
                "void setup() {",
                "  Serial.begin(115200);",
                "}",
                "",
                "void loop() {",
                "  if (timer) {",
                "    const double x = analogRead(A0);",
                "    const double y = filter(x);",
                "    Serial.println(y, 6);",
                "  }",
                "}",
                "",
            ]
        )

    def _export_bode_csv(self, item: Dict[str, Any]) -> str:
        response = item.get("response") or {}
        freq = response.get("freq_hz") or []
        mag = response.get("mag_db") or []
        phase = response.get("phase_deg") or []
        lines = ["freq_hz,mag_db,phase_deg"]
        for f_hz, mag_db, phase_deg in zip(freq, mag, phase):
            lines.append(f"{float(f_hz):.6f},{float(mag_db):.6f},{float(phase_deg):.6f}")
        return "\n".join(lines) + "\n"

    def _export_pole_zero_json(self, item: Dict[str, Any]) -> str:
        response = item.get("response") or {}
        payload = {
            "name": item.get("name"),
            "profile_key": item.get("profile_key"),
            "method": item.get("method"),
            "response_type": item.get("response_type"),
            "sample_rate": float(item.get("sample_rate") or config.sample_rate),
            "stable": bool(response.get("stable", False)),
            "gain": response.get("gain", 0.0),
            "zeros": response.get("zeros") or [],
            "poles": response.get("poles") or [],
        }
        return json.dumps(payload, indent=2)

    def _export_fixed_point_header(self, item: Dict[str, Any]) -> str:
        response = item.get("response") or {}
        quant = response.get("quantization") or self._quantization_payload(
            np.asarray(item.get("sos") or [], dtype=np.float64)
        )
        s16 = quant.get("recommended_s16") or {}
        s32 = quant.get("recommended_s32") or {}
        sections = item.get("sos") or []

        def _rows(rows: Any) -> List[List[int]]:
            if not isinstance(rows, list):
                return []
            clean: List[List[int]] = []
            for row in rows:
                if isinstance(row, list):
                    clean.append([int(v) for v in row])
            return clean

        s16_rows = _rows(s16.get("int_rows"))
        s32_rows = _rows(s32.get("int_rows"))
        lines = [
            "// Fixed-point coefficient export from KYMA Filter Lab",
            "#pragma once",
            "#include <stdint.h>",
            "",
            f"#define KYMA_FILTER_SECTIONS {len(sections)}",
            f"#define KYMA_FILTER_SAMPLE_RATE_HZ {float(item.get('sample_rate') or config.sample_rate):.6f}",
            f"#define KYMA_FILTER_S16_FRAC_BITS {int(s16.get('frac_bits') or 0)}",
            f"#define KYMA_FILTER_S32_FRAC_BITS {int(s32.get('frac_bits') or 0)}",
            "",
            "// Recommended 16-bit coefficient table",
            "static const int16_t KYMA_FILTER_S16_SOS[KYMA_FILTER_SECTIONS][6] = {",
        ]
        for row in s16_rows:
            lines.append("    {" + ", ".join(str(v) for v in row) + "},")
        lines += [
            "};",
            "",
            "// Recommended 32-bit coefficient table",
            "static const int32_t KYMA_FILTER_S32_SOS[KYMA_FILTER_SECTIONS][6] = {",
        ]
        for row in s32_rows:
            lines.append("    {" + ", ".join(str(v) for v in row) + "},")
        lines += [
            "};",
            "",
            f"// Max abs coeff: {float(quant.get('max_abs_coeff') or 0.0):.12g}",
            f"// Direct Q1.15 fits: {bool((quant.get('direct_q15') or {}).get('fits', False))}",
            f"// Direct Q1.31 fits: {bool((quant.get('direct_q31') or {}).get('fits', False))}",
            "",
        ]
        return "\n".join(lines)

    def _unsupported_arduino_filters(self, item: Dict[str, Any]) -> str:
        return "\n".join(
            [
                "// Arduino-Filters export is currently limited to Butterworth low/high-pass designs",
                "// for this integrated KYMA workflow.",
                "//",
                "// This design is still usable inside KYMA on the host.",
                "// For C++ coefficient import, use the iir1 export target instead.",
                "",
            ]
        )

    def _safe_name(self, item: Dict[str, Any]) -> str:
        raw = str(item.get("name") or item.get("id") or "filter")
        clean = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in raw)
        return clean.strip("_-") or "filter"
