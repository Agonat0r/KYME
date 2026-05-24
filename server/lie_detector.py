"""Prototype control-question matcher for arousal-based interview sessions.

This is intentionally not a forensic lie detector. It compares target question
responses against the current person's baseline and known-answer controls.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional


def _safe_score(prediction: Optional[dict]) -> Optional[float]:
    if not prediction:
        return None
    arousal = prediction.get("arousal") or {}
    try:
        return float(arousal.get("score"))
    except (TypeError, ValueError):
        return None


def _mean(values: List[float]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def _stdev(values: List[float]) -> float:
    return float(statistics.pstdev(values)) if len(values) > 1 else 0.0


@dataclass
class LieDetectorSession:
    baseline: List[float] = field(default_factory=list)
    truth_controls: List[float] = field(default_factory=list)
    lie_controls: List[float] = field(default_factory=list)
    active_question: Optional[dict] = None
    last_result: Optional[dict] = None

    def reset(self) -> Dict:
        self.baseline.clear()
        self.truth_controls.clear()
        self.lie_controls.clear()
        self.active_question = None
        self.last_result = None
        return self.status()

    def status(self) -> Dict:
        return {
            "ready": len(self.baseline) >= 5 and len(self.truth_controls) >= 2 and len(self.lie_controls) >= 2,
            "baseline_count": len(self.baseline),
            "truth_count": len(self.truth_controls),
            "lie_count": len(self.lie_controls),
            "baseline_mean": round(_mean(self.baseline), 2),
            "truth_mean": round(_mean(self.truth_controls), 2),
            "lie_mean": round(_mean(self.lie_controls), 2),
            "active_question": self.active_question,
            "last_result": self.last_result,
            "disclaimer": "Prototype only: compares arousal against controls; it does not prove truth or deception.",
        }

    def add_sample(self, kind: str, prediction: Optional[dict]) -> Dict:
        score = _safe_score(prediction)
        if score is None:
            raise ValueError("No arousal score is available yet. Use ECG or EDA/GSR and wait for decoded output.")
        if kind == "baseline":
            self.baseline.append(score)
        elif kind == "truth":
            self.truth_controls.append(score)
        elif kind == "lie":
            self.lie_controls.append(score)
        else:
            raise ValueError(f"Unknown sample kind: {kind}")
        return {"sample": {"kind": kind, "score": round(score, 2)}, **self.status()}

    def start_question(self, question: str = "") -> Dict:
        self.active_question = {
            "question": str(question or "").strip(),
            "started_at": time.time(),
        }
        return self.status()

    def score_question(self, prediction: Optional[dict], answer: str = "") -> Dict:
        score = _safe_score(prediction)
        if score is None:
            raise ValueError("No arousal score is available yet. Use ECG or EDA/GSR and wait for decoded output.")

        result = classify_response(
            score=score,
            baseline=self.baseline,
            truth_controls=self.truth_controls,
            lie_controls=self.lie_controls,
        )
        result.update(
            {
                "question": (self.active_question or {}).get("question", ""),
                "answer": str(answer or "").strip(),
                "started_at": (self.active_question or {}).get("started_at"),
                "ended_at": time.time(),
            }
        )
        self.last_result = result
        self.active_question = None
        return {"result": result, **self.status()}


def classify_response(score: float, baseline: List[float], truth_controls: List[float], lie_controls: List[float]) -> Dict:
    baseline_mean = _mean(baseline)
    baseline_std = max(_stdev(baseline), 3.0)
    truth_mean = _mean(truth_controls)
    lie_mean = _mean(lie_controls)
    response_delta = float(score) - baseline_mean

    if len(baseline) < 5 or len(truth_controls) < 2 or len(lie_controls) < 2:
        return {
            "label": "Inconclusive",
            "score": round(float(score), 2),
            "confidence": 0.0,
            "reason": "Train baseline plus at least two known-truth and two known-false controls first.",
            "response_delta": round(response_delta, 2),
        }

    truth_distance = abs(float(score) - truth_mean)
    lie_distance = abs(float(score) - lie_mean)
    separation = abs(lie_mean - truth_mean)
    min_separation = max(8.0, baseline_std * 1.5)

    if separation < min_separation:
        return {
            "label": "Inconclusive",
            "score": round(float(score), 2),
            "confidence": 0.25,
            "reason": "Known-truth and known-false controls are not separated enough for this person/session.",
            "response_delta": round(response_delta, 2),
        }

    margin = abs(truth_distance - lie_distance)
    confidence = min(0.92, max(0.35, margin / max(separation, 1e-6)))
    if lie_distance + max(4.0, baseline_std * 0.5) < truth_distance and response_delta > baseline_std:
        label = "Lie-like"
        reason = "Target response is closer to known-false controls and above baseline."
    elif truth_distance + max(4.0, baseline_std * 0.5) < lie_distance:
        label = "Truth-like"
        reason = "Target response is closer to known-truth controls."
    else:
        label = "Inconclusive"
        confidence = min(confidence, 0.45)
        reason = "Target response sits between truth and false controls."

    return {
        "label": label,
        "score": round(float(score), 2),
        "confidence": round(confidence, 3),
        "reason": reason,
        "response_delta": round(response_delta, 2),
        "truth_distance": round(truth_distance, 2),
        "lie_distance": round(lie_distance, 2),
    }
