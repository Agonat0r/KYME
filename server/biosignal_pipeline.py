"""Profile-aware pipeline with EMG training and live analyzers for all profiles."""

from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np

from biosignal_analyzers import ProfileWindowAnalyzer
from config import config
from eda_pipeline import EDAPipeline
from ecg_pipeline import ECGPipeline
from emg_pipeline import EMGPipeline
from eeg_pipeline import EEGPipeline
from eog_pipeline import EOGPipeline
from ppg_pipeline import PPGPipeline
from resp_pipeline import RespirationPipeline
from temp_pipeline import TemperaturePipeline


class BiosignalPipeline:
    """Expose a stable pipeline API across all biosignal profiles."""

    def __init__(self):
        self._profile = config.signal_profile
        self._pred_callbacks: List[Callable] = []
        if self._profile.key == "emg":
            self._impl = EMGPipeline()
        elif self._profile.key == "eeg":
            self._impl = EEGPipeline()
        elif self._profile.key == "ecg":
            self._impl = ECGPipeline()
        elif self._profile.key == "eog":
            self._impl = EOGPipeline()
        elif self._profile.key == "eda":
            self._impl = EDAPipeline()
        elif self._profile.key == "ppg":
            self._impl = PPGPipeline()
        elif self._profile.key == "resp":
            self._impl = RespirationPipeline()
        elif self._profile.key == "temp":
            self._impl = TemperaturePipeline()
        else:
            self._impl = None
        self._analyzer = None if self._profile.key == "emg" else ProfileWindowAnalyzer(self._profile)

        if self._impl:
            self._impl.add_prediction_callback(self._forward_classifier_prediction)

    @property
    def profile(self):
        return self._profile

    @property
    def supports_training(self) -> bool:
        return self._profile.training_supported and self._impl is not None

    @property
    def is_trained(self) -> bool:
        return self._impl.is_trained if self._impl else False

    @property
    def is_recording(self) -> bool:
        return self._impl.is_recording if self._impl else False

    @property
    def classifier_name(self) -> str:
        if not self._impl:
            return "rule-based"
        if self._analyzer is not None and not (self._impl.is_trained or self._impl.is_recording):
            return "rule-based"
        return self._impl.classifier_name

    def add_prediction_callback(self, fn: Callable) -> None:
        self._pred_callbacks.append(fn)

    def _emit_prediction(self, result: Dict, window: np.ndarray) -> None:
        for fn in self._pred_callbacks:
            fn(result, window)

    def _forward_classifier_prediction(self, class_idx: int, confidence: float, window: np.ndarray) -> None:
        label = config.class_labels[class_idx] if class_idx < len(config.class_labels) else "rest"
        result = {
            "label": label,
            "class_idx": class_idx,
            "confidence": confidence,
            "metrics": {
                "window_rms": round(float(np.mean(np.sqrt(np.mean(window.astype(np.float64) ** 2, axis=1)))), 5),
            },
            "summary": f"Classifier output {label}",
            "profile": self._profile.key,
        }
        self._emit_prediction(result, window)

    def on_window(self, window: np.ndarray) -> None:
        if self._impl:
            if self._analyzer is not None and not (self._impl.is_trained or self._impl.is_recording):
                result = self._analyzer.on_window(window) if self._analyzer else None
                if result is not None:
                    self._emit_prediction(result, window)
                return
            self._impl.on_window(window)
            return

        result = self._analyzer.on_window(window)
        if result is not None:
            self._emit_prediction(result, window)

    def load_model(self, path: str = None) -> bool:
        if not self._impl:
            return False
        return self._impl.load_model(path)

    def save_model(self, path: str = None) -> str:
        if not self._impl:
            raise RuntimeError(f"{self._profile.display_name} does not have a trainable decoder yet")
        return self._impl.save_model(path)

    def start_recording(self, label: str) -> None:
        if not self.supports_training:
            raise RuntimeError(f"{self._profile.display_name} training is not implemented yet")
        self._impl.start_recording(label)

    def stop_recording(self) -> None:
        if self._impl:
            self._impl.stop_recording()

    def clear_training_data(self) -> None:
        if self._impl:
            self._impl.clear_training_data()

    def train(self, classifier_name: str = "LDA") -> Dict:
        if not self.supports_training:
            return {
                "success": False,
                "error": f"{self._profile.display_name} training is not implemented yet",
                "profile": self._profile.key,
            }
        return self._impl.train(classifier_name=classifier_name)

    def get_training_summary(self) -> Dict:
        if self._impl:
            summary = self._impl.get_training_summary()
        else:
            summary = {
                "total_windows": 0,
                "per_gesture": {},
                "is_trained": False,
                "classifier": "rule-based",
            }
        summary["profile"] = self._profile.key
        summary["supports_training"] = self.supports_training
        return summary

    def get_channel_quality(self, window: np.ndarray) -> List[float]:
        if self._impl and (self._profile.key == "emg" or self._impl.is_trained or self._impl.is_recording):
            return self._impl.get_channel_quality(window)

        rms = np.sqrt(np.mean(window.astype(np.float64) ** 2, axis=1))
        scale = max(self._profile.metric_full_scale, 1e-6)
        return np.clip(rms / scale, 0.0, 1.0).tolist()
