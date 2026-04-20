"""Trainable temperature decoding pipeline for thermal trend classification."""

from __future__ import annotations

import collections
import logging
import os
import pickle
import threading
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from classifiers import LDAClassifier, get_classifier
from config import config

logger = logging.getLogger(__name__)


class TemperaturePipeline:
    """Buffered temperature decoder for stable, warming, and cooling trends."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._classifier = None
        self._classifier_name = "LDA"
        self._is_trained = False

        self._train_windows: List[np.ndarray] = []
        self._train_labels: List[int] = []
        self._recording = False
        self._current_label: Optional[int] = None

        self._pred_callbacks: List[Callable] = []
        self._vote_buffer: collections.deque[int] = collections.deque(maxlen=5)

        self._segment_seconds = 10.0
        self._segment_samples = max(int(config.sample_rate * self._segment_seconds), config.window_size_samples)
        self._history_size = max(int(config.sample_rate * 20.0), self._segment_samples)
        self._buf = np.zeros((config.n_channels, self._history_size), dtype=np.float64)
        self._buf_head = 0
        self._total_samples = 0
        self._initialized = False

        self._record_stride_samples = max(int(config.sample_rate * 2.0), config.window_increment_samples)
        self._predict_stride_samples = max(int(config.sample_rate * 1.0), config.window_increment_samples)
        self._last_record_total = 0
        self._last_predict_total = 0

        Path(config.model_dir).mkdir(exist_ok=True)

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def classifier_name(self) -> str:
        return self._classifier_name

    def add_prediction_callback(self, fn: Callable) -> None:
        self._pred_callbacks.append(fn)

    def on_window(self, window: np.ndarray) -> None:
        if not self._initialized:
            self._append(window)
            self._initialized = True
        else:
            inc = min(config.window_increment_samples, window.shape[1])
            self._append(window[:, -inc:])

        if self._total_samples < self._segment_samples:
            return

        recent = self._get_recent_segment()
        if recent is None:
            return

        if self._recording and self._current_label is not None:
            if self._total_samples - self._last_record_total >= self._record_stride_samples:
                self._train_windows.append(recent.copy())
                self._train_labels.append(self._current_label)
                self._last_record_total = self._total_samples
            return

        if not self._is_trained:
            return
        if self._total_samples - self._last_predict_total < self._predict_stride_samples:
            return
        self._last_predict_total = self._total_samples

        class_idx, confidence = self.predict(recent)
        if confidence < 0.45 and "stable temperature" in config.class_labels:
            class_idx = config.class_labels.index("stable temperature")
            confidence = max(confidence, 0.35)
            self._vote_buffer.clear()
        else:
            self._vote_buffer.append(class_idx)
            if len(self._vote_buffer) >= 3:
                voted, votes = collections.Counter(self._vote_buffer).most_common(1)[0]
                if votes >= len(self._vote_buffer) // 2 + 1:
                    class_idx = voted

        for fn in self._pred_callbacks:
            try:
                fn(class_idx, confidence, recent)
            except Exception as exc:
                logger.error("Prediction callback error: %s", exc)

    def _append(self, samples: np.ndarray) -> None:
        n_new = samples.shape[1]
        for idx in range(n_new):
            self._buf[:, self._buf_head % self._history_size] = samples[:, idx]
            self._buf_head += 1
        self._total_samples += n_new

    def _get_recent_segment(self) -> Optional[np.ndarray]:
        if self._total_samples < self._segment_samples:
            return None
        end = self._buf_head
        idx = np.arange(end - self._segment_samples, end) % self._history_size
        return self._buf[:, idx].astype(np.float32, copy=True)

    def _primary_signal(self, segment: np.ndarray) -> np.ndarray:
        return np.mean(segment.astype(np.float64, copy=False), axis=0)

    def _smooth(self, signal: np.ndarray, window_s: float) -> np.ndarray:
        width = max(1, int(config.sample_rate * window_s))
        kernel = np.ones(width, dtype=np.float64) / width
        return np.convolve(signal, kernel, mode="same")

    def _trend_features(self, signal: np.ndarray) -> Tuple[float, float, float]:
        x = np.linspace(0.0, 1.0, signal.size, dtype=np.float64)
        coeffs = np.polyfit(x, signal.astype(np.float64, copy=False), deg=2)
        quad = float(coeffs[0])
        slope = float(coeffs[1])
        fit = np.polyval(coeffs, x)
        residual = float(np.std(signal - fit))
        return slope, quad, residual

    def extract_features(self, segment: np.ndarray) -> np.ndarray:
        arr = segment.astype(np.float64, copy=False)
        signal = self._primary_signal(segment)
        smooth = self._smooth(signal, window_s=0.8)

        mean_level = float(np.mean(signal))
        std = float(np.std(signal))
        span = float(np.max(signal) - np.min(signal))
        rms = float(np.sqrt(np.mean(signal ** 2)))
        edge_slope = float(signal[-1] - signal[0])
        smooth_edge_slope = float(smooth[-1] - smooth[0])
        slope, quad, residual = self._trend_features(smooth)

        quarter = max(1, signal.size // 4)
        early_mean = float(np.mean(signal[:quarter]))
        late_mean = float(np.mean(signal[-quarter:]))
        delta = late_mean - early_mean

        mid = signal.size // 2
        half_delta = float(np.mean(signal[mid:]) - np.mean(signal[:mid])) if mid > 0 else 0.0

        diff = np.diff(signal)
        diff_std = float(np.std(diff)) if diff.size else 0.0
        diff_mean = float(np.mean(diff)) if diff.size else 0.0
        pos_frac = float(np.mean(diff > 0)) if diff.size else 0.0
        pos_mean = float(np.mean(diff[diff > 0])) if np.any(diff > 0) else 0.0
        neg_mean = float(np.mean(np.abs(diff[diff < 0]))) if np.any(diff < 0) else 0.0

        lead_mean = np.mean(arr, axis=1)
        lead_std = np.std(arr, axis=1)
        lead_span = np.ptp(arr, axis=1)
        lead_slopes = np.asarray([self._trend_features(self._smooth(ch, 0.8))[0] for ch in arr], dtype=np.float64)

        corr_values: List[float] = []
        for i in range(arr.shape[0]):
            for j in range(i + 1, arr.shape[0]):
                std_i = float(np.std(arr[i]))
                std_j = float(np.std(arr[j]))
                if std_i > 1e-9 and std_j > 1e-9:
                    corr_values.append(float(np.corrcoef(arr[i], arr[j])[0, 1]))
        mean_corr = float(np.mean(corr_values)) if corr_values else 0.0

        feats: List[float] = [
            mean_level,
            std,
            span,
            rms,
            edge_slope,
            smooth_edge_slope,
            slope,
            quad,
            residual,
            early_mean,
            late_mean,
            delta,
            half_delta,
            diff_std,
            diff_mean,
            pos_frac,
            pos_mean,
            neg_mean,
            mean_corr,
        ]
        feats.extend(lead_mean.tolist())
        feats.extend(lead_std.tolist())
        feats.extend(lead_span.tolist())
        feats.extend(lead_slopes.tolist())
        return np.asarray(feats, dtype=np.float32)

    def extract_features_batch(self, segments: List[np.ndarray]) -> np.ndarray:
        return np.stack([self.extract_features(segment) for segment in segments], axis=0)

    def start_recording(self, label_name: str) -> None:
        if label_name not in config.class_labels:
            raise ValueError(f"Unknown label '{label_name}'")
        self._current_label = config.class_labels.index(label_name)
        self._recording = True
        self._last_record_total = 0
        logger.info("Recording Temperature '%s' (label=%s)", label_name, self._current_label)

    def stop_recording(self) -> None:
        self._recording = False
        self._current_label = None
        logger.info("Temperature recording stopped. Buffer: %s windows", len(self._train_windows))

    def clear_training_data(self) -> None:
        self._train_windows = []
        self._train_labels = []
        self._classifier = None
        self._is_trained = False
        self._vote_buffer.clear()
        logger.info("Temperature training data cleared")

    def _augment_windows(
        self,
        windows: List[np.ndarray],
        labels: List[int],
        factor: int = 3,
    ) -> Tuple[List[np.ndarray], List[int]]:
        rng = np.random.default_rng(42)
        aug_windows, aug_labels = list(windows), list(labels)

        for _ in range(factor - 1):
            for window, label in zip(windows, labels):
                choice = int(rng.integers(0, 3))
                if choice == 0:
                    noise = rng.normal(0, max(float(window.std()) * 0.03, 1e-6), window.shape).astype(np.float32)
                    aug_windows.append(window + noise)
                elif choice == 1:
                    scale = float(rng.uniform(0.95, 1.05))
                    aug_windows.append((window * scale).astype(np.float32))
                else:
                    offset = float(rng.uniform(-0.01, 0.01))
                    aug_windows.append((window + offset).astype(np.float32))
                aug_labels.append(label)

        return aug_windows, aug_labels

    def train(self, classifier_name: str = "LDA") -> Dict:
        n = len(self._train_windows)
        if n < 8:
            return {"success": False, "error": f"Need >= 8 windows, have {n}"}

        self._classifier_name = classifier_name
        y = np.asarray(self._train_labels)

        if classifier_name == "LDA":
            X = self.extract_features_batch(self._train_windows)
            clf = LDAClassifier()
            result = clf.fit(X, y)
        elif classifier_name in ("TCN", "Mamba"):
            aug_wins, aug_lbls = self._augment_windows(self._train_windows, self._train_labels, factor=3)
            X = np.stack(aug_wins, axis=0)
            y = np.asarray(aug_lbls)
            clf = get_classifier(
                classifier_name,
                n_channels=config.n_channels,
                n_classes=len(config.class_labels),
            )
            result = clf.fit(X, y)
        else:
            return {"success": False, "error": f"Unknown classifier: {classifier_name}"}

        with self._lock:
            self._classifier = clf
            self._is_trained = True

        result["success"] = True
        result["classifier"] = classifier_name
        result["n_windows"] = n
        result["class_counts"] = {
            config.class_labels[int(idx)]: int((y == idx).sum()) for idx in np.unique(y)
        }
        return result

    def predict(self, segment: np.ndarray) -> Tuple[int, float]:
        if not self._is_trained or self._classifier is None:
            return 0, 0.0

        with self._lock:
            if self._classifier_name == "LDA":
                feats = self.extract_features(segment).reshape(1, -1)
                return self._classifier.predict(feats)

            raw = segment[np.newaxis, :, :]
            return self._classifier.predict(raw)

    def _meta_path_candidates(self, explicit_path: Optional[str] = None) -> List[str]:
        if explicit_path:
            return [explicit_path]
        profile = config.signal_profile_name
        return [
            os.path.join(config.model_dir, f"active_{profile}_classifier.pkl"),
            os.path.join(config.model_dir, "active_classifier.pkl"),
        ]

    def save_model(self, path: str = None) -> str:
        ext = ".pkl" if self._classifier_name == "LDA" else ".pt"
        profile = config.signal_profile_name
        path = path or os.path.join(config.model_dir, f"{profile}_{self._classifier_name.lower()}_model{ext}")
        meta_path = os.path.join(config.model_dir, f"active_{profile}_classifier.pkl")

        self._classifier.save(path)
        with open(meta_path, "wb") as fh:
            pickle.dump(
                {
                    "profile": profile,
                    "classifier": self._classifier_name,
                    "path": path,
                    "class_labels": list(config.class_labels),
                },
                fh,
            )
        logger.info("Temperature model saved -> %s", path)
        return path

    def load_model(self, path: str = None) -> bool:
        for meta_path in self._meta_path_candidates(path):
            if not os.path.exists(meta_path):
                continue
            try:
                with open(meta_path, "rb") as fh:
                    meta = pickle.load(fh)

                profile = meta.get("profile")
                if profile and profile != config.signal_profile_name:
                    continue

                name = meta["classifier"]
                model_path = meta["path"]
                if not os.path.exists(model_path):
                    continue

                kwargs = {}
                if name in ("TCN", "Mamba"):
                    kwargs = {"n_channels": config.n_channels, "n_classes": len(config.class_labels)}
                clf = get_classifier(name, **kwargs)
                clf.load(model_path)

                with self._lock:
                    self._classifier = clf
                    self._classifier_name = name
                    self._is_trained = True
                logger.info("Loaded %s model from %s", name, model_path)
                return True
            except Exception as exc:
                logger.error("Temperature load_model failed: %s", exc)
        return False

    def get_training_summary(self) -> Dict:
        counts: Dict[str, int] = {}
        for label in self._train_labels:
            name = config.class_labels[label]
            counts[name] = counts.get(name, 0) + 1
        return {
            "total_windows": len(self._train_windows),
            "per_gesture": counts,
            "is_trained": self._is_trained,
            "classifier": self._classifier_name,
        }

    def get_channel_quality(self, segment: np.ndarray) -> List[float]:
        rms = np.sqrt(np.mean(segment.astype(np.float64) ** 2, axis=1))
        scale = max(float(config.signal_profile.metric_full_scale), 1e-6)
        return np.clip(rms / scale, 0.0, 1.0).tolist()
