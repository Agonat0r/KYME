"""Trainable PPG decoding pipeline for pulse-state classification."""

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


class PPGPipeline:
    """Buffered PPG decoder that classifies pulse rate and signal quality."""

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

        self._segment_seconds = 5.0
        self._segment_samples = max(int(config.sample_rate * self._segment_seconds), config.window_size_samples)
        self._history_size = max(int(config.sample_rate * 8.0), self._segment_samples)
        self._buf = np.zeros((config.n_channels, self._history_size), dtype=np.float64)
        self._buf_head = 0
        self._total_samples = 0
        self._initialized = False

        self._record_stride_samples = max(int(config.sample_rate * 0.5), config.window_increment_samples)
        self._predict_stride_samples = max(int(config.sample_rate * 0.25), config.window_increment_samples)
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
        if confidence < 0.45 and "weak pulse" in config.class_labels:
            class_idx = config.class_labels.index("weak pulse")
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
        arr = segment.astype(np.float64, copy=False)
        signal = np.mean(arr, axis=0)
        return signal - float(np.mean(signal))

    def _find_peaks(self, signal: np.ndarray, min_interval_s: float, threshold: Optional[float] = None) -> np.ndarray:
        if signal.size < 3:
            return np.array([], dtype=int)
        x = signal.astype(np.float64, copy=False)
        candidates = np.where((x[1:-1] > x[:-2]) & (x[1:-1] >= x[2:]))[0] + 1
        if threshold is None:
            threshold = float(np.mean(x) + 0.25 * np.std(x))
        candidates = candidates[x[candidates] >= threshold]
        if candidates.size == 0:
            return candidates

        min_interval = max(1, int(min_interval_s * config.sample_rate))
        selected: List[int] = []
        last = -min_interval
        for idx in candidates:
            if idx - last >= min_interval:
                selected.append(int(idx))
                last = int(idx)
            elif x[idx] > x[selected[-1]]:
                selected[-1] = int(idx)
                last = int(idx)
        return np.asarray(selected, dtype=int)

    def _peak_widths(self, signal: np.ndarray, peaks: np.ndarray) -> np.ndarray:
        if peaks.size == 0:
            return np.array([], dtype=np.float64)

        widths: List[float] = []
        half_window = max(1, int(0.2 * config.sample_rate))
        n = signal.size
        for peak in peaks:
            amp = float(signal[peak])
            if amp <= 0.0:
                continue
            thresh = amp * 0.5
            left = int(peak)
            while left > max(0, peak - half_window) and signal[left] > thresh:
                left -= 1
            right = int(peak)
            while right < min(n - 1, peak + half_window) and signal[right] > thresh:
                right += 1
            widths.append((right - left) / config.sample_rate)
        return np.asarray(widths, dtype=np.float64)

    def extract_features(self, segment: np.ndarray) -> np.ndarray:
        signal = self._primary_signal(segment)
        span = float(np.max(signal) - np.min(signal))
        rms = float(np.sqrt(np.mean(signal ** 2)))
        std = float(np.std(signal))
        diff = np.diff(signal)
        derivative_rms = float(np.sqrt(np.mean(diff ** 2))) if diff.size else 0.0

        threshold = float(np.mean(signal) + 0.25 * np.std(signal))
        peaks = self._find_peaks(signal, min_interval_s=0.35, threshold=threshold)
        peak_count = int(peaks.size)
        peak_density = peak_count / self._segment_seconds

        if peak_count >= 2:
            rr = np.diff(peaks) / config.sample_rate
            rr_mean = float(np.mean(rr))
            rr_std = float(np.std(rr))
            rr_cv = rr_std / max(rr_mean, 1e-6)
            bpm = 60.0 / max(rr_mean, 1e-6)
        else:
            rr_mean = 0.0
            rr_std = 0.0
            rr_cv = 0.0
            bpm = 0.0

        peak_amps = signal[peaks] if peak_count else np.array([], dtype=np.float64)
        amp_mean = float(np.mean(peak_amps)) if peak_amps.size else 0.0
        amp_std = float(np.std(peak_amps)) if peak_amps.size else 0.0

        widths = self._peak_widths(signal, peaks)
        width_mean = float(np.mean(widths)) if widths.size else 0.0
        width_std = float(np.std(widths)) if widths.size else 0.0

        lead_rms = np.sqrt(np.mean(segment.astype(np.float64) ** 2, axis=1))
        lead_means = np.mean(segment.astype(np.float64), axis=1)
        lead_span = np.ptp(segment.astype(np.float64), axis=1)

        corr_values: List[float] = []
        arr = segment.astype(np.float64, copy=False)
        for i in range(arr.shape[0]):
            for j in range(i + 1, arr.shape[0]):
                std_i = float(np.std(arr[i]))
                std_j = float(np.std(arr[j]))
                if std_i > 1e-6 and std_j > 1e-6:
                    corr_values.append(float(np.corrcoef(arr[i], arr[j])[0, 1]))
        mean_corr = float(np.mean(corr_values)) if corr_values else 0.0

        perfusion = amp_mean / max(std, 1e-6)

        feats: List[float] = [
            bpm,
            peak_density,
            rr_mean,
            rr_std,
            rr_cv,
            span,
            rms,
            std,
            derivative_rms,
            amp_mean,
            amp_std,
            width_mean,
            width_std,
            perfusion,
            mean_corr,
            float(peak_count),
        ]
        feats.extend(lead_rms.tolist())
        feats.extend(lead_means.tolist())
        feats.extend(lead_span.tolist())
        return np.asarray(feats, dtype=np.float32)

    def extract_features_batch(self, segments: List[np.ndarray]) -> np.ndarray:
        return np.stack([self.extract_features(segment) for segment in segments], axis=0)

    def start_recording(self, label_name: str) -> None:
        if label_name not in config.class_labels:
            raise ValueError(f"Unknown label '{label_name}'")
        self._current_label = config.class_labels.index(label_name)
        self._recording = True
        self._last_record_total = 0
        logger.info("Recording PPG '%s' (label=%s)", label_name, self._current_label)

    def stop_recording(self) -> None:
        self._recording = False
        self._current_label = None
        logger.info("PPG recording stopped. Buffer: %s windows", len(self._train_windows))

    def clear_training_data(self) -> None:
        self._train_windows = []
        self._train_labels = []
        self._classifier = None
        self._is_trained = False
        self._vote_buffer.clear()
        logger.info("PPG training data cleared")

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
                    scale = float(rng.uniform(0.9, 1.1))
                    aug_windows.append((window * scale).astype(np.float32))
                else:
                    shift = int(rng.integers(-12, 13))
                    aug_windows.append(np.roll(window, shift=shift, axis=1).astype(np.float32))
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
        logger.info("PPG model saved -> %s", path)
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
                logger.error("PPG load_model failed: %s", exc)
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
