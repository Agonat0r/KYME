"""Trainable EEG decoding pipeline for band-state classification."""

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


class EEGPipeline:
    """Feature-based and deep-learning EEG decoder using the standard KYMA API."""

    BANDS: Tuple[Tuple[str, float, float], ...] = (
        ("delta", 1.0, 4.0),
        ("theta", 4.0, 8.0),
        ("alpha", 8.0, 12.0),
        ("beta", 12.0, 30.0),
        ("gamma", 30.0, 45.0),
    )

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
        if self._recording and self._current_label is not None:
            self._train_windows.append(window.copy())
            self._train_labels.append(self._current_label)
            return

        if not self._is_trained:
            return

        class_idx, confidence = self.predict(window)
        if confidence < 0.45 and "mixed activity" in config.class_labels:
            class_idx = config.class_labels.index("mixed activity")
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
                fn(class_idx, confidence, window)
            except Exception as exc:
                logger.error("Prediction callback error: %s", exc)

    def _bandpower_matrix(self, window: np.ndarray) -> np.ndarray:
        arr = window.astype(np.float64, copy=False)
        centered = arr - np.mean(arr, axis=1, keepdims=True)
        spec = np.fft.rfft(centered, axis=1)
        freqs = np.fft.rfftfreq(arr.shape[1], d=1.0 / config.sample_rate)
        power = np.abs(spec) ** 2

        total_mask = (freqs >= 1.0) & (freqs <= 45.0)
        total_power = power[:, total_mask].sum(axis=1, keepdims=True) + 1e-9

        parts = []
        for _, low_hz, high_hz in self.BANDS:
            mask = (freqs >= low_hz) & (freqs < high_hz)
            band = power[:, mask].sum(axis=1, keepdims=True) if np.any(mask) else np.zeros((arr.shape[0], 1))
            parts.append(band / total_power)
        return np.concatenate(parts, axis=1)

    def _dominant_frequency(self, signal: np.ndarray) -> float:
        x = signal.astype(np.float64, copy=False) - float(np.mean(signal))
        if x.size < 8:
            return 0.0
        spec = np.fft.rfft(x)
        freqs = np.fft.rfftfreq(x.size, d=1.0 / config.sample_rate)
        power = np.abs(spec) ** 2
        mask = (freqs >= 1.0) & (freqs <= 45.0)
        if not np.any(mask):
            return 0.0
        return float(freqs[mask][int(np.argmax(power[mask]))])

    def extract_features(self, window: np.ndarray) -> np.ndarray:
        bandpowers = self._bandpower_matrix(window)

        regions = []
        region_groups = ((0, 1), (2, 3), (4, 5), (6, 7))
        for lo, hi in region_groups:
            if hi < bandpowers.shape[0]:
                regions.append(np.mean(bandpowers[lo : hi + 1], axis=0))
        regional = np.stack(regions, axis=0) if regions else np.zeros((0, len(self.BANDS)))

        feats: List[float] = []
        feats.extend(bandpowers.flatten().tolist())
        feats.extend(regional.flatten().tolist())

        for lo, hi in region_groups:
            if hi < bandpowers.shape[0]:
                feats.append(float(bandpowers[lo, 2] - bandpowers[hi, 2]))
                feats.append(float(bandpowers[lo, 3] - bandpowers[hi, 3]))

        alpha_mean = float(np.mean(bandpowers[:, 2]))
        beta_mean = float(np.mean(bandpowers[:, 3]))
        theta_mean = float(np.mean(bandpowers[:, 1]))
        gamma_mean = float(np.mean(bandpowers[:, 4]))
        frontal_theta = float(np.mean(bandpowers[: min(2, bandpowers.shape[0]), 1]))
        posterior_alpha = float(np.mean(bandpowers[max(0, bandpowers.shape[0] - 2) :, 2]))
        beta_alpha_ratio = beta_mean / max(alpha_mean, 1e-6)
        theta_beta_ratio = theta_mean / max(beta_mean, 1e-6)
        dominant_freq = self._dominant_frequency(np.mean(window.astype(np.float64), axis=0))

        feats.extend(
            [
                alpha_mean,
                beta_mean,
                theta_mean,
                gamma_mean,
                frontal_theta,
                posterior_alpha,
                beta_alpha_ratio,
                theta_beta_ratio,
                dominant_freq,
            ]
        )
        return np.asarray(feats, dtype=np.float32)

    def extract_features_batch(self, windows: List[np.ndarray]) -> np.ndarray:
        return np.stack([self.extract_features(window) for window in windows], axis=0)

    def start_recording(self, label_name: str) -> None:
        if label_name not in config.class_labels:
            raise ValueError(f"Unknown label '{label_name}'")
        self._current_label = config.class_labels.index(label_name)
        self._recording = True
        logger.info("Recording EEG '%s' (label=%s)", label_name, self._current_label)

    def stop_recording(self) -> None:
        self._recording = False
        self._current_label = None
        logger.info("EEG recording stopped. Buffer: %s windows", len(self._train_windows))

    def clear_training_data(self) -> None:
        self._train_windows = []
        self._train_labels = []
        self._classifier = None
        self._is_trained = False
        self._vote_buffer.clear()
        logger.info("EEG training data cleared")

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
                    aug_windows.append(np.roll(window, shift=int(rng.integers(-4, 5)), axis=1).astype(np.float32))
                aug_labels.append(label)

        return aug_windows, aug_labels

    def train(self, classifier_name: str = "LDA") -> Dict:
        n = len(self._train_windows)
        if n < 12:
            return {"success": False, "error": f"Need >= 12 windows, have {n}"}

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

    def predict(self, window: np.ndarray) -> Tuple[int, float]:
        if not self._is_trained or self._classifier is None:
            return 0, 0.0

        with self._lock:
            if self._classifier_name == "LDA":
                feats = self.extract_features(window).reshape(1, -1)
                return self._classifier.predict(feats)

            raw = window[np.newaxis, :, :]
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
        logger.info("EEG model saved -> %s", path)
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
                logger.error("EEG load_model failed: %s", exc)
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

    def get_channel_quality(self, window: np.ndarray) -> List[float]:
        rms = np.sqrt(np.mean(window.astype(np.float64) ** 2, axis=1))
        scale = max(float(config.signal_profile.metric_full_scale), 1e-6)
        return np.clip(rms / scale, 0.0, 1.0).tolist()
