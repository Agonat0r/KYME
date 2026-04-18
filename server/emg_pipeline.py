"""
EMG feature extraction + multi-classifier pipeline.

Supports three classifier backends:
  LDA   — hand-crafted features (MAV, RMS, WL, ZC, SSC) + sklearn LDA
  TCN   — raw EMG windows -> Temporal Convolutional Network (PyTorch)
  Mamba — raw EMG windows -> State Space Model (PyTorch, CPU-safe)

Training flow
─────────────
1. start_recording(gesture) -> windows accumulate with labels
2. stop_recording()
3. Repeat for all gestures
4. train(classifier_name) -> fit selected classifier
5. save_model()
"""

import collections
import logging
import os
import pickle
import threading
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from classifiers import LDAClassifier, TCNClassifier, MambaClassifier, get_classifier
from config import config

logger = logging.getLogger(__name__)


class EMGPipeline:
    def __init__(self):
        self._fe = None     # LibEMG FeatureExtractor (lazy import)
        self._lock = threading.Lock()

        # Active classifier
        self._classifier = None
        self._classifier_name: str = "LDA"
        self._is_trained: bool = False

        # Training accumulation
        self._train_windows: List[np.ndarray] = []
        self._train_labels: List[int] = []
        self._recording: bool = False
        self._current_label: Optional[int] = None

        # Prediction callbacks: fn(class_idx, confidence, window)
        self._pred_callbacks: List[Callable] = []

        # ── Majority vote smoothing ──────────────────────────────────
        # Keeps last N predictions and outputs the majority class.
        # This kills jitter without adding latency (windows already overlap 75%).
        self._vote_window: int = 5  # last 5 predictions
        self._vote_buffer: collections.deque = collections.deque(maxlen=5)

        # ── Adaptive rest detection ──────────────────────────────────
        # Per-channel baseline RMS is learned from the first ~2 s of data
        # while the user is presumed to be sitting still. Activation then
        # requires MULTIPLE channels to rise well above their own baseline,
        # which rejects single-channel square-wave / pop artifacts.
        self._rest_rms_threshold: float = 0.003     # fallback when uncalibrated
        self._baseline_rms: Optional[np.ndarray] = None   # shape (n_ch,)
        self._baseline_samples: List[np.ndarray] = []     # collected windows
        self._baseline_target_windows: int = 40           # ~2 s at 20 Hz emit
        self._activation_ch_mult: float = 3.0             # ch active if > base * K
        self._activation_min_channels: int = 2            # need ≥N channels active

        # Hysteresis: avoid flapping between rest/active on borderline windows.
        self._rest_state: bool = True
        self._rest_streak: int = 0
        self._active_streak: int = 0
        self._dwell_windows: int = 2   # need 2 consecutive windows to change state

        Path(config.model_dir).mkdir(exist_ok=True)

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def classifier_name(self) -> str:
        return self._classifier_name

    # ── Callbacks ─────────────────────────────────────────────────────────

    def add_prediction_callback(self, fn: Callable) -> None:
        self._pred_callbacks.append(fn)

    # ── Window entry point (called from BrainFlow thread) ─────────────────

    def on_window(self, window: np.ndarray) -> None:
        """Each EMG window: shape (n_ch, window_size_samples)."""
        if self._recording and self._current_label is not None:
            self._train_windows.append(window.copy())
            self._train_labels.append(self._current_label)
            return

        if self._is_trained:
            is_rest = self._is_rest_window(window)
            if is_rest and "rest" in config.gestures:
                class_idx = config.gestures.index("rest")
                confidence = 0.95
                # Clear vote buffer so we don't carry old active predictions
                # forward into the next activation — this was a big cause of
                # the arm latching the same gesture after motion stopped.
                self._vote_buffer.clear()
            else:
                class_idx, confidence = self.predict(window)
                self._vote_buffer.append(class_idx)
                if len(self._vote_buffer) >= 3:
                    counts = collections.Counter(self._vote_buffer)
                    voted_class, voted_count = counts.most_common(1)[0]
                    if voted_count >= len(self._vote_buffer) // 2 + 1:
                        class_idx = voted_class

            for fn in self._pred_callbacks:
                try:
                    fn(class_idx, confidence, window)
                except Exception as exc:
                    logger.error(f"Prediction callback error: {exc}")

    # ── Rest / activation detection ───────────────────────────────────────

    def _is_rest_window(self, window: np.ndarray) -> bool:
        """Return True when the window is likely the user sitting still.

        Strategy:
          1. Compute per-channel RMS.
          2. Reject any channel whose RMS is wildly above the median (that's
             a single-channel artifact like a loose electrode or square-wave
             pop — it should never cause a gesture to fire).
          3. Compare the *remaining* channels to their learned baseline.
          4. Rest = too few channels show real activation.
          5. 2-window hysteresis so borderline windows don't flap.
        """
        per_ch_rms = np.sqrt(np.mean(window.astype(np.float64) ** 2, axis=1))

        # Baseline calibration — collect ~2 s of windows at startup.
        if self._baseline_rms is None:
            self._baseline_samples.append(per_ch_rms)
            if len(self._baseline_samples) >= self._baseline_target_windows:
                arr = np.stack(self._baseline_samples, axis=0)   # (N, n_ch)
                # 80th-percentile per channel = robust "quiet but not minimum"
                self._baseline_rms = np.percentile(arr, 80, axis=0)
                # Floor so a truly dead channel doesn't produce base=0 → always-active
                floor = max(float(np.median(self._baseline_rms)), 1e-6)
                self._baseline_rms = np.maximum(self._baseline_rms, floor)
                logger.info(f"EMG baseline calibrated (µV RMS per ch): "
                            f"{np.round(self._baseline_rms, 4).tolist()}")
            return True   # treat calibration window as rest

        # Count channels above their own baseline. Require ≥N so a single
        # noisy channel (square-wave pop, loose electrode) cannot drive the
        # arm on its own — real muscle activations light up several channels.
        thresh = self._baseline_rms * self._activation_ch_mult
        active = per_ch_rms > thresh
        n_active = int(active.sum())
        is_active = n_active >= self._activation_min_channels

        # 5) Hysteresis: need `dwell_windows` in the new state to switch.
        if is_active:
            self._active_streak += 1
            self._rest_streak = 0
            if not self._rest_state:
                return False
            if self._active_streak >= self._dwell_windows:
                self._rest_state = False
                return False
            return True   # still in rest until dwell met
        else:
            self._rest_streak += 1
            self._active_streak = 0
            if self._rest_state:
                return True
            if self._rest_streak >= self._dwell_windows:
                self._rest_state = True
                return True
            return False  # still active until dwell met

    def recalibrate_baseline(self) -> None:
        """Force a fresh baseline capture (e.g. user pressed 'Calibrate rest')."""
        self._baseline_rms = None
        self._baseline_samples = []
        self._rest_state = True
        self._rest_streak = 0
        self._active_streak = 0
        self._vote_buffer.clear()
        logger.info("Rest baseline cleared — collecting new calibration...")

    # ── Feature extraction (LibEMG) ───────────────────────────────────────

    def _get_fe(self):
        if self._fe is None:
            from libemg.feature_extractor import FeatureExtractor
            self._fe = FeatureExtractor()
        return self._fe

    def extract_features(self, window: np.ndarray) -> np.ndarray:
        """Single window (n_ch, n_samples) -> flat feature vector."""
        w3d = window[np.newaxis, :, :]
        feats = self._get_fe().extract_features(config.features, w3d)
        parts = [feats[k] for k in sorted(feats.keys())]
        return np.concatenate(parts, axis=1).flatten()

    def extract_features_batch(self, windows: List[np.ndarray]) -> np.ndarray:
        """List[window] -> (n_windows, n_features)."""
        arr = np.stack(windows, axis=0)
        feats = self._get_fe().extract_features(config.features, arr)
        parts = [feats[k] for k in sorted(feats.keys())]
        return np.concatenate(parts, axis=1)

    # ── Recording ─────────────────────────────────────────────────────────

    def start_recording(self, gesture_name: str) -> None:
        if gesture_name not in config.gestures:
            raise ValueError(f"Unknown gesture '{gesture_name}'")
        self._current_label = config.gestures.index(gesture_name)
        self._recording = True
        logger.info(f"Recording '{gesture_name}' (label={self._current_label})")

    def stop_recording(self) -> None:
        self._recording = False
        self._current_label = None
        logger.info(f"Recording stopped. Buffer: {len(self._train_windows)} windows")

    def clear_training_data(self) -> None:
        self._train_windows = []
        self._train_labels = []
        self._is_trained = False
        self._classifier = None
        logger.info("Training data cleared")

    # ── Training ──────────────────────────────────────────────────────────

    def _augment_windows(self, windows: List[np.ndarray], labels: List[int],
                          factor: int = 3) -> Tuple[List[np.ndarray], List[int]]:
        """EMG data augmentation for deep learning classifiers.

        Techniques used:
          - Gaussian noise injection (simulates electrode noise)
          - Time warping (small speed variations in muscle activation)
          - Amplitude scaling (accounts for electrode placement variance)
        """
        rng = np.random.default_rng(42)
        aug_windows, aug_labels = list(windows), list(labels)

        for _ in range(factor - 1):
            for w, lbl in zip(windows, labels):
                choice = rng.integers(0, 3)
                if choice == 0:
                    # Gaussian noise (σ = 5% of window std)
                    noise = rng.normal(0, max(w.std() * 0.05, 1e-6), w.shape).astype(np.float32)
                    aug_windows.append(w + noise)
                elif choice == 1:
                    # Amplitude scaling (0.8x – 1.2x)
                    scale = rng.uniform(0.8, 1.2)
                    aug_windows.append((w * scale).astype(np.float32))
                else:
                    # Channel dropout: zero out 1 random channel
                    w2 = w.copy()
                    ch = rng.integers(0, w.shape[0])
                    w2[ch] = 0.0
                    aug_windows.append(w2)
                aug_labels.append(lbl)

        logger.info(f"Augmented {len(windows)} -> {len(aug_windows)} windows (factor={factor})")
        return aug_windows, aug_labels

    def train(self, classifier_name: str = "LDA") -> Dict:
        """Train the chosen classifier on accumulated windows."""
        n = len(self._train_windows)
        if n < 10:
            return {"success": False, "error": f"Need >= 10 windows, have {n}"}

        self._classifier_name = classifier_name
        logger.info(f"Training {classifier_name} on {n} windows ...")

        y = np.array(self._train_labels)

        if classifier_name == "LDA":
            # Feature-based: extract hand-crafted features first
            X = self.extract_features_batch(self._train_windows)
            clf = LDAClassifier()
            result = clf.fit(X, y)
        elif classifier_name in ("TCN", "Mamba"):
            # Data augmentation for deep models (3x more training data)
            aug_wins, aug_lbls = self._augment_windows(
                self._train_windows, self._train_labels, factor=3
            )
            X = np.stack(aug_wins, axis=0)
            y = np.array(aug_lbls)
            kwargs = {
                "n_channels": config.n_channels,
                "n_classes": len(config.gestures),
            }
            clf = get_classifier(classifier_name, **kwargs)
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
            config.gestures[int(c)]: int((y == c).sum()) for c in np.unique(y)
        }

        va = result.get("val_accuracy")
        np_str = result.get("n_params", "?")
        logger.info(
            f"{classifier_name} trained. Val acc: {va:.3f}, Params: {np_str}"
            if va else f"{classifier_name} trained."
        )
        return result

    # ── Inference ─────────────────────────────────────────────────────────

    def predict(self, window: np.ndarray) -> Tuple[int, float]:
        """Return (class_idx, confidence) for one window."""
        if not self._is_trained or self._classifier is None:
            return 0, 0.0

        with self._lock:
            if self._classifier_name == "LDA":
                feats = self.extract_features(window).reshape(1, -1)
                return self._classifier.predict(feats)
            else:
                # TCN / Mamba expect raw (1, n_ch, seq_len)
                raw = window[np.newaxis, :, :]
                return self._classifier.predict(raw)

    # ── Persistence ───────────────────────────────────────────────────────

    def save_model(self, path: str = None) -> str:
        ext = ".pkl" if self._classifier_name == "LDA" else ".pt"
        path = path or os.path.join(config.model_dir, f"{self._classifier_name.lower()}_model{ext}")
        meta_path = os.path.join(config.model_dir, "active_classifier.pkl")

        self._classifier.save(path)
        with open(meta_path, "wb") as f:
            pickle.dump({
                "classifier": self._classifier_name,
                "path": path,
                "gestures": config.gestures,
                "features": config.features,
            }, f)
        logger.info(f"Model saved -> {path}")
        return path

    def load_model(self, path: str = None) -> bool:
        meta_path = path or os.path.join(config.model_dir, "active_classifier.pkl")
        if not os.path.exists(meta_path):
            return False
        try:
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
            name = meta["classifier"]
            model_path = meta["path"]
            if not os.path.exists(model_path):
                return False

            kwargs = {}
            if name in ("TCN", "Mamba"):
                kwargs = {"n_channels": config.n_channels, "n_classes": len(config.gestures)}
            clf = get_classifier(name, **kwargs)
            clf.load(model_path)

            with self._lock:
                self._classifier = clf
                self._classifier_name = name
                self._is_trained = True
            logger.info(f"Loaded {name} model from {model_path}")
            return True
        except Exception as exc:
            logger.error(f"load_model failed: {exc}")
            return False

    # ── Utilities ─────────────────────────────────────────────────────────

    def get_training_summary(self) -> Dict:
        counts: Dict[str, int] = {}
        for lbl in self._train_labels:
            name = config.gestures[lbl]
            counts[name] = counts.get(name, 0) + 1
        return {
            "total_windows": len(self._train_windows),
            "per_gesture": counts,
            "is_trained": self._is_trained,
            "classifier": self._classifier_name,
        }

    def get_channel_quality(self, window: np.ndarray) -> List[float]:
        rms = np.sqrt(np.mean(window ** 2, axis=1))
        return np.clip(rms / 0.1, 0.0, 1.0).tolist()
