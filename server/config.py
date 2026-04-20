"""Central configuration for the KYMA platform."""

import os
from dataclasses import dataclass, field
from typing import List

from biosignal_profiles import BiosignalProfile, get_profile


@dataclass
class Config:
    # Active biosignal profile. EMG is the only full end-to-end demo today,
    # but the registry keeps the rest of the stack profile-aware.
    signal_profile_name: str = os.getenv("SIGNAL_PROFILE", "emg")

    # BrainFlow / OpenBCI Cyton
    # Board ID 0 = CYTON_BOARD. For synthetic testing use -1 (SYNTHETIC_BOARD).
    board_id: int = int(os.getenv("BOARD_ID", "0"))
    serial_port: str = os.getenv("CYTON_PORT", "COM8")
    sample_rate: int = 250  # Cyton fixed at 250 Hz in the current hardware path

    # BrainFlow channel indices for the 8 EXG inputs on Cyton (1-indexed).
    signal_channels: List[int] = field(
        default_factory=lambda: [1, 2, 3, 4, 5, 6, 7, 8]
    )

    # Windowing
    window_size_ms: int = 200
    window_increment_ms: int = 50

    # Profile-derived decoding defaults. The legacy EMG names are kept as
    # read-only aliases for the existing code paths.
    features: List[str] = field(default_factory=list)
    class_labels: List[str] = field(default_factory=list)

    # Arduino serial bridge
    arduino_port: str = os.getenv("ARDUINO_PORT", "COM4")
    arduino_baud: int = 115200
    arduino_timeout: float = 1.0

    # Paths
    data_dir: str = "sessions"
    model_dir: str = "models"

    # FastAPI server
    host: str = "0.0.0.0"
    server_port: int = int(os.getenv("PORT", "8000"))

    # Misc
    prediction_confidence_threshold: float = 0.65
    channel_active_threshold: float = 1e-5

    def __post_init__(self) -> None:
        self.set_signal_profile(self.signal_profile_name)

    @property
    def signal_profile(self) -> BiosignalProfile:
        return get_profile(self.signal_profile_name)

    def set_signal_profile(self, name: str) -> BiosignalProfile:
        profile = get_profile(name)
        self.signal_profile_name = profile.key
        self.features = list(profile.default_features)
        self.class_labels = list(profile.class_labels)
        return profile

    @property
    def channel_labels(self) -> List[str]:
        labels = list(self.signal_profile.channel_labels)
        if len(labels) >= self.n_channels:
            return labels[: self.n_channels]
        return labels + [f"CH{i + 1}" for i in range(len(labels), self.n_channels)]

    @property
    def window_size_samples(self) -> int:
        return int(self.sample_rate * self.window_size_ms / 1000)

    @property
    def window_increment_samples(self) -> int:
        return int(self.sample_rate * self.window_increment_ms / 1000)

    @property
    def n_channels(self) -> int:
        return len(self.signal_channels)

    # Backward-compatible aliases used throughout the existing EMG code.
    @property
    def emg_channels(self) -> List[int]:
        return self.signal_channels

    @property
    def gestures(self) -> List[str]:
        return self.class_labels


config = Config()
