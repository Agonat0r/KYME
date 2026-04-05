"""
Central configuration for the KYMA platform.
Override via environment variables or edit defaults here.
"""
import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    # ── BrainFlow / OpenBCI Cyton ──────────────────────────────────────────
    # Board ID 0 = CYTON_BOARD. For synthetic testing use -1 (SYNTHETIC_BOARD).
    board_id: int = int(os.getenv("BOARD_ID", "0"))
    serial_port: str = os.getenv("CYTON_PORT", "COM8")
    sample_rate: int = 250  # Cyton fixed at 250 Hz

    # BrainFlow channel indices for the 8 EXG inputs on Cyton (1-indexed)
    emg_channels: List[int] = field(
        default_factory=lambda: [1, 2, 3, 4, 5, 6, 7, 8]
    )

    # ── Windowing ──────────────────────────────────────────────────────────
    window_size_ms: int = 200      # window length
    window_increment_ms: int = 50  # step between windows (80 % overlap)

    # ── EMG Feature Extraction (LibEMG feature names) ──────────────────────
    features: List[str] = field(
        default_factory=lambda: ["MAV", "RMS", "WL", "ZC", "SSC"]
    )

    # ── Gesture labels (index = class id) ─────────────────────────────────
    gestures: List[str] = field(
        default_factory=lambda: ["rest", "open", "close", "pinch", "point"]
    )

    # ── Arduino serial bridge ─────────────────────────────────────────────
    arduino_port: str = os.getenv("ARDUINO_PORT", "COM4")
    arduino_baud: int = 115200
    arduino_timeout: float = 1.0

    # ── Paths ─────────────────────────────────────────────────────────────
    data_dir: str = "sessions"
    model_dir: str = "models"

    # ── FastAPI server ────────────────────────────────────────────────────
    host: str = "0.0.0.0"
    server_port: int = int(os.getenv("PORT", "8000"))

    # ── Misc ──────────────────────────────────────────────────────────────
    prediction_confidence_threshold: float = 0.65
    # Minimum per-channel RMS (µV) before marking a channel as active
    channel_active_threshold: float = 1e-5

    # ── Derived (read-only) ───────────────────────────────────────────────
    @property
    def window_size_samples(self) -> int:
        return int(self.sample_rate * self.window_size_ms / 1000)

    @property
    def window_increment_samples(self) -> int:
        return int(self.sample_rate * self.window_increment_ms / 1000)

    @property
    def n_channels(self) -> int:
        return len(self.emg_channels)


config = Config()
