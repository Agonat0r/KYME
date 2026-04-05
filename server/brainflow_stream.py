"""
BrainFlow Cyton streaming layer.

Runs a background thread that pulls samples from the board, applies
bandpass + notch filters, fills a ring-buffer, and fires window callbacks
whenever enough new samples accumulate.

A SimulatedCytonStream (no hardware) is provided for offline dev/testing.
Set environment variable EMG_MOCK=1 to use it automatically.
"""

import logging
import threading
import time
from typing import Callable, List, Optional

import numpy as np

from config import config

logger = logging.getLogger(__name__)


class CytonStream:
    """Real Cyton board via BrainFlow."""

    def __init__(self):
        self.is_running: bool = False
        self._board = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Ring buffer — stores (n_channels × buffer_size) samples
        self._buf_size = config.sample_rate * 10          # 10 s
        self._buf = np.zeros((config.n_channels, self._buf_size), dtype=np.float64)
        self._buf_head = 0           # next write index
        self._total_samples = 0

        # Window emission tracking
        self._samples_since_emit = 0

        # Registered callbacks: fn(window: np.ndarray)  shape=(n_ch, win_size)
        self._callbacks: List[Callable[[np.ndarray], None]] = []

    # ── Public API ────────────────────────────────────────────────────────

    def add_window_callback(self, fn: Callable[[np.ndarray], None]) -> None:
        self._callbacks.append(fn)

    def connect(self, serial_port: str = None) -> bool:
        from brainflow.board_shim import BoardShim, BrainFlowInputParams, BrainFlowError

        port = serial_port or config.serial_port
        params = BrainFlowInputParams()
        params.serial_port = port

        BoardShim.disable_board_logger()
        self._board = BoardShim(config.board_id, params)
        try:
            self._board.prepare_session()
            logger.info(f"Cyton session ready on {port}")
            return True
        except BrainFlowError as exc:
            logger.error(f"BrainFlow prepare_session failed: {exc}")
            self._board = None
            return False

    def start(self) -> bool:
        from brainflow.board_shim import BrainFlowError
        if not self._board:
            return False
        try:
            self._board.start_stream()
        except BrainFlowError as exc:
            logger.error(f"BrainFlow start_stream failed: {exc}")
            return False
        self.is_running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True, name="cyton-read")
        self._thread.start()
        logger.info("Cyton stream started")
        return True

    def stop(self) -> None:
        self.is_running = False
        if self._thread:
            self._thread.join(timeout=3.0)
        if self._board:
            try:
                self._board.stop_stream()
                self._board.release_session()
            except Exception:
                pass
        logger.info("Cyton stream stopped")

    def get_window(self) -> Optional[np.ndarray]:
        """Return the most recent window as (n_ch, window_size_samples), or None."""
        with self._lock:
            n = config.window_size_samples
            if self._total_samples < n:
                return None
            end = self._buf_head
            idx = np.arange(end - n, end) % self._buf_size
            return self._buf[:, idx].copy()

    def get_latest_samples(self, n: int) -> Optional[np.ndarray]:
        """Return last n raw samples as (n_ch, n), or None."""
        with self._lock:
            if self._total_samples < n:
                return None
            end = self._buf_head
            idx = np.arange(end - n, end) % self._buf_size
            return self._buf[:, idx].copy()

    # ── Internal ──────────────────────────────────────────────────────────

    def _read_loop(self) -> None:
        from brainflow.board_shim import BrainFlowError
        from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations

        while self.is_running:
            try:
                raw = self._board.get_board_data()          # (all_ch, n_new)
                if raw.shape[1] == 0:
                    time.sleep(0.005)
                    continue

                emg = raw[config.emg_channels, :].astype(np.float64)  # (n_ch, n_new) — float64 required by DataFilter

                # Debug: log first data chunk stats
                if self._total_samples == 0 and emg.shape[1] > 0:
                    logger.info(f"First data chunk: shape={emg.shape}, ch1_range=[{emg[0].min():.2f}, {emg[0].max():.2f}] uV")

                # Per-channel filters (need at least ~20 samples for filter stability)
                if emg.shape[1] < 10:
                    self._append(emg)
                    continue

                for ch in range(emg.shape[0]):
                    DataFilter.detrend(emg[ch], DetrendOperations.CONSTANT.value)
                    DataFilter.perform_bandpass(
                        emg[ch], config.sample_rate,
                        20.0, 120.0, 2, FilterTypes.BUTTERWORTH.value, 0
                    )
                    # Notch filters: bandstop(data, sr, start_freq, stop_freq, order, type, ripple)
                    DataFilter.perform_bandstop(
                        emg[ch], config.sample_rate,
                        48.0, 52.0, 2, FilterTypes.BUTTERWORTH.value, 0
                    )
                    DataFilter.perform_bandstop(
                        emg[ch], config.sample_rate,
                        58.0, 62.0, 2, FilterTypes.BUTTERWORTH.value, 0
                    )

                self._append(emg)

            except BrainFlowError as exc:
                logger.error(f"BrainFlow read error: {exc}")
                time.sleep(0.01)

    def _append(self, emg: np.ndarray) -> None:
        n_new = emg.shape[1]
        with self._lock:
            for i in range(n_new):
                self._buf[:, self._buf_head % self._buf_size] = emg[:, i]
                self._buf_head += 1
            self._total_samples += n_new

        self._samples_since_emit += n_new
        while self._samples_since_emit >= config.window_increment_samples:
            self._samples_since_emit -= config.window_increment_samples
            window = self.get_window()
            if window is not None:
                self._fire(window)

    def _fire(self, window: np.ndarray) -> None:
        for fn in self._callbacks:
            try:
                fn(window)
            except Exception as exc:
                logger.error(f"Window callback raised: {exc}")


# ── Simulated stream (no hardware required) ───────────────────────────────────

class SimulatedCytonStream(CytonStream):
    """
    Generates synthetic EMG-like signals for testing without a Cyton board.
    Each channel: band-limited noise + a sinusoidal carrier.
    """

    def connect(self, serial_port: str = None) -> bool:
        logger.info("SimulatedCytonStream: no hardware needed")
        return True

    def start(self) -> bool:
        self.is_running = True
        self._thread = threading.Thread(target=self._sim_loop, daemon=True, name="sim-cyton")
        self._thread.start()
        logger.info("Simulated stream started")
        return True

    def stop(self) -> None:
        self.is_running = False
        if self._thread:
            self._thread.join(timeout=3.0)

    def _sim_loop(self) -> None:
        n_ch = config.n_channels
        sr = config.sample_rate
        batch = max(1, sr // 50)     # 50 Hz update cadence
        dt = batch / sr
        t = 0.0

        while self.is_running:
            t_vec = np.linspace(t, t + dt, batch, endpoint=False)
            emg = np.zeros((n_ch, batch), dtype=np.float32)
            for ch in range(n_ch):
                freq = 20.0 + ch * 15.0
                noise = np.random.randn(batch).astype(np.float32) * 0.04
                carrier = np.sin(2 * np.pi * freq * t_vec).astype(np.float32) * 0.08
                emg[ch] = noise + carrier
            t += dt
            self._append(emg)
            time.sleep(dt)
