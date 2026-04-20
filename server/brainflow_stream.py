"""
BrainFlow Cyton streaming layer.

Runs a background thread that pulls samples from the board, applies the
active profile's filter chain, fills a ring buffer, and fires window
callbacks whenever enough new samples accumulate.
"""

import json
import logging
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

from config import config

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency at runtime
    from scipy.signal import sosfilt

    _SCIPY_FILTER_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - optional dependency at runtime
    sosfilt = None  # type: ignore[assignment]
    _SCIPY_FILTER_ERROR = exc


class CytonStream:
    """Real Cyton board via BrainFlow."""

    def __init__(self):
        self.is_running: bool = False
        self._board = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        self._buf_size = config.sample_rate * 10
        self._buf = np.zeros((config.n_channels, self._buf_size), dtype=np.float64)
        self._buf_head = 0
        self._total_samples = 0
        self._samples_since_emit = 0
        self._callbacks: List[Callable[[np.ndarray], None]] = []
        self._chunk_callbacks: List[Callable[[np.ndarray], None]] = []
        self._custom_filter_name: str = ""
        self._custom_filter_mode: str = "append"
        self._custom_sos: Optional[np.ndarray] = None
        self._custom_zi: Optional[np.ndarray] = None

    def add_window_callback(self, fn: Callable[[np.ndarray], None]) -> None:
        if fn not in self._callbacks:
            self._callbacks.append(fn)

    def remove_window_callback(self, fn: Callable[[np.ndarray], None]) -> None:
        if fn in self._callbacks:
            self._callbacks.remove(fn)

    def add_chunk_callback(self, fn: Callable[[np.ndarray], None]) -> None:
        if fn not in self._chunk_callbacks:
            self._chunk_callbacks.append(fn)

    def remove_chunk_callback(self, fn: Callable[[np.ndarray], None]) -> None:
        if fn in self._chunk_callbacks:
            self._chunk_callbacks.remove(fn)

    def connect(self, serial_port: str = None) -> bool:
        from brainflow.board_shim import BoardShim, BrainFlowError, BrainFlowInputParams

        port = serial_port or config.serial_port
        params = BrainFlowInputParams()
        params.serial_port = port

        BoardShim.disable_board_logger()
        self._board = BoardShim(config.board_id, params)
        try:
            self._board.prepare_session()
            logger.info("Cyton session ready on %s", port)
            return True
        except BrainFlowError as exc:
            logger.error("BrainFlow prepare_session failed: %s", exc)
            self._board = None
            return False

    def start(self) -> bool:
        from brainflow.board_shim import BrainFlowError

        if not self._board:
            return False
        try:
            self._board.start_stream()
        except BrainFlowError as exc:
            logger.error("BrainFlow start_stream failed: %s", exc)
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
        with self._lock:
            n = config.window_size_samples
            if self._total_samples < n:
                return None
            end = self._buf_head
            idx = np.arange(end - n, end) % self._buf_size
            return self._buf[:, idx].copy()

    def get_latest_samples(self, n: int) -> Optional[np.ndarray]:
        with self._lock:
            if self._total_samples < n:
                return None
            end = self._buf_head
            idx = np.arange(end - n, end) % self._buf_size
            return self._buf[:, idx].copy()

    def set_custom_filter(self, runtime_filter: Optional[Dict[str, object]]) -> None:
        self._custom_filter_name = ""
        self._custom_filter_mode = "append"
        self._custom_sos = None
        self._custom_zi = None

        if not runtime_filter:
            return
        if sosfilt is None:
            logger.warning("Custom filter requested but SciPy is unavailable: %s", _SCIPY_FILTER_ERROR)
            return

        sos = np.asarray(runtime_filter.get("sos") or [], dtype=np.float64)
        if sos.ndim != 2 or sos.shape[1] != 6 or sos.size == 0:
            logger.warning("Ignoring invalid SOS custom filter payload")
            return

        self._custom_filter_name = str(runtime_filter.get("name") or "Custom Filter")
        self._custom_filter_mode = str(runtime_filter.get("apply_mode") or "append")
        self._custom_sos = sos
        self._custom_zi = np.zeros((config.n_channels, sos.shape[0], 2), dtype=np.float64)
        logger.info(
            "Custom filter armed: %s (%s sections, mode=%s)",
            self._custom_filter_name,
            sos.shape[0],
            self._custom_filter_mode,
        )

    def _read_loop(self) -> None:
        from brainflow.board_shim import BrainFlowError
        from brainflow.data_filter import DataFilter, DetrendOperations, FilterTypes

        profile = config.signal_profile
        while self.is_running:
            try:
                raw = self._board.get_board_data()
                if raw.shape[1] == 0:
                    time.sleep(0.005)
                    continue

                samples = raw[config.signal_channels, :].astype(np.float64)

                if self._total_samples == 0 and samples.shape[1] > 0:
                    logger.info(
                        "First %s chunk: shape=%s, ch1_range=[%.2f, %.2f] %s",
                        profile.display_name,
                        samples.shape,
                        samples[0].min(),
                        samples[0].max(),
                        profile.units,
                    )

                if samples.shape[1] < 10:
                    self._append(samples)
                    continue

                for ch in range(samples.shape[0]):
                    DataFilter.detrend(samples[ch], DetrendOperations.CONSTANT.value)
                    if self._custom_filter_mode != "replace_defaults":
                        self._apply_profile_filters(samples[ch], DataFilter, FilterTypes)

                self._append(samples)

            except BrainFlowError as exc:
                logger.error("BrainFlow read error: %s", exc)
                time.sleep(0.01)

    def _apply_profile_filters(self, data, DataFilter, FilterTypes) -> None:
        for stage in config.signal_profile.filters:
            if stage.kind == "bandpass" and stage.low_hz is not None and stage.high_hz is not None:
                DataFilter.perform_bandpass(
                    data,
                    config.sample_rate,
                    stage.low_hz,
                    stage.high_hz,
                    stage.order,
                    FilterTypes.BUTTERWORTH.value,
                    0,
                )
            elif stage.kind == "bandstop" and stage.low_hz is not None and stage.high_hz is not None:
                DataFilter.perform_bandstop(
                    data,
                    config.sample_rate,
                    stage.low_hz,
                    stage.high_hz,
                    stage.order,
                    FilterTypes.BUTTERWORTH.value,
                    0,
                )
            elif stage.kind == "lowpass" and stage.cutoff_hz is not None:
                DataFilter.perform_lowpass(
                    data,
                    config.sample_rate,
                    stage.cutoff_hz,
                    stage.order,
                    FilterTypes.BUTTERWORTH.value,
                    0,
                )
            elif stage.kind == "highpass" and stage.cutoff_hz is not None:
                DataFilter.perform_highpass(
                    data,
                    config.sample_rate,
                    stage.cutoff_hz,
                    stage.order,
                    FilterTypes.BUTTERWORTH.value,
                    0,
                )

    def _append(self, samples: np.ndarray) -> None:
        chunk = samples.astype(np.float64, copy=True)
        self._apply_custom_filter(chunk)
        n_new = chunk.shape[1]
        with self._lock:
            for i in range(n_new):
                self._buf[:, self._buf_head % self._buf_size] = chunk[:, i]
                self._buf_head += 1
            self._total_samples += n_new

        self._fire_chunk(chunk)
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
                logger.error("Window callback raised: %s", exc)

    def _fire_chunk(self, chunk: np.ndarray) -> None:
        for fn in self._chunk_callbacks:
            try:
                fn(chunk)
            except Exception as exc:
                logger.error("Chunk callback raised: %s", exc)

    def _apply_custom_filter(self, chunk: np.ndarray) -> None:
        if self._custom_sos is None or self._custom_zi is None or sosfilt is None:
            return
        for ch in range(min(chunk.shape[0], self._custom_zi.shape[0])):
            filtered, zi = sosfilt(self._custom_sos, chunk[ch], zi=self._custom_zi[ch])
            chunk[ch] = filtered
            self._custom_zi[ch] = zi


class SimulatedCytonStream(CytonStream):
    """Generate synthetic signals for the active biosignal profile."""

    def connect(self, serial_port: str = None) -> bool:
        logger.info("Simulated stream: no hardware needed")
        return True

    def start(self) -> bool:
        self.is_running = True
        self._thread = threading.Thread(target=self._sim_loop, daemon=True, name="sim-signal")
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
        batch = max(1, sr // 50)
        dt = batch / sr
        t = 0.0

        while self.is_running:
            t_vec = np.linspace(t, t + dt, batch, endpoint=False)
            samples = self._generate_chunk(config.signal_profile.key, n_ch, t_vec)
            t += dt
            self._append(samples.astype(np.float32))
            time.sleep(dt)

    def _generate_chunk(self, profile_key: str, n_ch: int, t_vec: np.ndarray) -> np.ndarray:
        if profile_key == "emg":
            return self._sim_emg(n_ch, t_vec)
        if profile_key == "eeg":
            return self._sim_eeg(n_ch, t_vec)
        if profile_key == "ecg":
            return self._sim_ecg(n_ch, t_vec)
        if profile_key == "eog":
            return self._sim_eog(n_ch, t_vec)
        if profile_key == "eda":
            return self._sim_eda(n_ch, t_vec)
        if profile_key == "ppg":
            return self._sim_ppg(n_ch, t_vec)
        if profile_key == "resp":
            return self._sim_resp(n_ch, t_vec)
        if profile_key == "temp":
            return self._sim_temp(n_ch, t_vec)
        return self._sim_emg(n_ch, t_vec)

    def _sim_emg(self, n_ch: int, t_vec: np.ndarray) -> np.ndarray:
        emg = np.zeros((n_ch, t_vec.size), dtype=np.float32)
        for ch in range(n_ch):
            carrier_freq = 18.0 + ch * 11.0
            env_freq = 0.18 + ch * 0.03
            phase = ch * 0.7
            envelope = 22.0 + 16.0 * (0.5 + 0.5 * np.sin(2 * np.pi * env_freq * t_vec + phase))
            carrier = np.sin(2 * np.pi * carrier_freq * t_vec + phase).astype(np.float32)
            noise = np.random.randn(t_vec.size).astype(np.float32) * 4.5
            emg[ch] = carrier * envelope.astype(np.float32) + noise
        return emg

    def _sim_eeg(self, n_ch: int, t_vec: np.ndarray) -> np.ndarray:
        eeg = np.zeros((n_ch, t_vec.size), dtype=np.float32)
        for ch in range(n_ch):
            phase = ch * 0.45
            alpha = 18.0 * np.sin(2 * np.pi * (8.0 + 0.4 * ch) * t_vec + phase)
            theta = 10.0 * np.sin(2 * np.pi * (4.5 + 0.1 * ch) * t_vec + phase * 0.7)
            beta = 6.0 * np.sin(2 * np.pi * (18.0 + ch) * t_vec + phase * 1.1)
            delta = 12.0 * np.sin(2 * np.pi * (1.2 + 0.03 * ch) * t_vec + phase * 0.2)
            noise = np.random.randn(t_vec.size).astype(np.float32) * 3.5
            eeg[ch] = (alpha + theta + beta + delta).astype(np.float32) * 0.45 + noise
        return eeg

    def _sim_ecg(self, n_ch: int, t_vec: np.ndarray) -> np.ndarray:
        ecg = np.zeros((n_ch, t_vec.size), dtype=np.float32)
        for ch in range(n_ch):
            beat_hz = 1.15 + ch * 0.01
            phase = (t_vec * beat_hz + ch * 0.015) % 1.0
            p = 120.0 * np.exp(-((phase - 0.16) / 0.035) ** 2)
            q = -140.0 * np.exp(-((phase - 0.29) / 0.012) ** 2)
            r = 900.0 * np.exp(-((phase - 0.31) / 0.008) ** 2)
            s = -220.0 * np.exp(-((phase - 0.34) / 0.015) ** 2)
            twave = 260.0 * np.exp(-((phase - 0.58) / 0.07) ** 2)
            baseline = 35.0 * np.sin(2 * np.pi * 0.18 * t_vec + ch * 0.4)
            noise = np.random.randn(t_vec.size).astype(np.float32) * 10.0
            ecg[ch] = (p + q + r + s + twave + baseline).astype(np.float32) + noise
        return ecg

    def _sim_eog(self, n_ch: int, t_vec: np.ndarray) -> np.ndarray:
        eog = np.zeros((n_ch, t_vec.size), dtype=np.float32)
        for ch in range(n_ch):
            phase = ch * 0.35
            drift = 75.0 * np.sin(2 * np.pi * 0.16 * t_vec + phase)
            saccade = 110.0 * np.tanh(4.5 * np.sin(2 * np.pi * (0.22 + ch * 0.01) * t_vec + phase))
            blink = 160.0 * np.maximum(0.0, np.sin(2 * np.pi * (0.42 + ch * 0.01) * t_vec + phase)) ** 8
            noise = np.random.randn(t_vec.size).astype(np.float32) * 3.0
            eog[ch] = (drift + saccade + blink).astype(np.float32) + noise
        return eog

    def _sim_eda(self, n_ch: int, t_vec: np.ndarray) -> np.ndarray:
        eda = np.zeros((n_ch, t_vec.size), dtype=np.float32)
        for ch in range(n_ch):
            phase = ch * 0.2
            tonic = 0.18 * np.sin(2 * np.pi * 0.02 * t_vec + phase)
            phasic = 0.95 * np.maximum(0.0, np.sin(2 * np.pi * (0.05 + ch * 0.004) * t_vec + phase)) ** 3
            noise = np.random.randn(t_vec.size).astype(np.float32) * 0.015
            eda[ch] = (tonic + phasic).astype(np.float32) + noise
        return eda

    def _sim_ppg(self, n_ch: int, t_vec: np.ndarray) -> np.ndarray:
        ppg = np.zeros((n_ch, t_vec.size), dtype=np.float32)
        for ch in range(n_ch):
            beat_hz = 1.2 + ch * 0.015
            phase = 2 * np.pi * beat_hz * t_vec + ch * 0.25
            fundamental = np.maximum(0.0, np.sin(phase)) ** 2
            harmonic = 0.28 * np.maximum(0.0, np.sin(2 * phase + 0.6)) ** 3
            wander = 0.12 * np.sin(2 * np.pi * 0.12 * t_vec + ch * 0.1)
            noise = np.random.randn(t_vec.size).astype(np.float32) * 0.02
            ppg[ch] = (fundamental + harmonic + wander).astype(np.float32) + noise
        return ppg

    def _sim_resp(self, n_ch: int, t_vec: np.ndarray) -> np.ndarray:
        resp = np.zeros((n_ch, t_vec.size), dtype=np.float32)
        for ch in range(n_ch):
            freq = 0.22 + ch * 0.006
            phase = ch * 0.4
            primary = 0.62 * np.sin(2 * np.pi * freq * t_vec + phase)
            harmonic = 0.14 * np.sin(2 * np.pi * freq * 2.0 * t_vec + phase * 0.7)
            noise = np.random.randn(t_vec.size).astype(np.float32) * 0.01
            resp[ch] = (primary + harmonic).astype(np.float32) + noise
        return resp

    def _sim_temp(self, n_ch: int, t_vec: np.ndarray) -> np.ndarray:
        temp = np.zeros((n_ch, t_vec.size), dtype=np.float32)
        for ch in range(n_ch):
            drift = 0.045 * np.sin(2 * np.pi * (0.01 + ch * 0.0008) * t_vec + ch * 0.25)
            ripple = 0.008 * np.sin(2 * np.pi * 0.07 * t_vec + ch * 0.4)
            noise = np.random.randn(t_vec.size).astype(np.float32) * 0.0015
            temp[ch] = (drift + ripple).astype(np.float32) + noise
        return temp


class PlaybackCytonStream(CytonStream):
    """Replay a recorded session from sessions/<id>/signal_raw.csv."""

    def __init__(self, session_id: Optional[str] = None, playback_rate: float = 1.0):
        super().__init__()
        self._session_id = session_id
        self._playback_rate = max(float(playback_rate or 1.0), 0.25)
        self._playback_data: Optional[np.ndarray] = None
        self._session_meta: dict = {}

    @property
    def session_id(self) -> Optional[str]:
        return self._session_id

    def connect(self, serial_port: str = None, session_id: str = None) -> bool:
        target = session_id or self._session_id
        if not target:
            logger.error("Playback connect failed: no session id provided")
            return False

        session_dir = Path(config.data_dir) / target
        meta_path = session_dir / "meta.json"
        raw_path = session_dir / "signal_raw.csv"
        if not raw_path.exists():
            raw_path = session_dir / "emg_raw.csv"
        if not meta_path.exists() or not raw_path.exists():
            logger.error("Playback session missing files: %s", target)
            return False

        try:
            with open(meta_path, encoding="utf-8") as fh:
                self._session_meta = json.load(fh)
            raw = np.loadtxt(raw_path, delimiter=",", skiprows=1)
        except Exception as exc:
            logger.error("Failed to load playback session %s: %s", target, exc)
            return False

        if raw.ndim == 1:
            raw = raw.reshape(1, -1)
        if raw.shape[1] < 2:
            logger.error("Playback session %s does not contain channel samples", target)
            return False

        cfg = self._session_meta.get("config", {})
        sample_rate = int(cfg.get("sample_rate") or config.sample_rate)
        if sample_rate != config.sample_rate:
            logger.error(
                "Playback session %s uses %s Hz, but the current runtime expects %s Hz",
                target,
                sample_rate,
                config.sample_rate,
            )
            return False

        data = raw[:, 1:].astype(np.float64)
        if data.shape[1] != config.n_channels:
            logger.error(
                "Playback session %s has %s channels, expected %s",
                target,
                data.shape[1],
                config.n_channels,
            )
            return False

        self._session_id = target
        self._playback_data = data.T
        logger.info(
            "Playback session ready: %s (%s samples, %s channels, %.2fx)",
            target,
            self._playback_data.shape[1],
            self._playback_data.shape[0],
            self._playback_rate,
        )
        return True

    def start(self) -> bool:
        if self._playback_data is None or self._playback_data.size == 0:
            return False
        self.is_running = True
        self._thread = threading.Thread(
            target=self._playback_loop,
            daemon=True,
            name="playback-signal",
        )
        self._thread.start()
        logger.info("Playback stream started")
        return True

    def stop(self) -> None:
        self.is_running = False
        if self._thread:
            self._thread.join(timeout=3.0)
        logger.info("Playback stream stopped")

    def _playback_loop(self) -> None:
        if self._playback_data is None:
            return

        total = self._playback_data.shape[1]
        batch = max(1, config.sample_rate // 20)
        cursor = 0

        while self.is_running:
            end = min(cursor + batch, total)
            chunk = self._playback_data[:, cursor:end]
            if chunk.size:
                self._append(chunk)
                sleep_s = chunk.shape[1] / config.sample_rate / self._playback_rate
                time.sleep(max(0.001, sleep_s))

            if end >= total:
                cursor = 0
                time.sleep(0.15)
            else:
                cursor = end
