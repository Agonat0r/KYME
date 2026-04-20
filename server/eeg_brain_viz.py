"""EEG visualization helpers backed by MNE and Nilearn."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm, colors as mcolors
except Exception as exc:  # pragma: no cover - import availability depends on runtime
    matplotlib = None
    plt = None
    cm = None
    mcolors = None
    _MATPLOTLIB_ERROR = str(exc)
else:
    _MATPLOTLIB_ERROR = ""

try:
    import mne
except Exception as exc:  # pragma: no cover - import availability depends on runtime
    mne = None
    _MNE_ERROR = str(exc)
else:
    _MNE_ERROR = ""

try:
    from nilearn import plotting as nilearn_plotting
except Exception as exc:  # pragma: no cover - import availability depends on runtime
    nilearn_plotting = None
    _NILEARN_ERROR = str(exc)
else:
    _NILEARN_ERROR = ""


class EEGBrainVisualizer:
    """Generate honest EEG visuals using established neuro tools."""

    FIG_BG = "#f7f3ed"
    FIG_TEXT = "#252831"
    FIG_DIM = "#727683"
    FIG_LINE = "#d8d2c8"

    BANDS: Tuple[Tuple[str, float, float], ...] = (
        ("delta", 1.0, 4.0),
        ("theta", 4.0, 8.0),
        ("alpha", 8.0, 12.0),
        ("beta", 12.0, 30.0),
        ("gamma", 30.0, 45.0),
    )

    def __init__(self, sample_rate: int, channel_labels: list[str], root_dir: Path) -> None:
        self.sample_rate = int(sample_rate)
        self.channel_labels = tuple(channel_labels[:8])
        self.root_dir = Path(root_dir)
        self.cache_dir = self.root_dir / "sessions" / "cache" / "eeg_brain"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.topomap_path = self.cache_dir / "topomap.png"
        self.sensors_path = self.cache_dir / "sensors.png"
        self.markers_path = self.cache_dir / "markers.html"
        self._lock = threading.Lock()
        self._last_payload: Dict[str, object] | None = None
        self._last_render_source_ts = 0.0
        self._info = None
        self._marker_coords_mm = None
        self._static_sensor_ready = False

        if self._mne_ready:
            self._build_montage_assets()

    @property
    def _mne_ready(self) -> bool:
        return bool(mne and plt and matplotlib)

    @property
    def _nilearn_ready(self) -> bool:
        return self._mne_ready and bool(nilearn_plotting and cm and mcolors)

    def _build_montage_assets(self) -> None:
        info = mne.create_info(
            list(self.channel_labels),
            sfreq=float(self.sample_rate),
            ch_types="eeg",
        )
        montage = mne.channels.make_standard_montage("standard_1020")
        info.set_montage(montage, match_case=False, on_missing="ignore")
        positions = montage.get_positions()["ch_pos"]
        coords_mm = []
        for label in self.channel_labels:
            coords = np.asarray(positions.get(label, np.zeros(3)), dtype=float)
            coords_mm.append(coords * 1000.0)
        self._info = info
        self._marker_coords_mm = np.asarray(coords_mm, dtype=float)

    def status(self, profile_key: str, has_window: bool) -> Dict[str, object]:
        if profile_key != "eeg":
            return {
                "available": False,
                "reason": "Switch to the EEG profile to use MNE/Nilearn brain visualization.",
                "surface_available": False,
                "surface_reason": "Nilearn marker view is only shown for EEG.",
            }
        if not self._mne_ready:
            reason = _MNE_ERROR or _MATPLOTLIB_ERROR or "MNE or Matplotlib is not available."
            return {
                "available": False,
                "reason": f"EEG visualization is unavailable: {reason}",
                "surface_available": False,
                "surface_reason": "Nilearn is disabled until the MNE base stack is available.",
            }
        if not has_window:
            return {
                "available": True,
                "reason": "Start an EEG stream to generate live scalp and marker views.",
                "surface_available": self._nilearn_ready,
                "surface_reason": "" if self._nilearn_ready else (_NILEARN_ERROR or "Nilearn is not installed in the active runtime."),
            }
        return {
            "available": True,
            "reason": "",
            "surface_available": self._nilearn_ready,
            "surface_reason": "" if self._nilearn_ready else (_NILEARN_ERROR or "Nilearn is not installed in the active runtime."),
        }

    def _bandpower(self, window: np.ndarray) -> Tuple[np.ndarray, str]:
        arr = window.astype(np.float64, copy=False)
        centered = arr - np.mean(arr, axis=1, keepdims=True)
        spectrum = np.fft.rfft(centered, axis=1)
        freqs = np.fft.rfftfreq(arr.shape[1], d=1.0 / max(self.sample_rate, 1))
        power = np.abs(spectrum) ** 2
        total_mask = (freqs >= 1.0) & (freqs <= 45.0)
        total = power[:, total_mask].sum(axis=1, keepdims=True) + 1e-9

        band_values = []
        band_means = []
        for _, low_hz, high_hz in self.BANDS:
            mask = (freqs >= low_hz) & (freqs < high_hz)
            band = power[:, mask].sum(axis=1) if np.any(mask) else np.zeros(arr.shape[0], dtype=float)
            rel = band / total[:, 0]
            band_values.append(rel)
            band_means.append(float(np.mean(rel)))

        dominant_idx = int(np.argmax(band_means))
        dominant_band = self.BANDS[dominant_idx][0]
        return np.asarray(band_values[dominant_idx], dtype=float), dominant_band

    def _render_topomap(self, values: np.ndarray, band_name: str) -> None:
        fig, ax = plt.subplots(figsize=(4.3, 4.1), dpi=120)
        fig.patch.set_facecolor(self.FIG_BG)
        ax.set_facecolor(self.FIG_BG)
        mne.viz.plot_topomap(
            values,
            self._info,
            axes=ax,
            show=False,
            sensors=True,
            contours=6,
            cmap="viridis",
        )
        ax.set_title(f"{band_name.upper()} Relative Power", color=self.FIG_TEXT, fontsize=10, pad=12)
        ax.tick_params(colors=self.FIG_DIM)
        self._save_figure_atomic(fig, self.topomap_path)
        plt.close(fig)

    def _render_sensors(self) -> None:
        if self._static_sensor_ready and self.sensors_path.exists():
            return
        fig = mne.viz.plot_sensors(
            self._info,
            kind="topomap",
            show_names=True,
            show=False,
        )
        fig.patch.set_facecolor(self.FIG_BG)
        for ax in fig.axes:
            ax.set_facecolor(self.FIG_BG)
            ax.tick_params(colors=self.FIG_DIM)
            ax.title.set_color(self.FIG_TEXT)
            ax.xaxis.label.set_color(self.FIG_TEXT)
            ax.yaxis.label.set_color(self.FIG_TEXT)
            for text in ax.texts:
                text.set_color(self.FIG_TEXT)
            for spine in ax.spines.values():
                spine.set_color(self.FIG_LINE)
        self._save_figure_atomic(fig, self.sensors_path, dpi=120)
        plt.close(fig)
        self._static_sensor_ready = True

    def _render_markers(self, values: np.ndarray, band_name: str) -> str:
        if not self._nilearn_ready:
            return _NILEARN_ERROR or "Nilearn is not installed in the active runtime."

        vmin = float(np.min(values))
        vmax = float(np.max(values))
        if abs(vmax - vmin) < 1e-9:
            vmax = vmin + 1e-9
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        palette = cm.get_cmap("viridis")
        marker_colors = [mcolors.to_hex(palette(norm(float(v)))) for v in values]
        marker_sizes = [6.0 + 8.0 * float(norm(float(v))) for v in values]
        title = f"EEG markers: {band_name}"
        view = nilearn_plotting.view_markers(
            self._marker_coords_mm,
            marker_color=marker_colors,
            marker_size=marker_sizes,
            marker_labels=list(self.channel_labels),
            title=title,
            title_fontsize=14,
        )
        view.resize(520, 230)
        tmp_path = self._temp_path_for(self.markers_path)
        view.save_as_html(str(tmp_path))
        tmp_path.replace(self.markers_path)
        return ""

    def _temp_path_for(self, target: Path) -> Path:
        return target.with_name(f".{target.stem}.{time.time_ns()}{target.suffix}")

    def _save_figure_atomic(self, fig, target: Path, **save_kwargs) -> None:
        tmp_path = self._temp_path_for(target)
        fig.savefig(
            tmp_path,
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
            **save_kwargs,
        )
        tmp_path.replace(target)

    def render(self, profile_key: str, window: np.ndarray | None, source_ts: float) -> Dict[str, object]:
        status = self.status(profile_key=profile_key, has_window=window is not None)
        if not status["available"] or window is None:
            payload = {
                "ok": bool(status["available"]),
                "available": bool(status["available"]),
                "note": status["reason"],
                "surface_available": bool(status["surface_available"]),
                "surface_note": status["surface_reason"],
                "updated_at": None,
                "dominant_band": None,
                "topomap_url": None,
                "sensors_url": None,
                "surface_url": None,
            }
            self._last_payload = payload
            return payload

        with self._lock:
            if (
                self._last_payload
                and source_ts <= self._last_render_source_ts
                and self.topomap_path.exists()
                and self.sensors_path.exists()
            ):
                return self._last_payload

            values, dominant_band = self._bandpower(window)
            self._render_topomap(values, dominant_band)
            self._render_sensors()
            surface_note = self._render_markers(values, dominant_band)
            updated_at = time.time()
            stamp = str(int(updated_at * 1000))
            payload = {
                "ok": True,
                "available": True,
                "note": (
                    "MNE topomap uses the dominant relative band from the latest EEG window. "
                    "Nilearn markers are a reference view only."
                ),
                "surface_available": self._nilearn_ready and self.markers_path.exists(),
                "surface_note": surface_note,
                "updated_at": updated_at,
                "dominant_band": dominant_band,
                "topomap_url": f"/api/eeg/brain-view/topomap.png?t={stamp}",
                "sensors_url": f"/api/eeg/brain-view/sensors.png?t={stamp}",
                "surface_url": f"/api/eeg/brain-view/markers.html?t={stamp}" if self._nilearn_ready else None,
            }
            self._last_render_source_ts = source_ts
            self._last_payload = payload
            return payload
