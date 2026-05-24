"""
Signal Confidence Layer — the core innovation of KYMA.

Instead of binary accept/reject, every channel gets a continuous trust score
that drives adaptive weighting throughout the pipeline.

Replace:
    ❌ "Bad data, retry"
With:
    ✅ "Signal usable at 72% confidence — proceeding with adaptive weighting"

Architecture
────────────
  SignalConfidenceLayer
    ├── per-channel metrics (updated every window)
    │   ├── stability       — variance of RMS over sliding window
    │   ├── snr             — signal-to-noise ratio vs baseline
    │   ├── activation_var  — useful activation pattern diversity
    │   ├── cross_talk      — correlation with adjacent channels
    │   └── saturation_risk — % samples near ADC limits
    │
    ├── composite trust score (weighted combination)
    │
    └── model viability assessment
        ├── current viability score
        └── improvement potential estimate
"""

import logging
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import config

logger = logging.getLogger(__name__)


@dataclass
class ChannelMetrics:
    """Per-channel signal quality metrics."""
    channel: int = 0
    stability: float = 0.0         # 0 = unstable, 1 = rock solid
    snr: float = 0.0               # signal-to-noise ratio (linear)
    activation_variance: float = 0.0  # diversity of activation patterns
    cross_talk: float = 0.0        # 0 = no cross-talk, 1 = fully correlated
    saturation_risk: float = 0.0   # 0 = safe, 1 = saturated
    trust_score: float = 0.0       # composite score 0-1
    raw_rms: float = 0.0           # current RMS value
    status: str = "unknown"        # "good" | "acceptable" | "degraded" | "failed"
    suggestion: str = ""           # actionable fix if degraded


@dataclass
class SignalConfidence:
    """System-wide signal confidence assessment."""
    channels: List[ChannelMetrics] = field(default_factory=list)
    overall_confidence: float = 0.0
    usable_channels: int = 0
    total_channels: int = 8
    model_viability: float = 0.0
    improvement_potential: float = 0.0
    message: str = ""
    suggestions: List[str] = field(default_factory=list)


class SignalConfidenceLayer:
    """Continuous signal quality assessment with adaptive weighting.

    Called every window from the EMG pipeline. Maintains sliding-window
    statistics and produces per-channel trust scores.
    """

    def __init__(self, n_channels: int = None):
        self.n_ch = n_channels or config.n_channels
        self._lock = threading.Lock()

        # Sliding window for per-channel RMS history
        self._history_len = 100  # ~5 seconds at 20 Hz window rate
        self._rms_history: np.ndarray = np.zeros((self.n_ch, self._history_len))
        self._hist_head: int = 0
        self._hist_count: int = 0

        # Baseline (learned from first ~2s of data)
        self._baseline_rms: Optional[np.ndarray] = None
        self._baseline_windows: List[np.ndarray] = []
        self._baseline_target: int = 40

        # Peak RMS seen per channel (for normalization)
        self._peak_rms: np.ndarray = np.full(self.n_ch, 1e-6)

        # Cross-talk correlation matrix (updated periodically)
        self._raw_history: List[np.ndarray] = []
        self._raw_history_max: int = 50
        self._cross_talk_matrix: Optional[np.ndarray] = None

        # Current metrics (thread-safe read)
        self._current: SignalConfidence = SignalConfidence(
            channels=[ChannelMetrics(channel=i) for i in range(self.n_ch)]
        )

        # Training data quality tracking
        self._train_window_scores: List[float] = []

    # ── Public API ────────────────────────────────────────────────────────

    def update(self, window: np.ndarray) -> SignalConfidence:
        """Process a new EMG window and update all metrics.

        Args:
            window: shape (n_ch, n_samples)

        Returns:
            SignalConfidence with current assessment
        """
        per_ch_rms = np.sqrt(np.mean(window.astype(np.float64) ** 2, axis=1))

        with self._lock:
            # Update RMS history
            self._rms_history[:, self._hist_head % self._history_len] = per_ch_rms
            self._hist_head += 1
            self._hist_count = min(self._hist_count + 1, self._history_len)

            # Update peak RMS
            self._peak_rms = np.maximum(self._peak_rms, per_ch_rms)

            # Baseline calibration
            if self._baseline_rms is None:
                self._baseline_windows.append(per_ch_rms.copy())
                if len(self._baseline_windows) >= self._baseline_target:
                    arr = np.stack(self._baseline_windows, axis=0)
                    self._baseline_rms = np.percentile(arr, 80, axis=0)
                    floor = max(float(np.median(self._baseline_rms)), 1e-6)
                    self._baseline_rms = np.maximum(self._baseline_rms, floor)
                    logger.info(f"Signal baseline calibrated: "
                                f"{np.round(self._baseline_rms, 5).tolist()}")

            # Store raw window for cross-talk analysis
            self._raw_history.append(window.copy())
            if len(self._raw_history) > self._raw_history_max:
                self._raw_history.pop(0)

            # Compute per-channel metrics
            metrics = []
            for ch in range(self.n_ch):
                m = self._compute_channel_metrics(ch, per_ch_rms[ch], window[ch])
                metrics.append(m)

            # Update cross-talk periodically
            if self._hist_count % 20 == 0 and len(self._raw_history) >= 10:
                self._update_cross_talk()

            # Apply cross-talk scores
            if self._cross_talk_matrix is not None:
                for ch in range(self.n_ch):
                    # Max cross-talk with any other channel
                    others = np.delete(self._cross_talk_matrix[ch], ch)
                    metrics[ch].cross_talk = float(np.max(others)) if len(others) > 0 else 0.0

            # Compute composite trust scores
            for m in metrics:
                m.trust_score = self._composite_trust(m)
                m.status = self._classify_status(m.trust_score)
                m.suggestion = self._generate_suggestion(m)

            # System-wide assessment
            trust_scores = [m.trust_score for m in metrics]
            usable = sum(1 for t in trust_scores if t > 0.3)
            overall = float(np.mean(sorted(trust_scores, reverse=True)[:max(usable, 2)]))

            # Model viability
            viability = self._compute_viability(metrics)
            improvement = self._compute_improvement_potential(metrics)

            # Generate message
            if overall > 0.8:
                message = f"Signal excellent — all channels clear ({overall:.0%})"
            elif overall > 0.6:
                message = f"Signal usable at {overall:.0%} confidence — proceeding with adaptive weighting"
            elif overall > 0.4:
                degraded = [m for m in metrics if m.trust_score < 0.4]
                message = (f"Signal degraded on {len(degraded)} channel(s) — "
                           f"auto-compensating ({overall:.0%})")
            else:
                message = f"Signal quality low ({overall:.0%}) — see suggestions"

            # Gather suggestions
            suggestions = [m.suggestion for m in metrics if m.suggestion]

            self._current = SignalConfidence(
                channels=metrics,
                overall_confidence=overall,
                usable_channels=usable,
                total_channels=self.n_ch,
                model_viability=viability,
                improvement_potential=improvement,
                message=message,
                suggestions=suggestions,
            )

        return self._current

    def get_current(self) -> SignalConfidence:
        """Thread-safe read of current confidence assessment."""
        with self._lock:
            return self._current

    def get_channel_weights(self) -> np.ndarray:
        """Return per-channel weights for training/inference (0-1)."""
        with self._lock:
            weights = np.array([m.trust_score for m in self._current.channels])
            # Normalize so best channel = 1.0
            max_w = weights.max()
            if max_w > 0:
                weights = weights / max_w
            return np.maximum(weights, 0.1)  # floor at 0.1 so no channel is fully ignored

    def record_training_window_quality(self, window: np.ndarray) -> float:
        """Score a training window and record it. Returns quality 0-1."""
        per_ch_rms = np.sqrt(np.mean(window.astype(np.float64) ** 2, axis=1))
        weights = self.get_channel_weights()
        # Weighted average activation
        weighted_activation = float(np.sum(per_ch_rms * weights) / np.sum(weights))
        # Normalize against peak
        peak = float(np.max(self._peak_rms))
        score = min(1.0, weighted_activation / max(peak, 1e-6))
        self._train_window_scores.append(score)
        return score

    def to_dict(self) -> dict:
        """Serialize current state for WebSocket broadcast."""
        c = self._current
        return {
            "channels": [
                {
                    "channel": m.channel,
                    "stability": round(m.stability, 3),
                    "snr": round(m.snr, 2),
                    "activation_variance": round(m.activation_variance, 3),
                    "cross_talk": round(m.cross_talk, 3),
                    "saturation_risk": round(m.saturation_risk, 3),
                    "trust_score": round(m.trust_score, 3),
                    "raw_rms": round(m.raw_rms, 6),
                    "status": m.status,
                    "suggestion": m.suggestion,
                }
                for m in c.channels
            ],
            "overall_confidence": round(c.overall_confidence, 3),
            "usable_channels": c.usable_channels,
            "total_channels": c.total_channels,
            "model_viability": round(c.model_viability, 3),
            "improvement_potential": round(c.improvement_potential, 3),
            "message": c.message,
            "suggestions": c.suggestions,
        }

    # ── Internal metrics computation ──────────────────────────────────────

    def _compute_channel_metrics(self, ch: int, rms: float,
                                  samples: np.ndarray) -> ChannelMetrics:
        m = ChannelMetrics(channel=ch, raw_rms=rms)

        # 1. Stability — coefficient of variation of RMS over history
        if self._hist_count >= 10:
            end = self._hist_head
            start = max(0, end - self._hist_count)
            idx = np.arange(start, end) % self._history_len
            hist = self._rms_history[ch, idx]
            mean_rms = float(np.mean(hist))
            std_rms = float(np.std(hist))
            cv = std_rms / max(mean_rms, 1e-8)
            # Lower CV = more stable. Map to 0-1 score.
            m.stability = float(np.clip(1.0 - cv * 2, 0, 1))

        # 2. SNR — signal vs baseline noise floor
        if self._baseline_rms is not None:
            baseline = float(self._baseline_rms[ch])
            m.snr = rms / max(baseline, 1e-8)
        else:
            m.snr = 1.0  # unknown, assume okay

        # 3. Activation variance — how diverse are the signal patterns?
        if self._hist_count >= 20:
            end = self._hist_head
            start = max(0, end - min(self._hist_count, 50))
            idx = np.arange(start, end) % self._history_len
            hist = self._rms_history[ch, idx]
            # Interquartile range normalized by median
            q75, q25 = np.percentile(hist, [75, 25])
            iqr = q75 - q25
            median = float(np.median(hist))
            m.activation_variance = float(np.clip(iqr / max(median, 1e-8), 0, 2)) / 2

        # 4. Saturation risk — samples near ADC limits
        abs_samples = np.abs(samples)
        # BrainFlow Cyton ADC range is roughly ±187.5 µV at 24x gain
        # Samples > 90% of range are concerning
        adc_limit = 180.0  # µV approximate
        near_limit = float(np.mean(abs_samples > adc_limit * 0.9))
        m.saturation_risk = near_limit

        return m

    def _composite_trust(self, m: ChannelMetrics) -> float:
        """Weighted combination of all metrics into a single trust score."""
        # Weights for each component
        w_stability = 0.30
        w_snr = 0.25
        w_activation = 0.20
        w_cross_talk = 0.10
        w_saturation = 0.15

        # SNR score: map ratio to 0-1 (SNR of 3+ is good)
        snr_score = float(np.clip((m.snr - 1) / 4, 0, 1))

        # Cross-talk penalty (high cross-talk = bad)
        ct_score = 1.0 - m.cross_talk

        # Saturation penalty
        sat_score = 1.0 - m.saturation_risk

        trust = (
            w_stability * m.stability
            + w_snr * snr_score
            + w_activation * m.activation_variance
            + w_cross_talk * ct_score
            + w_saturation * sat_score
        )
        return float(np.clip(trust, 0, 1))

    def _classify_status(self, trust: float) -> str:
        if trust > 0.7:
            return "good"
        elif trust > 0.4:
            return "acceptable"
        elif trust > 0.15:
            return "degraded"
        else:
            return "failed"

    def _generate_suggestion(self, m: ChannelMetrics) -> str:
        if m.trust_score > 0.7:
            return ""

        suggestions = []
        if m.stability < 0.3:
            suggestions.append(f"CH{m.channel + 1}: unstable — check electrode contact")
        if m.snr < 2.0 and self._baseline_rms is not None:
            suggestions.append(f"CH{m.channel + 1}: low SNR — increase contraction or reposition")
        if m.saturation_risk > 0.1:
            suggestions.append(f"CH{m.channel + 1}: near saturation — reduce gain or adjust placement")
        if m.cross_talk > 0.7:
            suggestions.append(f"CH{m.channel + 1}: high cross-talk — increase electrode spacing")
        if m.activation_variance < 0.1 and self._hist_count > 30:
            suggestions.append(f"CH{m.channel + 1}: low activation variance — possible misalignment")

        return "; ".join(suggestions) if suggestions else ""

    def _update_cross_talk(self) -> None:
        """Compute inter-channel correlation matrix from raw history."""
        if len(self._raw_history) < 10:
            return
        # Stack recent windows and compute correlation per channel pair
        recent = np.concatenate(self._raw_history[-20:], axis=1)  # (n_ch, lots_of_samples)
        # Pearson correlation
        try:
            corr = np.corrcoef(recent)
            # Diagonal is 1.0 (self-correlation) — zero it
            np.fill_diagonal(corr, 0)
            self._cross_talk_matrix = np.abs(corr)
        except Exception:
            pass

    def _compute_viability(self, metrics: List[ChannelMetrics]) -> float:
        """How viable is the current signal for building a model?"""
        trust_scores = sorted([m.trust_score for m in metrics], reverse=True)
        # Need at least 3 good channels
        top_3 = trust_scores[:3] if len(trust_scores) >= 3 else trust_scores
        base_viability = float(np.mean(top_3))

        # Bonus for training data quantity
        n_train = len(self._train_window_scores)
        data_bonus = min(0.2, n_train / 500)  # max 0.2 bonus at 500+ windows

        return float(np.clip(base_viability + data_bonus, 0, 1))

    def _compute_improvement_potential(self, metrics: List[ChannelMetrics]) -> float:
        """How much better could the model get with signal improvement?"""
        trust_scores = [m.trust_score for m in metrics]
        # Improvement = gap between current average and theoretical max
        current_avg = float(np.mean(trust_scores))
        # Max is if every degraded channel were brought to "good"
        improved = [max(t, 0.8) for t in trust_scores]
        improved_avg = float(np.mean(improved))
        return float(np.clip(improved_avg - current_avg, 0, 1))
