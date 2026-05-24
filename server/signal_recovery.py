"""
Signal Recovery Engine — intelligent signal recovery instead of forcing retries.

Instead of forcing retries on bad data, KYMA should:
  - Automatically re-weight noisy channels
  - Drop redundant channels
  - Apply adaptive filtering (bandpass tuned per channel)
  - Detect muscle inactivity vs sensor failure
  - Suggest minimal fixes ("Slightly adjust electrode 2")
  - Continue pipeline with partial data + uncertainty tracking
"""
import logging
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
from config import config

logger = logging.getLogger(__name__)

@dataclass
class RecoverySuggestion:
    channel: int
    severity: str
    action: str
    detail: str
    auto_applied: bool
    estimated_improvement: float

@dataclass
class RecoveryState:
    active_recoveries: List[str] = field(default_factory=list)
    suggestions: List[RecoverySuggestion] = field(default_factory=list)
    channels_reweighted: List[int] = field(default_factory=list)
    channels_dropped: List[int] = field(default_factory=list)
    adaptive_filters: Dict[int, dict] = field(default_factory=dict)
    overall_recovery_quality: float = 1.0

class SignalRecoveryEngine:
    def __init__(self, n_channels=None):
        self.n_ch = n_channels or config.n_channels
        self._lock = threading.Lock()
        self._channel_weights = np.ones(self.n_ch)
        self._dropped_channels = set()
        self._rms_history = []
        self._rms_history_max = 200
        self._consecutive_flat = np.zeros(self.n_ch, dtype=int)
        self._consecutive_saturated = np.zeros(self.n_ch, dtype=int)
        self._inactivity_threshold = 40
        self._state = RecoveryState()

    def process(self, window, trust_scores=None):
        per_ch_rms = np.sqrt(np.mean(window.astype(np.float64)**2, axis=1))
        with self._lock:
            self._rms_history.append(per_ch_rms.copy())
            if len(self._rms_history) > self._rms_history_max:
                self._rms_history.pop(0)
            suggestions, actions, reweighted, dropped = [], [], [], []
            for ch in range(self.n_ch):
                s, a = self._analyze_channel(ch, window[ch], per_ch_rms[ch],
                    trust_scores[ch] if trust_scores is not None else None)
                suggestions.extend(s)
                actions.extend(a)
                if ch in self._dropped_channels: dropped.append(ch)
                elif self._channel_weights[ch] < 0.9: reweighted.append(ch)
            active_w = [w for i,w in enumerate(self._channel_weights) if i not in self._dropped_channels]
            quality = float(np.mean(active_w)) if active_w else 0.0
            self._state = RecoveryState(actions, suggestions, reweighted, dropped, {}, quality)
        return self._state

    def get_weights(self):
        with self._lock:
            w = self._channel_weights.copy()
            for ch in self._dropped_channels: w[ch] = 0.0
            return w

    def apply_to_window(self, window):
        return window * self.get_weights()[:, np.newaxis]

    def get_state(self):
        with self._lock: return self._state

    def to_dict(self):
        s = self._state
        return {
            "active_recoveries": s.active_recoveries,
            "suggestions": [{"channel":sg.channel,"severity":sg.severity,"action":sg.action,
                "detail":sg.detail,"auto_applied":sg.auto_applied,
                "estimated_improvement":round(sg.estimated_improvement,2)} for sg in s.suggestions],
            "channels_reweighted": s.channels_reweighted,
            "channels_dropped": s.channels_dropped,
            "overall_recovery_quality": round(s.overall_recovery_quality, 3),
        }

    def _analyze_channel(self, ch, samples, rms, trust):
        suggestions, actions = [], []
        if rms < 1e-7:
            self._consecutive_flat[ch] += 1
            if self._consecutive_flat[ch] > self._inactivity_threshold:
                if ch not in self._dropped_channels:
                    self._dropped_channels.add(ch)
                    self._channel_weights[ch] = 0.0
                    actions.append(f"CH{ch+1}: auto-dropped (no signal)")
                suggestions.append(RecoverySuggestion(ch, "critical",
                    f"Check electrode {ch+1} — no signal detected",
                    "Zero-amplitude output for >2s", True, 0.15))
        else:
            self._consecutive_flat[ch] = 0
            if ch in self._dropped_channels and rms > 1e-5:
                self._dropped_channels.discard(ch)
                self._channel_weights[ch] = 0.5
                actions.append(f"CH{ch+1}: restored (signal returned)")

        abs_max = float(np.max(np.abs(samples)))
        if abs_max > 170:
            self._consecutive_saturated[ch] += 1
            if self._consecutive_saturated[ch] > 10:
                self._channel_weights[ch] = max(0.3, self._channel_weights[ch]*0.95)
                actions.append(f"CH{ch+1}: weight reduced (saturation)")
                suggestions.append(RecoverySuggestion(ch, "warning",
                    f"Reduce contact pressure on electrode {ch+1}",
                    f"Signal clipping at {abs_max:.1f} µV", True, 0.1))
        else:
            self._consecutive_saturated[ch] = max(0, self._consecutive_saturated[ch]-1)

        if len(self._rms_history) > 20 and rms > 1e-7:
            recent = np.array([h[ch] for h in self._rms_history[-20:]])
            cv = float(np.std(recent)/max(np.mean(recent),1e-8))
            if cv < 0.05:
                self._channel_weights[ch] = max(0.2, self._channel_weights[ch]*0.97)
                suggestions.append(RecoverySuggestion(ch, "info",
                    f"Electrode {ch+1} may be misaligned",
                    f"CV={cv:.3f} (expected >0.1)", True, 0.08))

        if trust is not None and trust > 0.6 and ch not in self._dropped_channels:
            target = min(1.0, trust)
            self._channel_weights[ch] += (target - self._channel_weights[ch]) * 0.05
        return suggestions, actions
