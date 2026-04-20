"""Protocol templates for repeatable KYMA biosignal collection runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

from biosignal_profiles import BiosignalProfile, get_profile


@dataclass(frozen=True)
class ProtocolTemplate:
    key: str
    title: str
    summary: str
    labels: Tuple[str, ...]
    repetitions: int
    trial_duration_s: float
    rest_duration_s: float
    instructions: Tuple[str, ...] = field(default_factory=tuple)
    recommended_source: str = "synthetic"
    recommended_split: str = "leave_one_session_out"

    def to_dict(self) -> Dict[str, object]:
        n_trials = len(self.labels) * max(self.repetitions, 0)
        estimated_duration_s = 0.0
        if n_trials > 0:
            estimated_duration_s = (
                n_trials * float(self.trial_duration_s)
                + max(n_trials - 1, 0) * float(self.rest_duration_s)
            )
        return {
            "key": self.key,
            "title": self.title,
            "summary": self.summary,
            "labels": list(self.labels),
            "repetitions": int(self.repetitions),
            "trial_duration_s": float(self.trial_duration_s),
            "rest_duration_s": float(self.rest_duration_s),
            "instructions": list(self.instructions),
            "recommended_source": self.recommended_source,
            "recommended_split": self.recommended_split,
            "estimated_duration_s": round(estimated_duration_s, 1),
        }


_PROFILE_TIMINGS: Dict[str, Dict[str, float]] = {
    "emg": {"quick": 3.0, "balanced": 4.0, "rest": 2.0},
    "eeg": {"quick": 6.0, "balanced": 8.0, "rest": 3.0},
    "ecg": {"quick": 8.0, "balanced": 12.0, "rest": 4.0},
    "eog": {"quick": 3.0, "balanced": 4.0, "rest": 2.0},
    "eda": {"quick": 8.0, "balanced": 12.0, "rest": 4.0},
    "ppg": {"quick": 8.0, "balanced": 12.0, "rest": 4.0},
    "resp": {"quick": 8.0, "balanced": 12.0, "rest": 4.0},
    "temp": {"quick": 10.0, "balanced": 15.0, "rest": 5.0},
}


def _profile_timings(profile_key: str) -> Dict[str, float]:
    return _PROFILE_TIMINGS.get(profile_key, {"quick": 4.0, "balanced": 6.0, "rest": 2.0})


def _source_for_profile(profile: BiosignalProfile) -> str:
    if profile.hardware_supported:
        return "hardware"
    return "synthetic"


def _full_label_order(profile: BiosignalProfile) -> Tuple[str, ...]:
    labels = tuple(str(label) for label in profile.class_labels)
    return labels or ("baseline",)


def _template_copy(profile: BiosignalProfile) -> Tuple[str, ...]:
    name = profile.display_name
    family = profile.family
    return (
        f"Use the active {name} profile and keep the sensor setup unchanged across the whole run.",
        f"Run all labels in order so the session group stays balanced for {family} evaluation.",
        "Finish each trial before starting the next one so session timing and metadata stay aligned.",
    )


def _build_templates(profile: BiosignalProfile) -> List[ProtocolTemplate]:
    labels = _full_label_order(profile)
    timings = _profile_timings(profile.key)
    source = _source_for_profile(profile)

    quick = ProtocolTemplate(
        key="quick_screen",
        title="Quick Screen",
        summary=f"Single-pass full-label check for {profile.display_name} signal quality and label separability.",
        labels=labels,
        repetitions=1,
        trial_duration_s=timings["quick"],
        rest_duration_s=timings["rest"],
        recommended_source=source,
        instructions=_template_copy(profile),
    )
    balanced = ProtocolTemplate(
        key="balanced_decoder_run",
        title="Balanced Decoder Run",
        summary=f"Repeated full-label protocol for {profile.display_name} training and leave-one-session-out evaluation.",
        labels=labels,
        repetitions=3,
        trial_duration_s=timings["balanced"],
        rest_duration_s=timings["rest"],
        recommended_source=source,
        instructions=_template_copy(profile),
    )
    return [quick, balanced]


def list_protocol_templates(profile_key: str) -> List[Dict[str, object]]:
    profile = get_profile(profile_key)
    return [template.to_dict() for template in _build_templates(profile)]


def get_protocol_template(profile_key: str, template_key: Optional[str]) -> Optional[Dict[str, object]]:
    if not template_key:
        return None
    profile = get_profile(profile_key)
    for template in _build_templates(profile):
        if template.key == template_key:
            return template.to_dict()
    return None


def default_protocol_template(profile_key: str) -> Optional[Dict[str, object]]:
    templates = list_protocol_templates(profile_key)
    return templates[0] if templates else None

