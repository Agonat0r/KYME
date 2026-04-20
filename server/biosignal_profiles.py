"""Profile registry for the KYMA biosignal platform."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class FilterStage:
    kind: str
    low_hz: Optional[float] = None
    high_hz: Optional[float] = None
    cutoff_hz: Optional[float] = None
    order: int = 2

    def to_dict(self) -> Dict[str, Optional[float]]:
        return {
            "kind": self.kind,
            "low_hz": self.low_hz,
            "high_hz": self.high_hz,
            "cutoff_hz": self.cutoff_hz,
            "order": self.order,
        }


@dataclass(frozen=True)
class BiosignalProfile:
    key: str
    display_name: str
    family: str
    description: str
    units: str
    hardware_supported: bool
    training_supported: bool
    robotic_arm_supported: bool
    support_level: str
    support_notes: str
    channel_labels: Tuple[str, ...]
    class_labels: Tuple[str, ...]
    default_features: Tuple[str, ...]
    filters: Tuple[FilterStage, ...]
    display_full_scale: float
    metric_label: str
    metric_full_scale: float
    mute_floor: float

    def to_dict(self) -> Dict[str, object]:
        return {
            "key": self.key,
            "display_name": self.display_name,
            "family": self.family,
            "description": self.description,
            "units": self.units,
            "hardware_supported": self.hardware_supported,
            "training_supported": self.training_supported,
            "robotic_arm_supported": self.robotic_arm_supported,
            "support_level": self.support_level,
            "support_notes": self.support_notes,
            "channel_labels": list(self.channel_labels),
            "class_labels": list(self.class_labels),
            "default_features": list(self.default_features),
            "filters": [stage.to_dict() for stage in self.filters],
            "display_full_scale": self.display_full_scale,
            "metric_label": self.metric_label,
            "metric_full_scale": self.metric_full_scale,
            "mute_floor": self.mute_floor,
        }


def _channels(prefix: str, count: int = 8) -> Tuple[str, ...]:
    return tuple(f"{prefix} {i + 1}" for i in range(count))


PROFILE_REGISTRY: Dict[str, BiosignalProfile] = {
    "emg": BiosignalProfile(
        key="emg",
        display_name="EMG",
        family="electromyography",
        description="Muscle activation for gesture decoding and direct motor control.",
        units="uV",
        hardware_supported=True,
        training_supported=True,
        robotic_arm_supported=True,
        support_level="full_demo",
        support_notes="Current KYMA demo path: acquisition, training, prediction, and robotic arm actuation.",
        channel_labels=_channels("Muscle"),
        class_labels=("rest", "open", "close", "pinch", "point"),
        default_features=("MAV", "RMS", "WL", "ZC", "SSC"),
        filters=(
            FilterStage("bandpass", low_hz=20.0, high_hz=120.0, order=2),
            FilterStage("bandstop", low_hz=48.0, high_hz=52.0, order=2),
            FilterStage("bandstop", low_hz=58.0, high_hz=62.0, order=2),
        ),
        display_full_scale=200.0,
        metric_label="Window RMS",
        metric_full_scale=200.0,
        mute_floor=0.5,
    ),
    "eeg": BiosignalProfile(
        key="eeg",
        display_name="EEG",
        family="electroencephalography",
        description="Cortical activity for rhythm analysis, workload, and attention decoding.",
        units="uV",
        hardware_supported=True,
        training_supported=True,
        robotic_arm_supported=False,
        support_level="trainable_profile",
        support_notes="Profile-aware filtering, synthetic data, live band-state decoding, and supervised training are available.",
        channel_labels=("Fp1", "Fp2", "C3", "C4", "P3", "P4", "O1", "O2"),
        class_labels=("delta dominant", "theta dominant", "alpha dominant", "beta dominant", "gamma dominant", "mixed activity"),
        default_features=("delta_band", "theta_band", "alpha_band", "beta_band", "gamma_band"),
        filters=(
            FilterStage("bandpass", low_hz=1.0, high_hz=45.0, order=2),
            FilterStage("bandstop", low_hz=48.0, high_hz=52.0, order=2),
            FilterStage("bandstop", low_hz=58.0, high_hz=62.0, order=2),
        ),
        display_full_scale=80.0,
        metric_label="Channel Activity",
        metric_full_scale=40.0,
        mute_floor=0.2,
    ),
    "ecg": BiosignalProfile(
        key="ecg",
        display_name="ECG",
        family="electrocardiography",
        description="Cardiac electrical activity for beat timing and rhythm analysis.",
        units="uV",
        hardware_supported=True,
        training_supported=True,
        robotic_arm_supported=False,
        support_level="trainable_profile",
        support_notes="Profile-aware streaming, synthetic beats, live rhythm/rate decoding, and supervised training are available.",
        channel_labels=("Lead I", "Lead II", "Lead III", "V1", "V2", "V3", "V4", "Ref"),
        class_labels=("steady rhythm", "elevated rate", "slow rate", "irregular rhythm", "artifact"),
        default_features=("r_peak_rate", "rr_interval", "qrs_width", "hrv"),
        filters=(
            FilterStage("bandpass", low_hz=0.5, high_hz=40.0, order=2),
            FilterStage("bandstop", low_hz=48.0, high_hz=52.0, order=2),
            FilterStage("bandstop", low_hz=58.0, high_hz=62.0, order=2),
        ),
        display_full_scale=1500.0,
        metric_label="Channel Activity",
        metric_full_scale=800.0,
        mute_floor=10.0,
    ),
    "eog": BiosignalProfile(
        key="eog",
        display_name="EOG",
        family="electrooculography",
        description="Eye motion and blink sensing for gaze or switch-style control.",
        units="uV",
        hardware_supported=True,
        training_supported=True,
        robotic_arm_supported=False,
        support_level="trainable_profile",
        support_notes="Profile-aware filtering, synthetic eye events, live motion-state decoding, and supervised training are available.",
        channel_labels=("Horiz L", "Horiz R", "Vert U", "Vert D", "Ref 1", "Ref 2", "Aux 1", "Aux 2"),
        class_labels=("fixation", "blink", "left saccade", "right saccade", "up gaze", "down gaze"),
        default_features=("blink_rate", "saccade_amp", "drift_rate"),
        filters=(
            FilterStage("bandpass", low_hz=0.1, high_hz=15.0, order=2),
            FilterStage("bandstop", low_hz=48.0, high_hz=52.0, order=2),
            FilterStage("bandstop", low_hz=58.0, high_hz=62.0, order=2),
        ),
        display_full_scale=350.0,
        metric_label="Channel Activity",
        metric_full_scale=180.0,
        mute_floor=2.0,
    ),
    "eda": BiosignalProfile(
        key="eda",
        display_name="EDA / GSR",
        family="electrodermal activity",
        description="Skin conductance for arousal, stress, and sympathetic response tracking.",
        units="uS",
        hardware_supported=False,
        training_supported=True,
        robotic_arm_supported=False,
        support_level="trainable_profile",
        support_notes="Profile metadata, synthetic streaming, live tonic/phasic state decoding, and supervised training are available.",
        channel_labels=_channels("EDA"),
        class_labels=("stable tonic", "rising arousal", "phasic peak", "recovering"),
        default_features=("tonic_level", "phasic_peaks", "rise_time"),
        filters=(FilterStage("lowpass", cutoff_hz=5.0, order=2),),
        display_full_scale=2.5,
        metric_label="Channel Activity",
        metric_full_scale=1.2,
        mute_floor=0.02,
    ),
    "ppg": BiosignalProfile(
        key="ppg",
        display_name="PPG",
        family="photoplethysmography",
        description="Optical blood-volume pulse sensing for heart rate and perfusion trends.",
        units="a.u.",
        hardware_supported=False,
        training_supported=True,
        robotic_arm_supported=False,
        support_level="trainable_profile",
        support_notes="Profile metadata, synthetic pulse generation, live pulse-state decoding, and supervised training are available.",
        channel_labels=_channels("PPG"),
        class_labels=("pulse stable", "elevated pulse", "slow pulse", "weak pulse"),
        default_features=("pulse_rate", "pulse_amplitude", "perfusion_index"),
        filters=(FilterStage("bandpass", low_hz=0.5, high_hz=8.0, order=2),),
        display_full_scale=1.5,
        metric_label="Channel Activity",
        metric_full_scale=0.8,
        mute_floor=0.02,
    ),
    "resp": BiosignalProfile(
        key="resp",
        display_name="Respiration",
        family="respiratory effort",
        description="Breathing waveform for rate, phase, and cycle analysis.",
        units="a.u.",
        hardware_supported=False,
        training_supported=True,
        robotic_arm_supported=False,
        support_level="trainable_profile",
        support_notes="Profile metadata, synthetic breathing traces, live breathing-state decoding, and supervised training are available.",
        channel_labels=_channels("Resp"),
        class_labels=("steady breathing", "fast breathing", "slow breathing", "inhalation phase", "exhalation phase"),
        default_features=("resp_rate", "insp_exp_ratio", "variability"),
        filters=(FilterStage("bandpass", low_hz=0.05, high_hz=2.0, order=2),),
        display_full_scale=1.0,
        metric_label="Channel Activity",
        metric_full_scale=0.5,
        mute_floor=0.01,
    ),
    "temp": BiosignalProfile(
        key="temp",
        display_name="Temperature",
        family="peripheral temperature",
        description="Slow thermal trends for recovery, stress, and skin-contact monitoring.",
        units="dC",
        hardware_supported=False,
        training_supported=True,
        robotic_arm_supported=False,
        support_level="trainable_profile",
        support_notes="Profile metadata, synthetic drift, live trend decoding, and supervised training are available.",
        channel_labels=_channels("Temp"),
        class_labels=("stable temperature", "warming", "cooling"),
        default_features=("mean_temp", "temp_drift", "slope"),
        filters=(FilterStage("lowpass", cutoff_hz=1.0, order=2),),
        display_full_scale=0.2,
        metric_label="Channel Activity",
        metric_full_scale=0.08,
        mute_floor=0.002,
    ),
}


def get_profile(name: str) -> BiosignalProfile:
    key = (name or "emg").strip().lower()
    if key not in PROFILE_REGISTRY:
        valid = ", ".join(sorted(PROFILE_REGISTRY))
        raise ValueError(f"Unknown biosignal profile '{name}'. Valid: {valid}")
    return PROFILE_REGISTRY[key]


def list_profiles() -> List[BiosignalProfile]:
    return list(PROFILE_REGISTRY.values())


def list_profile_dicts() -> List[Dict[str, object]]:
    return [profile.to_dict() for profile in list_profiles()]
