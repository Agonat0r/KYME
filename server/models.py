"""Pydantic request/response models and shared enums."""
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class SystemState(str, Enum):
    IDLE        = "idle"
    STREAMING   = "streaming"
    TRAINING    = "training"
    CALIBRATING = "calibrating"
    ESTOP       = "estop"


class TrainRequest(BaseModel):
    gesture: str
    duration_s: int = Field(default=5, ge=1, le=30)


class MoveCommand(BaseModel):
    joint_id: int = Field(ge=0, le=7)
    angle: int    = Field(ge=0, le=180)


class ConnectRequest(BaseModel):
    cyton_port: Optional[str] = None
    arduino_port: Optional[str] = None
    source: Optional[str] = None
    use_synthetic: bool = False
    lsl_stream_name: Optional[str] = None
    lsl_source_id: Optional[str] = None
    playback_session_id: Optional[str] = None
    playback_rate: float = Field(default=1.0, ge=0.25, le=4.0)


class ProfileRequest(BaseModel):
    profile: str


class SessionStartRequest(BaseModel):
    label: str = ""
    subject_id: str = ""
    condition: str = ""
    notes: str = ""
    protocol_key: str = ""
    protocol_title: str = ""
    session_group_id: str = ""
    trial_index: Optional[int] = None
    repetition_index: Optional[int] = None


class SubjectUpsertRequest(BaseModel):
    subject_id: str
    display_name: str = ""
    cohort: str = ""
    handedness: str = ""
    notes: str = ""


class DatasetCreateRequest(BaseModel):
    name: str = ""
    session_ids: List[str] = Field(default_factory=list)


class ExperimentRunRequest(BaseModel):
    dataset_id: str
    classifier: str = "LDA"
    notes: str = ""
    split_strategy: str = "temporal_holdout"
    holdout_fraction: float = Field(default=0.25, ge=0.1, le=0.5)
    holdout_gap_s: float = Field(default=0.2, ge=0.0, le=2.0)


class FilterDesignRequest(BaseModel):
    name: str = ""
    profile: str = ""
    method: str = "butter"
    response_type: str = "bandpass"
    order: int = Field(default=2, ge=1, le=10)
    sample_rate: float = Field(default=250.0, gt=1.0, le=5000.0)
    cutoff_hz: Optional[float] = Field(default=None, gt=0.0)
    low_hz: Optional[float] = Field(default=None, gt=0.0)
    high_hz: Optional[float] = Field(default=None, gt=0.0)
    rp_db: float = Field(default=1.0, gt=0.0, le=20.0)
    rs_db: float = Field(default=40.0, gt=0.0, le=160.0)
    apply_mode: str = "append"


class FilterActivateRequest(BaseModel):
    filter_id: str = Field(min_length=1, max_length=128)
    profile: str = ""


class LSLStartRequest(BaseModel):
    stream_name: Optional[str] = None
    include_markers: bool = True


class LSLMarkerRequest(BaseModel):
    event: str = Field(min_length=1, max_length=128)
    payload: Dict[str, Any] = Field(default_factory=dict)


class ReviewMarkerRequest(BaseModel):
    event: str = Field(min_length=1, max_length=128)
    note: str = ""
    selection_start_s: Optional[float] = Field(default=None, ge=0.0)
    selection_end_s: Optional[float] = Field(default=None, ge=0.0)
    selection_start_sample: Optional[int] = Field(default=None, ge=0)
    selection_end_sample: Optional[int] = Field(default=None, ge=0)
    metrics: Dict[str, Any] = Field(default_factory=dict)


class WorkshopAnalyzeRequest(BaseModel):
    profile: str = ""
    sample_rate: float = Field(default=250.0, gt=1.0, le=5000.0)
    channel_labels: List[str] = Field(default_factory=list)
    channels: List[List[float]] = Field(default_factory=list)
    focus_channel: int = Field(default=0, ge=0, le=63)
    selection_label: str = ""
    selection_start_s: Optional[float] = Field(default=None, ge=0.0)
    selection_end_s: Optional[float] = Field(default=None, ge=0.0)


class OSCStartRequest(BaseModel):
    host: str = "127.0.0.1"
    port: int = Field(default=9000, ge=1, le=65535)
    prefix: str = "/kyma"
    mirror_events: bool = True


class XDFInspectRequest(BaseModel):
    path: str


class XDFImportRequest(BaseModel):
    path: str
    stream_name: Optional[str] = None
    stream_id: Optional[str] = None
    signal_profile: Optional[str] = None
    label: str = ""


class DigitalWriteCommand(BaseModel):
    pin: int = Field(ge=0, le=53)
    value: int = Field(ge=0, le=1)  # 0=LOW, 1=HIGH


class AnalogWriteCommand(BaseModel):
    pin: int = Field(ge=0, le=53)
    value: int = Field(ge=0, le=255)


class FitRequest(BaseModel):
    classifier: str = "LDA"   # LDA | TCN | Mamba


class WSMessage(BaseModel):
    """Shape of every WebSocket message broadcast to clients."""
    type: str                      # emg | prediction | state | calibration | error
    data: Dict[str, Any]
    timestamp: float
