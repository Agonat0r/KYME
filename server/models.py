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
    mode: Optional[str] = None  # "mock" or "real"


class SessionStartRequest(BaseModel):
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
