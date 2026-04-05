"""
Arduino serial bridge — binary protocol.

Byte protocol (host → MCU):
  MOVE  [0x01, joint_id:u8, angle:u8]  → [0xAA, joint_id]   ACK
  ESTOP [0x02]                         → [0xAA, 0x00]        ACK
  HOME  [0x03]                         → [0xAA, 0x00]        ACK
  PING  [0x04]                         → [0xBB, 0x00]        PONG
  ERR response from MCU: [0xFF, reason]

Gesture→angle mapping is defined here so the server decides pose;
the MCU stays a dumb servo driver.
"""

import logging
import threading
import time
from typing import Dict, List, Optional

import serial
import serial.tools.list_ports

from config import config

logger = logging.getLogger(__name__)

# ── Protocol constants ────────────────────────────────────────────────────────
CMD_MOVE          = 0x01
CMD_ESTOP         = 0x02
CMD_HOME          = 0x03
CMD_PING          = 0x04
CMD_DIGITAL_WRITE = 0x05
CMD_ANALOG_WRITE  = 0x06
ACK_OK    = 0xAA
ACK_ERR   = 0xFF
ACK_PONG  = 0xBB

# ── Gesture → joint angles (degrees) ─────────────────────────────────────────
# Joints 0-7 map to servo pins 2-9 on the Arduino.
GESTURE_ANGLES: Dict[str, Dict[int, int]] = {
    "rest":  {i: 90 for i in range(8)},
    "open":  {0: 30,  1: 30,  2: 30,  3: 30,  4: 30,  5: 90, 6: 90, 7: 90},
    "close": {0: 150, 1: 150, 2: 150, 3: 150, 4: 150, 5: 90, 6: 90, 7: 90},
    "pinch": {0: 150, 1: 150, 2: 30,  3: 30,  4: 30,  5: 90, 6: 90, 7: 90},
    "point": {0: 30,  1: 150, 2: 150, 3: 150, 4: 150, 5: 90, 6: 90, 7: 90},
}


class ArduinoBridge:
    def __init__(self):
        self._serial: Optional[serial.Serial] = None
        self._lock = threading.Lock()
        self._connected: bool = False
        self._estop_active: bool = False

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def estop_active(self) -> bool:
        return self._estop_active

    # ── Connection ────────────────────────────────────────────────────────

    def connect(self, port: str = None, baud: int = None) -> bool:
        port = port or config.arduino_port
        baud = baud or config.arduino_baud
        try:
            self._serial = serial.Serial(
                port=port,
                baudrate=baud,
                timeout=config.arduino_timeout,
                write_timeout=1.0,
            )
            time.sleep(2.0)            # wait for Arduino bootloader reset
            self._connected = True
            logger.info(f"Arduino opened on {port}")
            # Drain the ready-signal byte the firmware sends at startup
            self._serial.reset_input_buffer()
            if not self.ping():
                logger.warning("Arduino PING failed after connect (firmware may not be flashed)")
            return True
        except serial.SerialException as exc:
            logger.error(f"Arduino connect failed: {exc}")
            return False

    def disconnect(self) -> None:
        if self._serial and self._serial.is_open:
            self._serial.close()
        self._connected = False

    # ── High-level commands ───────────────────────────────────────────────

    def ping(self) -> bool:
        resp = self._send(bytes([CMD_PING]))
        return resp is not None and len(resp) == 2 and resp[0] == ACK_PONG

    def move(self, joint_id: int, angle: int) -> bool:
        if self._estop_active:
            logger.warning("MOVE blocked — ESTOP active")
            return False
        if not 0 <= joint_id <= 7:
            logger.error(f"Invalid joint_id {joint_id}")
            return False
        angle = int(max(0, min(180, angle)))
        resp = self._send(bytes([CMD_MOVE, joint_id, angle]))
        return resp is not None and resp[0] == ACK_OK

    def execute_gesture(self, gesture_name: str) -> bool:
        """Send MOVE commands for all joints defined by the gesture."""
        angles = GESTURE_ANGLES.get(gesture_name)
        if angles is None:
            logger.warning(f"Unknown gesture '{gesture_name}'")
            return False
        return all(self.move(j, a) for j, a in angles.items())

    def estop(self) -> bool:
        self._estop_active = True
        resp = self._send(bytes([CMD_ESTOP]))
        ok = resp is not None and resp[0] == ACK_OK
        logger.warning(f"ESTOP {'ACK' if ok else 'no-ack'}")
        return ok

    def home(self) -> bool:
        resp = self._send(bytes([CMD_HOME]))
        ok = resp is not None and resp[0] == ACK_OK
        if ok:
            self._estop_active = False
        logger.info(f"HOME {'ACK' if ok else 'no-ack'}")
        return ok

    def digital_write(self, pin: int, value: int) -> bool:
        """Set a digital pin HIGH (1) or LOW (0). Cmd byte 0x05."""
        if self._estop_active:
            logger.warning("DIGITAL_WRITE blocked — ESTOP active")
            return False
        value = 1 if value else 0
        resp = self._send(bytes([CMD_DIGITAL_WRITE, pin, value]))
        return resp is not None and resp[0] == ACK_OK

    def analog_write(self, pin: int, value: int) -> bool:
        """PWM output 0-255 on a pin. Cmd byte 0x06."""
        if self._estop_active:
            logger.warning("ANALOG_WRITE blocked — ESTOP active")
            return False
        value = int(max(0, min(255, value)))
        resp = self._send(bytes([CMD_ANALOG_WRITE, pin, value]))
        return resp is not None and resp[0] == ACK_OK

    # ── Low-level ─────────────────────────────────────────────────────────

    def _send(self, data: bytes) -> Optional[bytes]:
        if not self._serial or not self._serial.is_open:
            return None
        try:
            with self._lock:
                self._serial.reset_input_buffer()
                self._serial.write(data)
                self._serial.flush()
                resp = self._serial.read(2)
            if len(resp) < 2:
                logger.debug(f"Short response ({len(resp)}B) for cmd=0x{data[0]:02X}")
                return None
            return resp
        except serial.SerialException as exc:
            logger.error(f"Serial error: {exc}")
            self._connected = False
            return None

    @staticmethod
    def list_ports() -> List[str]:
        return [p.device for p in serial.tools.list_ports.comports()]


# ── Mock bridge for testing without hardware ──────────────────────────────────

class MockArduinoBridge(ArduinoBridge):
    def connect(self, port=None, baud=None) -> bool:
        self._connected = True
        logger.info("MockArduinoBridge connected")
        return True

    def ping(self) -> bool:
        return True

    def move(self, joint_id: int, angle: int) -> bool:
        if self._estop_active:
            return False
        logger.debug(f"[MOCK] MOVE joint={joint_id} angle={angle}°")
        return True

    def estop(self) -> bool:
        self._estop_active = True
        logger.warning("[MOCK] ESTOP")
        return True

    def home(self) -> bool:
        self._estop_active = False
        logger.info("[MOCK] HOME")
        return True

    def digital_write(self, pin: int, value: int) -> bool:
        if self._estop_active:
            return False
        logger.debug(f"[MOCK] DIGITAL_WRITE pin={pin} value={value}")
        return True

    def analog_write(self, pin: int, value: int) -> bool:
        if self._estop_active:
            return False
        logger.debug(f"[MOCK] ANALOG_WRITE pin={pin} value={value}")
        return True
