/**
 * KYMA — Servo Controller Firmware
 * Servos on pins 2-9 (joints 0-7)
 *
 * Binary protocol (host → Arduino):
 *   0x01 MOVE  : [0x01, joint_id, angle]   → [0xAA, joint_id]  ACK
 *   0x02 ESTOP : [0x02]                    → [0xAA, 0x00]      ACK
 *   0x03 HOME  : [0x03]                    → [0xAA, 0x00]      ACK
 *   0x04 PING  : [0x04]                    → [0xBB, 0x00]      PONG
 *
 * Error responses: [0xFF, reason]
 *   0x01 = timeout waiting for payload
 *   0x02 = ESTOP active (move blocked)
 *   0x03 = invalid joint_id
 *   0xFF = unknown command
 */

#include <Servo.h>

// ── Config ────────────────────────────────────────────────────────────────────
#define NUM_JOINTS    8
#define BAUD_RATE     115200
#define CMD_TIMEOUT   100     // ms to wait for payload bytes

// Commands
#define CMD_MOVE      0x01
#define CMD_ESTOP     0x02
#define CMD_HOME      0x03
#define CMD_PING      0x04
#define CMD_DWRITE    0x05
#define CMD_AWRITE    0x06

// Responses
#define ACK_OK        0xAA
#define ACK_ERR       0xFF
#define ACK_PONG      0xBB

// ── Hardware ──────────────────────────────────────────────────────────────────
const uint8_t SERVO_PINS[NUM_JOINTS]  = { 2, 3, 4, 5, 6, 7, 8, 9 };
const uint8_t HOME_ANGLES[NUM_JOINTS] = { 90, 90, 90, 90, 90, 90, 90, 90 };
const uint8_t MIN_ANGLES[NUM_JOINTS]  = {  0,  0,  0,  0,  0,  0,  0,  0 };
const uint8_t MAX_ANGLES[NUM_JOINTS]  = {180,180,180,180,180,180,180,180 };

Servo    servos[NUM_JOINTS];
uint8_t  current_angles[NUM_JOINTS];
bool     estop_active = false;

// ── Helpers ───────────────────────────────────────────────────────────────────
inline void send_ack(uint8_t payload) {
  Serial.write(ACK_OK);
  Serial.write(payload);
}

inline void send_err(uint8_t reason) {
  Serial.write(ACK_ERR);
  Serial.write(reason);
}

bool wait_bytes(uint8_t n) {
  unsigned long t = millis();
  while (Serial.available() < n) {
    if (millis() - t > CMD_TIMEOUT) return false;
  }
  return true;
}

void do_home() {
  for (uint8_t i = 0; i < NUM_JOINTS; i++) {
    if (!servos[i].attached()) servos[i].attach(SERVO_PINS[i]);
    servos[i].write(HOME_ANGLES[i]);
    current_angles[i] = HOME_ANGLES[i];
  }
}

void do_estop() {
  estop_active = true;
  for (uint8_t i = 0; i < NUM_JOINTS; i++) {
    // Write current position then detach — removes PWM hold / reduces heat
    servos[i].write(current_angles[i]);
    delay(20);
    servos[i].detach();
  }
}

void do_resume() {
  estop_active = false;
  for (uint8_t i = 0; i < NUM_JOINTS; i++) {
    servos[i].attach(SERVO_PINS[i]);
    servos[i].write(current_angles[i]);
  }
}

// ── Setup / Loop ──────────────────────────────────────────────────────────────
void setup() {
  Serial.begin(BAUD_RATE);
  for (uint8_t i = 0; i < NUM_JOINTS; i++) {
    servos[i].attach(SERVO_PINS[i]);
    current_angles[i] = HOME_ANGLES[i];
    servos[i].write(HOME_ANGLES[i]);
  }
  delay(500);   // let servos reach home before accepting commands
  // Signal ready
  Serial.write(ACK_PONG);
  Serial.write(0xFF);
}

void loop() {
  if (Serial.available() < 1) return;

  uint8_t cmd = (uint8_t)Serial.read();

  switch (cmd) {

    case CMD_MOVE: {
      if (!wait_bytes(2)) { send_err(0x01); break; }
      uint8_t joint_id = (uint8_t)Serial.read();
      uint8_t angle    = (uint8_t)Serial.read();

      if (estop_active)        { send_err(0x02); break; }
      if (joint_id >= NUM_JOINTS) { send_err(0x03); break; }

      uint8_t clamped = constrain(angle, MIN_ANGLES[joint_id], MAX_ANGLES[joint_id]);
      servos[joint_id].write(clamped);
      current_angles[joint_id] = clamped;
      send_ack(joint_id);
      break;
    }

    case CMD_ESTOP: {
      do_estop();
      send_ack(0x00);
      break;
    }

    case CMD_HOME: {
      if (estop_active) do_resume();   // re-attach first
      do_home();
      send_ack(0x00);
      break;
    }

    case CMD_PING: {
      Serial.write(ACK_PONG);
      Serial.write(0x00);
      break;
    }

    case CMD_DWRITE: {
      if (!wait_bytes(2)) { send_err(0x01); break; }
      uint8_t pin   = (uint8_t)Serial.read();
      uint8_t value = (uint8_t)Serial.read();
      if (estop_active) { send_err(0x02); break; }
      pinMode(pin, OUTPUT);
      digitalWrite(pin, value ? HIGH : LOW);
      send_ack(pin);
      break;
    }

    case CMD_AWRITE: {
      if (!wait_bytes(2)) { send_err(0x01); break; }
      uint8_t pin   = (uint8_t)Serial.read();
      uint8_t value = (uint8_t)Serial.read();
      if (estop_active) { send_err(0x02); break; }
      pinMode(pin, OUTPUT);
      analogWrite(pin, value);
      send_ack(pin);
      break;
    }

    default:
      send_err(0xFF);
      break;
  }
}
