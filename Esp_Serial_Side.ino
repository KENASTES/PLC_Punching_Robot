#include "HUSKYLENS.h"
#include <HardwareSerial.h>

HUSKYLENS huskyRight;
HUSKYLENS huskyLeft;

const uint8_t relayPins[6] = { 13, 12, 14, 27, 26, 25 };

#define RXD1 16
#define TXD1 17

#define RXD2 21
#define TXD2 22

const uint32_t PC_SERIAL_BAUD = 115200;
const unsigned long HOLD_TIME = 300;
const unsigned long PC_COMM_TIMEOUT = 1000;
const unsigned long HUSKY_INIT_RETRY = 1000;

bool Camera_State = false;  // true = Webcam/PC serial, false = HuskyLens fallback
bool huskyReady = false;

int lastTriggeredZone = 0;
unsigned long activationTime = 0;
unsigned long lastPcPacketTime = 0;
unsigned long lastHuskyInitAttempt = 0;

String pcLine = "";

void triggerRelay(int zone) {
  for (int i = 0; i < 6; i++) {
    if (zone > 0 && i == (zone - 1)) {
      digitalWrite(relayPins[i], HIGH);
    } else {
      digitalWrite(relayPins[i], LOW);
    }
  }
}

void resetRelayState() {
  triggerRelay(0);
  lastTriggeredZone = 0;
  activationTime = 0;
}

void activateZone(int zone, const char* source) {
  if (zone < 1 || zone > 6) {
    resetRelayState();
    return;
  }

  if (lastTriggeredZone != zone) {
    triggerRelay(zone);
    lastTriggeredZone = zone;
    Serial.printf("RELAY,%s,%d\n", source, zone);
  }

  activationTime = millis();
}

void releaseRelayWhenExpired(bool flushHuskySerial) {
  if (lastTriggeredZone <= 0) {
    return;
  }

  if (millis() - activationTime < HOLD_TIME) {
    return;
  }

  resetRelayState();

  if (flushHuskySerial) {
    while (Serial1.available()) { Serial1.read(); }
    while (Serial2.available()) { Serial2.read(); }
  }

  Serial.println("READY");
}

void switchToWebcamMode() {
  if (!Camera_State) {
    resetRelayState();
    Camera_State = true;
    Serial.println("MODE,WEB");
  }

  lastPcPacketTime = millis();
}

void switchToHuskyMode(const char* reason) {
  if (Camera_State) {
    resetRelayState();
    Camera_State = false;
    Serial.printf("MODE,HUSKY,%s\n", reason);
  }
}

bool initHuskyIfNeeded() {
  if (huskyReady) {
    return true;
  }

  if (millis() - lastHuskyInitAttempt < HUSKY_INIT_RETRY) {
    return false;
  }

  lastHuskyInitAttempt = millis();

  bool leftReady = huskyLeft.begin(Serial1);
  bool rightReady = huskyRight.begin(Serial2);

  if (!leftReady || !rightReady) {
    Serial.println("HUSKY,NOT_READY");
    return false;
  }

  huskyReady = true;
  Serial.println("HUSKY,READY");

  huskyLeft.customText("1", 106, 120);
  huskyLeft.customText("1", 212, 120);

  huskyRight.customText("1", 106, 120);
  huskyRight.customText("1", 212, 120);

  return true;
}

void handlePcCommand(String line) {
  line.trim();

  if (line.length() == 0) {
    return;
  }

  if (line == "PING") {
    lastPcPacketTime = millis();
    Serial.println("PONG");
    return;
  }

  if (line == "MODE,WEB") {
    switchToWebcamMode();
    Serial.println("ACK,WEB");
    return;
  }

  if (line == "MODE,HUSKY") {
    lastPcPacketTime = millis();
    switchToHuskyMode("manual");
    Serial.println("ACK,HUSKY");
    return;
  }

  if (!line.startsWith("DATA,")) {
    Serial.printf("ERR,UNKNOWN,%s\n", line.c_str());
    return;
  }

  int firstComma = line.indexOf(',');
  int secondComma = line.indexOf(',', firstComma + 1);

  if (secondComma < 0) {
    Serial.printf("ERR,BAD_DATA,%s\n", line.c_str());
    return;
  }

  int targetZone = line.substring(firstComma + 1, secondComma).toInt();
  int boxState = line.substring(secondComma + 1).toInt();

  switchToWebcamMode();

  if (boxState == 1 && targetZone >= 1 && targetZone <= 6) {
    activateZone(targetZone, "WEB");
  } else {
    resetRelayState();
  }

  Serial.printf("ACK,DATA,%d,%d\n", targetZone, boxState);
}

void readPcSerial() {
  while (Serial.available()) {
    char c = (char)Serial.read();

    if (c == '\r') {
      continue;
    }

    if (c == '\n') {
      handlePcCommand(pcLine);
      pcLine = "";
      continue;
    }

    if (pcLine.length() < 64) {
      pcLine += c;
    } else {
      pcLine = "";
      Serial.println("ERR,LINE_TOO_LONG");
    }
  }
}

void runHuskyMode() {
  if (!initHuskyIfNeeded()) {
    resetRelayState();
    return;
  }

  int targetZone = 0;
  bool detected = false;

  if (huskyLeft.request() && huskyLeft.available()) {
    HUSKYLENSResult resultLeft = huskyLeft.read();

    if (resultLeft.ID > 0) {
      int x = resultLeft.xCenter;

      if (x >= 213) {
        targetZone = 6;
      } else if (x <= 210 && x >= 106) {
        targetZone = 5;
      } else {
        targetZone = 4;
      }

      detected = true;
    }
  }

  if (!detected && huskyRight.request() && huskyRight.available()) {
    HUSKYLENSResult resultRight = huskyRight.read();

    if (resultRight.ID > 0) {
      int x = resultRight.xCenter;

      if (x < 106) {
        targetZone = 1;
      } else if (x < 213) {
        targetZone = 2;
      } else {
        targetZone = 3;
      }

      detected = true;
    }
  }

  if (detected && targetZone > 0) {
    activateZone(targetZone, "HUSKY");
  }

  releaseRelayWhenExpired(true);
}

void setup() {
  Serial.begin(PC_SERIAL_BAUD);
  Serial2.begin(9600, SERIAL_8N1, RXD2, TXD2);
  Serial1.begin(9600, SERIAL_8N1, RXD1, TXD1);

  for (int i = 0; i < 6; i++) {
    pinMode(relayPins[i], OUTPUT);
    digitalWrite(relayPins[i], LOW);
  }

  lastPcPacketTime = millis();
  Serial.println("BOOT,HUSKY_FALLBACK");
}

void loop() {
  readPcSerial();

  if (Camera_State && (millis() - lastPcPacketTime > PC_COMM_TIMEOUT)) {
    switchToHuskyMode("timeout");
  }

  if (Camera_State) {
    releaseRelayWhenExpired(false);
  } else {
    runHuskyMode();
  }
}
