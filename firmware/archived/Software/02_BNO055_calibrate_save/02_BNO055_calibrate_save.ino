// Version: v1.0
// Purpose: Manual 6-face accelerometer calibration capture (BNO055 raw accel)
// Output: prints averaged raw accel for each face (m/s^2)

#include <Wire.h>
#include <Adafruit_BNO055.h>
#include <utility/imumaths.h>
#include <math.h>

Adafruit_BNO055 bno = Adafruit_BNO055(55, 0x28);

static const uint32_t SETTLE_MS = 2000;   // time to settle after moving to a face
static const uint32_t SAMPLE_MS = 5000;   // sampling duration per face
static const uint16_t SAMPLE_DT = 20;     // ~50 Hz sampling during capture

const char* faces[6] = {
  "Z+ (chip UP) flat on table",
  "Z- (chip DOWN) flat on table",
  "X+ (right edge DOWN)",
  "X- (left edge DOWN)",
  "Y+ (nose DOWN)",
  "Y- (nose UP)"
};

void waitForUser() {
  while (!Serial.available()) delay(10);
  while (Serial.available()) Serial.read();
}

void setup() {
  Serial.begin(115200);
  unsigned long t0 = millis();
  while (!Serial && millis() - t0 < 2000) {}

  Wire.begin();
  Wire.setClock(400000);

  if (!bno.begin()) {
    Serial.println("ERROR: BNO055 not detected.");
    while (1) delay(10);
  }
  delay(1000);
  bno.setExtCrystalUse(true);

  Serial.println("\n6-Face Accelerometer Capture (Manual Calibration)");
  Serial.println("For each face: place the device, don't touch it, then press ENTER.");
  Serial.println("It will settle, then record averages.\n");
}

void loop() {
  for (int i = 0; i < 6; i++) {
    Serial.println("--------------------------------------------------");
    Serial.print("Set face: "); Serial.println(faces[i]);
    Serial.println("Press ENTER when ready...");
    waitForUser();

    Serial.println("Settling...");
    delay(SETTLE_MS);

    // Accumulate samples
    double sx=0, sy=0, sz=0;
    uint32_t n = 0;
    uint32_t start = millis();

    while (millis() - start < SAMPLE_MS) {
      imu::Vector<3> a = bno.getVector(Adafruit_BNO055::VECTOR_ACCELEROMETER); // includes gravity
      sx += a.x();
      sy += a.y();
      sz += a.z();
      n++;
      delay(SAMPLE_DT);
    }

    double ax = sx / n;
    double ay = sy / n;
    double az = sz / n;
    double amag = sqrt(ax*ax + ay*ay + az*az);

    Serial.print("FACE "); Serial.print(i+1); Serial.print(" AVG aRaw = ");
    Serial.print(ax, 4); Serial.print(", ");
    Serial.print(ay, 4); Serial.print(", ");
    Serial.print(az, 4);
    Serial.print(" | |a|="); Serial.println(amag, 4);
  }

  Serial.println("\nDone. Copy the 6 averaged vectors into your notes and paste them to ChatGPT.");
  while (1) delay(1000);
}
