// CAR_LOGGER_BUTTON_v1.0
// 25 Hz CSV logger with on-device segmentation via push button (GPIO 9)
// Sensors: BNO055 (raw accel + gyro) + BMP390/BMP3XX
// Output: CSV rows only while logging==true, plus segment markers:
//   #SEGMENT_START,<segment_id>,<mode_code>,<t_ms>
//   #SEGMENT_END,<segment_id>,<t_ms>
//
// Notes:
// - Button wiring: GPIO 9 <-> momentary switch <-> GND
// - Uses INPUT_PULLUP, so idle=HIGH, pressed=LOW
// - No Wi-Fi / no laptop needed to start/stop segments in the field
//
// Version history:
//   v1.0 (Jan 23) Added button-controlled segmentation + mode codes
//
// ----------------------------

#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BNO055.h>
#include <utility/imumaths.h>
#include <Adafruit_BMP3XX.h>
#include <math.h>

Adafruit_BNO055 bno = Adafruit_BNO055(55, 0x28);
Adafruit_BMP3XX bmp;

// ---------- BMP ----------
#define BMP3_I2C_ADDR 0x77
const float SEA_LEVEL_HPA = 1013.25f; // placeholder; relative changes matter more

// ---------- ACCEL BIAS (m/s^2) ----------
// Recomputed after reassembly using full 6-face calibration (Jan 23)
const float ACC_BIAS_X = -0.1926f;
const float ACC_BIAS_Y = +0.1975f;
const float ACC_BIAS_Z = -0.3472f;

// ---------- Sampling ----------
const uint32_t FS_HZ = 25;
const uint32_t PERIOD_US = 1000000UL / FS_HZ;
uint32_t nextTickUs = 0;

// ---------- Button / Segmentation ----------
#define BUTTON_PIN 9
#define DEBOUNCE_MS 300

bool logging = false;
uint32_t segment_id = 0;
uint32_t lastButtonMs = 0;

// Mode codes (simple SHL-style integer labels for your own dataset)
#define MODE_TRAIN   3
#define MODE_SUBWAY  4
#define MODE_CAR     5
#define MODE_WALK    6

// IMPORTANT: Set this before collecting a mode session.
// For tomorrow's train trip, start with MODE_TRAIN.
// If you collect subway later, change to MODE_SUBWAY and re-upload.
uint8_t current_mode = MODE_TRAIN;

bool buttonPressed() {
  static bool lastState = HIGH;
  bool currentState = digitalRead(BUTTON_PIN);

  // Detect falling edge: HIGH -> LOW
  if (lastState == HIGH && currentState == LOW) {
    uint32_t now = millis();
    if (now - lastButtonMs > DEBOUNCE_MS) {
      lastButtonMs = now;
      lastState = currentState;
      return true;
    }
  }
  lastState = currentState;
  return false;
}

void printHeader() {
  Serial.println("t_ms,ax_raw,ay_raw,az_raw,ax_corr,ay_corr,az_corr,acc_mag_corr,gx,gy,gz,pressure_hpa,temp_C,alt_m");
}

void setup() {
  Serial.begin(115200);
  unsigned long t0 = millis();
  while (!Serial && millis() - t0 < 2000) {}

  // Button uses internal pullup (idle HIGH, pressed LOW)
  pinMode(BUTTON_PIN, INPUT_PULLUP);

  Wire.begin();
  Wire.setClock(400000);

  if (!bno.begin()) {
    Serial.println("#ERROR: BNO055 not detected");
    while (1) delay(10);
  }
  delay(1000);
  bno.setExtCrystalUse(true);

  if (!bmp.begin_I2C(BMP3_I2C_ADDR)) {
    Serial.println("#ERROR: BMP3XX not detected (try 0x76/0x77)");
    while (1) delay(10);
  }

  // Stable BMP settings
  bmp.setTemperatureOversampling(BMP3_OVERSAMPLING_2X);
  bmp.setPressureOversampling(BMP3_OVERSAMPLING_8X);
  bmp.setIIRFilterCoeff(BMP3_IIR_FILTER_COEFF_3);
  bmp.setOutputDataRate(BMP3_ODR_50_HZ);

  // Print header + status
  printHeader();
  Serial.print("#START_LOGGER,FS_HZ=");
  Serial.print(FS_HZ);
  Serial.print(",MODE_CODE=");
  Serial.println(current_mode);
  Serial.println("#READY_PRESS_BUTTON");

  nextTickUs = micros();
}

void loop() {
  // Handle segmentation button (can be checked as fast as possible)
  if (buttonPressed()) {
    logging = !logging;

    if (logging) {
      segment_id++;
      Serial.print("#SEGMENT_START,");
      Serial.print(segment_id);
      Serial.print(",");
      Serial.print(current_mode);
      Serial.print(",");
      Serial.println(millis());
    } else {
      Serial.print("#SEGMENT_END,");
      Serial.print(segment_id);
      Serial.print(",");
      Serial.println(millis());
    }
  }

  // Maintain 25 Hz sampling when actively logging
  if (!logging) return;

  uint32_t nowUs = micros();
  if ((int32_t)(nowUs - nextTickUs) < 0) return;
  nextTickUs += PERIOD_US;

  uint32_t nowMs = millis();

  // BNO055 raw accel includes gravity
  imu::Vector<3> aRaw = bno.getVector(Adafruit_BNO055::VECTOR_ACCELEROMETER);
  float axr = aRaw.x();
  float ayr = aRaw.y();
  float azr = aRaw.z();

  // Bias-corrected accel
  float ax = axr - ACC_BIAS_X;
  float ay = ayr - ACC_BIAS_Y;
  float az = azr - ACC_BIAS_Z;
  float amag = sqrtf(ax * ax + ay * ay + az * az);

  // Gyro (rad/s)
  imu::Vector<3> gRaw = bno.getVector(Adafruit_BNO055::VECTOR_GYROSCOPE);
  float gx = gRaw.x();
  float gy = gRaw.y();
  float gz = gRaw.z();

  // BMP readings
  float pressure_hpa = NAN, temp_C = NAN, alt_m = NAN;
  if (bmp.performReading()) {
    pressure_hpa = bmp.pressure / 100.0f;
    temp_C = bmp.temperature;
    alt_m = bmp.readAltitude(SEA_LEVEL_HPA);
  }

  // CSV line
  Serial.print(nowMs); Serial.print(",");
  Serial.print(axr, 4); Serial.print(",");
  Serial.print(ayr, 4); Serial.print(",");
  Serial.print(azr, 4); Serial.print(",");
  Serial.print(ax, 4);  Serial.print(",");
  Serial.print(ay, 4);  Serial.print(",");
  Serial.print(az, 4);  Serial.print(",");
  Serial.print(amag, 4); Serial.print(",");
  Serial.print(gx, 4);  Serial.print(",");
  Serial.print(gy, 4);  Serial.print(",");
  Serial.print(gz, 4);  Serial.print(",");
  Serial.print(pressure_hpa, 2); Serial.print(",");
  Serial.print(temp_C, 2);       Serial.print(",");
  Serial.println(alt_m, 2);
}
