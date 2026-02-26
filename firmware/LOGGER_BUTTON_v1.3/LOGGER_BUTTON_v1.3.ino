// LOGGER_BUTTON_v1.3
// - Writes CSV header at the START of EVERY segment (fixes missing headers in saved CSVs)
// - Keeps LED indicator for active logging
// - Emits logger metadata for Python to capture (FS, MODE_CODE)
// LED ON  = segment in progress
// LED OFF = idle

#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BNO055.h>
#include <utility/imumaths.h>
#include <Adafruit_BMP3XX.h>
#include <math.h>

// -------------------- HARDWARE --------------------
#define BUTTON_PIN 9
#define LED_PIN LED_BUILTIN
#define BMP3_I2C_ADDR 0x77

// -------------------- MODES -----------------------
#define MODE_TRAIN   3
#define MODE_SUBWAY  4
#define MODE_BUS     5
#define MODE_CAR     6
#define MODE_BIKE    7
#define MODE_WALK    8

// *** CHANGE THIS LINE ONLY PER MODE BUILD ***
uint8_t current_mode = MODE_CAR;

// -------------------- SENSORS ---------------------
Adafruit_BNO055 bno = Adafruit_BNO055(55, 0x28);
Adafruit_BMP3XX bmp;

// -------------------- CONSTANTS -------------------
const float SEA_LEVEL_HPA = 1013.25f;

// ---- ACCEL BIASES (m/s^2) ----
const float ACC_BIAS_X = -0.1926f;
const float ACC_BIAS_Y = -0.1975f;
const float ACC_BIAS_Z = -0.3472f;

// -------------------- TIMING ----------------------
const uint32_t FS_HZ = 25;
const uint32_t PERIOD_US = 1000000UL / FS_HZ;
uint32_t nextTickUs = 0;

// -------------------- STATE -----------------------
bool logging_active = false;
uint32_t segment_id = 0;
uint32_t segment_start_ms = 0;
bool last_button_state = HIGH;

// -------------------- HELPERS ---------------------
const char* modeName(uint8_t code) {
  switch (code) {
    case MODE_TRAIN:  return "train";
    case MODE_SUBWAY: return "subway";
    case MODE_BUS:    return "bus";
    case MODE_CAR:    return "car";
    case MODE_BIKE:   return "bike";
    case MODE_WALK:   return "walk";
    default:          return "unknown";
  }
}

void printCsvHeader() {
  Serial.println(
    "t_ms,"
    "ax_raw,ay_raw,az_raw,"
    "ax_corr,ay_corr,az_corr,"
    "acc_mag_corr,"
    "gx,gy,gz,"
    "pressure_hpa,temp_C,alt_m"
  );
}

void waitForButtonRelease() {
  while (digitalRead(BUTTON_PIN) == LOW) {
    delay(5);
  }
}

void setup() {
  Serial.begin(115200);
  unsigned long t0 = millis();
  while (!Serial && millis() - t0 < 2000) {}

  pinMode(BUTTON_PIN, INPUT_PULLUP);
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);

  Wire.begin();
  Wire.setClock(400000);

  if (!bno.begin()) {
    Serial.println("#ERROR:BNO055_NOT_FOUND");
    while (1) delay(10);
  }
  delay(1000);
  bno.setExtCrystalUse(true);

  if (!bmp.begin_I2C(BMP3_I2C_ADDR)) {
    Serial.println("#ERROR:BMP3XX_NOT_FOUND");
    while (1) delay(10);
  }

  bmp.setTemperatureOversampling(BMP3_OVERSAMPLING_2X);
  bmp.setPressureOversampling(BMP3_OVERSAMPLING_8X);
  bmp.setIIRFilterCoeff(BMP3_IIR_FILTER_COEFF_3);
  bmp.setOutputDataRate(BMP3_ODR_50_HZ);

  // Boot metadata (Python can parse this once)
  Serial.print("#START_LOGGER,FS_HZ=");
  Serial.print(FS_HZ);
  Serial.print(",MODE_CODE=");
  Serial.print(current_mode);
  Serial.print(",MODE_NAME=");
  Serial.println(modeName(current_mode));

  Serial.println("#READY_PRESS_BUTTON");

  nextTickUs = micros();
}

void loop() {
  // ---------------- BUTTON EDGE DETECT ----------------
  bool button_state = digitalRead(BUTTON_PIN);

  if (last_button_state == HIGH && button_state == LOW) {
    waitForButtonRelease();

    if (!logging_active) {
      // ---- START SEGMENT ----
      logging_active = true;
      digitalWrite(LED_PIN, HIGH);

      segment_id++;
      segment_start_ms = millis();

      Serial.print("#SEGMENT_START,");
      Serial.print(segment_id);
      Serial.print(",");
      Serial.print(current_mode);
      Serial.print(",");
      Serial.println(segment_start_ms);

      // >>> CRITICAL FIX: print CSV header at segment start
      printCsvHeader();

    } else {
      // ---- END SEGMENT ----
      logging_active = false;
      digitalWrite(LED_PIN, LOW);

      Serial.print("#SEGMENT_END,");
      Serial.print(segment_id);
      Serial.print(",");
      Serial.println(millis());
    }
  }
  last_button_state = button_state;

  // ---------------- SAMPLE CLOCK ----------------
  uint32_t nowUs = micros();
  if ((int32_t)(nowUs - nextTickUs) < 0) return;
  nextTickUs += PERIOD_US;

  if (!logging_active) return;

  uint32_t nowMs = millis();

  // ---------------- BNO055 ----------------
  imu::Vector<3> aRaw = bno.getVector(Adafruit_BNO055::VECTOR_ACCELEROMETER);
  float axr = aRaw.x();
  float ayr = aRaw.y();
  float azr = aRaw.z();

  float ax = axr - ACC_BIAS_X;
  float ay = ayr - ACC_BIAS_Y;
  float az = azr - ACC_BIAS_Z;
  float amag = sqrtf(ax*ax + ay*ay + az*az);

  imu::Vector<3> gRaw = bno.getVector(Adafruit_BNO055::VECTOR_GYROSCOPE);
  float gx = gRaw.x();
  float gy = gRaw.y();
  float gz = gRaw.z();

  // ---------------- BMP3XX ----------------
  float pressure_hpa = NAN, temp_C = NAN, alt_m = NAN;
  if (bmp.performReading()) {
    pressure_hpa = bmp.pressure / 100.0f;
    temp_C = bmp.temperature;
    alt_m = bmp.readAltitude(SEA_LEVEL_HPA);
  }

  // ---------------- CSV OUTPUT ----------------
  Serial.print(nowMs); Serial.print(",");
  Serial.print(axr,4); Serial.print(",");
  Serial.print(ayr,4); Serial.print(",");
  Serial.print(azr,4); Serial.print(",");
  Serial.print(ax,4);  Serial.print(",");
  Serial.print(ay,4);  Serial.print(",");
  Serial.print(az,4);  Serial.print(",");
  Serial.print(amag,4);Serial.print(",");
  Serial.print(gx,4);  Serial.print(",");
  Serial.print(gy,4);  Serial.print(",");
  Serial.print(gz,4);  Serial.print(",");
  Serial.print(pressure_hpa,2); Serial.print(",");
  Serial.print(temp_C,2);       Serial.print(",");
  Serial.println(alt_m,2);
}
