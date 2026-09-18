// beacon_inference.ino
// ECE499 Capstone — Edge CO₂ Beacon
// Spencer Aldrich | Union College | 2026

#define DEBUG_RAW 0

#include <bluefruit.h>
#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BNO055.h>
#include <utility/imumaths.h>
// #include <Adafruit_BMP3XX.h> 
#include <math.h>
#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "model_data.h"
#include "norm_stats.h"

// Hardware 
#define LED_PIN   LED_BUILTIN

Adafruit_BNO055 bno = Adafruit_BNO055(55, 0x28);

// Accelerometer bias (m/s²) — measured from logging firmware 
static const float ACC_BIAS_X = -0.1926f;
static const float ACC_BIAS_Y = -0.1975f;
static const float ACC_BIAS_Z = -0.3472f;

// Sampling 
static const uint32_t FS_HZ     = 25;
static const uint32_t PERIOD_US = 1000000UL / FS_HZ;  // 40 000 µs
static uint32_t nextTickUs      = 0;

// Window parameters 
static const int WIN_N      = 200;  // 8 s × 25 Hz
static const int STEP_N     = 100;  // 50% overlap → inference every 4 s

// INT8 quantization params 
static const float QUANT_SCALE      = 0.04095628f;
static const int   QUANT_ZERO_POINT = -56;

// Mode labels — must match model output layer order
static const int   N_MODES = 5;
static const char* MODE_NAMES[N_MODES] = {
    "train", "subway", "car", "bus", "walk"
};

// Speed model (m/s) used for distance accumulation
static const float SPEED_MPS[N_MODES] = {
    22.2f,  // 0: train  (80 km/h)
    8.3f,   // 1: subway (30 km/h)
    11.1f,  // 2: car    (40 km/h)
    5.6f,   // 3: bus    (20 km/h)
    1.4f    // 4: walk   ( 5 km/h)
};

// CO₂ emission factors (kg/km) — EPA eGRID 2023 
static const float CO2_KG_PER_KM[N_MODES] = {
    0.035f,  // 0: train
    0.041f,  // 1: subway
    0.089f,  // 2: car
    0.089f,  // 3: bus
    0.000f   // 4: walk
};

// Seconds of real time each inference step represents
static const float INTERVAL_S = (float)STEP_N / (float)FS_HZ;  // 4.0 s

// Trip accumulators
static float    trip_distance_m = 0.0f;
static float    trip_co2_g      = 0.0f;
static int      prev_mode_id    = -1;
static int      walk_streak     = 0;
static bool     trip_active     = false;

// BLE GATT
#define CO2_SVC_UUID  "4fafc201-1fb5-459e-8fcc-c5c9c3319100"
#define LIVE_CHR_UUID "4fafc201-1fb5-459e-8fcc-c5c9c3319101"
#define TRIP_CHR_UUID "4fafc201-1fb5-459e-8fcc-c5c9c3319102"

BLEService        co2_svc(CO2_SVC_UUID);
BLECharacteristic live_status_chr(LIVE_CHR_UUID);
BLECharacteristic trip_record_chr(TRIP_CHR_UUID);

static uint16_t trip_id            = 0;
static uint32_t trip_ts_start      = 0;
static uint8_t  trip_mode_id       = 0;
static uint8_t  cur_mode_id        = 0;
static uint8_t  cur_confidence     = 0;
static uint32_t ble_last_notify_ms = 0;

// Sliding window ring buffer
static float ring[WIN_N][N_CHANNELS];
static int   samples_in_buf      = 0;
static int   samples_since_infer = 0;

// TFLite Micro 
static const int ARENA_BYTES = 130 * 1024;
static uint8_t  tensor_arena[ARENA_BYTES];

static tflite::MicroMutableOpResolver<6> resolver;
static tflite::MicroInterpreter*         interpreter   = nullptr;
static TfLiteTensor*                     input_tensor  = nullptr;
static TfLiteTensor*                     output_tensor = nullptr;

// Forward declarations
static bool setupTFLite();
static void pushSample(float ax, float ay, float az,
                       float gx, float gy, float gz);
static void runInference();
static void setupBLE();
static void startAdv();
static void sendLiveStatus();
static void notifyTripRecord(uint32_t ts_end);
static void resetTrip();

// setup
void setup() {
    Serial.begin(115200);
    unsigned long t0 = millis();
    while (!Serial && millis() - t0 < 2000) {}

    pinMode(LED_PIN, OUTPUT);
    digitalWrite(LED_PIN, LOW);

    Wire.begin();
    Wire.setClock(400000);

    // BNO055
    if (!bno.begin()) {
        Serial.println("#ERROR:BNO055_NOT_FOUND");
        while (1) delay(10);
    }
    delay(1000);
    bno.setExtCrystalUse(true);
    Serial.println("#OK:BNO055");

    // TFLite Micro 
    if (!setupTFLite()) {
        Serial.println("#ERROR:TFLITE_INIT_FAILED");
        while (1) delay(10);
    }
    Serial.print("#OK:TFLITE  arena_used=");
    Serial.print(interpreter->arena_used_bytes());
    Serial.println("B");

    // BLE 
    setupBLE();
    startAdv();
    Serial.println("#OK:BLE  advertising as CO2-Beacon");

    // Boot summary
    Serial.print("#CONFIG  WIN=");  Serial.print(WIN_N);
    Serial.print(" STEP=");         Serial.print(STEP_N);
    Serial.print(" FS=");           Serial.print(FS_HZ);
    Serial.print("Hz  filling buffer (");
    Serial.print(WIN_N / FS_HZ);
    Serial.println("s)...");

#if DEBUG_RAW
    Serial.println("#DEBUG_RAW ON — raw sensor + softmax printed each inference");
#endif

    digitalWrite(LED_PIN, HIGH);
    nextTickUs = micros();
}
// loop
void loop() {
    // 1 Hz BLE LiveStatus
    uint32_t now_ms = millis();
    if (now_ms - ble_last_notify_ms >= 1000UL) {
        ble_last_notify_ms = now_ms;
        sendLiveStatus();
    }

    // 25 Hz tick
    uint32_t nowUs = micros();
    if ((int32_t)(nowUs - nextTickUs) < 0) return;
    nextTickUs += PERIOD_US;

    // Read sensors
    imu::Vector<3> aRaw = bno.getVector(Adafruit_BNO055::VECTOR_ACCELEROMETER);
    float ax = aRaw.x() - ACC_BIAS_X;
    float ay = aRaw.y() - ACC_BIAS_Y;
    float az = aRaw.z() - ACC_BIAS_Z;

    imu::Vector<3> gRaw = bno.getVector(Adafruit_BNO055::VECTOR_GYROSCOPE);
    float gx = gRaw.x();
    float gy = gRaw.y();
    float gz = gRaw.z();

    pushSample(ax, ay, az, gx, gy, gz);
}

// pushSample — maintains sliding window ring buffer, fires inference every STEP_N
static void pushSample(float ax, float ay, float az,
                       float gx, float gy, float gz) {
    // initial fill
    if (samples_in_buf < WIN_N) {
        ring[samples_in_buf][0] = ax;
        ring[samples_in_buf][1] = ay;
        ring[samples_in_buf][2] = az;
        ring[samples_in_buf][3] = gx;
        ring[samples_in_buf][4] = gy;
        ring[samples_in_buf][5] = gz;
        samples_in_buf++;
        if (samples_in_buf == WIN_N) {
            runInference();
            samples_since_infer = 0;
        }
        return;
    }

    // sliding — shift left by one sample, append new
    memmove(&ring[0][0], &ring[1][0],
            sizeof(float) * (WIN_N - 1) * N_CHANNELS);
    ring[WIN_N - 1][0] = ax;
    ring[WIN_N - 1][1] = ay;
    ring[WIN_N - 1][2] = az;
    ring[WIN_N - 1][3] = gx;
    ring[WIN_N - 1][4] = gy;
    ring[WIN_N - 1][5] = gz;

    samples_since_infer++;
    if (samples_since_infer >= STEP_N) {
        runInference();
        samples_since_infer = 0;
    }
}

// runInference — normalize → quantize → invoke → dequantize → accumulate
static void runInference() {
    uint32_t t_start = micros();

    // Z-score normalize + INT8 quantize into TFLite input tensor
    int8_t* inp = input_tensor->data.int8;
    for (int t = 0; t < WIN_N; t++) {
        for (int c = 0; c < N_CHANNELS; c++) {
            float x_norm = (ring[t][c] - NORM_MU[c]) / NORM_SIGMA[c];
            float q_f    = x_norm / QUANT_SCALE + (float)QUANT_ZERO_POINT;
            int   q      = (int)roundf(q_f);
            if (q < -128) q = -128;
            if (q >  127) q =  127;
            inp[t * N_CHANNELS + c] = (int8_t)q;
        }
    }

    // Run the model
    TfLiteStatus status = interpreter->Invoke();
    uint32_t latency_us = micros() - t_start;

    if (status != kTfLiteOk) {
        Serial.println("#ERROR:INVOKE_FAILED");
        return;
    }

    // Dequantize output softmax
    int8_t* out_q     = output_tensor->data.int8;
    float   out_scale = output_tensor->params.scale;
    int     out_zp    = output_tensor->params.zero_point;

    int   best_id   = 0;
    float best_prob = -1.0f;
    float all_probs[N_MODES];
    for (int i = 0; i < N_MODES; i++) {
        float prob   = ((float)out_q[i] - (float)out_zp) * out_scale;
        all_probs[i] = prob;
        if (prob > best_prob) { best_prob = prob; best_id = i; }
    }

    uint8_t  confidence = (uint8_t)constrain((int)(best_prob * 100.0f + 0.5f), 0, 100);
    uint32_t lat_ms     = latency_us / 1000;
    uint32_t lat_frac   = (latency_us % 1000) / 10;

    if (best_id == 1) best_id = 2;

    cur_mode_id    = (uint8_t)best_id;
    cur_confidence = confidence;

#if DEBUG_RAW
    Serial.print("#RAW  AX:"); Serial.print(ring[WIN_N-1][0], 3);
    Serial.print(" AY:");      Serial.print(ring[WIN_N-1][1], 3);
    Serial.print(" AZ:");      Serial.print(ring[WIN_N-1][2], 3);
    Serial.print(" GX:");      Serial.print(ring[WIN_N-1][3], 3);
    Serial.print(" GY:");      Serial.print(ring[WIN_N-1][4], 3);
    Serial.print(" GZ:");      Serial.println(ring[WIN_N-1][5], 3);
    Serial.print("#SOFTMAX  ");
    for (int i = 0; i < N_MODES; i++) {
        Serial.print(MODE_NAMES[i]); Serial.print(":");
        Serial.print((int)(all_probs[i] * 100.0f + 0.5f)); Serial.print("%  ");
    }
    Serial.println();
    Serial.print("MODE:"); Serial.print(MODE_NAMES[best_id]);
    Serial.print("  CONF:"); Serial.print(confidence);
    Serial.print(lat_frac); Serial.println("ms");
    return;  
#endif

    // Trip accumulation
    if (!trip_active) {
        trip_ts_start = millis() / 1000;
        trip_mode_id  = (uint8_t)best_id;
        trip_active   = true;
    }
    if (best_id != 4) trip_mode_id = (uint8_t)best_id;

    const float delta_m = SPEED_MPS[best_id] * INTERVAL_S;
    trip_distance_m += delta_m;
    trip_co2_g      += delta_m * CO2_KG_PER_KM[best_id];

    // Trip boundary detection
    bool trip_end = false;
    if (best_id == 4 && confidence >= 60) walk_streak++;
    else walk_streak = 0;
    if (walk_streak >= 5) { trip_end = true; walk_streak = 0; }

    if (trip_end && trip_distance_m > 0.0f) {
        uint32_t ts_end = millis() / 1000;
        Serial.print("TRIP_END  ID:"); Serial.print(trip_id);
        Serial.print("  DIST:");       Serial.print(trip_distance_m, 1);
        Serial.print("m  CO2:");       Serial.print(trip_co2_g, 1);
        Serial.println("g");
        notifyTripRecord(ts_end);
        trip_id++;
        resetTrip();
    }
    prev_mode_id = best_id;

    // Serial output — one line per inference
    Serial.print("MODE:"); Serial.print(MODE_NAMES[best_id]);
    Serial.print("  CONF:"); Serial.print(confidence);
    if (lat_frac < 10) Serial.print("0");
    Serial.print(lat_frac);
    Serial.print("ms  DIST:"); Serial.print(trip_distance_m, 1);
    Serial.print("m  CO2:");   Serial.print(trip_co2_g, 1);
    Serial.println("g");
}

// resetTrip — clear all trip accumulators after TRIP_END
static void resetTrip() {
    trip_distance_m = 0.0f;
    trip_co2_g      = 0.0f;
    trip_active     = false;
    trip_mode_id    = 0;
    trip_ts_start   = 0;
}

// setupTFLite — register ops, load model, allocate interpreter (all static)
static bool setupTFLite() {
    resolver.AddConv2D();
    resolver.AddMaxPool2D();
    resolver.AddFullyConnected();
    resolver.AddSoftmax();
    resolver.AddReshape();
    resolver.AddQuantize();

    const tflite::Model* model = tflite::GetModel(g_model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.print("#ERROR:SCHEMA_MISMATCH  got=");
        Serial.print(model->version()); Serial.print(" want=");
        Serial.println(TFLITE_SCHEMA_VERSION);
        return false;
    }

    static tflite::MicroInterpreter static_interp(
        model, resolver, tensor_arena, ARENA_BYTES);
    interpreter = &static_interp;

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("#ERROR:ALLOCATE_TENSORS_FAILED");
        return false;
    }

    input_tensor  = interpreter->input(0);
    output_tensor = interpreter->output(0);

    if (input_tensor->dims->size != 4 ||
        input_tensor->dims->data[1] != WIN_N ||
        input_tensor->dims->data[2] != N_CHANNELS) {
        Serial.print("#WARN:INPUT_SHAPE_UNEXPECTED  dims=");
        Serial.println(input_tensor->dims->size);
    }
    if (output_tensor->dims->data[output_tensor->dims->size - 1] != N_MODES) {
        Serial.print("#WARN:OUTPUT_SHAPE_UNEXPECTED  n=");
        Serial.println(output_tensor->dims->data[output_tensor->dims->size - 1]);
    }
    return true;
}

// BLE callbacks
static void ble_connect_callback(uint16_t conn_handle) {
    BLEConnection* conn = Bluefruit.Connection(conn_handle);
    char name[32] = {0};
    conn->getPeerName(name, sizeof(name));
    Serial.print("#BLE:CONNECTED  peer="); Serial.println(name);
}

static void ble_disconnect_callback(uint16_t conn_handle, uint8_t reason) {
    (void)conn_handle;
    Serial.print("#BLE:DISCONNECTED  reason=0x"); Serial.println(reason, HEX);
}

// setupBLE
static void setupBLE() {
    Bluefruit.begin();
    Bluefruit.setName("CO2-Beacon");
    Bluefruit.Periph.setConnectCallback(ble_connect_callback);
    Bluefruit.Periph.setDisconnectCallback(ble_disconnect_callback);

    co2_svc.begin();

    live_status_chr.setProperties(CHR_PROPS_NOTIFY);
    live_status_chr.setPermission(SECMODE_OPEN, SECMODE_NO_ACCESS);
    live_status_chr.setFixedLen(7);
    live_status_chr.begin();

    trip_record_chr.setProperties(CHR_PROPS_READ | CHR_PROPS_NOTIFY);
    trip_record_chr.setPermission(SECMODE_OPEN, SECMODE_NO_ACCESS);
    trip_record_chr.setFixedLen(20);
    uint8_t zeroes[20] = {0};
    trip_record_chr.begin();
    trip_record_chr.write(zeroes, 20);
}

// startAdv
static void startAdv() {
    Bluefruit.Advertising.addFlags(BLE_GAP_ADV_FLAGS_LE_ONLY_GENERAL_DISC_MODE);
    Bluefruit.Advertising.addTxPower();
    Bluefruit.Advertising.addService(co2_svc);
    Bluefruit.Advertising.addName();
    Bluefruit.Advertising.restartOnDisconnect(true);
    Bluefruit.Advertising.setInterval(160, 244);
    Bluefruit.Advertising.setFastTimeout(30);
    Bluefruit.Advertising.start(0);
}

// sendLiveStatus — 7-byte BLE notify at 1 Hz
// Payload: [mode_id, confidence, ts_s×4, trip_active]
static void sendLiveStatus() {
    if (!Bluefruit.connected()) return;

    uint32_t ts = millis() / 1000;
    uint8_t payload[7];
    payload[0] = cur_mode_id;
    payload[1] = cur_confidence;
    payload[2] = (uint8_t)(ts & 0xFF);
    payload[3] = (uint8_t)((ts >>  8) & 0xFF);
    payload[4] = (uint8_t)((ts >> 16) & 0xFF);
    payload[5] = (uint8_t)((ts >> 24) & 0xFF);
    payload[6] = trip_active ? 1 : 0;

    live_status_chr.notify(payload, 7);
}

// notifyTripRecord — 20-byte BLE notify on trip boundary
static void notifyTripRecord(uint32_t ts_end) {
    uint16_t dur_s  = (uint16_t)constrain((int32_t)(ts_end - trip_ts_start), 0, 65535);
    uint16_t dist_u = (uint16_t)constrain((int)trip_distance_m, 0, 65535);
    uint16_t co2_u  = (uint16_t)constrain((int)trip_co2_g,      0, 65535);

    uint8_t rec[20] = {0};
    rec[0]  = (uint8_t)(trip_id & 0xFF);
    rec[1]  = (uint8_t)(trip_id >> 8);
    rec[2]  = trip_mode_id;
    rec[3]  = cur_confidence;
    rec[4]  = (uint8_t)(trip_ts_start & 0xFF);
    rec[5]  = (uint8_t)(trip_ts_start >>  8);
    rec[6]  = (uint8_t)(trip_ts_start >> 16);
    rec[7]  = (uint8_t)(trip_ts_start >> 24);
    rec[8]  = (uint8_t)(ts_end & 0xFF);
    rec[9]  = (uint8_t)(ts_end >>  8);
    rec[10] = (uint8_t)(ts_end >> 16);
    rec[11] = (uint8_t)(ts_end >> 24);
    rec[12] = (uint8_t)(dur_s & 0xFF);
    rec[13] = (uint8_t)(dur_s >> 8);
    rec[14] = (uint8_t)(dist_u & 0xFF);
    rec[15] = (uint8_t)(dist_u >> 8);
    rec[16] = (uint8_t)(co2_u & 0xFF);
    rec[17] = (uint8_t)(co2_u >> 8);

    trip_record_chr.write(rec, 20);
    if (Bluefruit.connected()) {
        trip_record_chr.notify(rec, 20);
    }
}
