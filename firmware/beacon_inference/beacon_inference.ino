// beacon_inference.ino
// Issues #3 (25 Hz sampling), #4 (preprocessing), #5 (TFLite inference), #6 (latency)
//
// Hardware confirmed from LOGGER_BUTTON_v1.3:
//   BNO055 I2C 0x28 | BMP390 I2C 0x77 | 400 kHz bus
//
// Inference pipeline:
//   Sample 25 Hz → ring buffer (200 × 7) → z-score normalize → INT8 quantize
//   → TFLite Micro → argmax → Serial print (mode, confidence, latency)

#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BNO055.h>
#include <utility/imumaths.h>
#include <Adafruit_BMP3XX.h>
#include <math.h>

#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "model_data.h"   // g_model_data[], g_model_data_len  (xxd output)
#include "norm_stats.h"   // NORM_MU[7], NORM_SIGMA[7]

// ── Hardware ─────────────────────────────────────────────────────────────────
#define LED_PIN       LED_BUILTIN
#define BMP3_ADDR     0x77    // confirmed: matches LOGGER_BUTTON_v1.3

Adafruit_BNO055 bno = Adafruit_BNO055(55, 0x28);
Adafruit_BMP3XX bmp;

// ── Accelerometer bias (m/s²) — from logging firmware / docs/API_INTEGRATIONS.md
static const float ACC_BIAS_X = -0.1926f;
static const float ACC_BIAS_Y = -0.1975f;
static const float ACC_BIAS_Z = -0.3472f;

// ── Sampling ──────────────────────────────────────────────────────────────────
static const uint32_t FS_HZ     = 25;
static const uint32_t PERIOD_US = 1000000UL / FS_HZ;   // 40 000 µs
static uint32_t nextTickUs      = 0;

// ── Window parameters (must match Python windowing in window_build.py) ────────
static const int WIN_N      = 200;   // 8 s × 25 Hz
static const int STEP_N     = 100;   // 50% overlap → inference every 4 s
static const int N_CHANNELS = 7;     // ax, ay, az, gx, gy, gz, pressure_hpa

// ── INT8 quantization params (confirmed from convert_to_tflite.py output) ─────
static const float QUANT_SCALE      = 0.5251754522f;
static const int   QUANT_ZERO_POINT = 113;

// ── Mode labels — order must match model output layer ─────────────────────────
static const int   N_MODES = 5;
static const char* MODE_NAMES[N_MODES] = {
    "train", "subway", "car", "bus", "walk"
};

// ── Issue #10: speed model (m/s) — order matches MODE_NAMES ──────────────────
// walk=1.4, car=11.1 (40km/h urban avg), bus=8.3 (30km/h), train=16.7 (60km/h), subway=13.9 (50km/h)
static const float SPEED_MPS[N_MODES] = {
    16.7f,  // 0: train
    13.9f,  // 1: subway
    11.1f,  // 2: car
     8.3f,  // 3: bus
     1.4f   // 4: walk
};

// ── Issue #11: CO₂ emission factors (kg/km) — EPA eGRID 2023 ─────────────────
static const float CO2_KG_PER_KM[N_MODES] = {
    0.035f,  // 0: train
    0.041f,  // 1: subway
    0.089f,  // 2: car
    0.089f,  // 3: bus
    0.000f   // 4: walk
};

// Each inference covers STEP_N samples at FS_HZ = 4.0 seconds of travel
static const float INTERVAL_S = (float)STEP_N / (float)FS_HZ;   // 4.0 s

// ── Trip accumulators (Issues #10, #11) ───────────────────────────────────────
static float trip_distance_m = 0.0f;
static float trip_co2_g      = 0.0f;
static int   prev_mode_id    = -1;   // -1 = no prior inference
static int   walk_streak     = 0;    // consecutive high-conf walk inferences

// ── Sliding window ring buffer ────────────────────────────────────────────────
// ring[t][c]: t=0 is the oldest sample, t=WIN_N-1 is the newest.
// New samples are appended at ring[WIN_N-1] after shifting left by one row.
// This gives a contiguous, time-ordered buffer ready for normalization.
static float ring[WIN_N][N_CHANNELS];
static int   samples_in_buf      = 0;   // fills up to WIN_N on first pass
static int   samples_since_infer = 0;   // resets to 0 after each inference

// ── TFLite Micro ──────────────────────────────────────────────────────────────
// 50 KB static arena — no heap allocation anywhere in this file.
static const int ARENA_BYTES = 50 * 1024;
static uint8_t  tensor_arena[ARENA_BYTES];

static tflite::MicroMutableOpResolver<6> resolver;
static tflite::MicroInterpreter*         interpreter   = nullptr;
static TfLiteTensor*                     input_tensor  = nullptr;
static TfLiteTensor*                     output_tensor = nullptr;

// ── Forward declarations ──────────────────────────────────────────────────────
static bool setupTFLite();
static void pushSample(float ax, float ay, float az,
                       float gx, float gy, float gz, float pres);
static void runInference();

// =============================================================================
void setup() {
    Serial.begin(115200);
    unsigned long t0 = millis();
    while (!Serial && millis() - t0 < 2000) {}

    pinMode(LED_PIN, OUTPUT);
    digitalWrite(LED_PIN, LOW);

    // ── I²C — 400 kHz fast mode ───────────────────────────────────────────
    Wire.begin();
    Wire.setClock(400000);

    // ── BNO055 ────────────────────────────────────────────────────────────
    if (!bno.begin()) {
        Serial.println("#ERROR:BNO055_NOT_FOUND");
        while (1) delay(10);
    }
    delay(1000);                    // BNO055 datasheet: wait after begin()
    bno.setExtCrystalUse(true);     // use external crystal for accuracy
    Serial.println("#OK:BNO055");

    // ── BMP390 ────────────────────────────────────────────────────────────
    if (!bmp.begin_I2C(BMP3_ADDR)) {
        Serial.println("#ERROR:BMP390_NOT_FOUND");
        while (1) delay(10);
    }
    bmp.setTemperatureOversampling(BMP3_OVERSAMPLING_2X);
    bmp.setPressureOversampling(BMP3_OVERSAMPLING_8X);
    bmp.setIIRFilterCoeff(BMP3_IIR_FILTER_COEFF_3);
    bmp.setOutputDataRate(BMP3_ODR_50_HZ);
    Serial.println("#OK:BMP390");

    // ── TFLite Micro ──────────────────────────────────────────────────────
    if (!setupTFLite()) {
        Serial.println("#ERROR:TFLITE_INIT_FAILED");
        while (1) delay(10);
    }
    Serial.print("#OK:TFLITE  arena_used=");
    Serial.print(interpreter->arena_used_bytes());
    Serial.println("B");

    // ── Boot summary ──────────────────────────────────────────────────────
    Serial.print("#CONFIG  WIN=");   Serial.print(WIN_N);
    Serial.print(" STEP=");          Serial.print(STEP_N);
    Serial.print(" FS=");            Serial.print(FS_HZ);
    Serial.print("Hz  filling buffer (");
    Serial.print(WIN_N / FS_HZ);
    Serial.println("s)...");

    digitalWrite(LED_PIN, HIGH);
    nextTickUs = micros();
}

// =============================================================================
void loop() {
    // ── Exact 25 Hz tick via micros() — no delay() ───────────────────────
    uint32_t nowUs = micros();
    if ((int32_t)(nowUs - nextTickUs) < 0) return;
    nextTickUs += PERIOD_US;

    // ── BNO055: bias-corrected accelerometer + gyroscope ─────────────────
    imu::Vector<3> aRaw = bno.getVector(Adafruit_BNO055::VECTOR_ACCELEROMETER);
    float ax = aRaw.x() - ACC_BIAS_X;
    float ay = aRaw.y() - ACC_BIAS_Y;
    float az = aRaw.z() - ACC_BIAS_Z;

    imu::Vector<3> gRaw = bno.getVector(Adafruit_BNO055::VECTOR_GYROSCOPE);
    float gx = gRaw.x();
    float gy = gRaw.y();
    float gz = gRaw.z();

    // ── BMP390: pressure — fall back to training mean if read fails ───────
    float pressure_hpa = NORM_MU[6];   // 1014.45 hPa — won't skew normalize
    if (bmp.performReading()) {
        pressure_hpa = bmp.pressure / 100.0f;
    }

    pushSample(ax, ay, az, gx, gy, gz, pressure_hpa);
}

// =============================================================================
// pushSample — maintain sliding window, trigger inference every STEP_N samples
// =============================================================================
static void pushSample(float ax, float ay, float az,
                       float gx, float gy, float gz, float pres) {
    if (samples_in_buf < WIN_N) {
        // Phase 1: filling the buffer for the first time
        ring[samples_in_buf][0] = ax;
        ring[samples_in_buf][1] = ay;
        ring[samples_in_buf][2] = az;
        ring[samples_in_buf][3] = gx;
        ring[samples_in_buf][4] = gy;
        ring[samples_in_buf][5] = gz;
        ring[samples_in_buf][6] = pres;
        samples_in_buf++;

        if (samples_in_buf == WIN_N) {
            runInference();
            samples_since_infer = 0;
        }
        return;
    }

    // Phase 2: sliding window — shift left by one row, append at tail
    // (WIN_N-1) rows × N_CHANNELS floats = 1393 floats = 5572 bytes
    // On Cortex-M4 @ 64 MHz this takes ~5 µs, well within 40 ms tick budget.
    memmove(&ring[0][0], &ring[1][0],
            sizeof(float) * (WIN_N - 1) * N_CHANNELS);
    ring[WIN_N - 1][0] = ax;
    ring[WIN_N - 1][1] = ay;
    ring[WIN_N - 1][2] = az;
    ring[WIN_N - 1][3] = gx;
    ring[WIN_N - 1][4] = gy;
    ring[WIN_N - 1][5] = gz;
    ring[WIN_N - 1][6] = pres;

    samples_since_infer++;
    if (samples_since_infer >= STEP_N) {
        runInference();
        samples_since_infer = 0;
    }
}

// =============================================================================
// runInference — normalize → INT8 quantize → Invoke → dequantize → print
// =============================================================================
static void runInference() {
    uint32_t t_start = micros();

    // ── 1. Z-score normalize + INT8 quantize into input tensor ───────────
    // Input tensor shape: (1, 200, 7, 1) — flat index = t*N_CHANNELS + c
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

    // ── 2. Run inference ──────────────────────────────────────────────────
    TfLiteStatus status = interpreter->Invoke();
    uint32_t latency_us = micros() - t_start;

    if (status != kTfLiteOk) {
        Serial.println("#ERROR:INVOKE_FAILED");
        return;
    }

    // ── 3. Dequantize output softmax → find argmax ────────────────────────
    // Output tensor shape: (1, 5)
    int8_t* out_q     = output_tensor->data.int8;
    float   out_scale = output_tensor->params.scale;
    int     out_zp    = output_tensor->params.zero_point;

    int   best_id   = 0;
    float best_prob = -1.0f;
    for (int i = 0; i < N_MODES; i++) {
        float prob = ((float)out_q[i] - (float)out_zp) * out_scale;
        if (prob > best_prob) {
            best_prob = prob;
            best_id   = i;
        }
    }

    uint8_t confidence = (uint8_t)constrain((int)(best_prob * 100.0f + 0.5f), 0, 100);
    uint32_t lat_ms    = latency_us / 1000;
    uint32_t lat_frac  = (latency_us % 1000) / 10;   // tenths of ms

    // ── 4. Issue #10: distance accumulation ──────────────────────────────
    // Each step covers INTERVAL_S seconds at the mode's average speed.
    const float delta_m = SPEED_MPS[best_id] * INTERVAL_S;
    trip_distance_m += delta_m;

    // ── 5. Issue #11: CO₂ accumulation ───────────────────────────────────
    // co2_g += (delta_m / 1000) * factor_kg_per_km * 1000  = delta_m * factor
    trip_co2_g += delta_m * CO2_KG_PER_KM[best_id];

    // ── 6. Trip boundary detection — reset accumulators on trip end ──────
    // Condition A: motorised → walk (arrived at destination)
    bool trip_end = (prev_mode_id >= 0 && prev_mode_id != 4 && best_id == 4);

    // Condition B: sustained walk ≥ 60 s at conf ≥ 80% (new walking segment)
    if (best_id == 4 && confidence >= 80) {
        walk_streak++;
    } else {
        walk_streak = 0;
    }
    if (walk_streak >= 15) {   // 15 × 4 s = 60 s
        trip_end = true;
        walk_streak = 0;
    }

    if (trip_end && trip_distance_m > 0.0f) {
        Serial.print("TRIP_END  DIST:");
        Serial.print(trip_distance_m, 1);
        Serial.print("m  CO2:");
        Serial.print(trip_co2_g, 1);
        Serial.println("g");
        trip_distance_m = 0.0f;
        trip_co2_g      = 0.0f;
    }
    prev_mode_id = best_id;

    // ── 7. Serial output — running totals ─────────────────────────────────
    Serial.print("MODE:");
    Serial.print(MODE_NAMES[best_id]);
    Serial.print("  CONF:");
    Serial.print(confidence);
    Serial.print("%  LAT:");
    Serial.print(lat_ms);
    Serial.print(".");
    if (lat_frac < 10) Serial.print("0");
    Serial.print(lat_frac);
    Serial.print("ms  DIST:");
    Serial.print(trip_distance_m, 1);
    Serial.print("m  CO2:");
    Serial.print(trip_co2_g, 1);
    Serial.println("g");
}

// =============================================================================
// setupTFLite — register ops, allocate interpreter (all static, no heap)
// =============================================================================
static bool setupTFLite() {
    // Register only the ops this model uses:
    //   Conv2D → MaxPool2D → Conv2D → MaxPool2D → Flatten(Reshape) → Dense → Softmax
    resolver.AddConv2D();
    resolver.AddMaxPool2D();
    resolver.AddFullyConnected();
    resolver.AddSoftmax();
    resolver.AddReshape();
    resolver.AddQuantize();     // INT8 input dequantize node

    const tflite::Model* model = tflite::GetModel(model_co2_beacon_int8_tflite);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.print("#ERROR:SCHEMA_MISMATCH  got=");
        Serial.print(model->version());
        Serial.print(" want=");
        Serial.println(TFLITE_SCHEMA_VERSION);
        return false;
    }

    // Interpreter lives in static storage — zero heap
    static tflite::MicroInterpreter static_interp(
        model, resolver, tensor_arena, ARENA_BYTES);
    interpreter = &static_interp;

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("#ERROR:ALLOCATE_TENSORS_FAILED");
        return false;
    }

    input_tensor  = interpreter->input(0);
    output_tensor = interpreter->output(0);

    // Sanity-check tensor shapes
    // Input expected: (1, 200, 7, 1) = 1400 INT8 elements
    if (input_tensor->dims->size != 4 ||
        input_tensor->dims->data[1] != WIN_N ||
        input_tensor->dims->data[2] != N_CHANNELS) {
        Serial.print("#WARN:INPUT_SHAPE_UNEXPECTED  dims=");
        Serial.println(input_tensor->dims->size);
    }
    // Output expected: (1, 5) = 5 INT8 elements
    if (output_tensor->dims->data[output_tensor->dims->size - 1] != N_MODES) {
        Serial.print("#WARN:OUTPUT_SHAPE_UNEXPECTED  n=");
        Serial.println(output_tensor->dims->data[output_tensor->dims->size - 1]);
    }

    return true;
}
