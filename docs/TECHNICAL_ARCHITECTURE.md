# TECHNICAL_ARCHITECTURE.md — Edge CO₂ Beacon

## System Overview

```
[BNO055 IMU] ──┐
               ├─ I²C (400 kHz) ──► [nRF52840 MCU]
[BMP390 Baro] ─┘                        │
                                         ├─ Inference (TFLite Micro)
                                         ├─ CO₂ Estimation
                                         └─ BLE GATT ──► [Web Dashboard]
```

All classification and emission math runs on-device. The BLE link carries only
summarized trip records — never raw sensor samples.

---

## MCU: Adafruit Feather nRF52840

- CPU: ARM Cortex-M4F @ 64 MHz
- Flash: 1 MB
- RAM: 256 KB
- BLE: nRF52840 SoftDevice S140
- SDK: Adafruit nRF52 Arduino core

### Memory Budget (firmware/beacon_inference)

| Region          | Allocation | Notes                              |
|-----------------|------------|------------------------------------|
| TFLite arena    | 80 KB      | Static array in .bss               |
| Model flatbuffer| ~50 KB     | INT8 quantized, stored in flash    |
| norm_stats.h    | < 1 KB     | 7 mu + 7 sigma float constants     |
| Sensor ring buf | 7 KB       | 125 × 7 × float32 = 3500 B (×2 overlap) |
| BLE GATT bufs   | 2 KB       | TripRecord + LiveStatus            |
| Stack           | 4 KB       | Default nRF52 Arduino stack        |

Total static RAM: ~93 KB — fits in 256 KB with margin.

**Rule:** Zero heap allocation in firmware. All buffers are statically declared.

---

## Sensor Sampling Pipeline

```
25 Hz tick (micros-based)
  │
  ├─ BNO055: getVector(ACCELEROMETER) → ax_corr, ay_corr, az_corr
  ├─ BNO055: getVector(GYROSCOPE)     → gx, gy, gz
  └─ BMP390: performReading()         → pressure_hpa
  │
  ▼
Ring buffer (125 samples × 7 channels)
  │
  ▼ (every 62 new samples — 50% overlap)
Sliding window extraction
  │
  ▼
Z-score normalization (using frozen mu/sigma from norm_stats.h)
  │
  ▼
TFLite Micro inference → softmax[5]
  │
  ▼
Mode ID (argmax) + confidence (max prob × 100)
  │
  ▼
CO₂ Estimator → co2_g (per segment)
  │
  ▼
TripRecord accumulator → BLE notify on trip end
```

### Channel Order (must match model input exactly)

| Index | Channel       | Source  | Units   |
|-------|---------------|---------|---------|
| 0     | ax_corr       | BNO055  | m/s²    |
| 1     | ay_corr       | BNO055  | m/s²    |
| 2     | az_corr       | BNO055  | m/s²    |
| 3     | gx            | BNO055  | rad/s   |
| 4     | gy            | BNO055  | rad/s   |
| 5     | gz            | BNO055  | rad/s   |
| 6     | pressure_hpa  | BMP390  | hPa     |

`ax_corr = ax_raw − ACC_BIAS_X` (bias values burned in during calibration)

---

## ML Model Architecture

```
Input (125, 7, 1)
  │
Conv2D(16, kernel=(9,1), padding='same', activation='relu')
  │
MaxPooling2D(pool_size=(2,1))
  │
Conv2D(32, kernel=(9,1), padding='same', activation='relu')
  │
MaxPooling2D(pool_size=(2,1))
  │
Flatten
  │
Dense(64, activation='relu')
  │
Dropout(0.3)
  │
Dense(5, activation='softmax')
  │
Output: [train, subway, car, bus, walk] probabilities
```

- Pretrained on SHL dataset (coarse 5-class labels)
- Fine-tuned on self-collected data (segment-split v3, leakage-free)
- Converted to INT8 TFLite for MCU deployment

---

## Normalization

Constants computed once from train split, exported to C header:

```c
// norm_stats.h (auto-generated — DO NOT EDIT MANUALLY)
static const float NORM_MU[7]    = { ... };  // per-channel mean
static const float NORM_SIGMA[7] = { ... };  // per-channel std + 1e-8
```

Applied per sample at inference time:
```c
x_norm[t][c] = (x_raw[t][c] - NORM_MU[c]) / NORM_SIGMA[c];
```

---

## BLE Architecture

- Protocol: Bluetooth Low Energy 5.0 (nRF52840 SoftDevice S140)
- Role: Peripheral (beacon advertises, dashboard connects)
- Service UUID: custom 128-bit UUID (defined in API_INTEGRATIONS.md)

### Characteristics

| Name        | UUID suffix | Size    | Access   | Rate       |
|-------------|-------------|---------|----------|------------|
| LiveStatus  | 0x0001      | 7 bytes | Notify   | 1 Hz       |
| TripRecord  | 0x0002      | 20 bytes| Read/Ind | On request |
| Control     | 0x0003      | 1 byte  | Write    | On demand  |

### Advertising

- Interval: 100 ms
- Payload: device name "CO2-Beacon" + service UUID
- Connection interval: 7.5–15 ms (negotiated)

---

## Distance Estimation

Since GPS is test-only and unavailable during normal operation:

- **Walking:** stride length × step count (BNO055 accelerometer peak detection)
- **Transit modes:** speed model × duration (mode-specific average speeds)

Average speeds used:
| Mode   | Speed (km/h) |
|--------|-------------|
| walk   | 5           |
| bus    | 20          |
| car    | 40          |
| train  | 80          |
| subway | 30          |

Distance (km) = speed × duration_s / 3600

---

## CO₂ Estimation

```
co2_kg = distance_km × emission_factor_kg_per_km[mode_id]
co2_g  = co2_kg × 1000
```

Stored in TripRecord as `uint16` (grams, max 65.5 kg — sufficient for any single trip).

Emission factors (EPA eGRID 2023):
| Mode   | kg CO₂/km |
|--------|-----------|
| train  | 0.035     |
| subway | 0.041     |
| car    | 0.089     |
| bus    | 0.089     |
| walk   | 0.000     |

---

## Power Budget

Target: ≥ 24 hours on 1200 mAh Li-Ion battery.

| Component   | Current (est.) | Duty cycle | Avg mA |
|-------------|---------------|------------|--------|
| nRF52840    | 3.5 mA        | 100%       | 3.5    |
| BNO055      | 3.7 mA        | 100%       | 3.7    |
| BMP390      | 0.7 mA        | 100%       | 0.7    |
| BLE radio   | 7 mA          | ~5%        | 0.35   |
| **Total**   |               |            | ~8.25  |

Estimated runtime: 1200 / 8.25 ≈ **145 hours** (theoretical).
Actual measurement required (Issue #7).
