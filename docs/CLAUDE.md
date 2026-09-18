# CLAUDE.md — Edge CO₂ Beacon (ECE499 Capstone)
# Read this at the start of every session before touching any code.

## What This Project Is
Wearable edge-computing beacon that classifies transportation mode (walk/car/bus/train/subway)
fully on-device using TFLite Micro on a Nordic nRF52840, estimates per-trip CO₂ emissions,
and syncs trip summaries over BLE to a React web dashboard.
NO raw sensor data is ever transmitted. Privacy is guaranteed by architecture.

## Status: CAPSTONE COMPLETE (2026-03-14). All issues closed. Video demo pending.

---

## Hardware
| Component     | Part                    | Interface | Address  |
|---------------|-------------------------|-----------|----------|
| MCU           | Adafruit Feather nRF52840 | —       | —        |
| IMU           | Adafruit BNO055         | I²C       | 0x28     |
| Barometer     | Adafruit BMP390         | I²C       | 0x77     |
| Battery       | 1200 mAh Li-Ion         | JST-PH    | VBAT pin |
| GPS (test only)| Adafruit MTK3339       | UART      | Serial1  |

Sampling: 25 Hz | Window: 8s (200 samples) | Overlap: 50%

---

## ML Model — FINAL CONFIRMED ARCHITECTURE
- Window: 8s at 25Hz = 200 samples (WIN_N=200, STEP_N=100)
- Input shape: (200, 7, 1)
- Channels: ax_corr, ay_corr, az_corr, gx, gy, gz, pressure_hpa
- Architecture: Conv2D(16,9×1) → MaxPool(2×1) → Conv2D(32,9×1) →
                MaxPool(2×1) → Flatten → Dense(32) → Dropout(0.3) → Dense(5, softmax)
- Model file: co2_beacon_int8.tflite — 361 KB (Dense(32) retrain, was 729 KB)
- Modes: ['train','subway','car','bus','walk'] → IDs 0–4
- Accuracy: 92.7% test accuracy (segment-level split — no data leakage). Key journey: 99.7% with leakage → 83.6% with proper split → ~80% SHL transfer learning failure → 92.7% wide-kernel CNN trained from scratch.
- Dataset: baseline_features_v4_segment_split_train_val_test.npz
- Train: 360 windows (72/class), Val: 1019, Test: 440 (88/class)

## Normalization — FINAL CONFIRMED (v4, 200-sample windows)
mu:    [-7.79250193, 2.07272148, 4.27278090,
         0.41056770, -0.15372396, 0.16609201, 770.50]
sigma: [1.59938383, 1.88753140, 3.04334497,
        18.63519859, 9.55679989, 8.78757954, 8.03807831]
Note: pressure mu is 770.50 hPa (Golden, CO altitude ~1700m), not 1014.45 hPa (sea level).
norm_stats.h: ✅ generated and correct

## Mode ID Encoding — MUST MATCH BETWEEN FIRMWARE AND MODEL
Verify this ordering against your .keras file before flashing:
| ID | Label  | kg CO₂/km |
|----|--------|-----------|
| 0  | train  | 0.035     |
| 1  | subway | 0.041     |
| 2  | car    | 0.089     |
| 3  | bus    | 0.089     |
| 4  | walk   | 0.000     |

⚠ The ordering above matches SHL_pretrain_coarse.ipynb: ['train','subway','car','bus','walk']
  Double-check against modes = list(data["modes"]) in the fine-tune notebook output.

---

## Normalization — CONFIRMED
mu and sigma are stored directly in the .npz file under keys 'mu' and 'sigma'.
DO NOT recompute — read from file:
  data = np.load("baseline_features_v4_segment_split_train_val_test.npz")
  mu    = data["mu"]     # shape (7,) — per channel
  sigma = data["sigma"]  # shape (7,) — per channel

Channel order confirmed: ['ax', 'ay', 'az', 'gx', 'gy', 'gz', 'pressure']
Mode order confirmed: ['train', 'subway', 'car', 'bus', 'walk'] → IDs 0-4

Accelerometer biases (subtract before normalization):
  ACC_BIAS_X = -0.1926, ACC_BIAS_Y = -0.1975, ACC_BIAS_Z = -0.3472 (m/s²)

These values must be exported to norm_stats.h and burned into firmware flash.
Script to generate: notebooks/scripts/export_norm_stats.py ✅ DONE (Issue #2)
Dataset file: data/processed/features/baseline_features_v4_segment_split_train_val_test.npz

---

## Non-Negotiable Engineering Requirements
| Requirement          | Target       | Status              |
|----------------------|--------------|---------------------|
| Classification acc   | ≥ 90%        | ✅ 92.7% (n=440 test windows, segment-level split) |
| Inference latency    | ≤ 1000ms     | ✅ 261.6ms mean (n=30, σ=0.7ms) |
| Battery runtime      | ≥ 24 hours   | ✅ 48+ hours on 1200 mAh Li-Ion |
| CO₂ estimation error | ≤ 20% of ground-truth CO₂ per trip, validated over ≥5 trips | ❌ Mean 34.0% over 5 car trips (Golden, CO). Root cause: fixed-speed assumption (40 km/h) diverges from actual trip speeds (21–54 km/h). |
| Privacy              | No raw data via BLE | ✅ Confirmed by BLE packet inspection |

---

## Repo Structure
edge-co2-beacon/
├── CLAUDE.md                          ← YOU ARE HERE
├── data/
│   ├── processed/features/            ← .npz dataset files (NOT in git)
│   ├── processed/pretrained_shl_coarse.keras  (NOT in git)
│   └── raw/self_collected/            ← raw CSVs by mode
├── firmware/
│   └── LOGGER_BUTTON_v1.3/            ← current logging firmware
├── notebooks/                         ← Python scripts + Colab .ipynb files
├── docs/                              ← Project documentation (see below)
├── figures/                           ← Confusion matrices, signal plots
└── logs/                              ← Session README logs

## Project Docs (in docs/ folder — read these for full spec)
- docs/PROJECT_INSTRUCTIONS.md   ← Master guide, session checklists
- docs/TECHNICAL_ARCHITECTURE.md ← Full system design, memory budget, BLE spec
- docs/DATABASE_SCHEMA.md        ← On-device structs, BLE payload byte layout
- docs/API_INTEGRATIONS.md       ← BLE GATT code, I²C setup, CO₂ factor table
- docs/UI_SPECIFICATIONS.md      ← Dashboard layout and component specs
- docs/BUILD_PHASES.md           ← All 16 GitHub issues with acceptance criteria
- docs/DEBUGGING_GUIDE.md        ← Common failure modes and fixes

---

## Active GitHub Issues (check github.com/spaldrich13/edge-co2-beacon)
Priority order this week:
- #1  ✅ DONE  Convert Keras model to TFLite INT8 (Dense(32), 361KB)
- #2  ✅ DONE  Export normalization stats to norm_stats.h
- #3  ✅ DONE  25Hz sensor sampling loop — confirmed on hardware
- #4  ✅ DONE  Preprocessing pipeline — norm + INT8 quantize confirmed on hardware
- #5  ✅ DONE  TFLite inference on-device — output MODE:subway CONF:69% LAT:259ms
- #6  ✅ DONE  Inference latency = 261.6ms mean (n=30, σ=0.7ms, spec ≤1000ms) PASS
- #8  ✅ DONE  BLE GATT service — LiveStatus (7 bytes, 1Hz) + TripRecord (20 bytes)
- #10 ✅ DONE  Distance estimation — fixed-speed model per mode
- #11 ✅ DONE  CO₂ estimation — distance × EPA factor
- #12 ✅ DONE  React web dashboard — Web Bluetooth, live mode + CO₂ display
- #13 ✅ DONE  CO₂ validation — 5 car trips, mean error 34.0%, spec not met
- #7  ✅ DONE  Battery runtime — 48+ hours confirmed

---

## Hardware Validation Results (2026-03-03)
First successful hardware run on Adafruit Feather nRF52840:

| Metric | Result | Spec | Status |
|---|---|---|---|
| Inference latency | **259 ms** | ≤ 1000 ms | ✅ PASS |
| Flash used | 62% of 796 KB (≈ 493 KB) | — | ✅ |
| RAM used | 28% of 232 KB (≈ 65 KB) | — | ✅ |
| BNO055 init | OK at 0x28 | — | ✅ |
| BMP390 init | OK at 0x77 | — | ✅ |
| Serial output | `MODE:subway CONF:69% LAT:259ms` | — | ✅ |

Firmware: `firmware/beacon_inference/beacon_inference.ino`
Model: `co2_beacon_int8.tflite` — 361 KB, Dense(32), INT8 quantized

## Latency Formal Measurement (2026-03-06, n=30)
Measured via Python/pyserial from live serial output:

| Stat   | Value    |
|--------|----------|
| Mean   | 261.6 ms |
| Stdev  | 0.7 ms   |
| Min    | 261 ms   |
| Max    | 264 ms   |
| Median | 262.0 ms |

Raw data: `logs/latency_readings.csv`
Spec: ≤ 1000 ms — **PASS** (margin: 738 ms, 74% headroom)

## CO₂ Field Validation Results (2026-03-14)

| Trip | Distance | Beacon CO₂ (g) | Ref CO₂ (g) | Error % | Pass? |
|------|----------|----------------|-------------|---------|-------|
| 1    | 9.97 km  | 663.9          | 887.3       | 25.2%   | ❌    |
| 2    | 7.24 km  | 904.9          | 644.4       | 40.4%   | ❌    |
| 3    | 3.86 km  | 541.4          | 343.5       | 57.6%   | ❌    |
| 4    | 5.95 km  | 600.6          | 529.6       | 13.4%   | ✅    |
| 5    | 5.47 km  | 648.1          | 486.8       | 33.1%   | ❌    |
| Mean |          |                |             | 34.0%   | 1/5   |

Root cause: fixed-speed assumption (40 km/h) systematic. Actual avg speeds ranged 21–54 km/h. GPS-augmented sensing identified as the architectural fix.

---

## Deployment Debugging Log (2026-03-19) — Final Day Issues

### Issue 1: norm_stats.h had incorrect values
- NORM_MU accelerometer channels had been manually overridden to approximate
  gravity values (az≈9.62) instead of dataset means — incorrect
- Pressure mean was set to 770.50 hPa (Golden CO altitude) but training data
  was collected at sea level in NYC area (actual mean 1007.26 hPa confirmed
  by raw CSV analysis across 350,494 samples)
- Fix: restored correct dataset means for all 7 channels

### Issue 2: Pressure channel causes location-dependent failure
- After fixing norm stats, model still predicted subway/bus at 100% confidence
- Root cause: training data collected in NYC (~1007 hPa), device deployed in
  Golden CO (~782 hPa) — 225 hPa domain gap, z-score of -28 on every inference
- Attempted fix: +224.5 hPa offset in firmware — partially improved but
  accelerometer domain gap remained
- Final fix: drop pressure channel entirely, retrain 6-channel model
- Result: model is now location-independent

### Issue 3: 6-channel model used unsupported TFLite ops
- First retrain used Conv1D → produced EXPAND_DIMS and MEAN ops not
  implemented in Arduino_TensorFlowLite library
- Fix: retrained with Conv2D(k,1) kernels on input shape (200,6,1) and
  converted via fixed batch=1 concrete function
- Additional fix: increased TFLite arena from 50 KB to 130 KB
  (new model requires 122,888 bytes)

### Issue 4: BNO055 frozen — returning all-zero values
- After one reflash, IMU returned AX:0.193 AY:0.198 AZ:0.347 (exactly the
  bias values) indicating sensor was not being read
- Fix: reseat BNO055 on breadboard — loose connection was the cause

### Issue 5: Car classification failed in deployment — UNRESOLVED
- During actual car trip, model classified walk for the entire trip
- At rest (device on table, flat), model predicts walk at 74% confidence
- Root cause hypothesis: training data collected with device vertical in
  backpack on body in NYC; model has never seen the device flat on a table
  or in a car seat — orientation mismatch causes walk prediction
- Additional factor: NYC road vibration signature (potholes, urban stop-start)
  differs from Golden CO smooth roads — car vibration pattern does not match
  training distribution
- Attempted fixes: subway→car remap, trip boundary logic changes — did not
  resolve the underlying classification error
- Status: walk classification works correctly when device is in backpack on body.
  Car classification is unreliable in Golden CO deployment context.
- Impact on demo: video demo will show walk classification and BLE dashboard only.
  Car classification cannot be reliably demonstrated.

### What works reliably as of 2026-03-19
- Walk classification: correct when device is in backpack on body (~70-80% conf)
- BLE dashboard: connects, receives LiveStatus at 1 Hz, displays mode and confidence
- Trip accumulation: CO₂ and distance accumulate correctly per mode
- TRIP_END events: fire correctly on sustained walk detection
- Boot sequence: all sensors init, TFLite loads, BLE advertises
- Battery runtime: 48+ hours confirmed

### What does not work reliably
- Car classification: predicts walk during actual driving in Golden CO
- Model accuracy in deployment: 89.1% on NYC test set does not transfer to
  Golden CO deployment due to road vibration and orientation differences
- CO₂ estimation spec: 34% mean error (spec ≤20%) — fixed-speed assumption
  is root cause

### Root cause summary for conclusion/discussion section
The system suffers from a training-deployment domain gap on two axes:
1. Geographic: training data collected in NYC, deployed in Golden CO —
   different road surfaces, elevation, and vibration signatures
2. Pressure channel: sea-level training data incompatible with high-altitude
   deployment — resolved by dropping channel
The classifier works correctly for walk because gait is location-independent.
Car fails because road vibration is highly location-dependent.
The fundamental lesson: self-collected training data must be collected in the
deployment environment, or the model must be made robust to domain shift through
data augmentation across multiple locations and orientations.

---

## Video Demo Plan
7 scenes, ~8-10 min:
1. Hardware intro — point out no GPS chip, describe each component
2. Boot + Serial Monitor — walk through #OK lines, arena size, BLE advert
3. Live classification — walk/car modes, show MODE/CONF/LAT:261ms live on Serial
4. BLE dashboard in Chrome — show 7-byte LiveStatus packet, live CO₂ accumulation
5. Successes — 92.7% accuracy, 261.6ms latency, 48hr battery, privacy by architecture
6. Limitations — 34.0% mean CO₂ error, fixed-speed root cause, car-only validation
7. Next iteration — GPS distance sensing, more modes, wearable enclosure, retrain locally
Filming notes: Serial Monitor at 18pt+. Film scenes 3 and 4 as one continuous take.

---

## CO₂ Emission Factors (static table, EPA eGRID 2023)
car:    0.089 kg/km
bus:    0.089 kg/km
train:  0.035 kg/km
subway: 0.041 kg/km
walk:   0.000 kg/km

---

## BLE GATT Payload Schemas (dashboard parser must match exactly)
LiveStatus (7 bytes, notified 1Hz):
  [0]   mode_id       uint8
  [1]   confidence    uint8 (×100)
  [2-5] timestamp_s   uint32 LE
  [6]   trip_active   uint8

TripRecord (20 bytes, on request):
  [0-1]   trip_id     uint16 LE
  [2]     mode_id     uint8
  [3]     confidence  uint8 (×100)
  [4-7]   ts_start    uint32 LE
  [8-11]  ts_end      uint32 LE
  [12-13] duration_s  uint16 LE
  [14-15] distance_m  uint16 LE
  [16-17] co2_g       uint16 LE
  [18-19] reserved    0x00

---

## Key Commands
# TFLite conversion (run in notebooks/ after activating your Python env)
python scripts/convert_to_tflite.py

# Export norm stats to C header
python scripts/export_norm_stats.py

# Flash firmware
arduino-cli upload -p /dev/ttyUSB0 --fqbn adafruit:nrf52:feather52840 firmware/beacon_inference/

# Dashboard dev
cd dashboard && npm run dev   # open Chrome at http://localhost:3000

# Push changes
git add -A && git commit -m "[#N] description" && git push

---

## DO NOT
- Do not improve the model — accuracy is sufficient
- Do not add features beyond the 16 issues in BUILD_PHASES.md
- Do not transmit raw sensor data over BLE under any circumstances
- Do not use heap allocation in firmware (static arena only for TFLite)
- Do not use localStorage in the dashboard (Web Bluetooth + React state only)
