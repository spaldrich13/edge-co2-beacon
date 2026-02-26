# CLAUDE.md — Edge CO₂ Beacon (ECE499 Capstone)
# Read this at the start of every session before touching any code.

## What This Project Is
Wearable edge-computing beacon that classifies transportation mode (walk/car/bus/train/subway)
fully on-device using TFLite Micro on a Nordic nRF52840, estimates per-trip CO₂ emissions,
and syncs trip summaries over BLE to a React web dashboard.
NO raw sensor data is ever transmitted. Privacy is guaranteed by architecture.

## Deadline: ONE WEEK FROM NOW. Scope is locked. No model improvements.

---

## Hardware
| Component     | Part                    | Interface | Address  |
|---------------|-------------------------|-----------|----------|
| MCU           | Adafruit Feather nRF52840 | —       | —        |
| IMU           | Adafruit BNO055         | I²C       | 0x28     |
| Barometer     | Adafruit BMP390         | I²C       | 0x76     |
| Battery       | 1200 mAh Li-Ion         | JST-PH    | VBAT pin |
| GPS (test only)| Adafruit MTK3339       | UART      | Serial1  |

Sampling: 25 Hz | Window: 5s (125 samples) | Overlap: 50%

⚠ BMP390 address: CONFIRM before flashing — logging firmware uses 0x77,
  spec says 0x76. Run i2c_scanner first.

---

## ML Model — CONFIRMED ARCHITECTURE
- Input shape: (125, 7, 1) — 125 timesteps × 7 channels × 1 channel dim
- Channels (in order): ax, ay, az, gx, gy, gz, pressure_hpa
- Architecture: Conv2D(16, 9×1) → MaxPool → Conv2D(32, 9×1) → MaxPool → Flatten → Dense(64) → Dropout(0.3) → Dense(5, softmax)
- NOTE: Confirm kernel size (9×1 vs 5×1) by running model.summary() on pretrained_shl_coarse.keras before TFLite conversion
- Training: SHL pretraining (5×1 kernels) → transfer weights → linear probe → full fine-tune on self-data
- Test accuracy: ~95–96% (leakage-free, segment-split v3 dataset)
- Model file: data/processed/pretrained_shl_coarse.keras (locally on Mac, NOT in git)

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

## Normalization — CRITICAL, DO NOT RECOMPUTE AT RUNTIME
Z-score computed from TRAIN split only:
  mu    = X_train.mean(axis=(0,1), keepdims=True)  # shape (1,1,7,1)
  sigma = X_train.std(axis=(0,1),  keepdims=True) + 1e-8

Accelerometer biases (subtract before normalization):
  ACC_BIAS_X = -0.1926, ACC_BIAS_Y = -0.1975, ACC_BIAS_Z = -0.3472 (m/s²)

These values must be exported to norm_stats.h and burned into firmware flash.
Script to generate: notebooks/scripts/export_norm_stats.py (needs to be written — Issue #2)
Dataset file: data/processed/features/baseline_features_v3_segment_split_train_val_test.npz

---

## Non-Negotiable Engineering Requirements
| Requirement          | Target       | Status              |
|----------------------|--------------|---------------------|
| Classification acc   | ≥ 90%        | ✅ ~95–96% achieved |
| Inference latency    | ≤ 1000ms     | 🔲 Not yet measured |
| Battery runtime      | ≥ 24 hours   | 🔲 Not yet measured |
| CO₂ RMSE             | ≤ 0.20 kg    | 🔲 CRITICAL MISSING |
| Privacy              | No raw data via BLE | ✅ By design  |

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
- #1  [TODAY] Convert Keras model to TFLite INT8
- #2  [TODAY] Export normalization stats to norm_stats.h
- #3  [TODAY] 25Hz sensor sampling loop firmware
- #4  [DAY 2] Preprocessing pipeline in firmware
- #5  [DAY 2] Deploy TFLite + run inference on-device
- #6  [DAY 3] Measure inference latency (target ≤1000ms)
- #8  [DAY 3] BLE GATT service implementation
- #10 [DAY 4] Distance estimation module
- #11 [DAY 4] CO₂ estimation module
- #12 [DAY 4] Web dashboard (React + Web Bluetooth)
- #13 [DAY 5] ⚠ FORMAL CO₂ RMSE VALIDATION — cannot be skipped
- #7  [DAY 6] Battery runtime test — cannot be skipped

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
