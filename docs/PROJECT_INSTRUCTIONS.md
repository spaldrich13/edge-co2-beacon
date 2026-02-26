# PROJECT_INSTRUCTIONS.md — Edge CO₂ Beacon (ECE499 Capstone)

## Overview

Wearable edge-computing beacon that classifies transportation mode (walk/car/bus/train/subway)
fully on-device using TFLite Micro on a Nordic nRF52840, estimates per-trip CO₂ emissions,
and syncs trip summaries over BLE to a React web dashboard.

**Privacy guarantee:** NO raw sensor data is ever transmitted. Classification and CO₂ math
happen entirely on the MCU.

**Deadline:** One week from project start. Scope is locked.

---

## Hardware Summary

| Component      | Part                      | Interface | Address |
|----------------|---------------------------|-----------|---------|
| MCU            | Adafruit Feather nRF52840 | —         | —       |
| IMU            | Adafruit BNO055           | I²C       | 0x28    |
| Barometer      | Adafruit BMP390           | I²C       | 0x76    |
| Battery        | 1200 mAh Li-Ion           | JST-PH    | VBAT    |
| GPS (test only)| Adafruit MTK3339          | UART      | Serial1 |

Sampling: **25 Hz** | Window: **5 s (125 samples)** | Overlap: **50%**

---

## ML Model — Confirmed Architecture

- Input: `(125, 7, 1)` — 125 timesteps × 7 channels × 1 channel dim
- Channels (in order): `ax, ay, az, gx, gy, gz, pressure_hpa`
- Architecture: `Conv2D(16,9×1) → MaxPool → Conv2D(32,9×1) → MaxPool → Flatten → Dense(64) → Dropout(0.3) → Dense(5, softmax)`
- Training: SHL pretraining → linear probe → full fine-tune on self-collected data
- Test accuracy: ~95–96% (leakage-free, segment-split v3 dataset)
- Model file: `data/processed/pretrained_shl_coarse.keras` (local, NOT in git)

## Mode ID Encoding

| ID | Label  | kg CO₂/km |
|----|--------|-----------|
| 0  | train  | 0.035     |
| 1  | subway | 0.041     |
| 2  | car    | 0.089     |
| 3  | bus    | 0.089     |
| 4  | walk   | 0.000     |

⚠ Verify ordering against `modes = list(data["modes"])` in the fine-tune notebook before flashing.

---

## Normalization — Critical

Z-score computed from TRAIN split only:

```python
mu    = X_train.mean(axis=(0,1), keepdims=True)  # shape (1,1,7,1)
sigma = X_train.std(axis=(0,1),  keepdims=True) + 1e-8
```

These values must be exported to `firmware/beacon_inference/norm_stats.h` and burned
into firmware flash. **Do NOT recompute at runtime.**

Script: `notebooks/scripts/export_norm_stats.py`
Dataset: `data/processed/features/baseline_features_v3_segment_split_train_val_test.npz`

---

## Non-Negotiable Engineering Requirements

| Requirement        | Target             | Status              |
|--------------------|--------------------|---------------------|
| Classification acc | ≥ 90%              | ✅ ~95–96% achieved |
| Inference latency  | ≤ 1000 ms          | 🔲 Not yet measured |
| Battery runtime    | ≥ 24 hours         | 🔲 Not yet measured |
| CO₂ RMSE           | ≤ 0.20 kg per trip | 🔲 CRITICAL MISSING |
| Privacy            | No raw data via BLE| ✅ By design        |

---

## Session Checklist

Before starting any coding session:
- [ ] Read CLAUDE.md
- [ ] Check active GitHub issues at github.com/spaldrich13/edge-co2-beacon
- [ ] Confirm current issue priority (see BUILD_PHASES.md)
- [ ] Verify mode ID ordering has not changed

Before committing:
- [ ] Confirm no raw sensor data is exposed over BLE
- [ ] Commit message format: `[#N] description`
- [ ] No heap allocation added to firmware

Before flashing:
- [ ] norm_stats.h values match current train-split statistics
- [ ] Mode ID table in firmware matches model label ordering
- [ ] TFLite model passes byte-level checksum vs. converted file

---

## Active Issue Priority

1. **#1** [TODAY] Convert Keras model to TFLite INT8
2. **#2** [TODAY] Export normalization stats to norm_stats.h
3. **#3** [TODAY] 25 Hz sensor sampling loop firmware
4. **#4** [DAY 2] Preprocessing pipeline in firmware
5. **#5** [DAY 2] Deploy TFLite + run inference on-device
6. **#6** [DAY 3] Measure inference latency (target ≤ 1000 ms)
7. **#8** [DAY 3] BLE GATT service implementation
8. **#10** [DAY 4] Distance estimation module
9. **#11** [DAY 4] CO₂ estimation module
10. **#12** [DAY 4] Web dashboard (React + Web Bluetooth)
11. **#13** [DAY 5] ⚠ FORMAL CO₂ RMSE VALIDATION — cannot be skipped
12. **#7** [DAY 6] Battery runtime test — cannot be skipped

---

## Key Commands

```bash
# TFLite conversion
python notebooks/scripts/convert_to_tflite.py

# Export norm stats
python notebooks/scripts/export_norm_stats.py

# Flash firmware
arduino-cli upload -p /dev/ttyUSB0 --fqbn adafruit:nrf52:feather52840 firmware/beacon_inference/

# Dashboard dev server
cd dashboard && npm run dev   # Chrome at http://localhost:3000

# Commit
git add -A && git commit -m "[#N] description" && git push
```

---

## DO NOT

- Do not improve the model — accuracy is sufficient
- Do not add features beyond the 16 issues in BUILD_PHASES.md
- Do not transmit raw sensor data over BLE under any circumstances
- Do not use heap allocation in firmware (static arena only for TFLite)
- Do not use localStorage in the dashboard (Web Bluetooth + React state only)
