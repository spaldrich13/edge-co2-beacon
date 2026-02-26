# BUILD_PHASES.md — GitHub Issues & Acceptance Criteria

All 16 issues for the Edge CO₂ Beacon capstone. Scope is locked — do not add issues.

---

## Day 1 — Model Conversion & Normalization

### Issue #1 — Convert Keras Model to TFLite INT8

**Script:** `notebooks/scripts/convert_to_tflite.py`
**Input:** `data/processed/pretrained_shl_coarse.keras`
**Output:** `firmware/beacon_inference/model_data.h` (C byte array)

**Steps:**
1. Load `.keras` model
2. Create `tf.lite.TFLiteConverter.from_keras_model(model)`
3. Set `optimizations = [tf.lite.Optimize.DEFAULT]`
4. Provide representative dataset (subset of X_train) for INT8 calibration
5. Set `inference_input_type = tf.int8`
6. Set `inference_output_type = tf.int8`
7. Convert and save `.tflite` file
8. Run `xxd -i model.tflite > model_data.h`

**Acceptance Criteria:**
- [ ] `.tflite` file size ≤ 100 KB
- [ ] INT8 quantized (not float fallback)
- [ ] Python interpreter test accuracy ≥ 90% on test split
- [ ] `model_data.h` compiles without error in Arduino sketch

---

### Issue #2 — Export Normalization Stats to norm_stats.h

**Script:** `notebooks/scripts/export_norm_stats.py`
**Input:** `data/processed/features/baseline_features_v3_segment_split_train_val_test.npz`
**Output:** `firmware/beacon_inference/norm_stats.h`

**Steps:**
1. Load `X_train` from `.npz`
2. Compute `mu = X_train.mean(axis=(0,1))` — shape (7,)
3. Compute `sigma = X_train.std(axis=(0,1)) + 1e-8` — shape (7,)
4. Write C header with float arrays `NORM_MU[7]` and `NORM_SIGMA[7]`

**Acceptance Criteria:**
- [ ] Header compiles on nRF52840
- [ ] Channel order matches: ax, ay, az, gx, gy, gz, pressure_hpa
- [ ] Values printed to Serial during init and match Python-computed values

---

## Day 1 — Sensor Firmware

### Issue #3 — 25 Hz Sensor Sampling Loop

**File:** `firmware/beacon_inference/beacon_inference.ino`

**Requirements:**
- BNO055 and BMP390 initialized over I²C at 400 kHz
- Sampling at exactly 25 Hz using `micros()`-based tick (not `delay()`)
- Reads: `ax_corr, ay_corr, az_corr, gx, gy, gz, pressure_hpa`
- Bias subtraction applied to accelerometer
- Data fed into ring buffer

**Acceptance Criteria:**
- [ ] Serial output shows 25 samples/second (measure with Python timer)
- [ ] No sample jitter > 5 ms
- [ ] Both sensors initialize without error on boot

---

## Day 2 — Preprocessing & Inference

### Issue #4 — Preprocessing Pipeline in Firmware

**Requirements:**
- Sliding window extraction: 125 samples, 50% overlap (trigger every 62 samples)
- Z-score normalization using `NORM_MU` and `NORM_SIGMA` from `norm_stats.h`
- Output: normalized float array of shape (125, 7) ready for TFLite input

**Acceptance Criteria:**
- [ ] Window extraction triggers at correct rate (every 2.5 s at 25 Hz)
- [ ] Normalization verified: manually compute z-score for one window in Python and compare to firmware output via Serial

---

### Issue #5 — Deploy TFLite + Run Inference On-Device

**Requirements:**
- TFLite Micro interpreter initialized with 80 KB static arena
- Input tensor quantization: scale raw float input to INT8 using tensor quantization params
- Run `interpreter.Invoke()`
- Dequantize output to float softmax probabilities
- Argmax → mode_id, max → confidence

**Acceptance Criteria:**
- [ ] Inference runs without arena allocation error
- [ ] Serial prints `mode: car, conf: 92%` style output
- [ ] Classification matches Python interpreter on same window (within INT8 quantization error)

---

## Day 3 — Latency & BLE

### Issue #6 — Measure Inference Latency

**Method:** `micros()` before and after `interpreter.Invoke()`

**Acceptance Criteria:**
- [ ] Latency ≤ 1000 ms (target: < 500 ms)
- [ ] Measured over 100 inference calls, report mean ± std
- [ ] Log results in `logs/` with date

---

### Issue #8 — BLE GATT Service Implementation

**Requirements:**
- Service and characteristics as defined in `API_INTEGRATIONS.md`
- LiveStatus notified at 1 Hz
- TripRecord sent on SYNC_TRIPS control command
- Advertising interval: 100 ms, name: "CO2-Beacon"

**Acceptance Criteria:**
- [ ] nRF Connect app (mobile) can discover and connect to beacon
- [ ] LiveStatus notifications received at 1 Hz in nRF Connect
- [ ] TripRecord can be read manually via nRF Connect
- [ ] Dashboard can connect and receive LiveStatus

---

## Day 4 — CO₂ Pipeline

### Issue #10 — Distance Estimation Module

**Implementation:** Speed model (see `TECHNICAL_ARCHITECTURE.md`)

**Acceptance Criteria:**
- [ ] Distance computed correctly for all 5 modes
- [ ] Unit test: 10-min car trip → 6.67 km

---

### Issue #11 — CO₂ Estimation Module

**Implementation:** `co2_g = distance_km × factor_kg_per_km × 1000`

**Acceptance Criteria:**
- [ ] CO₂ value stored in TripRecord `co2_g` field
- [ ] Unit test: 10-min car at 40 km/h → 6.67 km × 0.089 × 1000 = 593 g
- [ ] Walk always returns 0 g

---

### Issue #12 — Web Dashboard (React + Web Bluetooth)

**Implementation:** See `UI_SPECIFICATIONS.md`

**Acceptance Criteria:**
- [ ] Connects to "CO2-Beacon" via Web Bluetooth in Chrome
- [ ] LiveStatus updates displayed in real time
- [ ] Trip history populates correctly after SYNC_TRIPS
- [ ] Session total CO₂ calculated correctly
- [ ] No localStorage used anywhere
- [ ] No raw sensor data displayed

---

## Day 5 — Validation

### Issue #13 — Formal CO₂ RMSE Validation ⚠ CANNOT BE SKIPPED

**Method:**
1. Take N ground-truth trips with known distance (use GPS or Maps)
2. Record beacon CO₂ estimate for each trip
3. Compute reference CO₂: `distance_km × factor`
4. RMSE = sqrt(mean((beacon_co2_kg − ref_co2_kg)²))

**Acceptance Criteria:**
- [ ] N ≥ 10 trips across at least 3 modes
- [ ] CO₂ RMSE ≤ 0.20 kg
- [ ] Results documented in `logs/` with raw data

---

## Day 6 — Hardware Validation

### Issue #7 — Battery Runtime Test ⚠ CANNOT BE SKIPPED

**Method:**
1. Charge to 100%
2. Run beacon with BLE advertising and 25 Hz sampling
3. Record time until power-off

**Acceptance Criteria:**
- [ ] Runtime ≥ 24 hours
- [ ] Results logged with start/end timestamps

---

## Remaining Issues (Reference)

| Issue | Title                          | Status    |
|-------|-------------------------------|-----------|
| #9    | GPS distance integration       | Deferred  |
| #14   | OTA firmware update            | Deferred  |
| #15   | Multiple-mode trip detection   | Deferred  |
| #16   | Cloud sync / export            | Deferred  |

Issues #9, #14, #15, #16 are out of scope for this deadline.
