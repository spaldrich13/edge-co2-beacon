# DEBUGGING_GUIDE.md — Common Failure Modes & Fixes

## Sensor Issues

### BNO055 not found at boot

**Symptom:** Serial prints `#ERROR:BNO055_NOT_FOUND`, halts.

**Causes & Fixes:**
- Wrong I²C address — check solder jumper on board; default is 0x28
- Wire not started — ensure `Wire.begin()` is called before `bno.begin()`
- I²C clock too high — try `Wire.setClock(100000)` for debugging
- Power issue — check 3.3V rail with multimeter
- Loose connection — reseat I²C wires

---

### BMP390 not found at boot

**Symptom:** Serial prints `#ERROR:BMP3XX_NOT_FOUND`, halts.

**Causes & Fixes:**
- Address mismatch — logging firmware uses `0x77`, spec says `0x76`.
  Check SDO pin: SDO=GND → 0x76, SDO=3.3V → 0x77
- Try `bmp.begin_I2C(0x77)` if hardware SDO is high
- Confirm `Wire.begin()` called first

---

### Pressure reads NaN

**Symptom:** `pressure_hpa` column in CSV contains `nan`.

**Cause:** `bmp.performReading()` returned false — sensor not ready.

**Fix:** Verify ODR setting. At 25 Hz sampling, BMP390 ODR must be ≥ 25 Hz.
`BMP3_ODR_50_HZ` is safe. Also check that the IIR filter doesn't stall output.

---

### Accelerometer values drift over time

**Symptom:** Steady-state acceleration not near 9.81 m/s².

**Cause:** BNO055 not using external crystal — internal oscillator drifts.

**Fix:** Ensure `bno.setExtCrystalUse(true)` is called after the 1-second delay.

---

## TFLite / Inference Issues

### Arena allocation failed

**Symptom:** `AllocateTensors()` returns error, inference never runs.

**Cause:** 80 KB arena is too small for the model + activations.

**Fix:**
- Run `interpreter.arena_used_bytes()` after `AllocateTensors()` to see actual usage
- Increase arena if needed, or reduce model size
- Ensure tensor_arena is declared as `static` (not on stack)

---

### Inference output is garbage / all zeros

**Symptoms:** Softmax outputs all 0.0 or unreasonable values.

**Causes & Fixes:**
- **Input quantization wrong:** INT8 TFLite expects scaled inputs.
  ```cpp
  // Get quantization params from input tensor
  float scale = input->params.scale;
  int zero_point = input->params.zero_point;
  input->data.int8[i] = (int8_t)(x_norm / scale + zero_point);
  ```
- **Channel order wrong:** Verify firmware channel order matches model's expected order:
  `ax, ay, az, gx, gy, gz, pressure_hpa`
- **Normalization not applied:** Check that NORM_MU/NORM_SIGMA values are non-zero
  and match Python-computed values

---

### Inference accuracy much lower than 95%

**Symptom:** Live classification is wrong most of the time.

**Causes:**
1. **Mode ID mismatch:** Firmware mode table doesn't match model's training label order.
   Fix: Re-check `modes = list(data["modes"])` from the fine-tune notebook.
2. **Normalization stats wrong:** If norm_stats.h was computed from all data (not train only),
   the normalization will be slightly off. Rerun `export_norm_stats.py` from train split only.
3. **Window overlap wrong:** Inference should trigger every 62 samples (50% of 125), not every 125.

---

### Inference latency > 1000 ms

**Symptom:** Serial shows classification takes > 1 second.

**Fixes:**
- Verify INT8 quantization was applied (not float fallback)
- Run `model.summary()` in Python to check actual parameter count
- Check that `MicroMutableOpResolver` registers all required ops (Conv2D, MaxPool2D, Dense, Softmax)
- Missing op registration causes fallback to slower reference implementation

---

## BLE Issues

### Dashboard cannot find beacon

**Symptom:** Web Bluetooth scan times out with no device found.

**Fixes:**
- Verify beacon is advertising: use nRF Connect mobile app to scan
- Check `BLE.setLocalName("CO2-Beacon")` matches the dashboard filter name exactly
- Ensure `BLE.advertise()` is called after service setup
- BLE advertising requires the SoftDevice to be running — check ArduinoBLE library version

---

### LiveStatus notifications not arriving

**Symptom:** Connected but dashboard shows no data updates.

**Fixes:**
- Dashboard must call `await char.startNotifications()` after connection
- Beacon must write to characteristic value AND the CCCD must be subscribed
- Check that `liveStatusChar.writeValue(...)` is called inside the main loop at 1 Hz

---

### TripRecord data appears garbled

**Symptom:** Trip history shows wrong mode, impossible CO₂ values.

**Causes:**
- Byte order mismatch: all multi-byte fields are little-endian. Ensure dashboard uses
  `dataView.getUint16(offset, true)` (true = little-endian)
- Struct padding: use `__attribute__((packed))` on firmware side to prevent alignment padding

---

## Normalization Issues

### norm_stats.h values don't match Python

**Diagnosis:**
```python
import numpy as np
data = np.load("data/processed/features/baseline_features_v3_segment_split_train_val_test.npz")
X_train = data["X_train"]
mu = X_train.mean(axis=(0,1))
sigma = X_train.std(axis=(0,1)) + 1e-8
print("mu:", mu)
print("sigma:", sigma)
```
Compare to values in `norm_stats.h`.

**Common mistake:** Using `axis=(0,1,2)` instead of `axis=(0,1)`. The keepdims shape must
be `(1,1,7,1)` — averaging over timesteps and samples, not channels.

---

## Data Collection Issues

### CSV has no header

**Symptom:** Pandas throws KeyError when reading column by name.

**Cause:** Old logging firmware did not write header at segment start.

**Fix:** LOGGER_BUTTON_v1.3 writes header at every segment start. If using older firmware,
the Python script falls back to writing the header itself. Regenerate segments with v1.3.

---

### Serial port not found

**Symptom:** `serial_logger_v1.3.py` shows "No serial ports found."

**Fixes:**
- Unplug and replug the USB cable
- Check `ls /dev/tty.*` on macOS — look for `tty.usbmodem*`
- Ensure Arduino Serial Monitor is closed (two processes cannot own the same port)
- Try a different USB cable (some are charge-only)

---

## Dashboard Issues

### Web Bluetooth not available

**Symptom:** `navigator.bluetooth is undefined`

**Fix:** Web Bluetooth only works in Chrome. Open Chrome, not Safari or Firefox.
Also requires HTTPS or localhost — `npm run dev` serves on localhost which is allowed.

---

### Trips disappear on page refresh

**Expected behavior:** This is by design. No localStorage is used. All trip data lives
in React state. A page refresh clears the session. If persistence is needed, the user
must reconnect and request SYNC_TRIPS again.
