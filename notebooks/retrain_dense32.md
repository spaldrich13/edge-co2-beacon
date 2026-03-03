# Retrain with Dense(32) — Flash Budget Fix

## Problem

The current model (`co2_beacon_widekernel_v4.keras` → `co2_beacon_int8.tflite`) is **729 KB**
in flash. With only **796 KB** of application flash on the nRF52840 (S140 SoftDevice consumes
the first 152 KB), the remaining ~67 KB cannot hold the TFLite runtime + sensor libraries
(~116 KB needed). Overflow: 49,500 bytes.

### Why the model is so large

The Flatten→Dense(64) layer dominates:

```
Flatten output: 50 × 7 × 32 = 11,200 activations
Dense(64): 11,200 × 64 weights (INT8) + 64 biases (INT32)
         = 716,800 + 256 = 717,056 bytes ≈ 700 KB
```

All other layers together (Conv2D×2, MaxPool×2, Dense(5)) total only ~12 KB.

---

## Fix: change Dense(64) → Dense(32)

### Step 1 — Edit the model definition in the Colab notebook

In `SHL_pretrain_coarse.ipynb` (or whichever notebook calls `build_widekernel_model()`),
find the Dense layer and change **one number**:

```python
# BEFORE
x = Dense(64, activation='relu')(x)

# AFTER
x = Dense(32, activation='relu')(x)
```

Everything else stays identical: same Conv2D(16, (9,1)), Conv2D(32, (9,1)), MaxPool2D,
Dropout(0.3), Dense(5, softmax), same optimizer, same dataset, same hyperparameters.

### Step 2 — Retrain from scratch

```python
# Same training call, new save path
model.fit(X_train, y_train,
          validation_data=(X_val, y_val),
          epochs=50, batch_size=32, ...)

model.save('data/processed/co2_beacon_widekernel_v4_d32.keras')
```

Expected accuracy: ~94–96% (Dense(32) has sufficient capacity for 5-class transport
classification; the bottleneck was always the convolutional feature extraction).

### Step 3 — TFLite INT8 conversion

Run the existing conversion script, pointing at the new model:

```bash
cd notebooks
# Edit convert_to_tflite.py: change MODEL_PATH to co2_beacon_widekernel_v4_d32.keras
python3 scripts/convert_to_tflite.py
```

Output: `model/co2_beacon_int8.tflite` (expected ~370 KB)

### Step 4 — Regenerate model_data.h

```bash
cd firmware/beacon_inference
xxd -i ../../model/co2_beacon_int8.tflite model_data.h
```

**CRITICAL**: The `xxd` tool generates a non-const array. Immediately add `const`:

```bash
# The generated first line looks like:
#   unsigned char model_co2_beacon_int8_tflite[] = {
# Change it to:
#   const unsigned char model_co2_beacon_int8_tflite[] = {
```

Without `const`, the 370 KB array is copied to RAM at startup and overflows the
232 KB RAM budget. The ARM linker places non-const initialized globals in `.data`
(copied from flash to RAM); `const` globals go to `.rodata` (flash only, zero RAM cost).

### Step 5 — Verify the length variable

`xxd` also generates a length variable. Confirm `model_data.h` ends with:

```c
const unsigned int model_co2_beacon_int8_tflite_len = <new_size>;
```

The variable name must match what `beacon_inference.ino` uses:
`tflite::GetModel(model_co2_beacon_int8_tflite)` (line 281).

---

## Expected flash budget after fix

| Item | Bytes | KB |
|---|---|---|
| Dense(32) weights (INT8) | 358,400 | 350.0 |
| Dense(32) biases (INT32) | 128 | 0.1 |
| Dense(5) weights (INT8) | 160 | 0.2 |
| Conv layers + metadata | ~12,000 | ~11.7 |
| **New model total** | **~370,000** | **~361 KB** |
| App flash budget | 815,104 | 796 KB |
| **Remaining for code** | **~445,000** | **~434 KB** |

Previous overflow: **49,500 bytes**. New headroom: **~434 KB** — comfortable margin for
TFLite runtime (~100 KB), Adafruit sensor libs (~30 KB), Arduino core (~20 KB), and
future BLE stack (~50 KB).

---

## Dense layer parameter count comparison

| Layer | Weights | Biases | Storage (INT8 w + INT32 b) |
|---|---|---|---|
| Dense(64) — old | 716,800 | 64 | 716,800 + 256 = **717,056 bytes** |
| Dense(32) — new | 358,400 | 32 | 358,400 + 128 = **358,528 bytes** |
| **Savings** | 358,400 | 32 | **358,528 bytes (≈ 350 KB)** |

---

## No firmware changes needed after model swap

`beacon_inference.ino` is model-agnostic. The resolver, arena, and inference loop
are all unchanged. `norm_stats.h` is also unchanged (same dataset, same train split).

The only file that changes is `model_data.h` (new `xxd` output with `const`).
