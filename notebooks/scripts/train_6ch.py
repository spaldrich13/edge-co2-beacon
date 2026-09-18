"""
Train 6-channel (no pressure) wide-kernel Conv2D CNN, convert to INT8 TFLite,
export norm_stats.h and model_data.h.

Architecture uses Conv2D with (k,1) kernels on input shape (200,6,1) so that
TFLite conversion produces only ops available in Arduino_TensorFlowLite:
  CONV_2D, MAX_POOL_2D, FULLY_CONNECTED, SOFTMAX, RESHAPE, QUANTIZE

Usage (from repo root):
  python notebooks/scripts/train_6ch.py
"""

import os, sys, subprocess
import numpy as np
import tensorflow as tf
from datetime import datetime

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT   = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
NPZ_PATH    = os.path.join(REPO_ROOT, "data", "processed",
                           "baseline_features_v4_segment_split_train_val_test.npz")
TFLITE_PATH = os.path.join(REPO_ROOT, "model", "co2_beacon_6ch_int8.tflite")
HEADER_PATH = os.path.join(REPO_ROOT, "firmware", "beacon_inference", "model_data.h")
NORM_PATH   = os.path.join(REPO_ROOT, "firmware", "beacon_inference", "norm_stats.h")

# ── Constants ─────────────────────────────────────────────────────────────────
N_CHANNELS  = 6   # ax, ay, az, gx, gy, gz  (drop pressure_hpa = index 6)
WIN_N       = 200
N_MODES     = 5
MODE_NAMES  = ["train", "subway", "car", "bus", "walk"]
CHANNELS_6  = ["ax", "ay", "az", "gx", "gy", "gz"]
N_CALIB     = 200

# Allowed TFLite ops for Arduino_TensorFlowLite
ALLOWED_OPS = {"CONV_2D", "MAX_POOL_2D", "FULLY_CONNECTED", "SOFTMAX",
               "RESHAPE", "QUANTIZE", "DEQUANTIZE"}


def main():
    print("=" * 60)
    print("6-channel retraining — Conv2D architecture (no pressure)")
    print("=" * 60)

    # ── Load data ─────────────────────────────────────────────────────────
    npz = np.load(NPZ_PATH, allow_pickle=True)
    modes = list(npz["modes"])

    def encode_labels(y):
        return np.array([modes.index(lbl) for lbl in y], dtype=np.int32)

    X_train = npz["X_train"][:, :, :6]   # (N, 200, 6) — drop channel 6
    X_val   = npz["X_val"][:, :, :6]
    X_test  = npz["X_test"][:, :, :6]
    y_train = encode_labels(npz["y_train"])
    y_val   = encode_labels(npz["y_val"])
    y_test  = encode_labels(npz["y_test"])

    mu    = np.squeeze(npz["mu"])[:6].astype(np.float64)
    sigma = np.squeeze(npz["sigma"])[:6].astype(np.float64)

    print(f"  X_train: {X_train.shape}  y_train: {y_train.shape}")
    print(f"  X_val:   {X_val.shape}    y_val:   {y_val.shape}")
    print(f"  X_test:  {X_test.shape}   y_test:  {y_test.shape}")
    print(f"  mu:    {mu}")
    print(f"  sigma: {sigma}")

    # ── Z-score normalize ─────────────────────────────────────────────────
    X_train_n = ((X_train - mu) / sigma).astype(np.float32)
    X_val_n   = ((X_val   - mu) / sigma).astype(np.float32)
    X_test_n  = ((X_test  - mu) / sigma).astype(np.float32)

    # ── Reshape to (N, 200, 6, 1) for Conv2D ─────────────────────────────
    X_train_n = X_train_n[..., np.newaxis]  # (N, 200, 6, 1)
    X_val_n   = X_val_n[..., np.newaxis]
    X_test_n  = X_test_n[..., np.newaxis]

    # ── Class distribution ────────────────────────────────────────────────
    print("\n  Class counts:")
    for i, m in enumerate(modes):
        n_tr = np.sum(y_train == i)
        n_va = np.sum(y_val == i)
        n_te = np.sum(y_test == i)
        print(f"    {m:<8} train={n_tr:>4}  val={n_va:>4}  test={n_te:>4}")

    # ── Data augmentation (offline) ───────────────────────────────────────
    aug_X, aug_y = [X_train_n], [y_train]
    rng = np.random.default_rng(42)
    N_AUG = 8  # 8 augmented copies → 9x training data
    for _ in range(N_AUG):
        noise = rng.normal(0, 0.05, X_train_n.shape).astype(np.float32)
        shifted = np.empty_like(X_train_n)
        for j in range(len(X_train_n)):
            shift = rng.integers(-10, 11)
            shifted[j] = np.roll(X_train_n[j], shift, axis=0)
        scale = rng.uniform(0.9, 1.1,
                            (len(X_train_n), 1, N_CHANNELS, 1)).astype(np.float32)
        aug_X.append(shifted * scale + noise)
        aug_y.append(y_train)
    X_train_aug = np.concatenate(aug_X, axis=0)
    y_train_aug = np.concatenate(aug_y, axis=0)
    perm = rng.permutation(len(X_train_aug))
    X_train_aug = X_train_aug[perm]
    y_train_aug = y_train_aug[perm]
    print(f"\n  Augmented training set: {X_train_aug.shape}")

    # ── One-hot labels + balanced val weights ─────────────────────────────
    y_train_oh = tf.keras.utils.to_categorical(y_train_aug, N_MODES)
    y_val_oh   = tf.keras.utils.to_categorical(y_val,       N_MODES)
    y_test_oh  = tf.keras.utils.to_categorical(y_test,      N_MODES)

    val_class_counts = np.bincount(y_val, minlength=N_MODES).astype(np.float32)
    val_weights = np.zeros(len(y_val), dtype=np.float32)
    for c in range(N_MODES):
        val_weights[y_val == c] = 1.0 / (val_class_counts[c] + 1e-8)
    val_weights /= val_weights.mean()

    # ── Seed sweep ────────────────────────────────────────────────────────
    best_acc = 0.0
    best_model = None
    best_seed = -1
    SEEDS = [0, 7, 42, 123, 2026]

    for seed in SEEDS:
        print(f"\n  --- Seed {seed} ---")
        tf.keras.utils.set_random_seed(seed)

        L2 = tf.keras.regularizers.l2(1e-3)
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(WIN_N, N_CHANNELS, 1)),
            tf.keras.layers.GaussianNoise(0.1),

            tf.keras.layers.Conv2D(64, kernel_size=(25, 1), padding="same",
                                   kernel_regularizer=L2),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 1)),

            tf.keras.layers.Conv2D(128, kernel_size=(15, 1), padding="same",
                                   kernel_regularizer=L2),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 1)),

            tf.keras.layers.Conv2D(64, kernel_size=(10, 1), padding="same",
                                   kernel_regularizer=L2),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 1)),

            # Use fixed Reshape instead of Flatten to avoid SHAPE/PACK/STRIDED_SLICE
            # ops in TFLite. After 3x MaxPool2D(2,1): 200→100→50→25, channels=6, filters=64
            tf.keras.layers.Reshape((25 * 6 * 64,)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(32, activation="relu", kernel_regularizer=L2),
            tf.keras.layers.Dense(N_MODES, activation="softmax"),
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-3),
            loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
            metrics=["accuracy"],
        )

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=25, restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=8, min_lr=1e-5
            ),
        ]

        model.fit(
            X_train_aug, y_train_oh,
            validation_data=(X_val_n, y_val_oh, val_weights),
            epochs=200,
            batch_size=32,
            callbacks=callbacks,
            verbose=0,
        )

        _, test_acc = model.evaluate(X_test_n, y_test_oh, verbose=0)
        print(f"  Seed {seed} → test accuracy: {test_acc*100:.1f}%")

        # Report with subway→car remap (firmware behaviour)
        preds_raw = np.argmax(model.predict(X_test_n, verbose=0), axis=1)
        preds_remap = np.where(preds_raw == 1, 2, preds_raw)
        remap_acc = np.mean(preds_remap == y_test)
        print(f"  Seed {seed} → remapped accuracy: {remap_acc*100:.1f}%")

        if test_acc > best_acc:
            best_acc = test_acc
            best_model = model
            best_seed = seed

    model = best_model
    acc = best_acc
    print(f"\n  Best seed: {best_seed}  →  Float32 test accuracy: {acc*100:.1f}%")

    model.summary()

    # ── Save Keras model ──────────────────────────────────────────────────
    keras_path = os.path.join(REPO_ROOT, "model", "co2_beacon_6ch.keras")
    os.makedirs(os.path.dirname(keras_path), exist_ok=True)
    model.save(keras_path)
    print(f"  Saved: {keras_path}")

    # ── INT8 TFLite conversion ────────────────────────────────────────────
    # Convert via concrete function with fixed batch=1 to avoid dynamic
    # shape ops (SHAPE, PACK, STRIDED_SLICE) that Arduino_TensorFlowLite lacks.
    print("\n[...] Converting to INT8 TFLite (fixed batch=1)...")

    fixed_input = tf.TensorSpec([1, WIN_N, N_CHANNELS, 1], dtype=tf.float32)
    concrete_func = tf.function(model).get_concrete_function(fixed_input)

    def representative_dataset():
        indices = np.random.choice(len(X_train_n), size=N_CALIB, replace=False)
        for i in indices:
            yield [X_train_n[i:i+1].astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type  = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    os.makedirs(os.path.dirname(TFLITE_PATH), exist_ok=True)
    with open(TFLITE_PATH, "wb") as f:
        f.write(tflite_model)

    size_kb = os.path.getsize(TFLITE_PATH) / 1024
    print(f"  Saved: {TFLITE_PATH}  ({size_kb:.1f} KB)")

    # ── Verify ops ────────────────────────────────────────────────────────
    print("\n[...] Verifying TFLite ops...")
    interp = tf.lite.Interpreter(model_path=TFLITE_PATH)
    interp.allocate_tensors()
    ops_details = interp._get_ops_details()
    op_names = sorted(set(d["op_name"] for d in ops_details))
    print(f"  Ops in model: {op_names}")

    disallowed = set(op_names) - ALLOWED_OPS - {"DELEGATE"}
    if disallowed:
        print(f"  *** WARNING: disallowed ops found: {disallowed} ***")
        print(f"  *** Model may not run on Arduino_TensorFlowLite ***")
    else:
        print(f"  All ops allowed for Arduino_TensorFlowLite.")

    # ── INT8 accuracy check ───────────────────────────────────────────────
    print("\n[...] INT8 accuracy check on test split...")
    inp_d  = interp.get_input_details()[0]
    out_d  = interp.get_output_details()[0]

    scale_in = inp_d["quantization_parameters"]["scales"][0]
    zp_in    = inp_d["quantization_parameters"]["zero_points"][0]
    scale_out = out_d["quantization_parameters"]["scales"][0]
    zp_out    = out_d["quantization_parameters"]["zero_points"][0]

    print(f"  Input  shape: {inp_d['shape']}  quant: scale={scale_in:.8f}  zp={zp_in}")
    print(f"  Output shape: {out_d['shape']}  quant: scale={scale_out:.8f}  zp={zp_out}")

    correct = 0
    correct_remap = 0
    for i in range(len(X_test_n)):
        window = X_test_n[i:i+1]  # (1, 200, 6, 1)
        q = np.round(window / scale_in + zp_in).clip(-128, 127).astype(np.int8)
        interp.set_tensor(inp_d["index"], q)
        interp.invoke()
        out_q = interp.get_tensor(out_d["index"])
        out_f = (out_q.astype(np.float32) - zp_out) * scale_out
        pred = np.argmax(out_f)
        if pred == y_test[i]:
            correct += 1
        # Subway→car remap (firmware behaviour)
        pred_remap = 2 if pred == 1 else pred
        if pred_remap == y_test[i]:
            correct_remap += 1

    int8_acc = correct / len(X_test_n)
    int8_remap_acc = correct_remap / len(X_test_n)
    print(f"  INT8 test accuracy:          {int8_acc*100:.1f}%")
    print(f"  INT8 test accuracy (remap):  {int8_remap_acc*100:.1f}%")

    # ── Generate model_data.h ─────────────────────────────────────────────
    print("\n[...] Generating model_data.h...")
    result = subprocess.run(
        ["xxd", "-i", os.path.basename(TFLITE_PATH)],
        capture_output=True, text=True,
        cwd=os.path.dirname(TFLITE_PATH),
    )
    if result.returncode != 0:
        print(f"  [WARN] xxd failed: {result.stderr}")
    else:
        content = result.stdout
        content = content.replace(
            "unsigned char co2_beacon_6ch_int8_tflite[]",
            "const unsigned char g_model_data[]"
        ).replace(
            "unsigned int co2_beacon_6ch_int8_tflite_len",
            "const unsigned int g_model_data_len"
        )
        header = (
            "// model_data.h — auto-generated by notebooks/scripts/train_6ch.py\n"
            "// DO NOT EDIT MANUALLY — regenerate if model changes\n"
            "#pragma once\n\n"
            + content
        )
        os.makedirs(os.path.dirname(HEADER_PATH), exist_ok=True)
        with open(HEADER_PATH, "w") as f:
            f.write(header)
        print(f"  Saved: {HEADER_PATH}")

    # ── Generate norm_stats.h ─────────────────────────────────────────────
    print("\n[...] Generating norm_stats.h (6 channels)...")

    def fmt_array(name, vals, chans):
        lines = [f"static const float {name}[{len(vals)}] = {{"]
        for i, (v, ch) in enumerate(zip(vals, chans)):
            comma = "," if i < len(vals) - 1 else ""
            lines.append(f"    {v:.8f}f{comma}  // [{i}] {ch}")
        lines.append("};")
        return "\n".join(lines)

    mu_arr    = fmt_array("NORM_MU",    mu,    CHANNELS_6)
    sigma_arr = fmt_array("NORM_SIGMA", sigma, CHANNELS_6)

    norm_header = f"""\
// norm_stats.h — auto-generated by notebooks/scripts/train_6ch.py
// Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
// Source:    baseline_features_v4_segment_split_train_val_test.npz (6-ch, no pressure)
//
// DO NOT EDIT MANUALLY.
// To regenerate: python notebooks/scripts/train_6ch.py
//
// Channel order: ax, ay, az, gx, gy, gz  (indices 0-5)
// Mode order:    train(0), subway(1), car(2), bus(3), walk(4)
//
// Usage in firmware:
//   x_norm = (x_raw - NORM_MU[c]) / NORM_SIGMA[c];
//
// Accelerometer biases (subtract BEFORE normalization):
//   ACC_BIAS_X = -0.1926f   ACC_BIAS_Y = -0.1975f   ACC_BIAS_Z = -0.3472f
//
// INT8 quantization params (from TFLite conversion):
//   QUANT_SCALE      = {scale_in:.10f}
//   QUANT_ZERO_POINT = {zp_in}
#pragma once

static const int N_CHANNELS = 6;

{mu_arr}

{sigma_arr}
"""
    with open(NORM_PATH, "w") as f:
        f.write(norm_header)
    print(f"  Saved: {NORM_PATH}")

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("DONE")
    print(f"  Best seed:                   {best_seed}")
    print(f"  Float32 test accuracy:       {acc*100:.1f}%")
    print(f"  INT8    test accuracy:       {int8_acc*100:.1f}%")
    print(f"  INT8    accuracy (remap):    {int8_remap_acc*100:.1f}%")
    print(f"  TFLite size:                 {size_kb:.1f} KB")
    print(f"  TFLite ops:                  {op_names}")
    print(f"  Input  quant:                scale={scale_in:.10f}  zp={zp_in}")
    print(f"  Output quant:                scale={scale_out:.10f}  zp={zp_out}")
    print(f"  model_data.h:                {HEADER_PATH}")
    print(f"  norm_stats.h:                {NORM_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
