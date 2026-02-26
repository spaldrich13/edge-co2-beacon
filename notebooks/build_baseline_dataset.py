import os
import numpy as np
import pandas as pd

# ---------------- CONFIG ----------------
IN_PATH = "/Users/spenceraldrich/Desktop/Union/Senior/Winter 2026/ECE-499/data/processed/windows_5s_50pct.npz"
OUT_DIR = "/Users/spenceraldrich/Desktop/Union/Senior/Winter 2026/ECE-499/data/processed"
OUT_NAME = "baseline_features_v1.npz"

# Modes to include right now (bike excluded)
MODES = ["train", "subway", "car", "bus", "walk"]

# Train/test split
TEST_FRAC = 0.30
RNG_SEED = 42

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)

    data = np.load(IN_PATH, allow_pickle=True)
    X = data["X"]  # (N, 125, 7)
    y = data["y"]  # (N,)
    feature_cols = list(data["feature_cols"])

    # Filter to desired modes only
    keep_idx = [i for i, yi in enumerate(y) if yi in MODES]
    X = X[keep_idx]
    y = y[keep_idx]

    # Count per mode
    counts = pd.Series(y).value_counts()
    print("\n=== Raw window counts ===")
    print(counts.to_string())

    # Balance: downsample each mode to min count
    min_count = int(counts.min())
    print(f"\nBalancing to min_count = {min_count} windows per mode")

    X_bal = []
    y_bal = []

    for m in MODES:
        idx = np.where(y == m)[0]
        sel = rng.choice(idx, size=min_count, replace=False)
        X_bal.append(X[sel])
        y_bal.append(np.array([m]*min_count, dtype=object))

    X_bal = np.concatenate(X_bal, axis=0)
    y_bal = np.concatenate(y_bal, axis=0)

    # Shuffle
    perm = rng.permutation(len(y_bal))
    X_bal = X_bal[perm]
    y_bal = y_bal[perm]

    # Train/test split
    n = len(y_bal)
    n_test = int(round(TEST_FRAC * n))
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]

    # NOTE: train_idx/test_idx are indices into the *pre-shuffled* data; we already shuffled,
    # so we can just slice instead:
    X_test = X_bal[:n_test]
    y_test = y_bal[:n_test]
    X_train = X_bal[n_test:]
    y_train = y_bal[n_test:]

    # Normalize (z-score) per channel using training set stats
    # Compute mean/std over (samples and time)
    mu = X_train.mean(axis=(0,1), keepdims=True)
    sigma = X_train.std(axis=(0,1), keepdims=True) + 1e-8

    X_train_z = (X_train - mu) / sigma
    X_test_z  = (X_test  - mu) / sigma

    # Save
    out_path = os.path.join(OUT_DIR, OUT_NAME)
    np.savez_compressed(
        out_path,
        X_train=X_train_z.astype(np.float32),
        y_train=y_train,
        X_test=X_test_z.astype(np.float32),
        y_test=y_test,
        feature_cols=np.array(feature_cols, dtype=object),
        modes=np.array(MODES, dtype=object),
        mu=mu.astype(np.float32),
        sigma=sigma.astype(np.float32),
        fs_hz=data["fs_hz"],
        win_s=data["win_s"],
        overlap=data["overlap"],
        note=np.array(["Balanced per-class to min_count; z-score using training stats"], dtype=object),
    )

    # Print summaries
    print("\n=== Balanced counts ===")
    print(pd.Series(y_bal).value_counts().to_string())
    print("\n=== Split ===")
    print(f"Train: {len(y_train)} windows")
    print(f"Test : {len(y_test)} windows")
    print(f"Saved -> {out_path}")

if __name__ == "__main__":
    main()

