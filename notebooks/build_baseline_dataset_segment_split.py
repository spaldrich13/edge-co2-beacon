import os
import numpy as np
import pandas as pd
from collections import defaultdict

# ---------------- CONFIG ----------------
IN_PATH = "/Users/spenceraldrich/Desktop/Union/Senior/Winter 2026/ECE-499/data/processed/windows_5s_50pct.npz"
OUT_DIR = "/Users/spenceraldrich/Desktop/Union/Senior/Winter 2026/ECE-499/data/processed"
OUT_NAME = "baseline_features_v2_segment_split.npz"

MODES = ["train", "subway", "car", "bus", "walk"]
TEST_FRAC = 0.30
RNG_SEED = 42

# ---------------- HELPERS ----------------
def zscore_train_test(X_train, X_test):
    mu = X_train.mean(axis=(0,1), keepdims=True)
    sigma = X_train.std(axis=(0,1), keepdims=True) + 1e-8
    return (X_train - mu)/sigma, (X_test - mu)/sigma, mu, sigma

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)

    d = np.load(IN_PATH, allow_pickle=True)
    X = d["X"]                # (N, 125, 7)
    y = d["y"]                # (N,)
    meta = d["meta"]          # array of dicts
    feature_cols = list(d["feature_cols"])

    # Filter to desired modes
    keep = [i for i, yi in enumerate(y) if yi in MODES]
    X = X[keep]
    y = y[keep]
    meta = meta[keep]

    # Build mapping: (mode, file) -> list of window indices
    groups = defaultdict(list)
    for i, (yi, mi) in enumerate(zip(y, meta)):
        file_id = mi.get("file", "UNKNOWN_FILE")
        groups[(yi, file_id)].append(i)

    # For each mode, list unique files
    mode_files = {m: sorted({f for (mm, f) in groups.keys() if mm == m}) for m in MODES}

    print("\n=== Segment inventory (unique files per mode) ===")
    for m in MODES:
        print(f"{m:>6}: {len(mode_files[m])} files")

    # Split files within each mode to approximate TEST_FRAC by windows (not by file count)
    test_idx = []
    train_idx = []

    split_report = []

    for m in MODES:
        files = mode_files[m]
        rng.shuffle(files)

        # compute windows per file
        file_counts = [(f, len(groups[(m,f)])) for f in files]
        total = sum(c for _, c in file_counts)
        target_test = int(round(TEST_FRAC * total))

        running = 0
        test_files = []
        train_files = []

        for f, c in file_counts:
            # Greedy fill to reach target_test
            if running < target_test:
                test_files.append(f)
                running += c
            else:
                train_files.append(f)

        # Collect indices
        for f in test_files:
            test_idx.extend(groups[(m,f)])
        for f in train_files:
            train_idx.extend(groups[(m,f)])

        split_report.append({
            "mode": m,
            "total_windows": total,
            "test_windows": sum(len(groups[(m,f)]) for f in test_files),
            "train_windows": sum(len(groups[(m,f)]) for f in train_files),
            "n_test_files": len(test_files),
            "n_train_files": len(train_files),
        })

    # Convert to arrays
    train_idx = np.array(train_idx, dtype=np.int64)
    test_idx = np.array(test_idx, dtype=np.int64)

    X_train = X[train_idx]
    y_train = y[train_idx]
    X_test  = X[test_idx]
    y_test  = y[test_idx]

    print("\n=== Pre-balance counts (windows) ===")
    print("TRAIN:\n", pd.Series(y_train).value_counts().to_string())
    print("\nTEST:\n", pd.Series(y_test).value_counts().to_string())

    # Balance within TRAIN and within TEST separately (downsample to min class inside each split)
    def balance_split(Xs, ys):
        counts = pd.Series(ys).value_counts()
        min_count = int(counts.min())
        Xb, yb = [], []
        for m in MODES:
            idx = np.where(ys == m)[0]
            sel = rng.choice(idx, size=min_count, replace=False)
            Xb.append(Xs[sel])
            yb.append(np.array([m]*min_count, dtype=object))
        Xb = np.concatenate(Xb, axis=0)
        yb = np.concatenate(yb, axis=0)
        perm = rng.permutation(len(yb))
        return Xb[perm], yb[perm], min_count

    X_train_b, y_train_b, train_min = balance_split(X_train, y_train)
    X_test_b,  y_test_b,  test_min  = balance_split(X_test,  y_test)

    print("\n=== Balanced counts ===")
    print("TRAIN min per class:", train_min)
    print(pd.Series(y_train_b).value_counts().to_string())
    print("\nTEST min per class:", test_min)
    print(pd.Series(y_test_b).value_counts().to_string())

    # Normalize (z-score) using TRAIN only
    X_train_z, X_test_z, mu, sigma = zscore_train_test(X_train_b, X_test_b)

    # Save dataset
    out_path = os.path.join(OUT_DIR, OUT_NAME)
    np.savez_compressed(
        out_path,
        X_train=X_train_z.astype(np.float32),
        y_train=y_train_b,
        X_test=X_test_z.astype(np.float32),
        y_test=y_test_b,
        feature_cols=np.array(feature_cols, dtype=object),
        modes=np.array(MODES, dtype=object),
        mu=mu.astype(np.float32),
        sigma=sigma.astype(np.float32),
        fs_hz=d["fs_hz"],
        win_s=d["win_s"],
        overlap=d["overlap"],
        split_report=np.array(split_report, dtype=object),
        note=np.array(["Segment-level split by source CSV file; balanced within train/test; z-score using train stats"], dtype=object),
    )

    # Print split report
    rep = pd.DataFrame(split_report)
    print("\n=== Split report (by mode) ===")
    print(rep.to_string(index=False))
    print(f"\nSaved -> {out_path}")

if __name__ == "__main__":
    main()

