# build_baseline_dataset_segment_split_v3.py
# Version: 3.1
#
# Purpose:
# - Load windowed data (windows_5s_50pct.npz)
# - Split by *file/segment* into TRAIN/VAL/TEST (no window leakage)
# - Handle small file-count modes (e.g., car has only 2 files)
# - Balance TRAIN and TEST only (VAL is unbalanced but segment-based, used for monitoring)
# - Z-score normalize using TRAIN statistics only
#
# Output:
# - baseline_features_v3_segment_split_train_val_test.npz

import os
import numpy as np
import pandas as pd
from collections import defaultdict

# ---------------- CONFIG ----------------
IN_PATH = "/Users/spenceraldrich/Desktop/Union/Senior/Winter 2026/ECE-499/data/processed/windows_8s_50pct.npz"
OUT_DIR = "/Users/spenceraldrich/Desktop/Union/Senior/Winter 2026/ECE-499/data/processed"
OUT_NAME = "baseline_features_v4_segment_split_train_val_test.npz"

MODES = ["train", "subway", "car", "bus", "walk"]

TEST_FRAC = 0.20
VAL_FRAC = 0.20
RNG_SEED = 42


# ---------------- HELPERS ----------------
def zscore_train_apply(X_train, X_val, X_test):
    mu = X_train.mean(axis=(0, 1), keepdims=True)
    sigma = X_train.std(axis=(0, 1), keepdims=True) + 1e-8
    return (X_train - mu) / sigma, (X_val - mu) / sigma, (X_test - mu) / sigma, mu, sigma


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)

    d = np.load(IN_PATH, allow_pickle=True)
    X = d["X"]
    y = d["y"]
    meta = d["meta"]
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

    mode_files = {m: sorted({f for (mm, f) in groups.keys() if mm == m}) for m in MODES}

    print("\n=== Segment inventory (unique files per mode) ===")
    for m in MODES:
        print(f"{m:>6}: {len(mode_files[m])} files")

    train_idx, val_idx, test_idx = [], [], []
    split_report = []

    for m in MODES:
        files = mode_files[m]
        rng.shuffle(files)

        file_counts = [(f, len(groups[(m, f)])) for f in files]
        total = sum(c for _, c in file_counts)
        n_files = len(files)

        test_files, val_files, train_files = [], [], []

        # ---- Small-file rules ----
        if n_files == 1:
            # Not ideal, but keep it in TRAIN so the model can see it.
            train_files = files[:]
        elif n_files == 2:
            # Force: 1 train, 1 test (no val)
            train_files = [files[0]]
            test_files = [files[1]]
        elif n_files == 3:
            # Force: 1 train, 1 val, 1 test
            train_files = [files[0]]
            val_files = [files[1]]
            test_files = [files[2]]
        else:
            # Greedy fill by windows (original behavior)
            target_test = int(round(TEST_FRAC * total))
            target_val = int(round(VAL_FRAC * total))

            running_test = 0
            running_val = 0

            for f, c in file_counts:
                if running_test < target_test:
                    test_files.append(f)
                    running_test += c
                elif running_val < target_val:
                    val_files.append(f)
                    running_val += c
                else:
                    train_files.append(f)

            # Safety: ensure at least 1 train file
            if len(train_files) == 0 and len(val_files) > 0:
                train_files.append(val_files.pop())
            if len(train_files) == 0 and len(test_files) > 0:
                train_files.append(test_files.pop())

        # Collect indices
        for f in train_files:
            train_idx.extend(groups[(m, f)])
        for f in val_files:
            val_idx.extend(groups[(m, f)])
        for f in test_files:
            test_idx.extend(groups[(m, f)])

        split_report.append(
            {
                "mode": m,
                "n_files": n_files,
                "total_windows": total,
                "train_windows": sum(len(groups[(m, f)]) for f in train_files),
                "val_windows": sum(len(groups[(m, f)]) for f in val_files),
                "test_windows": sum(len(groups[(m, f)]) for f in test_files),
                "n_train_files": len(train_files),
                "n_val_files": len(val_files),
                "n_test_files": len(test_files),
            }
        )

    train_idx = np.array(train_idx, dtype=np.int64)
    val_idx = np.array(val_idx, dtype=np.int64)
    test_idx = np.array(test_idx, dtype=np.int64)

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    print("\n=== Pre-balance counts (windows) ===")
    print("TRAIN:\n", pd.Series(y_train).value_counts().to_string())
    print("\nVAL:\n", pd.Series(y_val).value_counts().to_string())
    print("\nTEST:\n", pd.Series(y_test).value_counts().to_string())

    # Balance within TRAIN and within TEST separately
    def balance_split(Xs, ys):
        counts = pd.Series(ys).value_counts()
        min_count = int(counts.min())
        Xb, yb = [], []
        for mm in MODES:
            idx = np.where(ys == mm)[0]
            if len(idx) == 0:
                raise ValueError(
                    f"Split is missing mode '{mm}'. Cannot balance. "
                    f"Counts={counts.to_dict()}"
                )
            sel = rng.choice(idx, size=min_count, replace=False)
            Xb.append(Xs[sel])
            yb.append(np.array([mm] * min_count, dtype=object))
        Xb = np.concatenate(Xb, axis=0)
        yb = np.concatenate(yb, axis=0)
        perm = rng.permutation(len(yb))
        return Xb[perm], yb[perm], min_count

    X_train_b, y_train_b, train_min = balance_split(X_train, y_train)
    X_test_b, y_test_b, test_min = balance_split(X_test, y_test)

    # VAL remains unbalanced (segment-based) so small-file modes (like car) don't break it
    X_val_b, y_val_b = X_val, y_val
    val_min = None

    print("\n=== Balanced counts ===")
    print("TRAIN min per class:", train_min)
    print(pd.Series(y_train_b).value_counts().to_string())
    print("\nVAL: unbalanced (used for monitoring only)")
    print(pd.Series(y_val_b).value_counts().to_string())
    print("\nTEST min per class:", test_min)
    print(pd.Series(y_test_b).value_counts().to_string())

    # Normalize (z-score) using TRAIN only
    X_train_z, X_val_z, X_test_z, mu, sigma = zscore_train_apply(X_train_b, X_val_b, X_test_b)

    # Save dataset
    out_path = os.path.join(OUT_DIR, OUT_NAME)
    np.savez_compressed(
        out_path,
        X_train=X_train_z.astype(np.float32),
        y_train=y_train_b,
        X_val=X_val_z.astype(np.float32),
        y_val=y_val_b,
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
        note=np.array(
            [
                "Segment-level split by source CSV file into train/val/test; "
                "TRAIN+TEST balanced; VAL unbalanced; z-score using train stats"
            ],
            dtype=object,
        ),
    )

    # Print split report
    rep = pd.DataFrame(split_report)
    print("\n=== Split report (by mode) ===")
    print(rep.to_string(index=False))
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
