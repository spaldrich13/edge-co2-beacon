# shl_merge_pretrain_coarse.py
# Version: 1.0
#
# Purpose:
#   Merge multiple SHL window NPZ files (already in X:(N,125,7), y:mode strings)
#   into one coarse-label pretraining dataset.
#
# Output:
#   shl_pretrain_coarse_balanced.npz (by default)
#
# Notes:
# - Balances by downsampling each class to MAX_PER_CLASS (or to min count if MAX_PER_CLASS is None).
# - Keeps label set = ["train","subway","car","bus","walk"]

import os
import numpy as np
import pandas as pd

# -------- CONFIG --------
IN_FILES = [
    "/Users/spenceraldrich/Desktop/Union/Senior/Winter 2026/ECE-499/data/processed/windowed/shl_windows_5s_50pct_bag_User2_140717.npz",
    "/Users/spenceraldrich/Desktop/Union/Senior/Winter 2026/ECE-499/data/processed/windowed/shl_windows_5s_50pct_bag_User2_180717.npz",
]

OUT_DIR = "/Users/spenceraldrich/Desktop/Union/Senior/Winter 2026/ECE-499/data/processed/windowed"
OUT_NAME = "shl_pretrain_coarse_balanced.npz"

MODES = ["train", "subway", "car", "bus", "walk"]
RNG_SEED = 42

# Set a cap to avoid massive imbalance. 3000 is a reasonable starting point here.
MAX_PER_CLASS = 3000

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)

    X_list, y_list, meta_list = [], [], []

    for f in IN_FILES:
        if not os.path.exists(f):
            raise FileNotFoundError(f"Missing input NPZ: {f}")
        d = np.load(f, allow_pickle=True)
        X = d["X"]
        y = d["y"]
        meta = d["meta"] if "meta" in d.files else np.array([{"source_npz": os.path.basename(f)}] * len(y), dtype=object)

        # filter to MODES (safety)
        keep = np.array([yi in MODES for yi in y], dtype=bool)
        X = X[keep]
        y = y[keep]
        meta = meta[keep]

        X_list.append(X)
        y_list.append(y)
        meta_list.append(meta)

        print(f"\nLoaded {os.path.basename(f)}")
        print("  X:", X.shape)
        print("  y counts:\n", pd.Series(y).value_counts().to_string())

    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)
    meta_all = np.concatenate(meta_list, axis=0)

    print("\n=== Combined (pre-balance) ===")
    print("X:", X_all.shape)
    print(pd.Series(y_all).value_counts().to_string())

    # balance / cap
    counts = pd.Series(y_all).value_counts()
    if any(m not in counts.index for m in MODES):
        missing = [m for m in MODES if m not in counts.index]
        raise RuntimeError(f"Missing modes in merged SHL set: {missing}")

    if MAX_PER_CLASS is None:
        target = int(counts.min())
    else:
        target = int(min(counts.min(), MAX_PER_CLASS))

    Xb, yb, metab = [], [], []
    for m in MODES:
        idx = np.where(y_all == m)[0]
        sel = rng.choice(idx, size=target, replace=False)
        Xb.append(X_all[sel])
        yb.append(np.array([m] * target, dtype=object))
        metab.append(meta_all[sel])

    Xb = np.concatenate(Xb, axis=0)
    yb = np.concatenate(yb, axis=0)
    metab = np.concatenate(metab, axis=0)

    perm = rng.permutation(len(yb))
    Xb = Xb[perm]
    yb = yb[perm]
    metab = metab[perm]

    print("\n=== Balanced for pretrain ===")
    print("Target per class:", target)
    print("X:", Xb.shape)
    print(pd.Series(yb).value_counts().to_string())

    out_path = os.path.join(OUT_DIR, OUT_NAME)
    np.savez_compressed(
        out_path,
        X=Xb.astype(np.float32),
        y=yb,
        feature_cols=np.array(["ax", "ay", "az", "gx", "gy", "gz", "pressure"], dtype=object),
        fs_hz=np.array([25.0], dtype=np.float32),
        win_s=np.array([5.0], dtype=np.float32),
        overlap=np.array([0.5], dtype=np.float32),
        meta=np.array(metab, dtype=object),
        modes=np.array(MODES, dtype=object),
        note=np.array([f"Merged SHL bag windows from {len(IN_FILES)} recordings; balanced to {target}/class"], dtype=object),
    )

    print("\nSaved:", out_path)

if __name__ == "__main__":
    main()
