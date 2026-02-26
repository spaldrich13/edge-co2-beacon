# shl_build_windows_bag.py
# Version: 1.3
#
# Builds SHL bag-placement windows aligned to your self-collected shape:
#   X: (N, 125, 7) channels=[ax,ay,az,gx,gy,gz,pressure]
#   y: (N,) in {"train","subway","car","bus","walk"} (mapped from SHL fine labels)
#
# Uses INTERVAL labels from labels_track_main.txt (fine label track).
# This is robust when Label.txt is mismatched/not from the same recording.

import os
import numpy as np
import pandas as pd

# --------- CONFIG ----------
SHL_DIR = "/Users/spenceraldrich/Desktop/Union/Senior/Winter 2026/ECE-499/data/raw/shl/User2/180717"
BAG_MOTION = os.path.join(SHL_DIR, "Bag_Motion.txt")

# Interval fine-label file: start_ms end_ms fine_label_id
INTERVAL_LABELS_TXT = os.path.join(SHL_DIR, "labels_track_main.txt")

OUT_DIR = "/Users/spenceraldrich/Desktop/Union/Senior/Winter 2026/ECE-499/data/processed/windowed"
OUT_NAME = "shl_windows_5s_50pct_bag_User2_180717.npz"

FS_TARGET = 25.0
WIN_S = 5.0
OVERLAP = 0.50

ACC_IDX = [1, 2, 3]
GYR_IDX = [4, 5, 6]
PRESSURE_CANDIDATES = [20, 21, 22]  # per SHL docs: 20=pressure, 21=alt, 22=temp

# ---- Fine-label IDs from SHL documentation for labels_track_main.txt ----
# (We map fine IDs -> project modes.)
FINE_ID_TO_PROJECT = {
    # Walking
    4: "walk",   # Walking;Outside
    5: "walk",   # Walking;Inside

    # Car
    8: "car",    # Car;Driver
    9: "car",    # Car;Passenger

    # Bus
    10: "bus",   # Bus;Stand
    11: "bus",   # Bus;Sit
    12: "bus",   # Bus;Up;Stand
    13: "bus",   # Bus;Up;Sit

    # Train
    14: "train", # Train;Stand
    15: "train", # Train;Sit

    # Subway
    16: "subway",# Subway;Stand
    17: "subway" # Subway;Sit
}

PROJECT_KEEP = {"walk", "car", "bus", "train", "subway"}

# ---------------- HELPERS ----------------
def pick_pressure_column(motion_path, candidates):
    probe_n = 200000
    usecols = [0] + candidates
    df = pd.read_csv(
        motion_path, sep=r"\s+", header=None, usecols=usecols, nrows=probe_n,
        engine="python", on_bad_lines="skip"
    )
    df = df.apply(pd.to_numeric, errors="coerce")
    nan_fracs = {c: df[c].isna().mean() for c in candidates if c in df.columns}
    if not nan_fracs:
        raise ValueError("No pressure candidate columns found.")
    best = min(nan_fracs, key=nan_fracs.get)
    print("Pressure candidate NaN fractions:", nan_fracs)
    print("Picked pressure column:", best)
    return best

def load_interval_labels(path):
    # expects: start_ms end_ms fine_label_id
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python", on_bad_lines="skip")
    if df.shape[1] < 3:
        raise ValueError(f"Interval labels expected >=3 cols, got {df.shape[1]}")
    df = df.iloc[:, :3].copy()
    df.columns = ["t_start", "t_end", "fine_id"]

    df["t_start"] = pd.to_numeric(df["t_start"], errors="coerce")
    df["t_end"]   = pd.to_numeric(df["t_end"], errors="coerce")
    df["fine_id"] = pd.to_numeric(df["fine_id"], errors="coerce")
    df = df.dropna().copy()

    df["fine_id"] = df["fine_id"].astype(int)
    df["project_mode"] = df["fine_id"].map(FINE_ID_TO_PROJECT)
    df = df.dropna(subset=["project_mode"]).copy()

    return df.sort_values("t_start").reset_index(drop=True)

def assign_interval_labels(t_ms, intervals_df):
    starts = intervals_df["t_start"].to_numpy()
    ends   = intervals_df["t_end"].to_numpy()
    modes  = intervals_df["project_mode"].to_numpy()

    out = np.array([None] * len(t_ms), dtype=object)
    j = 0
    for i, t in enumerate(t_ms):
        while j < len(starts) and t > ends[j]:
            j += 1
        if j < len(starts) and (starts[j] <= t <= ends[j]):
            out[i] = modes[j]
    return out

def downsample_to_fs(t_ms, X, fs_target):
    t_s = (t_ms - t_ms[0]) / 1000.0
    dt = 1.0 / fs_target
    t_grid = np.arange(0, t_s[-1], dt)

    idx = np.searchsorted(t_s, t_grid)
    idx = np.clip(idx, 1, len(t_s) - 1)
    left, right = idx - 1, idx
    choose_right = (np.abs(t_s[right] - t_grid) < np.abs(t_s[left] - t_grid))
    pick = np.where(choose_right, right, left)
    pick = np.unique(pick)
    return t_ms[pick], X[pick]

def window_stack(X, y, t_ms, fs, win_s, overlap):
    win_len = int(round(fs * win_s))      # 125
    hop = int(round(win_len * (1 - overlap)))
    if hop <= 0:
        raise ValueError("Invalid overlap")

    Xw, yw, meta = [], [], []
    N = len(y)

    for start in range(0, N - win_len + 1, hop):
        end = start + win_len
        labels = y[start:end]

        if any(v is None for v in labels):
            continue
        if len(set(labels)) != 1:
            continue

        lab = labels[0]
        if lab not in PROJECT_KEEP:
            continue

        Xw.append(X[start:end])
        yw.append(lab)
        meta.append({"source": "SHL_Bag", "t_start_ms": float(t_ms[start]), "t_end_ms": float(t_ms[end - 1])})

    if not Xw:
        return None, None, None
    return np.stack(Xw).astype(np.float32), np.array(yw, dtype=object), np.array(meta, dtype=object)

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    if not os.path.exists(INTERVAL_LABELS_TXT):
        raise FileNotFoundError(f"Missing interval labels file: {INTERVAL_LABELS_TXT}")

    intervals = load_interval_labels(INTERVAL_LABELS_TXT)
    print("Interval labels kept (project modes):")
    print(intervals["project_mode"].value_counts().to_string())

    p_idx = pick_pressure_column(BAG_MOTION, PRESSURE_CANDIDATES)

    usecols = sorted(set([0] + ACC_IDX + GYR_IDX + [p_idx]))
    colpos = {c: i for i, c in enumerate(usecols)}
    print("Using columns:", usecols)

    all_t, all_X = [], []

    chunks = pd.read_csv(
        BAG_MOTION,
        sep=r"\s+",
        header=None,
        usecols=usecols,
        chunksize=200000,
        engine="python",
        on_bad_lines="skip",
    )

    for ch in chunks:
        ch = ch.apply(pd.to_numeric, errors="coerce")
        arr = ch.to_numpy(dtype=np.float64, copy=False)

        t = arr[:, colpos[0]]
        ax = arr[:, colpos[ACC_IDX[0]]]
        ay = arr[:, colpos[ACC_IDX[1]]]
        az = arr[:, colpos[ACC_IDX[2]]]
        gx = arr[:, colpos[GYR_IDX[0]]]
        gy = arr[:, colpos[GYR_IDX[1]]]
        gz = arr[:, colpos[GYR_IDX[2]]]
        pr = arr[:, colpos[p_idx]]

        X = np.stack([ax, ay, az, gx, gy, gz, pr], axis=1)

        good = ~np.isnan(t) & (~np.isnan(X).any(axis=1))
        t = t[good]
        X = X[good]
        if len(t) == 0:
            continue

        all_t.append(t)
        all_X.append(X)

    t_ms = np.concatenate(all_t)
    X = np.concatenate(all_X)

    order = np.argsort(t_ms)
    t_ms = t_ms[order]
    X = X[order]

    # label by fine-label intervals
    y = assign_interval_labels(t_ms, intervals)

    # downsample then relabel
    t_ds, X_ds = downsample_to_fs(t_ms, X, FS_TARGET)
    y_ds = assign_interval_labels(t_ds, intervals)

    Xw, yw, meta = window_stack(X_ds, y_ds, t_ds, FS_TARGET, WIN_S, OVERLAP)
    if Xw is None:
        raise RuntimeError("No windows produced. Likely wrong interval file for this day or no overlapping labels.")

    out_path = os.path.join(OUT_DIR, OUT_NAME)
    np.savez_compressed(
        out_path,
        X=Xw,
        y=yw,
        feature_cols=np.array(["ax", "ay", "az", "gx", "gy", "gz", "pressure"], dtype=object),
        fs_hz=np.array([FS_TARGET], dtype=np.float32),
        win_s=np.array([WIN_S], dtype=np.float32),
        overlap=np.array([OVERLAP], dtype=np.float32),
        meta=np.array(meta, dtype=object),
        note=np.array(["SHL Bag; labels_track_main (fine interval labels); downsample 25Hz; 5s windows 50% overlap; single-label windows"], dtype=object),
    )

    print("\n=== SHL WINDOWING COMPLETE ===")
    print("Saved:", out_path)
    print("X shape:", Xw.shape)
    print("Windows per mode:\n", pd.Series(yw).value_counts().to_string())

if __name__ == "__main__":
    main()
