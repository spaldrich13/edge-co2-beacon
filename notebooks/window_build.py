import os
import glob
import numpy as np
import pandas as pd

BASE_DIR = "/Users/spenceraldrich/Desktop/Union/Senior/Winter 2026/ECE-499/data/raw/self_collected"
OUT_DIR  = "/Users/spenceraldrich/Desktop/Union/Senior/Winter 2026/ECE-499/data/processed"

VALID_MODES = ["train", "subway", "car", "bus", "bike", "walk"]

FS_HZ = 25
WIN_S = 8.0
OVERLAP = 0.50

WIN_N = int(round(WIN_S * FS_HZ))                  # 125
STEP_N = int(round(WIN_N * (1.0 - OVERLAP)))       # ~62

DROP_START_S = 5.0
GAP_MS = 200
PRESSURE_MIN_HPA = 900
PRESSURE_MAX_HPA = 1100
TEMP_MIN_C = -10
TEMP_MAX_C = 60

EXPECTED_COLS = [
    "t_ms",
    "ax_raw","ay_raw","az_raw",
    "ax_corr","ay_corr","az_corr",
    "acc_mag_corr",
    "gx","gy","gz",
    "pressure_hpa","temp_C","alt_m"
]

FEATURE_COLS = ["ax_corr", "ay_corr", "az_corr", "gx", "gy", "gz", "pressure_hpa"]
OUT_NPZ_NAME = f"windows_{int(WIN_S)}s_{int(OVERLAP*100)}pct.npz"
OUT_SUMMARY_NAME = "window_summary.csv"

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def read_flexible_csv(path: str) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        first = f.readline().strip()

    if first.startswith("#"):
        return pd.read_csv(path, comment="#")

    if any(c.isalpha() for c in first):
        return pd.read_csv(path)

    # legacy numeric-only (no header)
    return pd.read_csv(path, header=None, names=EXPECTED_COLS)

def load_and_clean_segment(path: str) -> pd.DataFrame:
    df = read_flexible_csv(path)

    missing = [c for c in (FEATURE_COLS + ["t_ms","temp_C"]) if c not in df.columns]
    if missing:
        raise ValueError(f"Missing cols {missing} in {os.path.basename(path)}")

    df = df.sort_values("t_ms").reset_index(drop=True)

    # drop first 5 seconds relative to segment start
    t0 = int(df["t_ms"].iloc[0])
    df = df[df["t_ms"] >= (t0 + int(DROP_START_S * 1000))].reset_index(drop=True)
    if len(df) < WIN_N:
        return df

    # drop implausible pressure/temp rows
    p = df["pressure_hpa"].astype(float)
    temp = df["temp_C"].astype(float)
    mask_ok = (
        (p >= PRESSURE_MIN_HPA) & (p <= PRESSURE_MAX_HPA) &
        (temp >= TEMP_MIN_C) & (temp <= TEMP_MAX_C)
    )
    df = df[mask_ok].reset_index(drop=True)
    return df

def split_on_gaps(df: pd.DataFrame):
    if len(df) < 2:
        return []
    t = df["t_ms"].astype(np.int64).values
    dt = np.diff(t)
    gap_idxs = np.where(dt > GAP_MS)[0]
    if len(gap_idxs) == 0:
        return [df]

    blocks = []
    start = 0
    for gi in gap_idxs:
        end = gi + 1
        blk = df.iloc[start:end].reset_index(drop=True)
        if len(blk) >= WIN_N:
            blocks.append(blk)
        start = end

    blk = df.iloc[start:].reset_index(drop=True)
    if len(blk) >= WIN_N:
        blocks.append(blk)

    return blocks

def window_block(df: pd.DataFrame, mode: str, file_id: str):
    X = df[FEATURE_COLS].astype(np.float32).values
    n = X.shape[0]
    windows = []
    i = 0
    w_idx = 0
    while i + WIN_N <= n:
        Xw = X[i:i+WIN_N]
        meta = {
            "mode": mode,
            "file": file_id,
            "start_row": i,
            "end_row": i + WIN_N,
            "window_index": w_idx,
            "t_start_ms": int(df["t_ms"].iloc[i]),
            "t_end_ms": int(df["t_ms"].iloc[i + WIN_N - 1]),
        }
        windows.append((Xw, mode, meta))
        i += STEP_N
        w_idx += 1
    return windows

def main():
    ensure_dir(OUT_DIR)

    all_windows = []
    summary_rows = []

    for mode in VALID_MODES:
        mode_dir = os.path.join(BASE_DIR, mode)
        if not os.path.isdir(mode_dir):
            continue

        paths = sorted(glob.glob(os.path.join(mode_dir, "*.csv")))
        for p in paths:
            file_id = os.path.basename(p)
            try:
                df = load_and_clean_segment(p)
                if len(df) < WIN_N:
                    summary_rows.append({
                        "mode": mode, "file": file_id,
                        "status": "SKIP_TOO_SHORT_AFTER_CLEAN",
                        "n_rows_after_clean": int(len(df)),
                        "n_blocks": 0, "n_windows": 0
                    })
                    continue

                blocks = split_on_gaps(df)
                n_windows_file = 0
                for blk in blocks:
                    ws = window_block(blk, mode, file_id)
                    n_windows_file += len(ws)
                    all_windows.extend(ws)

                summary_rows.append({
                    "mode": mode, "file": file_id,
                    "status": "OK",
                    "n_rows_after_clean": int(len(df)),
                    "n_blocks": int(len(blocks)),
                    "n_windows": int(n_windows_file)
                })

            except Exception as e:
                summary_rows.append({
                    "mode": mode, "file": file_id,
                    "status": f"ERROR:{type(e).__name__}",
                    "n_rows_after_clean": "", "n_blocks": "", "n_windows": 0
                })

    if len(all_windows) == 0:
        raise RuntimeError("No windows created. Check BASE_DIR and parsing rules.")

    X = np.stack([w[0] for w in all_windows], axis=0)
    y = np.array([w[1] for w in all_windows], dtype=object)
    meta = np.array([w[2] for w in all_windows], dtype=object)

    out_npz = os.path.join(OUT_DIR, OUT_NPZ_NAME)
    np.savez_compressed(
        out_npz,
        X=X,
        y=y,
        feature_cols=np.array(FEATURE_COLS, dtype=object),
        fs_hz=np.array([FS_HZ], dtype=np.int32),
        win_s=np.array([WIN_S], dtype=np.float32),
        overlap=np.array([OVERLAP], dtype=np.float32),
        meta=meta,
    )

    df_sum = pd.DataFrame(summary_rows)
    out_sum_csv = os.path.join(OUT_DIR, OUT_SUMMARY_NAME)
    df_sum.to_csv(out_sum_csv, index=False)

    totals = df_sum[df_sum["status"] == "OK"].groupby("mode")["n_windows"].sum().sort_values(ascending=False)

    print("\n=== WINDOWING COMPLETE (flex reader) ===")
    print(f"Saved: {out_npz}")
    print(f"Saved: {out_sum_csv}")
    print(f"X shape: {X.shape} (N_windows, {WIN_N} samples, {len(FEATURE_COLS)} features)")
    print("\nWindows per mode:")
    print(totals.to_string())
    print("\nDone.")

if __name__ == "__main__":
    main()
