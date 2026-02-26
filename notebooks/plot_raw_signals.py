# plot_raw_signals.py
# Version: 1.1 (Fix ParserError: auto-skip preamble / inconsistent CSV lines)
#
# Purpose:
# - Plot raw time-series sensor channels (ax,ay,az,gx,gy,gz,pressure) from representative CSVs
# - Robust CSV loading for files that include metadata/preamble lines before the real header
#
# Output:
# - For each mode, one PNG per channel: <mode>_<channel>_raw.png
# - Optionally, a single multi-channel figure per mode: <mode>_all_channels_raw.png

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------
# Config
# -----------------------
RAW_ROOT = os.path.expanduser(
    "~/Desktop/Union/Senior/Winter 2026/ECE-499/data/raw/self_collected"
)
OUT_DIR = os.path.expanduser(
    "~/Desktop/Union/Senior/Winter 2026/ECE-499/figures_raw_signals"
)

# Plot duration
PLOT_SECONDS = 10.0
SKIP_SECONDS = 0.0  # set >0 if you want to skip initial settling

# Sampling rate used if no timestamp column exists
FS_HZ = 25.0

# Target channels (preferred)
CHANNELS = [
    "ax_corr", "ay_corr", "az_corr",
    "gx", "gy", "gz",
    "pressure_hpa",
]

# Fallback column candidates if preferred names are missing
FALLBACKS = {
    "ax_corr": ["ax_corr", "ax", "accel_x", "acc_x", "a_x"],
    "ay_corr": ["ay_corr", "ay", "accel_y", "acc_y", "a_y"],
    "az_corr": ["az_corr", "az", "accel_z", "acc_z", "a_z"],
    "gx": ["gx", "gyro_x", "g_x"],
    "gy": ["gy", "gyro_y", "g_y"],
    "gz": ["gz", "gyro_z", "g_z"],
    "pressure_hpa": ["pressure_hpa", "pressure", "press_hpa", "bmp_pressure_hpa"],
}

# Timestamp column candidates
TIME_COLS = ["t_ms", "time_ms", "timestamp_ms", "timestamp", "t"]


# -----------------------
# Helpers
# -----------------------
def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def pick_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    return None


def sniff_header_start(path: str, max_lines: int = 60, min_commas: int = 3) -> int:
    """
    Returns the line index (0-based) where the 'real' CSV header likely starts.
    Many of your logs have a few metadata lines (1 field), then the actual header
    with many comma-separated columns.
    """
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = []
            for _ in range(max_lines):
                line = f.readline()
                if not line:
                    break
                lines.append(line.rstrip("\n"))
    except OSError:
        return 0

    # Find first line that looks like a real header: enough commas and not empty
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        if line.count(",") >= min_commas:
            return i

    # If nothing obvious, default to 0
    return 0


def load_csv_flex(path: str) -> pd.DataFrame:
    """
    Robust CSV loader:
    - Detect and skip preamble lines before the real header
    - Use python engine for tolerance
    - Skip malformed lines instead of crashing
    """
    start = sniff_header_start(path, max_lines=80, min_commas=3)

    # First attempt: normal comma CSV from detected header line
    try:
        return pd.read_csv(
            path,
            skiprows=start,
            engine="python",
            sep=",",
            on_bad_lines="skip",
        )
    except Exception:
        pass

    # Second attempt: sometimes there are extra non-CSV lines scattered; increase tolerance
    try:
        return pd.read_csv(
            path,
            skiprows=start,
            engine="python",
            sep=",",
            on_bad_lines="skip",
            skip_blank_lines=True,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to parse CSV: {os.path.basename(path)} (start line {start}) -> {e}") from e


def build_time_vector(df: pd.DataFrame) -> np.ndarray:
    tcol = pick_column(df, TIME_COLS)
    if tcol is not None:
        t = pd.to_numeric(df[tcol], errors="coerce").to_numpy(dtype=float)
        # drop NaNs at start if present
        if np.all(np.isnan(t)):
            # fallback to index-based time
            n = len(df)
            return np.arange(n) / FS_HZ

        # if milliseconds, convert to seconds (heuristic: large values)
        med = np.nanmedian(t)
        if med > 1e3:
            t = (t - np.nanmin(t)) / 1000.0
        else:
            t = t - np.nanmin(t)
        return t

    # Fallback: derive time from sample index
    n = len(df)
    return np.arange(n) / FS_HZ


def choose_representative_file(mode_dir: str) -> str | None:
    # Prefer *_seg*.csv if present, otherwise any CSV
    segs = sorted(glob.glob(os.path.join(mode_dir, "*_seg*.csv")))
    if segs:
        return segs[0]
    all_csv = sorted(glob.glob(os.path.join(mode_dir, "*.csv")))
    return all_csv[0] if all_csv else None


def slice_time(df: pd.DataFrame, t: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    t0 = SKIP_SECONDS
    t1 = SKIP_SECONDS + PLOT_SECONDS
    mask = (t >= t0) & (t <= t1)
    # If time vector is weird (all NaN), fallback to first N samples
    if mask.sum() < 5:
        n = min(int(PLOT_SECONDS * FS_HZ), len(df))
        return df.iloc[:n].reset_index(drop=True), np.arange(n) / FS_HZ
    return df.loc[mask].reset_index(drop=True), t[mask] - t0


def plot_channel(mode: str, ch_name: str, t: np.ndarray, y: np.ndarray, out_path: str) -> None:
    fig = plt.figure(figsize=(10, 4))
    ax = plt.gca()
    ax.plot(t, y)
    ax.set_title(f"{mode.upper()} raw signal: {ch_name} ({PLOT_SECONDS:.0f}s window)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ch_name)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_all_channels(mode: str, t: np.ndarray, series: dict[str, np.ndarray], out_path: str) -> None:
    fig, axes = plt.subplots(len(CHANNELS), 1, figsize=(10, 2.0 * len(CHANNELS)), sharex=True)
    if len(CHANNELS) == 1:
        axes = [axes]
    for i, ch in enumerate(CHANNELS):
        ax = axes[i]
        ax.plot(t, series[ch])
        ax.set_ylabel(ch)
        ax.grid(True, alpha=0.3)
    axes[0].set_title(f"{mode.upper()} raw signals ({PLOT_SECONDS:.0f}s)")
    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


# -----------------------
# Main
# -----------------------
def main() -> None:
    ensure_dir(OUT_DIR)

    mode_folders = [d for d in sorted(os.listdir(RAW_ROOT)) if os.path.isdir(os.path.join(RAW_ROOT, d))]
    if not mode_folders:
        raise RuntimeError(f"No mode folders found under {RAW_ROOT}")

    for mode in mode_folders:
        mode_dir = os.path.join(RAW_ROOT, mode)
        fpath = choose_representative_file(mode_dir)
        if fpath is None:
            print(f"[WARN] No CSV found for mode: {mode}")
            continue

        df = load_csv_flex(fpath)

        # If the file parsed but produced 1 column, it likely still started too early.
        # Re-sniff with a stricter threshold (more commas) and retry once.
        if df.shape[1] <= 1:
            start = sniff_header_start(fpath, max_lines=120, min_commas=6)
            df = pd.read_csv(fpath, skiprows=start, engine="python", sep=",", on_bad_lines="skip")

        t = build_time_vector(df)
        df_s, t_s = slice_time(df, t)

        series = {}
        missing = []
        for ch in CHANNELS:
            col = pick_column(df_s, FALLBACKS.get(ch, [ch]))
            if col is None:
                missing.append(ch)
                continue
            series[ch] = pd.to_numeric(df_s[col], errors="coerce").to_numpy(dtype=float)

        if missing:
            print(f"[WARN] {mode}: missing channels {missing} in {os.path.basename(fpath)}")

        # Save per-channel plots (only for channels that exist)
        for ch, y in series.items():
            out_path = os.path.join(OUT_DIR, f"{mode}_{ch}_raw.png")
            plot_channel(mode, ch, t_s[:len(y)], y, out_path)
            print(f"Saved: {out_path}")

        # Save combined figure only if all channels exist
        if all(ch in series for ch in CHANNELS):
            out_path = os.path.join(OUT_DIR, f"{mode}_all_channels_raw.png")
            plot_all_channels(mode, t_s, series, out_path)
            print(f"Saved: {out_path}")

        print(f"[INFO] Mode {mode}: source file used -> {fpath}")


if __name__ == "__main__":
    main()
