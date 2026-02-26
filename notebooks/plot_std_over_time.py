# plot_std_over_time.py
# Purpose:
# Option A: Plot per-window standard deviation over time (aligned to t=0 per segment/file),
# overlaying multiple modes on the same axes with different colors/markers.
#
# Notes:
# - Uses windows_5s_50pct.npz, which stores per-window meta dicts including:
#   meta[i]["mode"], meta[i]["file"], meta[i]["t_start_ms"], meta[i]["t_end_ms"]
# - Time on x-axis is elapsed seconds from the START of each file (segment), so all files align at t=0.
# - This is an exploratory plot; it compares "time since segment start" across modes.
#
# Output:
# - One PNG per feature: <feature>_std_vs_time.png
# - Optional combined plot: std_vs_time_all_features.png

import os
import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# Config (match your repo structure)
# -----------------------
NPZ_PATH = os.path.expanduser(
    "~/Desktop/Union/Senior/Winter 2026/ECE-499/data/processed/windowed/windows_5s_50pct.npz"
)
OUT_DIR = os.path.expanduser(
    "~/Desktop/Union/Senior/Winter 2026/ECE-499/figures"
)

MODE_ORDER = ["train", "subway", "car", "bus", "walk"]

# Pick which feature(s) to plot; use None to plot all features in separate figures
FEATURES_TO_PLOT = None  # e.g., ["gx", "gy", "gz"] or ["pressure_hpa"]

# How to define the time for a window:
# - "center": (t_start + t_end)/2
# - "start": t_start
TIME_ANCHOR = "center"

# Convert ms to seconds
MS_TO_S = 1e-3

# Optional: subsample points per mode (per file) to reduce clutter
MAX_POINTS_PER_FILE = None  # e.g., 300 or None for all

# Optional: show only first N seconds of each file
T_MAX_SECONDS = None  # e.g., 120 or None

# Save combined multi-panel plot
SAVE_COMBINED = True


# -----------------------
# Helpers
# -----------------------
def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def to_str_list(arr) -> list[str]:
    out = []
    for x in arr:
        if isinstance(x, bytes):
            out.append(x.decode("utf-8"))
        else:
            out.append(str(x))
    return out


def extract_modes_from_meta(meta_arr: np.ndarray) -> np.ndarray:
    return np.array([m.get("mode", "") for m in meta_arr], dtype=object)


def extract_files_from_meta(meta_arr: np.ndarray) -> np.ndarray:
    return np.array([m.get("file", "") for m in meta_arr], dtype=object)


def extract_times_from_meta(meta_arr: np.ndarray, anchor: str = "center") -> np.ndarray:
    # Returns window time in ms (absolute, per-file timestamp basis)
    t_start = np.array([m.get("t_start_ms", np.nan) for m in meta_arr], dtype=float)
    t_end = np.array([m.get("t_end_ms", np.nan) for m in meta_arr], dtype=float)
    if anchor == "start":
        return t_start
    return 0.5 * (t_start + t_end)


def stable_mode_style_map(modes: list[str]):
    # Keep consistent styles across plots
    markers = {
        "train": "o",
        "subway": "s",
        "car": "^",
        "bus": "D",
        "walk": "x",
    }
    # Let matplotlib pick colors; we only fix marker shapes
    return markers


def downsample_idx(n: int, max_n: int) -> np.ndarray:
    if max_n is None or n <= max_n:
        return np.arange(n)
    # Evenly spaced indices
    return np.linspace(0, n - 1, max_n).astype(int)


# -----------------------
# Main
# -----------------------
def main() -> None:
    ensure_dir(OUT_DIR)

    d = np.load(NPZ_PATH, allow_pickle=True)
    X = d["X"]  # (N, 125, 7)
    meta = d["meta"]
    feature_cols = to_str_list(d["feature_cols"])

    modes = extract_modes_from_meta(meta)
    files = extract_files_from_meta(meta)
    t_ms = extract_times_from_meta(meta, anchor=TIME_ANCHOR)

    # Per-window std dev across time axis
    # stds shape = (N, 7)
    stds = np.std(X, axis=1)

    # Which features are we plotting?
    if FEATURES_TO_PLOT is None:
        feats = feature_cols
    else:
        feats = FEATURES_TO_PLOT
        missing = [f for f in feats if f not in feature_cols]
        if missing:
            raise ValueError(f"Requested features not found in NPZ: {missing}. Available={feature_cols}")

    feat_to_idx = {f: i for i, f in enumerate(feature_cols)}
    markers = stable_mode_style_map(MODE_ORDER)

    # Precompute: per-file min time so we can convert absolute window time -> elapsed seconds per file
    # This aligns each file to t=0.
    unique_files = np.unique(files)
    file_t0 = {}
    for f in unique_files:
        idx = np.where(files == f)[0]
        # robust min ignoring NaN
        t0 = np.nanmin(t_ms[idx])
        file_t0[f] = t0

    def plot_one_feature(feat: str, out_path: str) -> None:
        j = feat_to_idx[feat]
        fig = plt.figure(figsize=(10, 5))
        ax = plt.gca()

        # Overlay points by mode, but keep each file aligned to t=0
        for m in MODE_ORDER:
            idx_m = np.where(modes == m)[0]
            if idx_m.size == 0:
                continue

            # We'll build aggregated (t_elapsed, std) across all files for this mode
            t_all = []
            s_all = []

            # loop per file to do per-file clipping/subsampling cleanly
            for f in np.unique(files[idx_m]):
                idx_fm = idx_m[files[idx_m] == f]
                if idx_fm.size == 0:
                    continue

                # elapsed time in seconds from file start
                t_elapsed = (t_ms[idx_fm] - file_t0[f]) * MS_TO_S
                s = stds[idx_fm, j]

                # optional: limit time range
                if T_MAX_SECONDS is not None:
                    keep = t_elapsed <= T_MAX_SECONDS
                    t_elapsed = t_elapsed[keep]
                    s = s[keep]
                    if t_elapsed.size == 0:
                        continue

                # optional: subsample to reduce clutter
                keep_idx = downsample_idx(len(t_elapsed), MAX_POINTS_PER_FILE)
                t_elapsed = t_elapsed[keep_idx]
                s = s[keep_idx]

                t_all.append(t_elapsed)
                s_all.append(s)

            if len(t_all) == 0:
                continue

            t_all = np.concatenate(t_all)
            s_all = np.concatenate(s_all)

            ax.scatter(
                t_all,
                s_all,
                s=14,
                alpha=0.6,
                marker=markers.get(m, "o"),
                label=m,
            )

        ax.set_title(f"Std Dev vs Time (aligned per segment): {feat}")
        ax.set_xlabel("Elapsed time since segment start (s)")
        ax.set_ylabel("Std Dev within 5-second window")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", ncols=3, frameon=True)
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"Saved: {out_path}")

    # One figure per feature
    for feat in feats:
        out_path = os.path.join(OUT_DIR, f"{feat}_std_vs_time.png")
        plot_one_feature(feat, out_path)

    # Optional combined multi-panel figure (all features)
    if SAVE_COMBINED and FEATURES_TO_PLOT is None:
        n = len(feature_cols)
        fig, axes = plt.subplots(n, 1, figsize=(11, 2.4 * n), sharex=True)
        if n == 1:
            axes = [axes]

        for i, feat in enumerate(feature_cols):
            ax = axes[i]
            j = feat_to_idx[feat]

            for m in MODE_ORDER:
                idx_m = np.where(modes == m)[0]
                if idx_m.size == 0:
                    continue

                t_all = []
                s_all = []
                for f in np.unique(files[idx_m]):
                    idx_fm = idx_m[files[idx_m] == f]
                    if idx_fm.size == 0:
                        continue
                    t_elapsed = (t_ms[idx_fm] - file_t0[f]) * MS_TO_S
                    s = stds[idx_fm, j]

                    if T_MAX_SECONDS is not None:
                        keep = t_elapsed <= T_MAX_SECONDS
                        t_elapsed = t_elapsed[keep]
                        s = s[keep]
                        if t_elapsed.size == 0:
                            continue

                    keep_idx = downsample_idx(len(t_elapsed), MAX_POINTS_PER_FILE)
                    t_elapsed = t_elapsed[keep_idx]
                    s = s[keep_idx]

                    t_all.append(t_elapsed)
                    s_all.append(s)

                if len(t_all) == 0:
                    continue

                t_all = np.concatenate(t_all)
                s_all = np.concatenate(s_all)

                ax.scatter(
                    t_all, s_all,
                    s=10, alpha=0.55,
                    marker=markers.get(m, "o"),
                    label=m if i == 0 else None  # legend once
                )

            ax.set_title(feat)
            ax.set_ylabel("Std")
            ax.grid(True, alpha=0.25)

        axes[-1].set_xlabel("Elapsed time since segment start (s)")
        if n > 0:
            axes[0].legend(loc="best", ncols=3, frameon=True)
        plt.tight_layout()
        out_path = os.path.join(OUT_DIR, "std_vs_time_all_features.png")
        plt.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

