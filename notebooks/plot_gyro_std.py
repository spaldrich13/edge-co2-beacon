# plot_gyro_std.py
# Version: 1.0
#
# Purpose:
# - Load windowed data (windows_5s_50pct.npz)
# - Compute per-window standard deviation for gyro channels (gx, gy, gz)
# - Create one slide-ready figure:
#     1) Three boxplots (gx/gy/gz std by mode)
#     2) One summary boxplot for mean gyro std (mean of gx,gy,gz std per window)
#
# Output:
# - gyro_std_boxplots.png
# - gyro_std_mean_boxplot.png

import os
import numpy as np
import matplotlib.pyplot as plt


# -----------------------
# Config
# -----------------------
NPZ_PATH = os.path.expanduser(
    "~/Desktop/Union/Senior/Winter 2026/ECE-499/data/processed/windowed/windows_5s_50pct.npz"
)
OUT_DIR = os.path.expanduser(
    "~/Desktop/Union/Senior/Winter 2026/ECE-499/figures"
)

MODE_ORDER = ["train", "subway", "car", "bus", "walk"]

GYRO_COLS = ["gx", "gy", "gz"]

SAVE_MEAN_GYRO = True  # creates an additional single-boxplot summary


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


def get_mode_data(values_1d: np.ndarray, modes: np.ndarray, mode_order: list[str]) -> tuple[list[np.ndarray], list[str]]:
    plot_data = []
    labels = []
    for m in mode_order:
        idx = np.where(modes == m)[0]
        if idx.size == 0:
            continue
        plot_data.append(values_1d[idx])
        labels.append(m)
    return plot_data, labels


# -----------------------
# Plotting
# -----------------------
def plot_three_gyro_boxplots(stds: np.ndarray, feature_cols: list[str], modes: np.ndarray) -> str:
    """
    stds: (N_windows, C) std across time axis for each channel
    """
    # map gyro names -> column index
    col_to_idx = {name: i for i, name in enumerate(feature_cols)}
    missing = [c for c in GYRO_COLS if c not in col_to_idx]
    if missing:
        raise ValueError(f"Missing gyro columns in NPZ feature_cols: {missing}. Available={feature_cols}")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)

    for ax, g in zip(axes, GYRO_COLS):
        gi = col_to_idx[g]
        vals = stds[:, gi]
        plot_data, labels = get_mode_data(vals, modes, MODE_ORDER)

        ax.boxplot(plot_data, labels=labels, showfliers=True, whis=1.5)
        ax.set_title(f"{g} std (per 5s window)")
        ax.set_xlabel("Mode")
        ax.set_ylabel("Std Dev")
        ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "gyro_std_boxplots.png")
    plt.savefig(out_path, dpi=250)
    plt.close(fig)
    return out_path


def plot_mean_gyro_boxplot(stds: np.ndarray, feature_cols: list[str], modes: np.ndarray) -> str:
    col_to_idx = {name: i for i, name in enumerate(feature_cols)}
    gi = [col_to_idx[g] for g in GYRO_COLS]
    mean_vals = np.mean(stds[:, gi], axis=1)

    fig = plt.figure(figsize=(8.5, 4.5))
    ax = plt.gca()
    plot_data, labels = get_mode_data(mean_vals, modes, MODE_ORDER)

    ax.boxplot(plot_data, labels=labels, showfliers=True, whis=1.5)
    ax.set_title("Mean Gyro Std Dev (avg of gx, gy, gz) per 5s window")
    ax.set_xlabel("Mode")
    ax.set_ylabel("Std Dev")
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "gyro_std_mean_boxplot.png")
    plt.savefig(out_path, dpi=250)
    plt.close(fig)
    return out_path


# -----------------------
# Main
# -----------------------
def main() -> None:
    ensure_dir(OUT_DIR)

    d = np.load(NPZ_PATH, allow_pickle=True)
    X = d["X"]  # (N, 125, 7)
    feature_cols = to_str_list(d["feature_cols"])
    meta = d["meta"]
    modes = extract_modes_from_meta(meta)

    # std across time axis
    stds = np.std(X, axis=1)  # (N, C)

    p1 = plot_three_gyro_boxplots(stds, feature_cols, modes)
    print(f"Saved: {p1}")

    if SAVE_MEAN_GYRO:
        p2 = plot_mean_gyro_boxplot(stds, feature_cols, modes)
        print(f"Saved: {p2}")


if __name__ == "__main__":
    main()

