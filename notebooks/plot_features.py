# plot_features.py
# Version: 2.0 (Boxplots, no density)
#
# Purpose:
# - Load windowed data (windows_5s_50pct.npz)
# - Compute per-window standard deviation for each channel
# - Create advisor-friendly boxplots (counts/density not used)
#
# Output:
# - One PNG per feature: <feature>_std_boxplot.png
# - (Optional) a combined figure: all_features_std_boxplots.png

import os
import numpy as np
import matplotlib.pyplot as plt


# -----------------------
# Config
# -----------------------
# Adjust these paths to match your repo structure if needed
NPZ_PATH = os.path.expanduser(
    "~/Desktop/Union/Senior/Winter 2026/ECE-499/data/processed/windowed/windows_5s_50pct.npz"
)
OUT_DIR = os.path.expanduser(
    "~/Desktop/Union/Senior/Winter 2026/ECE-499/figures"
)

# Mode order for consistent plots
MODE_ORDER = ["train", "subway", "car", "bus", "walk"]

# If True, also saves a combined figure with 7 rows (one per feature)
SAVE_COMBINED = True


# -----------------------
# Helpers
# -----------------------
def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def to_str_list(arr) -> list[str]:
    # Handles numpy arrays of strings or bytes
    out = []
    for x in arr:
        if isinstance(x, bytes):
            out.append(x.decode("utf-8"))
        else:
            out.append(str(x))
    return out


def extract_modes_from_meta(meta_arr: np.ndarray) -> np.ndarray:
    # meta is an array of dicts (allow_pickle=True)
    return np.array([m.get("mode", "") for m in meta_arr], dtype=object)


def boxplot_feature(
    feature_name: str,
    data_by_mode: dict[str, np.ndarray],
    out_path: str,
    mode_order: list[str],
) -> None:
    fig = plt.figure(figsize=(9, 5))
    ax = plt.gca()

    plot_data = [data_by_mode[m] for m in mode_order if m in data_by_mode]
    labels = [m for m in mode_order if m in data_by_mode]

    ax.boxplot(
        plot_data,
        labels=labels,
        showfliers=True,
        whis=1.5,
        patch_artist=False,
    )
    ax.set_title(f"Window Std Dev by Mode: {feature_name}")
    ax.set_xlabel("Mode")
    ax.set_ylabel("Std Dev (within 5-second window)")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


# -----------------------
# Main
# -----------------------
def main() -> None:
    ensure_dir(OUT_DIR)

    d = np.load(NPZ_PATH, allow_pickle=True)
    X = d["X"]  # shape: (N_windows, 125, 7)
    feature_cols = to_str_list(d["feature_cols"])
    meta = d["meta"]

    # Modes per window
    modes = extract_modes_from_meta(meta)

    # Compute std per window per feature across time axis (125 samples)
    # stds shape: (N_windows, 7)
    stds = np.std(X, axis=1)

    # Build per-feature dicts: feature -> mode -> array of std values
    per_feature_mode = {feat: {} for feat in feature_cols}

    # Only keep modes we care about (and enforce consistent ordering)
    unique_modes = set(modes.tolist())
    for m in MODE_ORDER:
        if m not in unique_modes:
            continue
        idx = np.where(modes == m)[0]
        if idx.size == 0:
            continue
        for j, feat in enumerate(feature_cols):
            per_feature_mode[feat][m] = stds[idx, j]

    # Save one plot per feature
    for feat in feature_cols:
        out_path = os.path.join(OUT_DIR, f"{feat}_std_boxplot.png")
        boxplot_feature(feat, per_feature_mode[feat], out_path, MODE_ORDER)
        print(f"Saved: {out_path}")

    # Optional combined figure (one row per feature)
    if SAVE_COMBINED:
        fig, axes = plt.subplots(len(feature_cols), 1, figsize=(10, 2.4 * len(feature_cols)))
        if len(feature_cols) == 1:
            axes = [axes]

        for i, feat in enumerate(feature_cols):
            ax = axes[i]
            data_by_mode = per_feature_mode[feat]
            plot_data = [data_by_mode[m] for m in MODE_ORDER if m in data_by_mode]
            labels = [m for m in MODE_ORDER if m in data_by_mode]

            ax.boxplot(plot_data, labels=labels, showfliers=True, whis=1.5)
            ax.set_title(feat)
            ax.set_ylabel("Std")
            ax.grid(True, axis="y", alpha=0.3)

        plt.tight_layout()
        out_path = os.path.join(OUT_DIR, "all_features_std_boxplots.png")
        plt.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
