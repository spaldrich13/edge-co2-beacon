#!/usr/bin/env python3
# plot_pca_projection.py
# v2 - adds --channels, supports feature_cols, supports string labels in y.

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt


def _first_present(d, keys):
    for k in keys:
        if k in d:
            return k
    return None


def _map_labels(y_raw):
    y_raw = np.asarray(y_raw).reshape(-1)
    if y_raw.dtype.kind in ("i", "u"):
        y_int = y_raw.astype(int)
        uniq = sorted(np.unique(y_int))
        max_id = int(max(uniq)) if uniq else -1
        label_names = [str(i) for i in range(max_id + 1)]
        return y_int, label_names

    y_str = np.array([str(v) for v in y_raw], dtype=str)
    uniq = sorted(np.unique(y_str))
    name_to_id = {name: i for i, name in enumerate(uniq)}
    y_int = np.array([name_to_id[s] for s in y_str], dtype=int)
    return y_int, uniq


def load_windows_npz(npz_path, split="all"):
    data = np.load(npz_path, allow_pickle=True)

    X_key = _first_present(data, ["X", "windows", "X_all", "data", "signals"])
    y_key = _first_present(data, ["y", "labels", "y_all", "modes"])
    if X_key is None or y_key is None:
        raise KeyError(f"Could not locate X/y. Keys: {list(data.keys())}")

    X = np.asarray(data[X_key])
    y_raw = data[y_key]

    # channel names come from feature_cols in your file
    ch_key = _first_present(data, ["feature_cols", "channel_names", "channels", "feature_names", "sensor_names"])
    if ch_key is not None:
        channel_names = [str(c) for c in np.asarray(data[ch_key]).reshape(-1)]
    else:
        channel_names = [f"ch{i}" for i in range(X.shape[-1])]

    # labels mapping
    y, label_names = _map_labels(y_raw)

    if X.ndim != 3:
        raise ValueError(f"Expected X 3D, got {X.shape}")

    # Ensure (N,T,C)
    if X.shape[1] <= 16 and X.shape[2] >= 50:
        X = np.transpose(X, (0, 2, 1))

    return X, y, channel_names, label_names


def ensure_outdir(path):
    os.makedirs(path, exist_ok=True)


def pca_2d_svd(Z):
    U, S, Vt = np.linalg.svd(Z, full_matrices=False)
    P = U[:, :2] * S[:2]
    eigvals = (S ** 2) / (Z.shape[0] - 1)
    evr = eigvals / eigvals.sum()
    return P, evr[:2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--outdir", default="figs_pca")
    ap.add_argument("--split", default="all", choices=["all", "train", "test"])
    ap.add_argument("--max_points_per_mode", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--channels", nargs="*", default=None,
                    help="Optional subset of channels (names from feature_cols). If omitted, uses all.")
    args = ap.parse_args()

    ensure_outdir(args.outdir)
    X, y, channel_names, label_names = load_windows_npz(args.npz, split=args.split)

    # Channel subset selection
    if args.channels:
        wanted = set(args.channels)
        ch_idx = [i for i, nm in enumerate(channel_names) if nm in wanted]
        if not ch_idx:
            raise ValueError(f"No matching channels. Available: {channel_names}")
        X = X[:, :, ch_idx]
        used_channels = [channel_names[i] for i in ch_idx]
    else:
        used_channels = channel_names

    rng = np.random.default_rng(args.seed)

    # Downsample per mode for plot clarity
    keep = []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        take = min(args.max_points_per_mode, len(idx))
        keep.append(rng.choice(idx, size=take, replace=False))
    keep = np.concatenate(keep)

    Xs = X[keep]
    ys = y[keep]

    # Flatten windows
    Z = Xs.reshape(Xs.shape[0], -1).astype(float)

    # Standardize features globally
    mu = Z.mean(axis=0, keepdims=True)
    sd = Z.std(axis=0, keepdims=True) + 1e-8
    Z = (Z - mu) / sd

    P, evr2 = pca_2d_svd(Z)

    fig = plt.figure(figsize=(10, 8))
    for cls in np.unique(ys):
        idx = np.where(ys == cls)[0]
        name = label_names[int(cls)] if label_names and int(cls) < len(label_names) else str(int(cls))
        plt.scatter(P[idx, 0], P[idx, 1], s=10, alpha=0.35, label=name)

    plt.title(f"PCA 2D of Window Vectors | PC1={evr2[0]:.3f}, PC2={evr2[1]:.3f}\nChannels: {', '.join(used_channels)}")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True, alpha=0.3)
    plt.legend()

    out = os.path.join(args.outdir, "pca_2d_projection.png")
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)

    print(f"Saved {out}")


if __name__ == "__main__":
    main()
