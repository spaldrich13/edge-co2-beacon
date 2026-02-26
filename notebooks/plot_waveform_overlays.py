#!/usr/bin/env python3
# plot_waveform_overlays.py
# v2 - robust NPZ loader: handles split strings and string labels safely.

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

SPLIT_TOKENS = {"train", "test", "val", "valid", "validation"}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", required=True, help="Path to windowed .npz (e.g., windows_5s_50pct.npz)")
    p.add_argument("--outdir", default="plots_waveforms", help="Output directory")
    p.add_argument("--split", choices=["all", "train", "test"], default="all", help="Which split to plot")
    p.add_argument("--fs", type=float, default=25.0, help="Sampling rate (Hz)")
    p.add_argument("--n_per_class", type=int, default=20, help="Number of windows per class to overlay")
    p.add_argument("--normalize_per_window", action="store_true",
                   help="If set, z-normalize each window (per-channel) before plotting (shape-only)")
    p.add_argument("--channels", nargs="*", default=None,
                   help="Subset of channels to plot by name (e.g., ax ay az gx gy gz pressure_hpa)")
    return p.parse_args()

def _npz_keys(npz):
    return list(npz.keys())

def _pick_first_existing(npz, candidates):
    for k in candidates:
        if k in npz:
            return k
    return None

def _as_1d(arr):
    arr = np.asarray(arr)
    return arr.reshape(-1)

def _looks_like_split(arr_1d):
    if arr_1d.dtype.kind not in ("U", "S", "O"):
        return False
    vals = set(str(v).strip().lower() for v in np.unique(arr_1d))
    # If all unique values are split tokens, it's a split vector.
    return len(vals) > 0 and vals.issubset(SPLIT_TOKENS)

def _map_string_labels_to_int(y_str):
    y_str = np.asarray(y_str).astype(str)
    uniq = sorted(np.unique(y_str))
    name_to_id = {name: i for i, name in enumerate(uniq)}
    y_int = np.array([name_to_id[s] for s in y_str], dtype=int)
    return y_int, uniq

def load_windows_npz(path, split="all"):
    npz = np.load(path, allow_pickle=True)
    keys = _npz_keys(npz)

    # --- Find X ---
    x_key = _pick_first_existing(npz, ["X", "windows", "data", "signals"])
    if x_key is None:
        raise KeyError(f"Could not find X in NPZ. Keys={keys}")
    X = np.asarray(npz[x_key])

    # Expect X shape [N, T, C] or [N, C, T]; handle both
    if X.ndim != 3:
        raise ValueError(f"Expected X to be 3D [N,T,C] or [N,C,T]. Got shape={X.shape} (key={x_key})")

    # --- Channel names (optional) ---
    ch_key = _pick_first_existing(npz, ["feature_cols", "channel_names", "channels", "feature_names"])
    if ch_key is not None:
        channel_names = [str(x) for x in np.asarray(npz[ch_key]).reshape(-1)]
    else:
        # default generic names
        channel_names = [f"ch{i}" for i in range(X.shape[-1])]

    # --- Split vector (optional) ---
    split_key = _pick_first_existing(npz, ["split", "splits", "partition", "set"])
    split_vec = None
    if split_key is not None:
        split_vec = _as_1d(npz[split_key]).astype(str)

    # --- Label names (optional) ---
    label_names_key = _pick_first_existing(npz, ["label_names", "labels_names", "class_names", "classes"])
    label_names = None
    if label_names_key is not None:
        label_names = [str(x) for x in np.asarray(npz[label_names_key]).reshape(-1)]

    # --- Find y (true labels) ---
    # Candidate keys for labels:
    y_key = _pick_first_existing(npz, ["y", "labels", "label", "mode", "modes", "class_id"])
    if y_key is None:
        # if no explicit y, try to infer from any 1D array matching N
        N = X.shape[0]
        candidates = []
        for k in keys:
            a = np.asarray(npz[k])
            if a.ndim == 1 and len(a) == N and k != split_key and k != ch_key and k != label_names_key:
                candidates.append(k)
        if candidates:
            y_key = candidates[0]

    if y_key is None:
        raise KeyError(f"Could not find labels (y) in NPZ. Keys={keys}")

    y_raw = _as_1d(npz[y_key])

    # If y_raw looks like split ("train"/"test"), treat it as split and search for another label key
    if _looks_like_split(y_raw):
        if split_vec is None:
            split_vec = y_raw.astype(str)
            split_key = y_key
        # try again for true labels by excluding this key
        alt_keys = [k for k in ["labels", "label", "mode", "modes", "class_id"] if k in npz and k != y_key]
        if alt_keys:
            y_key2 = alt_keys[0]
            y_raw = _as_1d(npz[y_key2])
        else:
            raise ValueError(
                f"NPZ key '{y_key}' contains split strings (train/test), but no other label key found.\n"
                f"Keys={keys}\n"
                f"Fix: ensure NPZ stores true labels under 'y'/'labels'/'mode' and split under 'split'."
            )

    # Convert labels to int if needed
    if y_raw.dtype.kind in ("i", "u"):
        y = y_raw.astype(int)
        if label_names is None:
            # create names as strings of ids
            uniq = sorted(np.unique(y))
            label_names = [str(i) for i in uniq]
    else:
        # string labels like "walk", "bus"
        y = np.asarray(y_raw).astype(str)
        y, inferred_names = _map_string_labels_to_int(y)
        if label_names is None:
            label_names = inferred_names

    # --- Apply split filter (if requested) ---
    if split != "all":
        if split_vec is None:
            raise ValueError(
                f"--split={split} requested but NPZ has no split vector.\n"
                f"Keys={keys}\n"
                f"Expected a key like 'split' containing train/test per window."
            )
        mask = np.array([s.strip().lower() == split for s in split_vec], dtype=bool)
        X = X[mask]
        y = y[mask]

    # Ensure X is [N, T, C]
    # If currently [N, C, T], transpose
    if X.shape[1] == len(channel_names) and X.shape[2] != len(channel_names):
        # likely [N, C, T]
        X = np.transpose(X, (0, 2, 1))

    return X, y, channel_names, label_names

def normalize_window_per_channel(win):
    # win shape [T, C]
    mu = win.mean(axis=0, keepdims=True)
    sd = win.std(axis=0, keepdims=True)
    sd = np.where(sd == 0, 1.0, sd)
    return (win - mu) / sd

def plot_overlays(X, y, channel_names, label_names, outdir, fs, n_per_class, normalize_per_window, selected_channels):
    os.makedirs(outdir, exist_ok=True)

    # channel selection
    if selected_channels is not None:
        keep_idx = []
        for name in selected_channels:
            if name not in channel_names:
                raise ValueError(f"Channel '{name}' not found. Available={channel_names}")
            keep_idx.append(channel_names.index(name))
    else:
        keep_idx = list(range(len(channel_names)))

    t = np.arange(X.shape[1]) / fs

    classes = np.unique(y)
    for c in classes:
        class_name = label_names[c] if (label_names is not None and c < len(label_names)) else str(c)
        idx = np.where(y == c)[0]
        if len(idx) == 0:
            continue
        np.random.shuffle(idx)
        idx = idx[:min(n_per_class, len(idx))]

        for ch in keep_idx:
            plt.figure()
            for i in idx:
                win = X[i, :, :]
                if normalize_per_window:
                    win = normalize_window_per_channel(win)
                plt.plot(t, win[:, ch], alpha=0.3)
            plt.title(f"Waveform Overlays | class={class_name} | channel={channel_names[ch]} | n={len(idx)}")
            plt.xlabel("Time (s)")
            plt.ylabel("Amplitude" + (" (per-window normalized)" if normalize_per_window else ""))
            plt.tight_layout()

            fname = f"overlay_{class_name}_{channel_names[ch]}.png".replace(" ", "_")
            plt.savefig(os.path.join(outdir, fname), dpi=200)
            plt.close()

def main():
    args = parse_args()
    X, y, channel_names, label_names = load_windows_npz(args.npz, split=args.split)

    plot_overlays(
        X=X,
        y=y,
        channel_names=channel_names,
        label_names=label_names,
        outdir=args.outdir,
        fs=args.fs,
        n_per_class=args.n_per_class,
        normalize_per_window=args.normalize_per_window,
        selected_channels=args.channels
    )

    print(f"Saved overlays to: {args.outdir}")
    print(f"Loaded: X={X.shape}, classes={len(np.unique(y))}, channels={len(channel_names)}")

if __name__ == "__main__":
    main()
