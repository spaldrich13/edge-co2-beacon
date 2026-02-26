#!/usr/bin/env python3
# plot_fft_spectra.py
# v2 - supports string labels (e.g., "train") and feature_cols channel names.

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
    """
    Accepts y_raw as int or str labels.
    Returns:
      y_int: (N,) int
      label_names: list[str] where label_names[i] is class name
    """
    y_raw = np.asarray(y_raw).reshape(-1)

    # already integer labels
    if y_raw.dtype.kind in ("i", "u"):
        y_int = y_raw.astype(int)
        uniq = sorted(np.unique(y_int))
        # label_names indexed by class id; ensure length = max_id+1
        max_id = int(max(uniq)) if uniq else -1
        label_names = [str(i) for i in range(max_id + 1)]
        return y_int, label_names

    # string/object labels
    y_str = np.array([str(v) for v in y_raw], dtype=str)
    uniq = sorted(np.unique(y_str))
    name_to_id = {name: i for i, name in enumerate(uniq)}
    y_int = np.array([name_to_id[s] for s in y_str], dtype=int)
    label_names = uniq
    return y_int, label_names


def load_windows_npz(npz_path, split="all"):
    data = np.load(npz_path, allow_pickle=True)

    # windows_5s_50pct.npz uses X, y, feature_cols, fs_hz...
    if split in ("train", "test"):
        Xk, yk = f"X_{split}", f"y_{split}"
        if Xk in data and yk in data:
            X = data[Xk]
            y_raw = data[yk]
        else:
            raise KeyError(f"Split keys not found: {Xk}/{yk}")
    else:
        X_key = _first_present(data, ["X", "windows", "X_all", "data", "signals"])
        y_key = _first_present(data, ["y", "labels", "y_all", "modes"])
        if X_key is None or y_key is None:
            raise KeyError(f"Could not locate X/y. Keys: {list(data.keys())}")
        X = data[X_key]
        y_raw = data[y_key]

    # channel names
    ch_key = _first_present(data, ["feature_cols", "channel_names", "channels", "feature_names", "sensor_names"])
    if ch_key is not None:
        channel_names = [str(c) for c in np.asarray(data[ch_key]).reshape(-1)]
    else:
        channel_names = [f"ch{i}" for i in range(X.shape[-1])]

    # labels (map strings -> ints)
    y, label_names = _map_labels(y_raw)

    # ensure X is (N,T,C)
    X = np.asarray(X)
    if X.ndim != 3:
        raise ValueError(f"Expected X 3D, got {X.shape}")

    # transpose if (N,C,T)
    if X.shape[1] <= 16 and X.shape[2] >= 50:
        X = np.transpose(X, (0, 2, 1))

    return X, y, channel_names, label_names


def ensure_outdir(path):
    os.makedirs(path, exist_ok=True)


def average_power_spectrum(x, fs):
    x = x.astype(float)
    x = x - x.mean()
    T = x.shape[0]
    freqs = np.fft.rfftfreq(T, d=1.0 / fs)
    Xf = np.fft.rfft(x)
    power = (np.abs(Xf) ** 2) / T
    return freqs, power


def plot_fft_by_mode(X, y, channel_idx, channel_name, label_names, outdir,
                     fs=25.0, fmax=None, max_windows_per_mode=200, seed=7):
    rng = np.random.default_rng(seed)
    classes = np.unique(y)

    fig = plt.figure(figsize=(12, 6))

    for cls in classes:
        idx = np.where(y == cls)[0]
        if len(idx) == 0:
            continue

        take = min(max_windows_per_mode, len(idx))
        chosen = rng.choice(idx, size=take, replace=False)

        ps_list = []
        freqs_ref = None
        for i in chosen:
            freqs, ps = average_power_spectrum(X[i, :, channel_idx], fs)
            if freqs_ref is None:
                freqs_ref = freqs
            ps_list.append(ps)

        ps_mean = np.mean(np.stack(ps_list, axis=0), axis=0)

        if fmax is not None:
            mask = freqs_ref <= fmax
            freqs_plot = freqs_ref[mask]
            ps_plot = ps_mean[mask]
        else:
            freqs_plot = freqs_ref
            ps_plot = ps_mean

        name = label_names[int(cls)] if label_names and int(cls) < len(label_names) else str(int(cls))
        plt.plot(freqs_plot, ps_plot, linewidth=2.0, label=name)

    plt.title(f"Average Power Spectrum by Mode — {channel_name}")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power (arb. units)")
    plt.grid(True, alpha=0.3)
    plt.legend()

    fname = os.path.join(outdir, f"fft_{channel_name}.png")
    fig.tight_layout()
    fig.savefig(fname, dpi=200)
    plt.close(fig)
    return fname


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--outdir", default="figs_fft")
    ap.add_argument("--split", default="all", choices=["all", "train", "test"])
    ap.add_argument("--fs", type=float, default=25.0)
    ap.add_argument("--fmax", type=float, default=None,
                    help="Max frequency to plot (Hz). If omitted, plots full rFFT range.")
    ap.add_argument("--max_windows_per_mode", type=int, default=200)
    ap.add_argument("--channels", nargs="*", default=None)
    args = ap.parse_args()

    ensure_outdir(args.outdir)
    X, y, channel_names, label_names = load_windows_npz(args.npz, split=args.split)

    # default fmax = Nyquist
    if args.fmax is None:
        args.fmax = args.fs / 2.0

    if args.channels:
        wanted = set(args.channels)
        ch_indices = [i for i, nm in enumerate(channel_names) if nm in wanted]
        if not ch_indices:
            raise ValueError(f"No matching channels found. Available: {channel_names}")
    else:
        ch_indices = list(range(len(channel_names)))

    print(f"Loaded X={X.shape}, y={y.shape}, classes={len(np.unique(y))}")
    print(f"Channel names: {channel_names}")

    for ci in ch_indices:
        out = plot_fft_by_mode(
            X, y, ci, channel_names[ci], label_names, args.outdir,
            fs=args.fs, fmax=args.fmax, max_windows_per_mode=args.max_windows_per_mode
        )
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
