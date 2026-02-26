#!/usr/bin/env python3
"""
feature_probe_train.py (v2.0)

Purpose:
- Compute engineered features from windows (mean/std/energy + FFT-derived metrics)
- Train a simple probe classifier (logistic regression if sklearn is installed; else nearest-centroid)
- Report probe accuracy and rank engineered features.

Key fix vs v1:
- Correctly loads labels from meta['mode'] when NPZ stores split tags in npz['y'].
- Uses feature_cols for channel names when present.
- --split now correctly filters by split tags when available (train/test), without colliding with the "train" transportation mode.

Usage:
  python3 feature_probe_train.py --npz windows_5s_50pct.npz --outdir probe_out --split all
"""

import argparse
import os
import numpy as np
import csv


SPLIT_TOKENS = {"train", "test", "val", "valid", "validation"}


def _first_present(d, keys):
    for k in keys:
        if k in d:
            return k
    return None


def _as_1d(a):
    return np.asarray(a).reshape(-1)


def _looks_like_split_vec(arr_1d):
    if arr_1d.dtype.kind not in ("U", "S", "O"):
        return False
    vals = set(str(v).strip().lower() for v in np.unique(arr_1d))
    return len(vals) > 0 and vals.issubset(SPLIT_TOKENS)


def _map_str_labels_to_int(y_str):
    y_str = np.asarray(y_str).astype(str)
    names = sorted(np.unique(y_str).tolist())
    lut = {name: i for i, name in enumerate(names)}
    y = np.array([lut[s] for s in y_str], dtype=int)
    return y, names


def load_windows_npz(npz_path, split="all"):
    """
    Supports two NPZ styles:
    A) Window NPZ (your windows_5s_50pct.npz):
        keys: X, y (split tags), feature_cols, meta (array of dicts with 'mode')
    B) Train/test NPZ:
        keys: X_train,y_train,X_test,y_test,(optional) modes/label_names/channel names

    Returns:
      X: (N,T,C) float
      y: (N,) int labels
      channel_names: list[str] length C
      label_names: list[str] where label_names[i] is class name
      split_tags: (N,) str or None
    """
    data = np.load(npz_path, allow_pickle=True)
    keys = list(data.files)

    # --- Load X ---
    if "X" in data.files:
        X = np.asarray(data["X"])
    elif all(k in data.files for k in ["X_train", "X_test"]):
        X = np.concatenate([data["X_train"], data["X_test"]], axis=0)
    else:
        xk = _first_present(data, ["windows", "X_all", "data", "signals"])
        if xk is None:
            raise KeyError(f"Could not locate X. Keys={keys}")
        X = np.asarray(data[xk])

    if X.ndim != 3:
        raise ValueError(f"Expected X to be 3D. Got {X.shape}")

    # ensure (N,T,C)
    if X.shape[1] <= 16 and X.shape[2] >= 50:
        X = np.transpose(X, (0, 2, 1))

    # --- Channel names ---
    ch_key = _first_present(data, ["feature_cols", "channel_names", "channels", "feature_names", "sensor_names"])
    if ch_key is not None:
        channel_names = [str(c) for c in _as_1d(data[ch_key]).tolist()]
    else:
        channel_names = [f"ch{i}" for i in range(X.shape[2])]

    # --- Determine labels + split tags ---
    split_tags = None

    # Case: window NPZ with meta
    if "meta" in data.files:
        meta = data["meta"]  # shape (N,), each is dict
        if not (isinstance(meta, np.ndarray) and len(meta) == X.shape[0]):
            raise ValueError(f"meta shape mismatch: meta={getattr(meta,'shape',None)} X={X.shape}")

        # split tags are often stored as npz['y'] in your file
        if "y" in data.files:
            y_raw = _as_1d(data["y"])
            if _looks_like_split_vec(y_raw):
                split_tags = y_raw.astype(str)

        # true class labels from meta['mode']
        y_str = np.array([m["mode"] for m in meta], dtype=str)
        y, label_names = _map_str_labels_to_int(y_str)

    else:
        # Case: train/test NPZ where y is actual labels
        if all(k in data.files for k in ["y_train", "y_test"]):
            y_raw = np.concatenate([data["y_train"], data["y_test"]], axis=0)
        else:
            yk = _first_present(data, ["y", "labels", "y_all", "modes"])
            if yk is None:
                raise KeyError(f"Could not locate y. Keys={keys}")
            y_raw = data[yk]

        y_raw = _as_1d(y_raw)

        if y_raw.dtype.kind in ("i", "u"):
            y = y_raw.astype(int)
            label_names = [str(i) for i in sorted(np.unique(y).tolist())]
        else:
            y, label_names = _map_str_labels_to_int(y_raw.astype(str))

        # if this NPZ also contains split tags, try to find them
        sk = _first_present(data, ["split", "splits", "partition", "set"])
        if sk is not None:
            split_tags = _as_1d(data[sk]).astype(str)

    # Apply split filtering if requested and available
    if split != "all":
        if split_tags is None:
            raise ValueError(f"--split={split} requested, but no split tags were found in this NPZ.")
        split_l = split.strip().lower()
        mask = np.array([s.strip().lower() == split_l for s in split_tags], dtype=bool)
        X = X[mask]
        y = y[mask]
        if X.shape[0] == 0:
            raise ValueError(f"--split={split} produced zero windows. Check split tags content.")

    return X, y, channel_names, label_names, split_tags


def ensure_outdir(path):
    os.makedirs(path, exist_ok=True)


def spectral_entropy(power, eps=1e-12):
    p = power / (power.sum() + eps)
    p = np.clip(p, eps, 1.0)
    return float(-np.sum(p * np.log(p)))


def bandpower(freqs, power, f_lo, f_hi):
    mask = (freqs >= f_lo) & (freqs <= f_hi)
    if not np.any(mask):
        return 0.0
    return float(np.trapezoid(power[mask], freqs[mask]))


def extract_features_per_window(X, fs, channel_names):
    """
    X: (N,T,C)
    Returns:
      F: (N, D)
      feat_names: list[str]
    """
    N, T, C = X.shape
    freqs = np.fft.rfftfreq(T, d=1.0 / fs)

    feat_list = []
    feat_names = []

    for c in range(C):
        sig = X[:, :, c].astype(float)

        mu = sig.mean(axis=1)
        sd = sig.std(axis=1)
        energy = np.mean(sig ** 2, axis=1)

        feat_list += [mu, sd, energy]
        feat_names += [f"{channel_names[c]}__mean", f"{channel_names[c]}__std", f"{channel_names[c]}__energy"]

        sig0 = sig - mu[:, None]
        Xf = np.fft.rfft(sig0, axis=1)
        power = (np.abs(Xf) ** 2) / T

        power_no_dc = power.copy()
        power_no_dc[:, 0] = 0.0
        dom_idx = np.argmax(power_no_dc, axis=1)
        dom_freq = freqs[dom_idx]

        ent = np.array([spectral_entropy(power[i]) for i in range(N)], dtype=float)

        bp_0_2 = np.array([bandpower(freqs, power[i], 0.0, 2.0) for i in range(N)], dtype=float)
        bp_2_5 = np.array([bandpower(freqs, power[i], 2.0, 5.0) for i in range(N)], dtype=float)
        bp_5_10 = np.array([bandpower(freqs, power[i], 5.0, 10.0) for i in range(N)], dtype=float)

        feat_list += [dom_freq, ent, bp_0_2, bp_2_5, bp_5_10]
        feat_names += [
            f"{channel_names[c]}__dom_freq",
            f"{channel_names[c]}__spec_entropy",
            f"{channel_names[c]}__bp_0_2",
            f"{channel_names[c]}__bp_2_5",
            f"{channel_names[c]}__bp_5_10",
        ]

    F = np.stack(feat_list, axis=1)
    return F, feat_names


def train_probe_model(F, y, seed=7):
    rng = np.random.default_rng(seed)
    N = F.shape[0]
    idx = rng.permutation(N)
    split = int(0.7 * N)
    tr, te = idx[:split], idx[split:]

    mu = F[tr].mean(axis=0, keepdims=True)
    sd = F[tr].std(axis=0, keepdims=True) + 1e-8
    Ftr = (F[tr] - mu) / sd
    Fte = (F[te] - mu) / sd
    ytr, yte = y[tr], y[te]

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score

        clf = LogisticRegression(max_iter=3000, multi_class="auto")
        clf.fit(Ftr, ytr)
        pred = clf.predict(Fte)
        acc = float(accuracy_score(yte, pred))

        coef = clf.coef_
        importance = np.mean(np.abs(coef), axis=0)

        return acc, importance, "logreg"
    except Exception:
        classes = np.unique(ytr)
        centroids = np.stack([Ftr[ytr == c].mean(axis=0) for c in classes], axis=0)
        dists = np.sum((Fte[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        pred = classes[np.argmin(dists, axis=1)]
        acc = float(np.mean(pred == yte))
        importance = np.var(centroids, axis=0)
        return acc, importance, "nearest_centroid"


def save_feature_ranking(outdir, feat_names, importance, top_k=200):
    order = np.argsort(-importance)
    out_csv = os.path.join(outdir, "feature_ranking.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "feature", "importance"])
        for r, j in enumerate(order[:top_k], start=1):
            w.writerow([r, feat_names[j], float(importance[j])])
    return out_csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--outdir", default="probe_out")
    ap.add_argument("--split", default="all", choices=["all", "train", "test"])
    ap.add_argument("--fs", type=float, default=25.0)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    ensure_outdir(args.outdir)

    X, y, channel_names, label_names, split_tags = load_windows_npz(args.npz, split=args.split)
    print(f"Loaded X={X.shape}, y={y.shape}, classes={len(np.unique(y))}")
    print(f"Channels: {channel_names}")
    print(f"Labels  : {label_names}")

    F, feat_names = extract_features_per_window(X, fs=args.fs, channel_names=channel_names)
    acc, importance, model_used = train_probe_model(F, y, seed=args.seed)

    out_csv = save_feature_ranking(args.outdir, feat_names, importance)

    print(f"Probe model: {model_used}")
    print(f"Probe accuracy (random 70/30 window split): {acc:.4f}")
    print(f"Saved feature ranking: {out_csv}")
    print("Top 20 features:")
    for j in np.argsort(-importance)[:20]:
        print(f"  {feat_names[j]:34s}  {importance[j]:.6f}")


if __name__ == "__main__":
    main()
