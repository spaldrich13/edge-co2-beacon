# shl_inspect_bag_motion.py
# Version: 1.0
# Purpose: Inspect SHL Bag_Motion.txt column count + NaN pattern + likely sensor blocks.

import numpy as np

BAG_MOTION = "/Users/spenceraldrich/Desktop/Union/Senior/Winter 2026/ECE-499/data/raw/shl/User1/220617/Bag_Motion.txt"

def main():
    # Read only first ~50k lines to infer structure quickly
    n_lines = 50000
    rows = []
    with open(BAG_MOTION, "r") as f:
        for i, line in enumerate(f):
            if i >= n_lines:
                break
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            rows.append(parts)

    # Find the maximum column count seen
    max_cols = max(len(r) for r in rows)
    print("Max columns seen (including timestamp):", max_cols)

    # Convert to float array padding with NaN
    arr = np.full((len(rows), max_cols), np.nan, dtype=float)
    for i, r in enumerate(rows):
        for j, v in enumerate(r):
            try:
                arr[i, j] = float(v)
            except:
                arr[i, j] = np.nan

    nan_frac = np.mean(np.isnan(arr), axis=0)
    print("\nNaN fraction per column (first 25 cols):")
    for j in range(min(25, max_cols)):
        print(f"col {j:02d}: nan_frac={nan_frac[j]:.3f}")

    # Print a few first non-NaN rows for sanity
    idx = np.where(~np.isnan(arr).all(axis=1))[0][:5]
    print("\nExample non-empty rows (first 25 cols):")
    for k in idx:
        print(arr[k, :min(25, max_cols)])

if __name__ == "__main__":
    main()

