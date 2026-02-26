import os
import numpy as np
import pandas as pd

SHL_DIR = "/Users/spenceraldrich/Desktop/Union/Senior/Winter 2026/ECE-499/data/raw/shl/User1/220617"
BAG_MOTION = os.path.join(SHL_DIR, "Bag_Motion.txt")
LABEL_TXT  = os.path.join(SHL_DIR, "Label.txt")
LABEL_MAP_TXT = os.path.join(SHL_DIR, "labels_track_main.txt")

# --- Bag Motion time range (probe) ---
bm = pd.read_csv(BAG_MOTION, sep=r"\s+", header=None, usecols=[0], nrows=200000, engine="python", on_bad_lines="skip")
bm = pd.to_numeric(bm[0], errors="coerce").dropna()
print("Bag_Motion t range (probe):", float(bm.min()), "to", float(bm.max()))

# --- Label.txt shape + range ---
lab = pd.read_csv(LABEL_TXT, sep=r"\s+", header=None, engine="python", on_bad_lines="skip")
print("Label.txt shape:", lab.shape)
print("Label.txt head:\n", lab.head(10))

# try numeric for first 3 cols
for c in range(min(3, lab.shape[1])):
    lab[c] = pd.to_numeric(lab[c], errors="coerce")
print("Label col0 range:", float(lab[0].min()), "to", float(lab[0].max()))
if lab.shape[1] >= 2:
    print("Label col1 range:", float(lab[1].min()), "to", float(lab[1].max()))
if lab.shape[1] >= 3:
    print("Label col2 unique (first 20):", lab[2].dropna().unique()[:20])

# --- Label map head ---
with open(LABEL_MAP_TXT, "r") as f:
    lines = [next(f) for _ in range(15)]
print("labels_track_main head:\n", "".join(lines))

