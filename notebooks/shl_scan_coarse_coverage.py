# shl_scan_coarse_coverage.py
# Version: 1.0
#
# Purpose:
#   Scan SHL folders and report which recordings contain coarse-label time
#   for your 5 target modes: walk, car, bus, train, subway (Bag placement).
#
# Notes:
# - Uses Label.txt coarse label codes:
#   walk=2, car=5, bus=6, train=7, subway=8
# - Label.txt is aligned line-by-line with <position>_Motion.txt. (SHL doc)

import os
import pandas as pd

# -------- CONFIG --------
SHL_ROOT = "/Users/spenceraldrich/Desktop/Union/Senior/Winter 2026/ECE-499/data/raw/shl"
POSITION = "Bag"  # Bag, Hand, Hips, Torso
LABEL_FILENAME = "Label.txt"

TARGET = {
    "walk": 2,
    "car": 5,
    "bus": 6,
    "train": 7,
    "subway": 8,
}

MIN_SECONDS_PER_MODE = 60  # require at least this much per mode to consider it "present"
FS_HZ = 100  # SHL motion/label grid is 100Hz (per docs)

# ------------------------
def iter_recordings(root):
    for user in sorted(os.listdir(root)):
        if not user.lower().startswith("user"):
            continue
        user_dir = os.path.join(root, user)
        if not os.path.isdir(user_dir):
            continue
        for recid in sorted(os.listdir(user_dir)):
            rec_dir = os.path.join(user_dir, recid)
            if not os.path.isdir(rec_dir):
                continue
            yield user, recid, rec_dir

def scan_label_txt(label_path):
    # Read only the coarse label column (0-based col=1), whitespace-delimited.
    # Use chunks to handle big files.
    counts = {k: 0 for k in TARGET.keys()}
    total = 0

    for chunk in pd.read_csv(
        label_path,
        sep=r"\s+",
        header=None,
        usecols=[1],           # coarse label
        engine="python",
        chunksize=1_000_000,
    ):
        s = chunk.iloc[:, 0]
        total += len(s)
        for name, code in TARGET.items():
            counts[name] += int((s == code).sum())

    return total, counts

def main():
    rows = []
    for user, recid, rec_dir in iter_recordings(SHL_ROOT):
        label_path = os.path.join(rec_dir, LABEL_FILENAME)
        motion_path = os.path.join(rec_dir, f"{POSITION}_Motion.txt")

        if not os.path.exists(label_path):
            continue
        if not os.path.exists(motion_path):
            # You can still scan labels even if motion isn't downloaded yet,
            # but it won't be usable for training until motion is present.
            motion_ok = False
        else:
            motion_ok = True

        try:
            total, counts = scan_label_txt(label_path)
        except Exception as e:
            print(f"Skip {user}/{recid}: failed reading Label.txt: {e}")
            continue

        secs = {k: counts[k] / FS_HZ for k in counts}
        present = {k: (secs[k] >= MIN_SECONDS_PER_MODE) for k in secs}

        rows.append({
            "user": user,
            "recid": recid,
            "motion_ok": motion_ok,
            "total_min": total / FS_HZ / 60.0,
            **{f"{k}_min": secs[k] / 60.0 for k in secs},
            **{f"has_{k}": present[k] for k in present},
        })

    if not rows:
        print("No recordings found. Check SHL_ROOT path.")
        return

    df = pd.DataFrame(rows)

    cols = ["user", "recid", "motion_ok", "total_min"] + [f"{k}_min" for k in TARGET.keys()]

    # 1) Top overall (what you already saw)
    df_overall = df.sort_values(["motion_ok", "total_min"], ascending=[False, False])
    print("\n=== Top recordings overall (minutes per mode) ===")
    print(df_overall[cols].head(30).to_string(index=False))

    # 2) Top by CAR
    df_car = df.sort_values(["car_min", "motion_ok", "total_min"], ascending=[False, False, False])
    print("\n=== Top recordings by CAR minutes ===")
    print(df_car[cols].head(50).to_string(index=False))

    # 3) Recordings that contain ALL 5 modes (>= threshold)
    print("\n=== Recordings that contain ALL 5 modes (>= MIN_SECONDS_PER_MODE each) ===")
    mask_all = True
    for k in TARGET.keys():
        mask_all = mask_all & (df[f"{k}_min"] >= (MIN_SECONDS_PER_MODE / 60.0))
    df_all = df[mask_all].sort_values(["motion_ok", "total_min"], ascending=[False, False])

    if len(df_all) == 0:
        print("None found with all 5 modes at this threshold.")
    else:
        print(df_all[cols].head(50).to_string(index=False))

    # 4) If no single recording has all 5, find a small UNION set
    print("\n=== Quick union suggestion (greedy) ===")
    remaining = set(TARGET.keys())
    picked = []
    df_tmp = df.copy()

    while remaining:
        # score each row by how many remaining modes it covers
        scores = []
        for i, r in df_tmp.iterrows():
            covers = [m for m in remaining if r[f"{m}_min"] >= (MIN_SECONDS_PER_MODE / 60.0)]
            scores.append((len(covers), r["motion_ok"], r["total_min"], i, covers))
        scores.sort(reverse=True)
        best_n, _, _, best_i, covers = scores[0]
        if best_n == 0:
            break
        r = df_tmp.loc[best_i]
        picked.append((r["user"], r["recid"], covers))
        for m in covers:
            remaining.discard(m)
        df_tmp = df_tmp.drop(index=best_i)

    if remaining:
        print("Could not cover modes:", remaining)
    else:
        for u, rid, covers in picked:
            print(f"- {u}/{rid} covers {covers}")

if __name__ == "__main__":
    main()
