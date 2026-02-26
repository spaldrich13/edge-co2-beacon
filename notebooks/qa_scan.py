import os
import glob
import pandas as pd
import numpy as np

BASE_DIR = "/Users/spenceraldrich/Desktop/Union/Senior/Winter 2026/ECE-499/data/raw/self_collected"
OUT_DIR  = "/Users/spenceraldrich/Desktop/Union/Senior/Winter 2026/ECE-499/qa_outputs"

VALID_MODES = {"train","subway","car","bus","bike","walk"}

PRESSURE_MIN_HPA = 900
PRESSURE_MAX_HPA = 1100
TEMP_MIN_C = -10
TEMP_MAX_C = 60
GAP_MS = 200

EXPECTED_COLS = [
    "t_ms",
    "ax_raw","ay_raw","az_raw",
    "ax_corr","ay_corr","az_corr",
    "acc_mag_corr",
    "gx","gy","gz",
    "pressure_hpa","temp_C","alt_m"
]

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def parse_metadata(lines):
    meta = {}
    for ln in lines:
        if ln.startswith("#") and "=" in ln:
            k,v = ln[1:].strip().split("=", 1)
            meta[k.strip()] = v.strip()
    return meta

def read_flexible_csv(path: str):
    # Peek first line: if it contains letters, it’s probably a header or metadata
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        first = f.readline().strip()

    meta = {}
    if first.startswith("#"):
        # New format: comments + header
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            head = [f.readline().strip() for _ in range(60)]
        meta = parse_metadata(head)
        df = pd.read_csv(path, comment="#")
        return meta, df

    # If first line contains alphabetic characters, assume it’s a header line
    if any(c.isalpha() for c in first):
        df = pd.read_csv(path)
        return meta, df

    # Otherwise: legacy numeric-only file, no header
    df = pd.read_csv(path, header=None, names=EXPECTED_COLS)
    return meta, df

def main():
    ensure_dir(OUT_DIR)

    csv_paths = []
    for mode in sorted(VALID_MODES):
        csv_paths.extend(glob.glob(os.path.join(BASE_DIR, mode, "*.csv")))
    csv_paths = sorted(csv_paths)
    if not csv_paths:
        raise RuntimeError(f"No CSVs found under {BASE_DIR}")

    summary_rows, gaps_rows, outlier_rows = [], [], []

    for p in csv_paths:
        fname = os.path.basename(p)
        mode_folder = os.path.basename(os.path.dirname(p))

        row = {"file": fname, "mode_folder": mode_folder, "path": p, "ok": True, "reason": ""}

        try:
            meta, df = read_flexible_csv(p)

            missing_cols = [c for c in EXPECTED_COLS if c not in df.columns]
            if missing_cols:
                row["ok"] = False
                row["reason"] = f"missing_cols:{missing_cols}"
                summary_rows.append(row)
                continue

            row["storage_mode_tag"] = meta.get("storage_mode_tag", "")
            row["device_mode_tag"]  = meta.get("device_mode_tag", "")
            row["mode_mismatch"]    = meta.get("mode_mismatch", "")
            row["fs_hz"]            = meta.get("fs_hz", "")
            row["fs_hz_inferred"]   = meta.get("fs_hz_inferred", "")
            row["segment_id"]       = meta.get("segment_id", "")
            row["saved_at"]         = meta.get("saved_at", "")

            t = df["t_ms"].astype(np.int64).values
            if len(t) < 10:
                row["ok"] = False
                row["reason"] = "too_few_rows"
                summary_rows.append(row)
                continue

            dt = np.diff(t)
            row["n_rows"] = int(len(df))
            row["t_start_ms"] = int(t[0])
            row["t_end_ms"]   = int(t[-1])
            row["duration_s"] = float((t[-1] - t[0]) / 1000.0)

            med_dt = float(np.median(dt))
            row["median_dt_ms"] = med_dt
            row["fs_est_hz"] = float(1000.0 / med_dt) if med_dt > 0 else np.nan

            max_dt = float(np.max(dt))
            row["max_dt_ms"] = max_dt
            n_gaps = int(np.sum(dt > GAP_MS))
            row["n_gaps_gt_200ms"] = n_gaps

            if n_gaps > 0:
                gap_idxs = np.where(dt > GAP_MS)[0]
                for gi in gap_idxs[:10]:
                    gaps_rows.append({
                        "file": fname, "mode_folder": mode_folder,
                        "gap_index": int(gi),
                        "t_before_ms": int(t[gi]),
                        "t_after_ms": int(t[gi+1]),
                        "dt_ms": int(dt[gi]),
                    })

            pressure = df["pressure_hpa"].astype(float).values
            temp = df["temp_C"].astype(float).values

            p_out = np.where((pressure < PRESSURE_MIN_HPA) | (pressure > PRESSURE_MAX_HPA))[0]
            t_out = np.where((temp < TEMP_MIN_C) | (temp > TEMP_MAX_C))[0]

            row["n_pressure_outliers"] = int(len(p_out))
            row["n_temp_outliers"] = int(len(t_out))

            if len(p_out) > 0:
                for oi in p_out[:10]:
                    outlier_rows.append({
                        "file": fname, "mode_folder": mode_folder,
                        "type": "pressure",
                        "row_index": int(oi),
                        "t_ms": int(df["t_ms"].iloc[oi]),
                        "value": float(pressure[oi]),
                    })
            if len(t_out) > 0:
                for oi in t_out[:10]:
                    outlier_rows.append({
                        "file": fname, "mode_folder": mode_folder,
                        "type": "temp",
                        "row_index": int(oi),
                        "t_ms": int(df["t_ms"].iloc[oi]),
                        "value": float(temp[oi]),
                    })

            warn = []
            if n_gaps > 0: warn.append("timestamp_gaps")
            if len(p_out) > 0: warn.append("pressure_outliers")
            if len(t_out) > 0: warn.append("temp_outliers")
            if warn:
                row["reason"] = ",".join(warn)

            if row.get("mode_mismatch","") == "1":
                row["ok"] = False
                row["reason"] = "mode_mismatch=1"

            summary_rows.append(row)

        except Exception as e:
            row["ok"] = False
            row["reason"] = f"exception:{type(e).__name__}"
            summary_rows.append(row)

    df_sum = pd.DataFrame(summary_rows)
    df_sum.to_csv(os.path.join(OUT_DIR, "qa_summary.csv"), index=False)

    pd.DataFrame(gaps_rows).to_csv(os.path.join(OUT_DIR, "qa_gaps.csv"), index=False)
    pd.DataFrame(outlier_rows).to_csv(os.path.join(OUT_DIR, "qa_outliers.csv"), index=False)

    mode_totals = (
        df_sum.groupby("mode_folder", dropna=False)
        .agg(
            n_files=("file", "count"),
            total_minutes=("duration_s", lambda s: float(np.nansum(s) / 60.0)),
            median_minutes=("duration_s", lambda s: float(np.nanmedian(s) / 60.0)),
            max_gap_ms=("max_dt_ms", "max"),
            total_gaps=("n_gaps_gt_200ms", "sum"),
            pressure_outliers=("n_pressure_outliers", "sum"),
            temp_outliers=("n_temp_outliers", "sum"),
        )
        .reset_index()
        .sort_values("mode_folder")
    )
    mode_totals.to_csv(os.path.join(OUT_DIR, "qa_mode_totals.csv"), index=False)

    print("\n=== QA COMPLETE (flex reader) ===")
    print(f"Files scanned: {len(df_sum)}")
    print("Outputs written to:", OUT_DIR)
    print("\nMode totals (minutes):")
    print(mode_totals[["mode_folder","n_files","total_minutes","total_gaps","pressure_outliers","temp_outliers"]].to_string(index=False))

    fails = df_sum[df_sum["ok"] == False]
    if len(fails) > 0:
        print("\n[ATTN] Some files flagged ok=False:")
        print(fails[["file","mode_folder","reason"]].head(20).to_string(index=False))
    else:
        print("\nAll files passed basic integrity checks (some may have warnings).")

if __name__ == "__main__":
    main()
