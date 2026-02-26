import serial
import serial.tools.list_ports
import time
from datetime import datetime
import os

# SERIAL
BAUD = 115200
READ_TIMEOUT = 1.0

# FILE SAFETY
FLUSH_INTERVAL = 1.0  # seconds
WRITE_METADATA_COMMENTS = True

# >>> IMPORTANT: Base output directory for ALL saved segments
BASE_OUTPUT_DIR = "/Users/spenceraldrich/Desktop/Union/Senior/Winter 2026/ECE-499/data/raw/self_collected"

# Map Arduino MODE_CODE -> canonical mode name
MODE_MAP = {
    "3": "train",
    "4": "subway",
    "5": "bus",
    "6": "car",
    "7": "bike",
    "8": "walk",
}

CSV_HEADER = (
    "t_ms,ax_raw,ay_raw,az_raw,"
    "ax_corr,ay_corr,az_corr,"
    "acc_mag_corr,"
    "gx,gy,gz,"
    "pressure_hpa,temp_C,alt_m"
)


def pick_port():
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        raise RuntimeError("No serial ports found.")
    print("Available ports:")
    for i, p in enumerate(ports):
        print(f"  [{i}] {p.device} - {p.description}")
    idx = int(input("Select port index: "))
    if idx < 0 or idx >= len(ports):
        raise ValueError("Invalid port index.")
    return ports[idx].device


def safe_flush(f):
    try:
        f.flush()
        os.fsync(f.fileno())
    except Exception:
        pass


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def parse_start_logger(line: str):
    # Example: #START_LOGGER,FS_HZ=25,MODE_CODE=4,MODE_NAME=subway
    fs_hz = None
    mode_code = None
    mode_name = None
    try:
        parts = line.strip().split(",")
        for p in parts:
            if p.startswith("FS_HZ="):
                fs_hz = p.split("=", 1)[1]
            elif p.startswith("MODE_CODE="):
                mode_code = p.split("=", 1)[1]
            elif p.startswith("MODE_NAME="):
                mode_name = p.split("=", 1)[1]
    except Exception:
        pass
    return fs_hz, mode_code, mode_name


def main():
    ensure_dir(BASE_OUTPUT_DIR)

    port = pick_port()

    override_mode = input(
        "Select mode folder (train/subway/car/bus/bike/walk) or press Enter to auto-detect from Arduino: "
    ).strip().lower()

    if override_mode and override_mode not in {"train", "subway", "car", "bus", "bike", "walk"}:
        print(f"[WARN] Unknown override '{override_mode}'. Ignoring override and auto-detecting instead.")
        override_mode = ""

    print("\nOpening serial port...")
    ser = serial.Serial(port, BAUD, timeout=READ_TIMEOUT)
    time.sleep(2)

    print("[OK] Connected. Waiting for #START_LOGGER ...")
    print("Press Ctrl+C to stop logger safely.\n")

    fs_hz = None
    mode_code = None
    mode_name = None

    current_file = None
    current_fname = None
    segment_id = None
    segment_start_ms = None

    last_flush = time.time()
    line_count = 0
    header_written = False

    # For visibility
    last_status = time.time()

    def close_active_file(reason: str):
        nonlocal current_file, current_fname, segment_id, line_count, header_written
        if current_file:
            safe_flush(current_file)
            current_file.close()
            print(f"[CLOSE] {reason} -> {current_fname} (data_lines={line_count})")
        current_file = None
        current_fname = None
        segment_id = None
        header_written = False

    try:
        while True:
            try:
                raw = ser.readline()
                if not raw:
                    # periodic heartbeat
                    if time.time() - last_status > 5.0:
                        print("[STATUS] Waiting... (no data)")
                        last_status = time.time()
                    continue

                line = raw.decode(errors="ignore").strip()
                if not line:
                    continue

                # Capture logger config
                if line.startswith("#START_LOGGER"):
                    fs_hz, mode_code, mode_name = parse_start_logger(line)
                    if not mode_name and mode_code in MODE_MAP:
                        mode_name = MODE_MAP[mode_code]
                    print(f"[INFO] {line}")
                    print(f"[INFO] Detected: FS_HZ={fs_hz}, MODE_CODE={mode_code}, MODE_NAME={mode_name}")
                    continue

                # Segment start
                if line.startswith("#SEGMENT_START"):
                    # Format: #SEGMENT_START,<id>,<mode_code>,<t_ms>
                    parts = line.split(",")
                    if len(parts) != 4:
                        print(f"[WARN] Malformed SEGMENT_START: {line}")
                        continue

                    _, seg_id, seg_mode_code, t_ms = parts
                    segment_id = seg_id
                    segment_start_ms = t_ms

                    # Determine mode folder/tag
                    seg_mode_tag = override_mode if override_mode else MODE_MAP.get(seg_mode_code, f"mode{seg_mode_code}")

                    # Create mode folder under your dataset directory
                    current_dir = os.path.join(BASE_OUTPUT_DIR, seg_mode_tag)
                    ensure_dir(current_dir)

                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    current_fname = os.path.join(current_dir, f"{seg_mode_tag}_seg{segment_id}_{ts}.csv")
                    current_file = open(current_fname, "w", newline="")

                    # Metadata comment block (safe to load in pandas with comment='#')
                    if WRITE_METADATA_COMMENTS:
                        current_file.write("#logger=serial_logger_v1.2\n")
                        current_file.write(f"#fs_hz={fs_hz}\n")
                        current_file.write(f"#mode_code={seg_mode_code}\n")
                        current_file.write(f"#mode_tag={seg_mode_tag}\n")
                        current_file.write(f"#segment_id={segment_id}\n")
                        current_file.write(f"#segment_start_ms={segment_start_ms}\n")
                        current_file.write(f"#saved_at={datetime.now().isoformat()}\n")

                    header_written = False
                    line_count = 0
                    last_flush = time.time()

                    print(f"[START] Segment {segment_id} ({seg_mode_tag}) -> {current_fname}")
                    continue

                # Segment end
                if line.startswith("#SEGMENT_END"):
                    close_active_file("Segment ended")
                    continue

                # If inside a segment file:
                if current_file:
                    # If Arduino prints the header line right after SEGMENT_START,
                    # capture it and ensure it exists in the file.
                    if (not header_written) and (line == CSV_HEADER):
                        current_file.write(line + "\n")
                        header_written = True
                        continue

                    # If we missed Arduino header, enforce it on first data line
                    if not header_written:
                        current_file.write(CSV_HEADER + "\n")
                        header_written = True

                    # Ignore any other internal # messages during segment
                    if line.startswith("#"):
                        continue

                    # Write numeric CSV row
                    current_file.write(line + "\n")
                    line_count += 1

                    now = time.time()
                    if now - last_flush >= FLUSH_INTERVAL:
                        safe_flush(current_file)
                        last_flush = now

                    if now - last_status > 5.0:
                        print(f"[STATUS] seg={segment_id} lines={line_count}")
                        last_status = now

            except serial.SerialException as e:
                print(f"[WARN] Serial error: {e}")
                close_active_file("Serial exception")
                time.sleep(1)

    except KeyboardInterrupt:
        print("\n[STOP] Logger interrupted by user.")
    finally:
        close_active_file("Shutdown")
        try:
            ser.close()
        except Exception:
            pass
        print("[DONE] Serial port closed.")


if __name__ == "__main__":
    main()
