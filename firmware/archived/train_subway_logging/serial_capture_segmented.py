import serial
import serial.tools.list_ports
import time
from datetime import datetime
import os
import sys

BAUD = 115200
FLUSH_INTERVAL = 1.0  # seconds

def pick_port():
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        raise RuntimeError("No serial ports found.")
    print("Available ports:")
    for i, p in enumerate(ports):
        print(f"  [{i}] {p.device} - {p.description}")
    idx = int(input("Select port index: "))
    return ports[idx].device

def safe_flush(f):
    try:
        f.flush()
        os.fsync(f.fileno())
    except Exception:
        pass

def main():
    port = pick_port()
    mode_tag = input("Enter mode tag (train / subway / bus / etc): ").strip()

    print("\nOpening serial port...")
    ser = serial.Serial(port, BAUD, timeout=1)
    time.sleep(2)

    print("[OK] Connected. Waiting for segments...")
    print("Press Ctrl+C to stop logger safely.\n")

    current_file = None
    current_fname = None
    segment_id = None
    last_flush = time.time()
    line_count = 0

    try:
        while True:
            try:
                raw = ser.readline()
                if not raw:
                    continue

                line = raw.decode(errors="ignore").strip()

                # ---- SEGMENT START ----
                if line.startswith("#SEGMENT_START"):
                    _, seg_id, mode_code, t_ms = line.split(",")
                    segment_id = seg_id

                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    current_fname = f"{mode_tag}_seg{segment_id}_{ts}.csv"
                    current_file = open(current_fname, "w")

                    print(f"[START] Segment {segment_id} -> writing {current_fname}")
                    line_count = 0
                    last_flush = time.time()
                    continue

                # ---- SEGMENT END ----
                if line.startswith("#SEGMENT_END"):
                    if current_file:
                        safe_flush(current_file)
                        current_file.close()
                        print(f"[END] Segment {segment_id} saved -> {current_fname} (lines={line_count})")
                    current_file = None
                    current_fname = None
                    segment_id = None
                    continue

                # ---- DATA LINE ----
                if current_file:
                    current_file.write(line + "\n")
                    line_count += 1

                    now = time.time()
                    if now - last_flush >= FLUSH_INTERVAL:
                        safe_flush(current_file)
                        last_flush = now

            except serial.SerialException as e:
                print(f"[WARN] Serial error: {e}")
                if current_file:
                    print("[WARN] Closing active file due to serial error.")
                    safe_flush(current_file)
                    current_file.close()
                    current_file = None
                time.sleep(1)

    except KeyboardInterrupt:
        print("\n[STOP] Logger interrupted by user.")
    finally:
        if current_file:
            safe_flush(current_file)
            current_file.close()
            print(f"[FINAL] File closed safely -> {current_fname}")
        ser.close()
        print("[DONE] Serial port closed.")

if __name__ == "__main__":
    main()
