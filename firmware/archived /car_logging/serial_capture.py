import serial
import serial.tools.list_ports
import time
from datetime import datetime

BAUD = 115200

def pick_port():
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        raise RuntimeError("No serial ports found.")
    print("Available ports:")
    for i, p in enumerate(ports):
        print(f"  [{i}] {p.device} - {p.description}")
    idx = int(input("Select port index: "))
    return ports[idx].device

def main():
    port = pick_port()
    fname = datetime.now().strftime("train_prelim_%Y%m%d_%H%M%S.csv")
    print(f"\nLogging to: {fname}")
    print("Press Ctrl+C to stop.\n")

    with serial.Serial(port, BAUD, timeout=1) as ser, open(fname, "w") as f:
        # Give the Feather time to reset when serial opens
        time.sleep(2)

        while True:
            line = ser.readline().decode(errors="ignore").strip()
            if not line:
                continue

            # Skip comment lines except START marker
            if line.startswith("#") and not line.startswith("#START_LOG"):
                continue

            f.write(line + "\n")
            f.flush()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped logging.")

