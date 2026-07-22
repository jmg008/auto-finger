import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path


COLLECT_ORDER = [
    ("none", "none.txt"),
    ("rock", "rock.txt"),
    ("paper", "paper.txt"),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect EMG serial data for none, rock, and paper classes."
    )
    parser.add_argument("--port", default="COM15", help="Serial port, for example COM15")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--seconds", type=float, default=8.0, help="Seconds to collect per class")
    parser.add_argument("--sets", type=int, default=1, help="Number of none/rock/paper collection sets")
    parser.add_argument("--settle", type=float, default=2.0, help="Seconds to wait before recording")
    parser.add_argument("--arduino-cli", default="arduino-cli", help="Path to arduino-cli executable")
    parser.add_argument("--fqbn", default="arduino:avr:uno", help="Arduino board FQBN")
    parser.add_argument("--sensor-sketch", default="EMG_Sensor", help="Raw EMG serial sketch folder")
    parser.add_argument("--no-upload", action="store_true", help="Do not upload the raw sensor sketch")
    parser.add_argument("--upload-wait", type=float, default=2.0, help="Seconds to wait after upload")
    return parser.parse_args()


def run_command(command, cwd=None):
    print("$ " + " ".join(str(part) for part in command))
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True)

    if result.stdout:
        print(result.stdout.rstrip())
    if result.stderr:
        print(result.stderr.rstrip())

    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(command)}")


def resolve_arduino_cli(cli_name):
    path = Path(cli_name)
    if path.exists():
        return str(path)

    found = shutil.which(cli_name)
    if found:
        return found

    candidates = [
        Path.home() / "Documents" / "Codex" / "tools" / "arduino-cli" / "arduino-cli.exe",
        Path.home() / "AppData" / "Local" / "Programs" / "Arduino CLI" / "arduino-cli.exe",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    raise FileNotFoundError(
        "arduino-cli executable was not found. Pass the full path with "
        "--arduino-cli, for example: --arduino-cli "
        r"C:\Users\j\Documents\Codex\tools\arduino-cli\arduino-cli.exe"
    )


def normalize_sketch_path(sketch_path):
    path = Path(sketch_path)
    if path.suffix.lower() == ".ino":
        path = path.parent
    return path


def upload_sensor_sketch(args):
    path = normalize_sketch_path(args.sensor_sketch)
    if not path.exists():
        raise FileNotFoundError(f"Raw sensor sketch not found: {path}")

    print("\n[Raw sensor sketch] Compile")
    run_command([args.arduino_cli, "compile", "--fqbn", args.fqbn, str(path)])

    print("\n[Raw sensor sketch] Upload")
    run_command(
        [args.arduino_cli, "upload", "-p", args.port, "--fqbn", args.fqbn, str(path)]
    )

    if args.upload_wait > 0:
        print(f"Waiting {args.upload_wait:.1f}s after upload...")
        time.sleep(args.upload_wait)


def ensure_pyserial_available():
    try:
        import serial  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("pyserial is required for collection: pip install pyserial") from exc


def parse_sensor_line(raw_line):
    line = raw_line.decode("utf-8", errors="ignore").strip()
    if not line:
        return None

    parts = line.split(",")
    if len(parts) < 2:
        return None

    try:
        return int(float(parts[0])), int(float(parts[1]))
    except ValueError:
        return None


def collect_one_class(ser, name, out_path, seconds, settle, append=False, set_index=1, set_count=1):
    input(f"\nPrepare '{name}' ({set_index}/{set_count}), then press Enter to collect.")
    ser.reset_input_buffer()

    if settle > 0:
        print(f"Settling for {settle:.1f}s...")
        time.sleep(settle)
        ser.reset_input_buffer()

    rows = []
    end_at = time.monotonic() + seconds
    next_report = time.monotonic() + 1.0

    while time.monotonic() < end_at:
        parsed = parse_sensor_line(ser.readline())
        if parsed is not None:
            rows.append(parsed)

        if time.monotonic() >= next_report:
            print(f"  {name}: {len(rows)} samples")
            next_report += 1.0

    if not rows:
        raise RuntimeError(
            "No numeric sensor rows were collected. Upload the raw serial sketch first; "
            "it should print lines like '123,456'."
        )

    mode = "a" if append else "w"
    with open(out_path, mode, encoding="utf-8") as file:
        for inside, outside in rows:
            file.write(f"{inside},{outside}\n")

    action = "Appended" if append else "Saved"
    print(f"{action} {len(rows)} samples to {out_path}")


def collect_data(args):
    import serial

    print("Expected serial format: inside,outside")

    with serial.Serial(port=args.port, baudrate=args.baud, timeout=0.1) as ser:
        time.sleep(2.0)
        ser.reset_input_buffer()

        written_files = set()
        for set_index in range(1, args.sets + 1):
            print(f"\n[Collection set {set_index}/{args.sets}]")
            for name, filename in COLLECT_ORDER:
                path = Path(filename)
                collect_one_class(
                    ser,
                    name,
                    path,
                    args.seconds,
                    args.settle,
                    append=path in written_files,
                    set_index=set_index,
                    set_count=args.sets,
                )
                written_files.add(path)


def main():
    args = parse_args()
    if args.sets < 1:
        raise ValueError("--sets must be at least 1")
    if args.seconds <= 0:
        raise ValueError("--seconds must be greater than 0")
    if args.settle < 0:
        raise ValueError("--settle must be 0 or greater")

    ensure_pyserial_available()

    if not args.no_upload:
        args.arduino_cli = resolve_arduino_cli(args.arduino_cli)
        upload_sensor_sketch(args)

    collect_data(args)
    print("\nDone.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
        sys.exit(130)
