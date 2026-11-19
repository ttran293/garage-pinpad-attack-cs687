"""Local keypad recording app and dataset utilities. Requires Python 3.10+."""

from __future__ import annotations

import argparse
import csv
import io
import json
import random
import re
import struct
import threading
import wave
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parent
DATASET = "pins"
DATA = ROOT / "data"
PLAN = DATA / "plan.csv"
MANIFEST = DATA / "manifest.csv"
AUDIO = DATA / "audio"
WEB = ROOT / "web"
PIN_FIELDS = ["trial_id", "pin", "audio_path", "session_id", "keypad_id",
          "distance_cm", "recorded_at_utc", "sample_rate", "duration_s", "peak", "clipped_pct"]
CALIBRATION_FIELDS = ["trial_id", "digit", "block", "repeat", *PIN_FIELDS[2:]]
FIELDS = PIN_FIELDS
LOCK = threading.Lock()


def configure_dataset(name: str) -> None:
    """Select a dataset once at startup; PIN training paths remain unchanged."""
    global DATASET, DATA, PLAN, MANIFEST, AUDIO, FIELDS
    if name not in ("pins", "calibration"):
        raise ValueError("Dataset must be pins or calibration")
    DATASET = name
    DATA = ROOT / ("calibration" if name == "calibration" else "data")
    PLAN, MANIFEST, AUDIO = DATA / "plan.csv", DATA / "manifest.csv", DATA / "audio"
    FIELDS = CALIBRATION_FIELDS if name == "calibration" else PIN_FIELDS


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def make_plan(count: int, seed: int) -> list[dict[str, str]]:
    if count < 10:
        raise ValueError("Count must be at least 10")
    rng = random.Random(seed)
    for _ in range(100):
        columns = []
        for _position in range(4):
            digits = list(range(10)) * (count // 10)
            digits += rng.sample(range(10), count % 10)
            rng.shuffle(digits)
            columns.append(digits)
        pins = ["".join(str(columns[position][i]) for position in range(4))
                for i in range(count)]
        if len(set(pins)) == count:
            return [{"trial_id": f"T{i + 1:04d}", "pin": pin}
                    for i, pin in enumerate(pins)]
    raise ValueError("Could not generate unique PINs; try a smaller count or another seed")


def make_calibration_plan(count: int = 200, seed: int = 687) -> list[dict[str, str]]:
    """One of each digit per randomized block; twenty examples/digit by default."""
    if count < 10 or count % 10:
        raise ValueError("Calibration count must be a positive multiple of 10")
    rng = random.Random(seed)
    rows = []
    for block in range(1, count // 10 + 1):
        digits = list("0123456789")
        rng.shuffle(digits)
        if rows and digits[0] == rows[-1]["digit"]:
            digits[0], digits[1] = digits[1], digits[0]
        for digit in digits:
            rows.append({"trial_id": f"C{len(rows) + 1:04d}", "digit": digit,
                         "block": str(block), "repeat": str(block)})
    return rows


def init(count: int, seed: int) -> None:
    DATA.mkdir(exist_ok=True)
    AUDIO.mkdir(exist_ok=True)
    if PLAN.exists() or MANIFEST.exists():
        raise FileExistsError("Dataset already initialized; existing recordings were preserved")
    calibration = DATASET == "calibration"
    rows = make_calibration_plan(count, seed) if calibration else make_plan(count, seed)
    with PLAN.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["trial_id", "digit", "block", "repeat"]
                                if calibration else ["trial_id", "pin"])
        writer.writeheader()
        writer.writerows(rows)
    with MANIFEST.open("w", newline="", encoding="utf-8") as file:
        csv.DictWriter(file, fieldnames=FIELDS).writeheader()
    print(f"Created {count} {'single-key calibration' if calibration else 'four-digit'} trials in {PLAN}")


def audio_stats(blob: bytes) -> tuple[int, float, float, float]:
    try:
        with wave.open(io.BytesIO(blob), "rb") as wav:
            if wav.getnchannels() != 1 or wav.getsampwidth() != 2 or wav.getcomptype() != "NONE":
                raise ValueError("Audio must be mono, uncompressed 16-bit PCM WAV")
            rate, frames = wav.getframerate(), wav.getnframes()
            if not 16000 <= rate <= 96000 or not 0.25 <= frames / rate <= 30:
                raise ValueError("Audio sample rate or duration is outside the allowed range")
            pcm = wav.readframes(frames)
    except (wave.Error, EOFError, struct.error) as exc:
        raise ValueError("Invalid WAV file") from exc
    samples = struct.iter_unpack("<h", pcm)
    peak = 0
    clipped = 0
    for (sample,) in samples:
        magnitude = abs(sample)
        peak = max(peak, magnitude)
        clipped += magnitude >= 32760
    return rate, frames / rate, peak / 32768, 100 * clipped / frames


def safe_id(value: str, name: str) -> str:
    value = value.strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", value):
        raise ValueError(f"{name} must be 1-40 letters, numbers, underscores, or hyphens")
    return value


def save_recording(blob: bytes, headers) -> dict:
    calibration = DATASET == "calibration"
    if headers.get("X-Dataset", "pins") != DATASET:
        raise ValueError("Wrong dataset; reload the recording page for this server")
    trial_id = safe_id(headers.get("X-Trial-Id", ""), "Trial ID")
    session_id = safe_id(headers.get("X-Session-Id", ""), "Session ID")
    keypad_id = safe_id(headers.get("X-Keypad-Id", ""), "Keypad ID")
    distance = headers.get("X-Distance-Cm", "").strip()
    if not re.fullmatch(r"\d{1,3}", distance) or not 1 <= int(distance) <= 999:
        raise ValueError("Distance must be 1-999 cm")
    rate, duration, peak, clipped = audio_stats(blob)
    if calibration and duration < 3:
        raise ValueError("Calibration takes need at least 3 seconds to retain press and release audio")
    with LOCK:
        plan = {row["trial_id"]: row for row in read_csv(PLAN)}
        if trial_id not in plan:
            raise ValueError("Unknown trial ID")
        planned = plan[trial_id]
        if calibration and headers.get("X-Expected-Label", "") != planned["digit"]:
            raise ValueError("Displayed digit does not match this trial; reload the page")
        if trial_id in {row["trial_id"] for row in read_csv(MANIFEST)}:
            raise ValueError("This trial was already saved")
        AUDIO.mkdir(exist_ok=True)
        relative = f"audio/{trial_id}.wav"
        path = DATA / relative
        if path.exists():
            raise ValueError("Audio file already exists")
        path.write_bytes(blob)
        row = {
            "trial_id": trial_id, "audio_path": relative,
            "session_id": session_id, "keypad_id": keypad_id, "distance_cm": distance,
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "sample_rate": str(rate), "duration_s": f"{duration:.3f}",
            "peak": f"{peak:.4f}", "clipped_pct": f"{clipped:.3f}",
        }
        if calibration:
            row.update(digit=planned["digit"], block=planned["block"], repeat=planned["repeat"])
        else:
            row["pin"] = planned["pin"]
        try:
            with MANIFEST.open("a", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=FIELDS)
                writer.writerow(row)
        except Exception:
            path.unlink(missing_ok=True)
            raise
    return row


class Handler(BaseHTTPRequestHandler):
    def send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        route = urlparse(self.path).path
        if route == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
            return
        if route == "/**":
            self.send_response(302)
            self.send_header("Location", "/")
            self.end_headers()
            return
        if route == "/api/state":
            plan = read_csv(PLAN)
            done = {row["trial_id"] for row in read_csv(MANIFEST)}
            pending = next((row for row in plan if row["trial_id"] not in done), None)
            self.send_json(200, {"dataset": DATASET, "total": len(plan),
                                 "completed": len(done), "next": pending})
            return
        files = {"/": (WEB / "index.html", "text/html; charset=utf-8"),
                 "/recorder-worklet.js": (WEB / "recorder-worklet.js", "text/javascript; charset=utf-8")}
        if route not in files:
            self.send_error(404)
            return
        path, content_type = files[route]
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/record":
            self.send_error(404)
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 44 <= size <= 6_000_000:
                raise ValueError("Recording must be a WAV file under 6 MB")
            row = save_recording(self.rfile.read(size), self.headers)
            self.send_json(201, {"saved": row})
        except (ValueError, OSError) as exc:
            self.send_json(400, {"error": str(exc)})


def audit() -> None:
    plan = read_csv(PLAN)
    records = read_csv(MANIFEST)
    if not plan:
        raise ValueError("Run 'init' first")
    print(f"Saved: {len(records)}/{len(plan)}")
    if DATASET == "calibration":
        print("Single-key counts:", {str(d): sum(row["digit"] == str(d) for row in records)
                                     for d in range(10)})
        print("Completed blocks:", sum(all(row["trial_id"] in {r["trial_id"] for r in records}
                                          for row in plan if row["block"] == block)
                                      for block in {row["block"] for row in plan}))
    else:
        for position in range(4):
            counts = {str(d): sum(row["pin"][position] == str(d) for row in records)
                      for d in range(10)}
            print(f"Position {position + 1}: {counts}")
    print("Sessions:", {session: sum(row["session_id"] == session for row in records)
                         for session in sorted({row["session_id"] for row in records})})
    issues = []
    for row in records:
        path = DATA / row["audio_path"]
        if not path.is_file():
            issues.append(f"{row['trial_id']}: missing audio")
        elif float(row["peak"]) < 0.01:
            issues.append(f"{row['trial_id']}: very quiet")
        elif float(row["clipped_pct"]) > 0.1:
            issues.append(f"{row['trial_id']}: clipping")
    print("Quality issues:", issues if issues else "none detected")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    initialize = sub.add_parser("init", help="Create a balanced recording plan")
    initialize.add_argument("--count", type=int, help="Trials: default 50 PINs or 200 calibration presses")
    initialize.add_argument("--seed", type=int, default=687)
    serve = sub.add_parser("serve", help="Start the local recording app")
    serve.add_argument("--port", type=int, help="Default 8765 for PINs, 8766 for calibration")
    audit_parser = sub.add_parser("audit", help="Summarize saved recordings")
    for command in (initialize, serve, audit_parser):
        command.add_argument("--dataset", choices=["pins", "calibration"], default="pins")
    args = parser.parse_args()
    configure_dataset(args.dataset)
    if args.command == "init":
        count = args.count if args.count is not None else (200 if DATASET == "calibration" else 50)
        init(count, args.seed)
    elif args.command == "audit":
        audit()
    else:
        if not PLAN.exists():
            parser.error("Run 'init' before 'serve'")
        port = args.port if args.port is not None else (8766 if DATASET == "calibration" else 8765)
        server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        print(f"Dataset: {DATASET}; open http://localhost:{port}/ and press Ctrl+C to stop")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            server.server_close()


if __name__ == "__main__":
    main()
