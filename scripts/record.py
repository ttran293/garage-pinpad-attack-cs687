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
DATA = ROOT / "data"
PLAN = DATA / "plan.csv"
MANIFEST = DATA / "manifest.csv"
AUDIO = DATA / "audio"
WEB = ROOT / "web"
FIELDS = ["trial_id", "pin", "audio_path", "session_id", "keypad_id",
          "distance_cm", "recorded_at_utc", "sample_rate", "duration_s", "peak", "clipped_pct"]
LOCK = threading.Lock()


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


def init(count: int, seed: int) -> None:
    DATA.mkdir(exist_ok=True)
    AUDIO.mkdir(exist_ok=True)
    if PLAN.exists() or MANIFEST.exists():
        raise FileExistsError("Dataset already initialized; existing recordings were preserved")
    rows = make_plan(count, seed)
    with PLAN.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["trial_id", "pin"])
        writer.writeheader()
        writer.writerows(rows)
    with MANIFEST.open("w", newline="", encoding="utf-8") as file:
        csv.DictWriter(file, fieldnames=FIELDS).writeheader()
    print(f"Created {count} balanced four-digit trials in {PLAN}")


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
    trial_id = safe_id(headers.get("X-Trial-Id", ""), "Trial ID")
    session_id = safe_id(headers.get("X-Session-Id", ""), "Session ID")
    keypad_id = safe_id(headers.get("X-Keypad-Id", ""), "Keypad ID")
    distance = headers.get("X-Distance-Cm", "").strip()
    if not re.fullmatch(r"\d{1,3}", distance) or not 1 <= int(distance) <= 999:
        raise ValueError("Distance must be 1-999 cm")
    rate, duration, peak, clipped = audio_stats(blob)
    with LOCK:
        plan = {row["trial_id"]: row["pin"] for row in read_csv(PLAN)}
        if trial_id not in plan:
            raise ValueError("Unknown trial ID")
        if trial_id in {row["trial_id"] for row in read_csv(MANIFEST)}:
            raise ValueError("This trial was already saved")
        AUDIO.mkdir(exist_ok=True)
        relative = f"audio/{trial_id}.wav"
        path = DATA / relative
        if path.exists():
            raise ValueError("Audio file already exists")
        path.write_bytes(blob)
        row = {
            "trial_id": trial_id, "pin": plan[trial_id], "audio_path": relative,
            "session_id": session_id, "keypad_id": keypad_id, "distance_cm": distance,
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "sample_rate": str(rate), "duration_s": f"{duration:.3f}",
            "peak": f"{peak:.4f}", "clipped_pct": f"{clipped:.3f}",
        }
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
            self.send_json(200, {"total": len(plan), "completed": len(done), "next": pending})
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
    initialize.add_argument("--count", type=int, default=50)
    initialize.add_argument("--seed", type=int, default=687)
    serve = sub.add_parser("serve", help="Start the local recording app")
    serve.add_argument("--port", type=int, default=8765)
    sub.add_parser("audit", help="Summarize saved recordings")
    args = parser.parse_args()
    if args.command == "init":
        init(args.count, args.seed)
    elif args.command == "audit":
        audit()
    else:
        if not PLAN.exists():
            parser.error("Run 'init' before 'serve'")
        server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
        print(f"Open http://localhost:{args.port}/ and press Ctrl+C to stop")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            server.server_close()


if __name__ == "__main__":
    main()
