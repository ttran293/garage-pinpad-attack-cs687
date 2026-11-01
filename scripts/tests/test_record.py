import csv
import io
import sys
import tempfile
import unittest
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import record


def wav_bytes() -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(48000)
        wav.writeframes((b"\x00\x10" * 48000))
    return buffer.getvalue()


class RecordingTest(unittest.TestCase):
    def test_browser_extra_requests(self):
        class FakeHandler:
            def __init__(self, path):
                self.path = path
                self.status = None
                self.headers = {}

            def send_response(self, status):
                self.status = status

            def send_header(self, name, value):
                self.headers[name] = value

            def end_headers(self):
                pass

        accidental_url = FakeHandler("/**")
        record.Handler.do_GET(accidental_url)
        self.assertEqual(accidental_url.status, 302)
        self.assertEqual(accidental_url.headers["Location"], "/")
        favicon = FakeHandler("/favicon.ico")
        record.Handler.do_GET(favicon)
        self.assertEqual(favicon.status, 204)

    def test_plan_is_unique_and_balanced_by_position(self):
        plan = record.make_plan(50, 687)
        pins = [row["pin"] for row in plan]
        self.assertEqual(len(set(pins)), 50)
        for position in range(4):
            self.assertEqual({digit: sum(pin[position] == digit for pin in pins)
                              for digit in "0123456789"},
                             {digit: 5 for digit in "0123456789"})

    def test_save_and_reject_duplicate(self):
        original = record.DATA, record.PLAN, record.MANIFEST, record.AUDIO
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temp:
            base = Path(temp)
            record.DATA, record.PLAN = base, base / "plan.csv"
            record.MANIFEST, record.AUDIO = base / "manifest.csv", base / "audio"
            try:
                with record.PLAN.open("w", newline="", encoding="utf-8") as file:
                    writer = csv.DictWriter(file, fieldnames=["trial_id", "pin"])
                    writer.writeheader()
                    writer.writerow({"trial_id": "T0001", "pin": "0047"})
                with record.MANIFEST.open("w", newline="", encoding="utf-8") as file:
                    csv.DictWriter(file, fieldnames=record.FIELDS).writeheader()
                headers = {"X-Trial-Id": "T0001", "X-Session-Id": "session1",
                           "X-Keypad-Id": "keypad1",
                           "X-Distance-Cm": "45"}
                row = record.save_recording(wav_bytes(), headers)
                self.assertEqual(row["pin"], "0047")
                self.assertEqual(record.read_csv(record.MANIFEST)[0]["pin"], "0047")
                self.assertTrue((base / row["audio_path"]).is_file())
                with self.assertRaisesRegex(ValueError, "already saved"):
                    record.save_recording(wav_bytes(), headers)
                self.assertEqual(len(record.read_csv(record.MANIFEST)), 1)
            finally:
                record.DATA, record.PLAN, record.MANIFEST, record.AUDIO = original


if __name__ == "__main__":
    unittest.main()
