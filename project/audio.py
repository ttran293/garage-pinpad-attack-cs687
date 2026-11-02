"""Read keypad recordings and turn each press into a mel-spectrogram."""

import csv
import math
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import find_peaks, resample_poly

if __package__:
    from .mel_spectrogram import make_mel_spectrogram
else:
    from mel_spectrogram import make_mel_spectrogram


DATA_DIR = Path(__file__).resolve().parent.parent / "scripts" / "data"
SAMPLE_RATE = 48000
FRAME_SIZE = 480             # 10 milliseconds of audio
SECONDS_BEFORE_PRESS = 0.10
SECONDS_AFTER_PRESS = 0.14


def read_manifest():
    manifest_path = DATA_DIR / "manifest.csv"

    with open(manifest_path, newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    if len(rows) == 0:
        raise ValueError("The manifest has no saved recordings.")

    seen_ids = set()
    for row in rows:
        trial_id = row["trial_id"]
        pin = row["pin"]
        audio_path = DATA_DIR / row["audio_path"]

        if trial_id in seen_ids:
            raise ValueError("Duplicate trial ID: " + trial_id)
        if len(pin) != 4 or not pin.isdigit():
            raise ValueError("Invalid four-digit PIN for " + trial_id)
        if not audio_path.is_file():
            raise FileNotFoundError("Missing audio file: " + str(audio_path))

        seen_ids.add(trial_id)

    return rows


def read_audio(audio_path):
    audio, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)

    if audio.shape[1] != 1:
        raise ValueError("Expected mono audio: " + str(audio_path))

    audio = audio[:, 0]

    # Browsers can record at different rates. Convert them to one rate.
    if sample_rate != SAMPLE_RATE:
        divisor = math.gcd(sample_rate, SAMPLE_RATE)
        up = SAMPLE_RATE // divisor
        down = sample_rate // divisor
        audio = resample_poly(audio, up, down).astype(np.float32)

    return audio


def detect_presses(audio):
    """Find four strong sounds and return their times in seconds."""
    if len(audio) < SAMPLE_RATE // 2:
        raise ValueError("Recording is too short.")

    frame_count = len(audio) // FRAME_SIZE
    frames = audio[:frame_count * FRAME_SIZE].reshape(frame_count, FRAME_SIZE)
    frame_rms = np.sqrt(np.mean(frames * frames, axis=1))
    smoothed_rms = np.convolve(frame_rms, np.ones(5) / 5, mode="same")
    noise_level = max(float(np.quantile(smoothed_rms, 0.25)), 0.000001)

    peaks, _ = find_peaks(
        smoothed_rms,
        height=noise_level * 4,
        prominence=noise_level * 3,
        distance=18,              # 180 milliseconds between possible presses
    )

    if len(peaks) < 4:
        raise ValueError("Fewer than four possible presses were found.")

    # Rank the possible presses by loudness. This ignores a faint start click.
    ranked_peaks = []
    for peak in peaks:
        ranked_peaks.append((smoothed_rms[peak], peak))
    ranked_peaks.sort(reverse=True)

    fourth_loudest = ranked_peaks[3][0]
    if fourth_loudest < noise_level * 8:
        raise ValueError("The fourth press is too close to background noise.")

    if len(ranked_peaks) > 4:
        fifth_loudest = ranked_peaks[4][0]
        if fourth_loudest < fifth_loudest * 1.7:
            raise ValueError("The fourth and fifth sounds are too similar.")

    press_times = []
    for loudness, peak in ranked_peaks[:4]:
        press_times.append(float(peak * FRAME_SIZE / SAMPLE_RATE))
    press_times.sort()

    return press_times


def make_clip(audio, press_time):
    clip_length = round((SECONDS_BEFORE_PRESS + SECONDS_AFTER_PRESS) * SAMPLE_RATE)
    start = round((press_time - SECONDS_BEFORE_PRESS) * SAMPLE_RATE)
    clip = np.zeros(clip_length, dtype=np.float32)

    source_start = max(0, start)
    source_end = min(len(audio), start + clip_length)
    if source_end > source_start:
        destination_start = source_start - start
        destination_end = destination_start + (source_end - source_start)
        clip[destination_start:destination_end] = audio[source_start:source_end]

    return clip


def process_recording(row):
    audio_path = DATA_DIR / row["audio_path"]
    audio = read_audio(audio_path)
    press_times = detect_presses(audio)

    images = []
    for press_time in press_times:
        clip = make_clip(audio, press_time)
        images.append(make_mel_spectrogram(clip))

    return torch.stack(images), press_times
