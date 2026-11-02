"""Save pictures of a WAV recording and its individual keypad presses."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from project.audio import (
    SAMPLE_RATE,
    SECONDS_BEFORE_PRESS,
    detect_presses,
    make_clip,
    make_spectrogram,
)


OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def load_audio(path):
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)  # Use one channel if the file is stereo.

    if sample_rate != SAMPLE_RATE:
        audio = resample_poly(audio, SAMPLE_RATE, sample_rate).astype(np.float32)

    return audio


def draw_clip(wave_ax, spectrogram_ax, audio, start_time):
    end_time = start_time + len(audio) / SAMPLE_RATE
    times = np.arange(len(audio)) / SAMPLE_RATE + start_time
    wave_ax.plot(times, audio, linewidth=0.5)
    wave_ax.set_ylabel("Amplitude")
    wave_ax.set_xlim(start_time, end_time)

    image = make_spectrogram(audio)[0].numpy()
    spectrogram_ax.imshow(image, origin="lower", aspect="auto", cmap="magma",
                          extent=(start_time, end_time, 0, SAMPLE_RATE / 2),
                          vmin=-7, vmax=2)
    spectrogram_ax.set_ylim(0, 12000)
    spectrogram_ax.set_ylabel("Frequency (Hz)")
    spectrogram_ax.set_xlabel("Time (seconds)")


def main():
    parser = argparse.ArgumentParser(description="Save a WAV spectrogram as a PNG image.")
    parser.add_argument("wav", type=Path, help="WAV file to view")
    parser.add_argument("--presses", action="store_true",
                        help="Show the four clips used by the model")
    parser.add_argument("--output", type=Path, help="PNG output path")
    args = parser.parse_args()

    if not args.wav.is_file():
        parser.error("WAV file does not exist: " + str(args.wav))

    audio = load_audio(args.wav)
    suffix = "_presses.png" if args.presses else "_spectrogram.png"
    output_path = args.output or OUTPUT_DIR / (args.wav.stem + suffix)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.presses:
        press_times = detect_presses(audio)
        clips = [make_clip(audio, time) for time in press_times]
        start_time = -SECONDS_BEFORE_PRESS
    else:
        clips = [audio]
        start_time = 0

    width = 16 if args.presses else 12
    figure, axes = plt.subplots(2, len(clips), figsize=(width, 6), squeeze=False)
    for position, clip in enumerate(clips):
        wave_ax = axes[0, position]
        spectrogram_ax = axes[1, position]
        draw_clip(wave_ax, spectrogram_ax, clip, start_time)
        if args.presses:
            wave_ax.axvline(0, color="red", linewidth=1)
            spectrogram_ax.axvline(0, color="cyan", linewidth=1)
            wave_ax.set_title(f"Press {position + 1} at {press_times[position]:.2f} s")

    figure.suptitle(args.wav.name)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    print("Saved:", output_path.resolve())


if __name__ == "__main__":
    main()
