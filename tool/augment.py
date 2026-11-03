"""Preview simple audio changes on one keypad press."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

from project.audio import SAMPLE_RATE, detect_presses, make_clip
from project.mel_spectrogram import make_mel_spectrogram
from tool.spectrogram import load_audio


OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "augmentation"


def change_volume(clip, gain_db):
    changed = clip * (10 ** (gain_db / 20))
    if np.max(np.abs(changed)) >= 1:
        raise ValueError("Volume change would clip the audio. Use a smaller --gain-db.")
    return changed.astype(np.float32)


def shift_clip(clip, shift_ms):
    samples = round(shift_ms * SAMPLE_RATE / 1000)
    if abs(samples) >= len(clip):
        raise ValueError("The timing shift is longer than the clip.")

    changed = np.zeros_like(clip)
    if samples > 0:
        changed[samples:] = clip[:-samples]
    elif samples < 0:
        changed[:samples] = clip[-samples:]
    else:
        changed[:] = clip
    return changed


def add_noise(clip, snr_db, rng):
    signal_rms = np.sqrt(np.mean(clip ** 2))
    noise_rms = signal_rms / (10 ** (snr_db / 20))
    noise = rng.normal(0, noise_rms, len(clip))
    changed = clip + noise
    if np.max(np.abs(changed)) >= 1:
        raise ValueError("Noise would clip the audio. Use a larger --noise-snr-db.")
    return changed.astype(np.float32)


def save_preview(clips, output_dir, name):
    output_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(len(clips), 2, figsize=(12, 13))
    max_amplitude = max(np.max(np.abs(clip)) for clip in clips.values()) * 1.1
    times = np.arange(len(next(iter(clips.values())))) / SAMPLE_RATE

    for row, (label, clip) in enumerate(clips.items()):
        wav_path = output_dir / f"{name}_{label}.wav"
        sf.write(wav_path, clip, SAMPLE_RATE)

        axes[row, 0].plot(times, clip, linewidth=0.5)
        axes[row, 0].set_ylim(-max_amplitude, max_amplitude)
        axes[row, 0].set_ylabel(label + "\nAmplitude")

        image = make_mel_spectrogram(clip)[0].numpy()
        axes[row, 1].imshow(image, origin="lower", aspect="auto",
                            cmap="magma", vmin=-7, vmax=2,
                            extent=(0, times[-1], 0, image.shape[0]))
        axes[row, 1].set_ylabel("Mel band")

    axes[-1, 0].set_xlabel("Time (seconds)")
    axes[-1, 1].set_xlabel("Time (seconds)")
    figure.suptitle(name + ": original and augmented press")
    figure.tight_layout()
    image_path = output_dir / f"{name}_comparison.png"
    figure.savefig(image_path, dpi=150)
    plt.close(figure)
    return image_path


def main():
    parser = argparse.ArgumentParser(description="Preview keypad audio augmentation.")
    parser.add_argument("wav", type=Path, help="Original four-digit WAV recording")
    parser.add_argument("--press", type=int, choices=[1, 2, 3, 4], default=1)
    parser.add_argument("--gain-db", type=float, default=6)
    parser.add_argument("--shift-ms", type=float, default=10)
    parser.add_argument("--noise-snr-db", type=float, default=30)
    parser.add_argument("--seed", type=int, default=687)
    args = parser.parse_args()

    if not args.wav.is_file():
        parser.error("WAV file does not exist: " + str(args.wav))
    if args.noise_snr_db <= 0:
        parser.error("--noise-snr-db must be greater than zero")

    audio = load_audio(args.wav)
    press_time = detect_presses(audio)[args.press - 1]
    original = make_clip(audio, press_time)
    rng = np.random.default_rng(args.seed)

    clips = {
        "original": original,
        "volume": change_volume(original, args.gain_db),
        "shift": shift_clip(original, args.shift_ms),
        "noise": add_noise(original, args.noise_snr_db, rng),
    }
    combined = shift_clip(change_volume(original, args.gain_db), args.shift_ms)
    clips["combined"] = add_noise(combined, args.noise_snr_db, rng)

    name = f"{args.wav.stem}_press{args.press}"
    output_dir = OUTPUT_DIR / name
    image_path = save_preview(clips, output_dir, name)
    print("Press:", args.press, "at", round(press_time, 3), "seconds")
    print("Saved WAV previews and comparison image to:", output_dir.resolve())
    print("Comparison image:", image_path.resolve())


if __name__ == "__main__":
    main()
