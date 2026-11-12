"""Acoustic features for four-press recordings."""

import numpy as np
import torch
from scipy.signal import stft, welch
from torch.nn import functional as F

if __package__:
    from .audio import SAMPLE_RATE, detect_presses, make_clip, read_audio
else:
    from audio import SAMPLE_RATE, detect_presses, make_clip, read_audio


ATTACK_BANDS = ((0, 1000), (1000, 2400), (2400, 3100),
                (3100, 4800), (4800, 6000), (6000, 12000))


def window(audio, center, before, after):
    start = round((center - before) * SAMPLE_RATE)
    result = np.zeros(round((before + after) * SAMPLE_RATE), dtype=np.float32)
    lo = max(0, start)
    hi = min(len(audio), start + len(result))
    if hi > lo:
        result[lo - start:hi - start] = audio[lo:hi]
    return result


def onset_time(audio, peak):
    lo = max(0, round((peak - 0.080) * SAMPLE_RATE))
    hi = min(len(audio), round((peak + 0.025) * SAMPLE_RATE))
    segment = audio[lo:hi]
    if not len(segment):
        return peak

    energy = np.convolve(segment ** 2, np.ones(96) / 96, mode="same")
    maximum = int(np.argmax(energy))
    below = np.flatnonzero(energy[:maximum] < energy[maximum] * 0.10)
    edge = int(below[-1] + 1) if len(below) else 0
    return (lo + edge) / SAMPLE_RATE


def acoustic_summary(audio, peak, onset):
    clip = make_clip(audio, peak)
    rms = max(float(np.sqrt(np.mean(clip ** 2))), 1e-8)
    normalized = clip / rms
    frequencies, power = welch(normalized, fs=SAMPLE_RATE, nperseg=2048, noverlap=1536)
    power /= max(float(power.sum()), 1e-12)

    coarse = []
    for lo in range(0, 12000, 250):
        bins = (frequencies >= lo) & (frequencies < lo + 250)
        coarse.append(np.log10(power[bins].sum() + 1e-10))
    fine = np.log10(power[frequencies <= 9000] + 1e-10)
    envelope = np.sqrt(np.mean(normalized.reshape(120, 96) ** 2, axis=1))

    attack = window(audio, onset, 0.060, 0.100) / rms
    frequencies, _, spectrum = stft(attack, fs=SAMPLE_RATE, nperseg=512, noverlap=384)
    attack_power = np.abs(spectrum) ** 2
    bands = []
    for lo, hi in ATTACK_BANDS:
        bins = (frequencies >= lo) & (frequencies < hi)
        band = np.log10(attack_power[bins].sum(axis=0) + 1e-10)
        image = torch.as_tensor(band, dtype=torch.float32).reshape(1, 1, 1, -1)
        resized = F.interpolate(image, size=(1, 16), mode="bilinear", align_corners=False)
        bands.extend(resized.numpy().ravel())

    level_and_onset = [np.log10(rms), peak - onset]
    return np.concatenate([coarse, fine, envelope, bands, level_and_onset]).astype(np.float32)


def recording_features(path):
    audio = read_audio(path)
    peaks = detect_presses(audio)
    return np.stack([acoustic_summary(audio, peak, onset_time(audio, peak)) for peak in peaks])
