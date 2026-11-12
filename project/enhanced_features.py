"""Acoustic features for four-press recordings."""

import numpy as np
import torch
from scipy.signal import butter, sosfiltfilt, stft, welch
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from torch.nn import functional as F

if __package__:
    from .audio import SAMPLE_RATE, detect_presses, make_clip, read_audio
else:
    from audio import SAMPLE_RATE, detect_presses, make_clip, read_audio


ATTACK_BANDS = ((0, 1000), (1000, 2400), (2400, 3100),
                (3100, 4800), (4800, 6000), (6000, 12000))
HARMONIC_LOWPASS = butter(3, 300, fs=SAMPLE_RATE, output="sos")
HARMONIC_BLOCKS = (651, 161, 161)
HARMONIC_FEATURE_SET = "baseline_h12_relative"


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


def carrier_frequency(clip):
    if float(np.mean(clip ** 2)) < 1e-12:
        return 2720.0

    fft_size = 32768
    spectrum = np.abs(np.fft.rfft(clip * np.hanning(len(clip)), n=fft_size)) ** 2
    frequencies = np.fft.rfftfreq(fft_size, 1 / SAMPLE_RATE)
    candidates = np.flatnonzero((frequencies >= 2500) & (frequencies <= 2950))
    index = int(candidates[np.argmax(spectrum[candidates])])
    left, center, right = np.log(spectrum[index - 1:index + 2] + 1e-20)
    curvature = left - 2 * center + right
    offset = 0.0
    if abs(curvature) > 1e-12:
        offset = 0.5 * (left - right) / curvature
    return float((index + np.clip(offset, -0.5, 0.5)) * SAMPLE_RATE / fft_size)


def average_bins(values, count):
    return np.array([part.mean() for part in np.array_split(values, count)])


def first_two_harmonics(clip):
    frequency = carrier_frequency(clip)
    time = np.arange(len(clip)) / SAMPLE_RATE
    basebands = []
    for harmonic in (1, 2):
        shifted = clip * np.exp(-2j * np.pi * frequency * harmonic * time)
        basebands.append(sosfiltfilt(HARMONIC_LOWPASS, shifted))
    envelopes = 2 * np.abs(basebands)
    reference = max(float(np.sqrt(np.mean(envelopes[0] ** 2))), 1e-8)
    attack_samples = round(0.060 * SAMPLE_RATE)
    features = [(frequency - 2720) / 100]

    for raw_envelope, baseband in zip(envelopes, basebands):
        envelope = raw_envelope / reference
        features.extend(np.log1p(average_bins(envelope, 32)))
        features.extend(np.log1p(average_bins(envelope[:attack_samples], 24)))
        features.extend(np.quantile(envelope, [0.1, 0.25, 0.5, 0.75, 0.9, 0.99]))
        energy = envelope ** 2
        cumulative = np.cumsum(energy) / (float(energy.sum()) + 1e-12)
        features.extend(np.searchsorted(cumulative, q) / len(clip) for q in (0.1, 0.25, 0.5, 0.75, 0.9))
        features.append(float(np.log10(np.mean(energy) + 1e-10)))

        increments = np.angle(baseband[1:] * baseband[:-1].conj()) * SAMPLE_RATE / (2 * np.pi)
        active = ((raw_envelope[1:] > 0.1 * max(float(raw_envelope.max()), 1e-8))
                  & (raw_envelope[1:] > reference * 0.005))
        features.extend(average_bins(np.where(active, np.clip(increments, -300, 300) / 300, 0.0), 12))

    return np.asarray(features, dtype=np.float32)


def recording_features(path, *, harmonics=False):
    audio = read_audio(path)
    peaks = detect_presses(audio)
    onsets = [onset_time(audio, peak) for peak in peaks]
    summary = np.stack([acoustic_summary(audio, peak, onset) for peak, onset in zip(peaks, onsets)])
    if not harmonics:
        return summary

    harmonic = np.stack([first_two_harmonics(window(audio, onset, 0.030, 0.130)) for onset in onsets])
    relative = harmonic - harmonic.mean(axis=0)
    return np.concatenate([summary, harmonic, relative], axis=1)


class BlockScaler(BaseEstimator, TransformerMixin):
    """The archived SVM checkpoint loads this class by name."""

    def __init__(self, blocks):
        self.blocks = blocks

    def fit(self, x, y=None):
        if sum(self.blocks) != x.shape[1]:
            raise ValueError("Feature block sizes do not match input.")
        self.scaler_ = StandardScaler().fit(x)
        self.weights_ = np.repeat(1 / np.sqrt(self.blocks), self.blocks)
        return self

    def transform(self, x):
        return self.scaler_.transform(x) * self.weights_
