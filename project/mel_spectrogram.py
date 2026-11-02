"""Convert a sound clip to a mel-spectrogram for the model."""

import torch
import torchaudio


SAMPLE_RATE = 48000
FFT_SIZE = 1024
HOP_SIZE = 180
MEL_BANDS = 64

mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=FFT_SIZE,
    hop_length=HOP_SIZE,
    n_mels=MEL_BANDS,
)


def make_mel_spectrogram(clip):
    waveform = torch.from_numpy(clip)
    power = mel_transform(waveform)
    return torch.log10(power + 0.00000001).unsqueeze(0)
