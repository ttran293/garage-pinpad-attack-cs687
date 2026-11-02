"""Convert a sound clip to the original linear-frequency spectrogram."""

import torch


FFT_SIZE = 512
HOP_SIZE = 128


def make_spectrogram(clip):
    waveform = torch.from_numpy(clip)
    window = torch.hann_window(FFT_SIZE)
    spectrum = torch.stft(
        waveform,
        n_fft=FFT_SIZE,
        hop_length=HOP_SIZE,
        window=window,
        return_complex=True,
    )
    power = spectrum.abs().square()
    return torch.log10(power + 0.00000001).unsqueeze(0)
