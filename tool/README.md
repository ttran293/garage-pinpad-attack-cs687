# Spectrogram tool

Run these commands from the repository root. They save PNG images under `tool/output/`.

```powershell
.\.venv\Scripts\python.exe -m tool.spectrogram scripts/data/audio/T0001.wav
.\.venv\Scripts\python.exe -m tool.spectrogram scripts/data/audio/T0001.wav --presses
```

The first command shows the whole recording as a waveform and mel-spectrogram. The
second shows the four short clips that the model uses, with a line at each
detected press peak. This view uses the same press detector, clip length, and
mel-spectrogram conversion as the model in `project/mel_spectrogram.py`.

To save an image somewhere else, add `--output path/to/image.png`. Install the
shared environment dependencies with `.\.venv\Scripts\python.exe -m pip install -r project/requirements.txt`.

## Augmentation preview

Preview volume, timing, and light white-noise changes on one detected press:

```powershell
.\.venv\Scripts\python.exe -m tool.augment scripts/data/audio/T0001.wav --press 1
```

This saves five short WAV files (original, volume, shift, noise, combined) and
one comparison PNG under `tool/output/augmentation/T0001_press1/`. The source
recording and the manifest are not changed. Adjust the example with
`--gain-db -6`, `--shift-ms -10`, `--noise-snr-db 30`, or `--seed 687`.
The shift is applied to the short clip, after the press has been detected.
The preview does not add augmented recordings to the training dataset.
