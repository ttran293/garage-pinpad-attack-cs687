# Spectrogram tool

Run these commands from the repository root. They save PNG images under `tool/output/`.

```powershell
.\.venv\Scripts\python.exe -m tool.spectrogram scripts/data/audio/T0001.wav
.\.venv\Scripts\python.exe -m tool.spectrogram scripts/data/audio/T0001.wav --presses
```

The first command shows the whole recording as a waveform and spectrogram. The
second shows the four short clips that the model uses, with a line at each
detected press peak. This view uses the same press detector, clip length, and
spectrogram conversion as `project/audio.py`.

To save an image somewhere else, add `--output path/to/image.png`. Install the
shared environment dependencies with `.\.venv\Scripts\python.exe -m pip install -r project/requirements.txt`.
