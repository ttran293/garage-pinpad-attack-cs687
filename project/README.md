# Garage keypad audio model

This project trains an original, randomly initialized PyTorch CNN to classify keypress sounds. The input is a full four-digit WAV recording. The audio code detects four press events, crops one short clip per event, and converts each clip to a log power spectrogram. The CNN learns to classify those clips as digits.

All recorded sessions feed one dataset. Session ID remains metadata. `train` uses a seeded 80/20 split of complete recordings, so all four clips from one PIN stay together. The current seven recordings are only a pilot; the resulting test set has two recordings and cannot support a reliable accuracy claim. Collect substantially more recordings before reporting model performance.

## Commands

Run these from the repository root:

```powershell
.\.venv\Scripts\python.exe project/train.py --inspect
.\.venv\Scripts\python.exe project/train.py --epochs 30
```

`--inspect` reports detected press times without training. The training command makes the 80/20 recording split, calculates normalization from training clips, trains the CNN on the GPU, and saves one model under `project/models/`. The 20% test recordings are held out for later evaluation. The saved file contains the trial IDs in each split and the normalization values.

The code is split into three files: `audio.py` prepares recordings, `model.py` defines the CNN, and `train.py` splits data, trains, and saves the model.

Create the GPU environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r project/requirements.txt
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

The requirements pin the CUDA 12.8 build of PyTorch 2.10.0. The training command will require CUDA rather than silently falling back to CPU.

The onset detector must be checked against more recording conditions before relying on it for a final result. A rejected recording is reported rather than silently assigned four guessed timestamps. Raw WAV files remain unchanged.
