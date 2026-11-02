# Garage keypad audio model

This project trains a CNN or CoAtNet to classify keypress sounds. The input is a full four-digit WAV recording. The audio code detects four press events, crops one short clip per event, and converts each clip to a mel-spectrogram. Each model learns to classify those clips as digits.

All recorded sessions feed one dataset. Session ID remains metadata. `train` uses a seeded 80/20 split of complete recordings, so all four clips from one PIN stay together. The current pilot has 50 recordings. Its test set is still small, so results should be treated as preliminary.

## Commands

Run these from the repository root:

```powershell
.\.venv\Scripts\python.exe project/train.py --inspect
.\.venv\Scripts\python.exe project/train.py --model both --epochs 30
```

`--inspect` reports detected press times without training. The training command makes one 80/20 recording split, calculates normalization from training clips, then trains both models from scratch on the GPU. It reports training and held-out test accuracy for individual digits and complete four-digit PINs. Each model is saved separately under `project/models/`. The saved files contain the trial IDs in each split and the normalization values. Use `--model cnn` or `--model coatnet` to train just one model.

`audio.py` prepares the recordings. `mel_spectrogram.py` creates the model input. `spectrogram.py` keeps the original linear-frequency version for comparison. `model.py` defines both models, and `train.py` trains and saves them. CoAtNet comes from the `timm` library and starts with random weights; it is not an ImageNet-pretrained model.

Create the GPU environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r project/requirements.txt
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

The requirements pin the CUDA 12.8 builds of PyTorch, torchvision, and torchaudio, plus `timm`. The training command requires CUDA rather than silently falling back to CPU.

The onset detector must be checked against more recording conditions before relying on it for a final result. A rejected recording is reported rather than silently assigned four guessed timestamps. Raw WAV files remain unchanged.
