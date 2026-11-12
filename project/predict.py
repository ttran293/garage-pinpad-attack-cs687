"""Predict a four-digit PIN from a WAV and a saved checkpoint."""

import argparse
import json
from pathlib import Path

import torch

if __package__:
    from .audio import recording_spectrograms
    from .enhanced_features import recording_features
    from .model import KeypadCNN, KeypadCoAtNet, KeypadGRU
else:
    from audio import recording_spectrograms
    from enhanced_features import recording_features
    from model import KeypadCNN, KeypadCoAtNet, KeypadGRU


SPECTROGRAM_MODELS = {"cnn": KeypadCNN, "coatnet": KeypadCoAtNet}


@torch.inference_mode()
def predict(checkpoint, wav):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model_type = saved.get("model_type")
    if model_type == "gru":
        model = KeypadGRU(saved["features"])
        features = torch.from_numpy(recording_features(wav)).unsqueeze(0)
    elif model_type in SPECTROGRAM_MODELS:
        if saved.get("spectrogram_type") != "mel":
            raise ValueError("Expected a mel-spectrogram checkpoint.")
        model = SPECTROGRAM_MODELS[model_type]()
        features, _ = recording_spectrograms(wav)
    else:
        raise ValueError(f"Unsupported checkpoint model: {model_type!r}")

    model.load_state_dict(saved["model_state"])
    model.eval()
    features = (features - saved["mean"]) / saved["standard_deviation"]
    digits = model(features).argmax(dim=-1).flatten().tolist()
    return "".join(map(str, digits))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("wav", type=Path)
    args = parser.parse_args()
    print(json.dumps({"predicted_pin": predict(args.checkpoint, args.wav)}))


if __name__ == "__main__":
    main()
