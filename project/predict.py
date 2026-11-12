"""Predict a four-digit PIN from a WAV and a GRU checkpoint."""

import argparse
import json
from pathlib import Path

import torch

if __package__:
    from .enhanced_features import recording_features
    from .model import KeypadGRU
else:
    from enhanced_features import recording_features
    from model import KeypadGRU


@torch.inference_mode()
def predict(checkpoint, wav):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved.get("model_type") != "gru":
        raise ValueError("Expected a GRU checkpoint.")

    model = KeypadGRU(saved["features"])
    model.load_state_dict(saved["model_state"])
    model.eval()
    features = torch.from_numpy(recording_features(wav)).unsqueeze(0)
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
