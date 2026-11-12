"""Predict a four-digit PIN from a WAV and a saved checkpoint."""

import argparse
import json
import sys
from pathlib import Path

import joblib
import torch

if __package__:
    from . import enhanced_features
    from .audio import recording_spectrograms
    from .model import KeypadCNN, KeypadCoAtNet, KeypadGRU
else:
    import enhanced_features
    from audio import recording_spectrograms
    from model import KeypadCNN, KeypadCoAtNet, KeypadGRU


SPECTROGRAM_MODELS = {"cnn": KeypadCNN, "coatnet": KeypadCoAtNet}


def predict_svm(checkpoint, wav):
    # Joblib records the scaler's import path from the training command.
    if __package__:
        sys.modules.setdefault("enhanced_features", enhanced_features)
    else:
        root = str(Path(__file__).resolve().parents[1])
        if root not in sys.path:
            sys.path.append(root)

    saved = joblib.load(checkpoint)
    if saved.get("feature_set") != enhanced_features.HARMONIC_FEATURE_SET:
        raise ValueError("Expected a harmonic SVM checkpoint.")

    features = enhanced_features.recording_features(wav, harmonics=True)
    digits = saved["classifier"].predict(features)
    return "".join(map(str, digits))


@torch.inference_mode()
def predict(checkpoint, wav):
    if checkpoint.suffix.lower() == ".joblib":
        return predict_svm(checkpoint, wav)

    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model_type = saved.get("model_type")
    if model_type == "gru":
        model = KeypadGRU(saved["features"])
        features = torch.from_numpy(enhanced_features.recording_features(wav)).unsqueeze(0)
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
