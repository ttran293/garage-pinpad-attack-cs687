"""Train a keypad model and score it on held-out recordings."""

import argparse
import math
import random
from datetime import datetime
from pathlib import Path

import joblib
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

if __package__:
    from .audio import DATA_DIR, process_recording, read_manifest
    from .enhanced_features import HARMONIC_FEATURE_SET, recording_features
    from .model import KeypadCNN, KeypadCoAtNet, KeypadGRU, make_keypad_svm
else:
    from audio import DATA_DIR, process_recording, read_manifest
    from enhanced_features import HARMONIC_FEATURE_SET, recording_features
    from model import KeypadCNN, KeypadCoAtNet, KeypadGRU, make_keypad_svm


PROJECT_DIR = Path(__file__).resolve().parent
RANDOM_SEED = 687


def split_recordings(rows, seed=RANDOM_SEED):
    """Split whole recordings, never individual digit clips."""
    if len(rows) < 2:
        raise ValueError("At least two recordings are needed for a train/test split.")

    shuffled_rows = rows.copy()
    random.Random(seed).shuffle(shuffled_rows)

    test_count = math.ceil(len(shuffled_rows) * 0.20)
    test_rows = shuffled_rows[:test_count]
    train_rows = shuffled_rows[test_count:]

    return train_rows, test_rows


def prepare_training_data(rows, *, acoustic=False, harmonics=False):
    use_acoustic = acoustic or harmonics
    features = []
    labels = []
    for row in rows:
        if use_acoustic:
            wav = DATA_DIR / row["audio_path"]
            recording = torch.from_numpy(recording_features(wav, harmonics=harmonics))
        else:
            recording, _ = process_recording(row)
        features.append(recording)
        labels.append([int(digit) for digit in row["pin"]])
    features = torch.stack(features)
    labels = torch.tensor(labels, dtype=torch.long)
    if use_acoustic:
        return features, labels
    return features.flatten(0, 1), labels.flatten()


def check_training_digits(labels):
    missing_digits = set(range(10)) - set(labels.flatten().tolist())
    if missing_digits:
        raise ValueError("Training recordings have no examples of: " + str(sorted(missing_digits)))


def train_model(train_rows, epochs, seed, model_name):
    is_gru = model_name == "gru"
    cuda_available = torch.cuda.is_available()
    if not cuda_available and not is_gru:
        raise RuntimeError("CUDA is unavailable. Use the project .venv.")
    if epochs < 1:
        raise ValueError("Epochs must be at least 1.")

    torch.manual_seed(seed)
    device = torch.device("cuda:0" if cuda_available else "cpu")

    features, labels = prepare_training_data(train_rows, acoustic=is_gru)
    check_training_digits(labels)

    if is_gru:
        mean = features.mean(dim=(0, 1), keepdim=True)
        standard_deviation = features.std(dim=(0, 1), correction=0, keepdim=True).clamp_min(0.1)
        model = KeypadGRU(features.shape[-1]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    else:
        mean = features.mean()
        standard_deviation = features.std().clamp_min(0.000001)
        if model_name == "cnn":
            model = KeypadCNN().to(device)
            learning_rate = 0.001
        else:
            model = KeypadCoAtNet().to(device)
            learning_rate = 0.0005
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    dataset = TensorDataset((features - mean) / standard_deviation, labels)
    loader = DataLoader(dataset, batch_size=16, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    loss_function = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch_features, batch_labels in loader:
            optimizer.zero_grad()
            logits = model(batch_features.to(device)).reshape(-1, 10)
            loss = loss_function(logits, batch_labels.to(device).flatten())
            loss.backward()
            if is_gru:
                nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()

            total_loss += loss.item()

        if epoch == 0 or (epoch + 1) % 5 == 0 or epoch + 1 == epochs:
            average_loss = total_loss / len(loader)
            print(f"Epoch {epoch + 1}/{epochs}: loss {average_loss:.4f}")

    return model, mean, standard_deviation


def evaluate_model(model, rows, mean, standard_deviation):
    features, labels = prepare_training_data(rows, acoustic=isinstance(model, KeypadGRU))
    features = (features - mean) / standard_deviation
    loader = DataLoader(TensorDataset(features), batch_size=16)
    device = next(model.parameters()).device

    model.eval()
    predicted_digits = []
    with torch.no_grad():
        for (batch_features,) in loader:
            predictions = model(batch_features.to(device))
            predicted_digits.extend(predictions.argmax(dim=-1).cpu().flatten().tolist())

    return score_predictions(predicted_digits, labels)


def score_predictions(predictions, labels):
    correct = torch.as_tensor(predictions).reshape(-1, 4) == labels.reshape(-1, 4)
    return correct.sum().item(), correct.numel(), correct.all(dim=1).sum().item(), len(correct)


def train_svm(train_rows, test_rows):
    features, labels = prepare_training_data(train_rows, harmonics=True)
    check_training_digits(labels)
    train_features = features.flatten(0, 1).numpy()
    model = make_keypad_svm().fit(train_features, labels.flatten().numpy())
    train_result = score_predictions(model.predict(train_features), labels)

    test_features, test_labels = prepare_training_data(test_rows, harmonics=True)
    predictions = model.predict(test_features.flatten(0, 1).numpy())
    return model, train_result, score_predictions(predictions, test_labels)


def save_model(model, model_name, mean, standard_deviation,
               train_rows, test_rows, epochs, seed):
    model_dir = PROJECT_DIR / "models"
    model_dir.mkdir(exist_ok=True)
    representation = {"gru": "acoustic", "svm": "harmonic"}.get(model_name, "mel")
    suffix = ".joblib" if model_name == "svm" else ".pt"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"keypad_{model_name}_{representation}_{timestamp}{suffix}"
    model_path = model_dir / filename

    checkpoint = {
        "model_type": model_name,
        "seed": seed,
        "train_ids": [row["trial_id"] for row in train_rows],
        "test_ids": [row["trial_id"] for row in test_rows],
    }
    if model_name == "svm":
        checkpoint.update(classifier=model, feature_set=HARMONIC_FEATURE_SET)
        joblib.dump(checkpoint, model_path)
        return model_path

    checkpoint.update(model_state=model.cpu().state_dict(), mean=mean,
                      standard_deviation=standard_deviation, epochs=epochs)
    if model_name == "gru":
        checkpoint.update(feature="acoustic_summary", features=mean.shape[-1])
    else:
        checkpoint["spectrogram_type"] = "mel"
    torch.save(checkpoint, model_path)
    return model_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inspect", action="store_true", help="Only show detected press times")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs for neural models")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--model", choices=["cnn", "coatnet", "both", "gru", "svm"], default="coatnet")
    args = parser.parse_args()

    rows = read_manifest()

    if args.inspect:
        for row in rows:
            try:
                images, times = process_recording(row)
                rounded_times = [round(time, 2) for time in times]
                print(row["trial_id"], row["pin"], "presses:", rounded_times)
            except ValueError as error:
                print(row["trial_id"], "could not be segmented:", error)
        return

    train_rows, test_rows = split_recordings(rows, args.seed)
    print("Training recordings:", len(train_rows))
    print("Test recordings:", len(test_rows))

    if args.model != "svm" and torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    model_names = ["cnn", "coatnet"] if args.model == "both" else [args.model]
    for model_name in model_names:
        print("\nTraining:", model_name)
        if model_name == "svm":
            model, train_result, test_result = train_svm(train_rows, test_rows)
            mean = standard_deviation = None
        else:
            model, mean, standard_deviation = train_model(
                train_rows, args.epochs, args.seed, model_name)
            train_result = evaluate_model(model, train_rows, mean, standard_deviation)
            test_result = evaluate_model(model, test_rows, mean, standard_deviation)
        model_path = save_model(model, model_name, mean, standard_deviation,
                                train_rows, test_rows, args.epochs, args.seed)

        print("Training digits: {}/{}; complete PINs: {}/{}".format(*train_result))
        print("Test digits: {}/{}; complete PINs: {}/{}".format(*test_result))
        print("Model saved:", model_path)


if __name__ == "__main__":
    main()
