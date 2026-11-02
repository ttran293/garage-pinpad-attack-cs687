"""Train a CNN or CoAtNet on mel-spectrograms and compare predictions."""

import argparse
import math
import random
from datetime import datetime
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

if __package__:
    from .audio import process_recording, read_manifest
    from .model import KeypadCNN, KeypadCoAtNet
else:
    from audio import process_recording, read_manifest
    from model import KeypadCNN, KeypadCoAtNet


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


def prepare_training_data(train_rows):
    images = []
    labels = []

    for row in train_rows:
        recording_images, _ = process_recording(row)

        for position in range(4):
            images.append(recording_images[position])
            labels.append(int(row["pin"][position]))

    image_tensor = torch.stack(images)
    label_tensor = torch.tensor(labels, dtype=torch.long)
    return image_tensor, label_tensor


def train_model(train_rows, epochs, seed, model_name):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Use the project .venv.")
    if epochs < 1:
        raise ValueError("Epochs must be at least 1.")

    torch.manual_seed(seed)
    device = torch.device("cuda:0")

    images, labels = prepare_training_data(train_rows)
    missing_digits = set(range(10)) - set(labels.tolist())
    if missing_digits:
        raise ValueError("Training recordings have no examples of: " + str(sorted(missing_digits)))

    # Use only training recordings to calculate normalization values.
    mean = images.mean()
    standard_deviation = images.std().clamp_min(0.000001)
    images = (images - mean) / standard_deviation

    dataset = TensorDataset(images, labels)
    loader = DataLoader(dataset, batch_size=16, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))

    if model_name == "cnn":
        model = KeypadCNN().to(device)
        learning_rate = 0.001
    else:
        model = KeypadCoAtNet().to(device)
        learning_rate = 0.0005

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch_images, batch_labels in loader:
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            predictions = model(batch_images)
            loss = loss_function(predictions, batch_labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if epoch == 0 or (epoch + 1) % 5 == 0 or epoch + 1 == epochs:
            average_loss = total_loss / len(loader)
            print(f"Epoch {epoch + 1}/{epochs}: loss {average_loss:.4f}")

    return model, mean, standard_deviation


def evaluate_model(model, rows, mean, standard_deviation):
    images, labels = prepare_training_data(rows)
    images = (images - mean) / standard_deviation
    loader = DataLoader(TensorDataset(images, labels), batch_size=16)

    model.eval()
    predicted_digits = []
    with torch.no_grad():
        for batch_images, _ in loader:
            predictions = model(batch_images.to("cuda:0"))
            predicted_digits.extend(predictions.argmax(dim=1).cpu().tolist())

    correct_digits = 0
    correct_pins = 0
    true_digits = labels.tolist()
    for position in range(len(true_digits)):
        if predicted_digits[position] == true_digits[position]:
            correct_digits += 1

    for position in range(0, len(true_digits), 4):
        if predicted_digits[position:position + 4] == true_digits[position:position + 4]:
            correct_pins += 1

    return correct_digits, len(true_digits), correct_pins, len(rows)


def save_model(model, model_name, mean, standard_deviation,
               train_rows, test_rows, epochs, seed):
    model_dir = PROJECT_DIR / "models"
    model_dir.mkdir(exist_ok=True)
    filename = "keypad_" + model_name + "_mel_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".pt"
    model_path = model_dir / filename

    train_ids = []
    for row in train_rows:
        train_ids.append(row["trial_id"])

    test_ids = []
    for row in test_rows:
        test_ids.append(row["trial_id"])

    # Store weights on CPU so this file can also be loaded without a GPU.
    model = model.to("cpu")
    checkpoint = {
        "model_state": model.state_dict(),
        "model_type": model_name,
        "spectrogram_type": "mel",
        "mean": mean,
        "standard_deviation": standard_deviation,
        "epochs": epochs,
        "seed": seed,
        "train_ids": train_ids,
        "test_ids": test_ids,
    }
    torch.save(checkpoint, model_path)
    return model_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inspect", action="store_true", help="Only show detected press times")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--model", choices=["cnn", "coatnet", "both"], default="coatnet")
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

    print("GPU:", torch.cuda.get_device_name(0))

    model_names = ["cnn", "coatnet"] if args.model == "both" else [args.model]
    for model_name in model_names:
        print("\nTraining:", model_name)
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
