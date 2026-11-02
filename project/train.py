"""Split the recordings, train the CNN on the GPU, and save its weights."""

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
    from .model import KeypadCNN
else:
    from audio import process_recording, read_manifest
    from model import KeypadCNN


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


def train_model(train_rows, epochs, seed):
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
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = KeypadCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
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


def save_model(model, mean, standard_deviation, train_rows, test_rows, epochs, seed):
    model_dir = PROJECT_DIR / "models"
    model_dir.mkdir(exist_ok=True)
    filename = "keypad_cnn_mel_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".pt"
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

    model, mean, standard_deviation = train_model(train_rows, args.epochs, args.seed)
    model_path = save_model(model, mean, standard_deviation,
                            train_rows, test_rows, args.epochs, args.seed)
    print("GPU:", torch.cuda.get_device_name(0))
    print("Model saved:", model_path)


if __name__ == "__main__":
    main()
