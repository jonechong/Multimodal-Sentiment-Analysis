# RNN
import os
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import h5py
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt


## Model Definition
class AudioClassifier(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AudioClassifier, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size, batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 3)

    def forward(self, x):
        # x shape: (batch, sequence_length, input_size)
        lstm_out, (hn, cn) = self.lstm(x)

        # output = self.fc(lstm_out)
        # Global Average Pooling across the sequence_length dimension
        gap = torch.mean(lstm_out, dim=1)
        output = self.fc(gap)

        return output


## Dataset Definition
class AudioDataset(Dataset):
    label_mapping = {"Positive": 2, "Negative": 0, "Neutral": 1}

    def __init__(self, hdf5_file, split="train"):
        self.hdf5_file = hdf5_file
        self.split = split
        self.keys = []
        self.labels = []

        with h5py.File(hdf5_file, "r") as f:
            for group_key in f.keys():
                if f[group_key].attrs["split"] == split:
                    self.keys.append(group_key)
                    self.labels.append(self.label_mapping[f[group_key].attrs["label"]])

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        with h5py.File(self.hdf5_file, "r") as f:
            group_key = self.keys[idx]
            melspectrogram = f[group_key]["normalized_mels"][()]
            label = self.labels[idx]

        melspectrogram_tensor = torch.tensor(melspectrogram, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return melspectrogram_tensor.T, label_tensor


def audio_collate_fn(batch):
    melspectrograms, labels = zip(*batch)
    melspectrograms_padded = pad_sequence(melspectrograms, batch_first=True)

    labels = torch.stack(labels)
    return melspectrograms_padded, labels


## Training Functions
def train(model, data_loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    correct_predictions = 0

    for inputs, labels in data_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == labels).sum().item()

    avg_loss = total_loss / len(data_loader)
    accuracy = correct_predictions / len(data_loader.dataset)
    return avg_loss, accuracy


def validate(model, data_loader, loss_fn, device):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()

    avg_loss = total_loss / len(data_loader)
    accuracy = correct_predictions / len(data_loader.dataset)
    return avg_loss, accuracy


if __name__ == "__main__":
    ## Initialise Datasets and DataLoaders
    hdf5_path = r"output_h5/mels.h5"
    batch_size = 64
    num_workers = 4
    train_dataset = AudioDataset(hdf5_path, split="train")
    val_dataset = AudioDataset(hdf5_path, split="validate")
    test_dataset = AudioDataset(hdf5_path, split="test")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=audio_collate_fn,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=audio_collate_fn,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=audio_collate_fn,
        num_workers=num_workers,
    )

    for melspectrograms, labels in train_loader:
        print("Batch shape:", melspectrograms.shape)
        print("Labels shape:", labels.shape)
        break  # Only check the first batch

    print("Number of batches in train_loader:", len(train_loader))
    print("Number of batches in val_loader:", len(val_loader))
    print("Number of batches in test_loader:", len(test_loader))
    ## Execution
    # Params
    learning_rate = 1e-4
    hidden_size = 1024
    num_epochs = 30
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Model setup
    input_size = 128
    model = AudioClassifier(input_size=input_size, hidden_size=hidden_size).to(device)

    # Optimizer setup
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Loss function setup
    loss_fn = torch.nn.CrossEntropyLoss()
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    model_save_path = "rnn_models"
    os.makedirs(model_save_path, exist_ok=True)

    for epoch in range(num_epochs):
        print(f"Starting Epoch {epoch+1}/{num_epochs}")
        train_loss, train_accuracy = train(
            model, train_loader, optimizer, loss_fn, device
        )
        val_loss, val_accuracy = validate(model, val_loader, loss_fn, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

        torch.save(
            model.state_dict(), os.path.join(model_save_path, f"epoch{epoch+1}.pth")
        )

        # Logging
        print(
            f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}"
        )
        print(
            f"Epoch {epoch+1}/{num_epochs} - Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}"
        )

    # Calculate Test Accuracy
    # Find the epoch number with the lowest validation loss
    max_val_accuray, best_epoch = max(
        (val, idx) for (idx, val) in enumerate(val_accuracies, 1)
    )

    # Construct the filename for the model with the lowest validation loss
    best_model_path = os.path.join(model_save_path, f"epoch{best_epoch}.pth")
    print(
        f"Loading best model from epoch {best_epoch} with validation accuracy {max_val_accuray}"
    )

    # Load the model state
    model.load_state_dict(torch.load(best_model_path))
    test_loss, test_accuracy = validate(model, test_loader, loss_fn, device)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    epochs_range = range(1, num_epochs + 1)

    plt.figure(figsize=(10, 4))
    plt.plot(epochs_range, train_losses, label="Training Loss")
    plt.plot(epochs_range, val_losses, label="Validation Loss")
    plt.title("Training vs Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.xticks(epochs_range)
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(epochs_range, train_accuracies, label="Training Accuracy")
    plt.plot(epochs_range, val_accuracies, label="Validation Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.xticks(epochs_range)
    plt.legend()
    plt.show()
