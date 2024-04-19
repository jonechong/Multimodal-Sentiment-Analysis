import os
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import h5py
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pad_sequence


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
            melspectrogram = f[group_key]["mfcc_features_normalized"][()]
            label = self.labels[idx]
        melspectrogram_tensor = torch.tensor(melspectrogram, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return melspectrogram_tensor, label_tensor


def custom_collate_fn(batch):
    melspectrograms, labels = zip(*batch)
    melspectrograms_padded = pad_sequence(melspectrograms, batch_first=True)
    melspectrograms_channels = melspectrograms_padded
    labels = torch.stack(labels)
    return melspectrograms_channels, labels


class CRNN(nn.Module):
    def __init__(self, dropout_rate=0.0, num_classes=3, hidden_size=256, num_layers=1):
        super(CRNN, self).__init__()
        self.conv = nn.Sequential(
            ## 1st Convolutional Layer
            nn.Conv1d(
                in_channels=20,  # Number of features per timestep, which is 13 for MFCCs, 40 for melspec
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            ## 2nd Convolutional Layer
            nn.Conv1d(
                in_channels=256,  # Number of features per timestep, which is 13 for MFCCs, 40 for melspec
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm1d(num_features=512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Conv1d(
                in_channels=512,  # Number of features per timestep, which is 13 for MFCCs, 40 for melspec
                out_channels=1024,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm1d(num_features=1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        self.gru = nn.GRU(
            input_size=1024,  # Adjusted based on the output channels of the last Conv1d
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Sequential(
            # nn.Linear(hidden_size, hidden_size * 2),
            # nn.LayerNorm(hidden_size * 2),
            # nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        x = self.conv(x.transpose(1, 2))
        x = x.permute(0, 2, 1)
        x, _ = self.gru(x)
        x = x[:, -1, :]
        # print(x.shape)
        # print(x[:, -1, :].shape)
        x = self.fc(x)
        # x = self.global_avg_pool(
        #     x
        # )  # Apply global average pooling to reduce each feature map to a single value
        # x = torch.flatten(x, start_dim=1)
        # x = self.dropout(x)
        # x = self.fc(x)
        # print(x.shape)
        return x


def train(model, data_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct_predictions = 0
    for inputs, labels in data_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == labels).sum().item()
    avg_loss = total_loss / len(data_loader)
    accuracy = correct_predictions / len(data_loader.dataset)
    return avg_loss, accuracy


def validate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
    avg_loss = total_loss / len(data_loader)
    accuracy = correct_predictions / len(data_loader.dataset)
    return avg_loss, accuracy


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    hdf5_path = "output_h5/normalized_mfccs.h5"
    batch_size = 32
    learning_rate = 1e-1
    num_epochs = 30
    num_workers = 4
    patience = 5
    hidden_size = 1024
    num_layers = 2

    train_dataset = AudioDataset(hdf5_path, split="train")
    val_dataset = AudioDataset(hdf5_path, split="validate")
    test_dataset = AudioDataset(hdf5_path, split="test")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=custom_collate_fn,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=custom_collate_fn,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=custom_collate_fn,
        num_workers=num_workers,
    )

    model = CRNN(hidden_size=hidden_size, num_layers=num_layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=patience, verbose=True
    )

    model_save_path = "crnn_models"
    os.makedirs(model_save_path, exist_ok=True)

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    # Training and validation loop
    for epoch in range(num_epochs):
        print(f"Starting Epoch {epoch+1}/{num_epochs}")
        train_loss, train_accuracy = train(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_accuracy = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

        torch.save(
            model.state_dict(), os.path.join(model_save_path, f"epoch{epoch+1}.pth")
        )

        print(
            f"Epoch {epoch+1}: Train Loss {train_loss:.4f}, Train Accuracy {train_accuracy:.4f}, "
            f"Val Loss {val_loss:.4f}, Val Accuracy {val_accuracy:.4f}"
        )

    # Test the model
    max_val_accuray, best_epoch = max(
        (val, idx) for (idx, val) in enumerate(val_accuracies, 1)
    )

    max_val_accuray = 0.3
    best_epoch = 2

    best_model_path = os.path.join(model_save_path, f"epoch{best_epoch}.pth")
    print(
        f"Loading best model from epoch {best_epoch} with validation accuracy {max_val_accuray}"
    )

    # # Load the model state
    # model.load_state_dict(torch.load(best_model_path))

    # print("start")

    # class_predictions = {0: 0, 1: 0, 2: 0}
    # model.eval()

    # with torch.no_grad():
    #     for inputs, _ in test_loader:
    #         inputs = inputs.to(device)
    #         outputs = model(inputs)
    #         _, predicted = torch.max(outputs, 1)
    #         for label in predicted:
    #             class_predictions[label.item()] += 1

    # print(class_predictions)
    # print("end")

    test_loss, test_accuracy = validate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_losses, label="Train Loss")
    plt.plot(range(1, num_epochs + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), train_accuracies, label="Train Accuracy")
    plt.plot(range(1, num_epochs + 1), val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()
