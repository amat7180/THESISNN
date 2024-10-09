import os
import matplotlib.pyplot as plt
import h5py
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Subset

class Simple3DCNNClassification(nn.Module):
    def __init__(self, input_shape, num_classes=4):
        super(Simple3DCNNClassification, self).__init__()

        self.conv1 = nn.Conv3d(in_channels=input_shape[0], out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, padding=1)

        self.residual_conv1 = nn.Conv3d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.residual_conv2 = nn.Conv3d(in_channels=128, out_channels=128, kernel_size=3, padding=1)

        self.conv4 = nn.Conv3d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv3d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv6 = nn.Conv3d(in_channels=256, out_channels=512, kernel_size=3, padding=1)

        self.residual_conv3 = nn.Conv3d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.residual_conv4 = nn.Conv3d(in_channels=512, out_channels=512, kernel_size=3, padding=1)

        self.conv7 = nn.Conv3d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv8 = nn.Conv3d(in_channels=512, out_channels=512, kernel_size=3, padding=1)

        self.flatten_size = self._calculate_flatten_size(input_shape)

        self.fc1 = nn.Linear(self.flatten_size, num_classes)

    def _calculate_flatten_size(self, input_shape):
        # Create a dummy tensor with the same shape as input to calculate the output size after convolutions
        dummy_input = torch.randn(1, *input_shape)
        dummy_output = self._forward_conv_layers(dummy_input)
        return int(torch.flatten(dummy_output, 1).size(1))

    def _forward_conv_layers(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        residual = F.relu(self.residual_conv1(x))
        residual = F.relu(self.residual_conv2(residual))
        x = x + residual  # First residual connection

        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        residual = F.relu(self.residual_conv3(x))
        residual = F.relu(self.residual_conv4(residual))
        x = x + residual  # Second residual connection

        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        return x

    def forward(self, x):
        x = self._forward_conv_layers(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        confidence = F.softmax(x, dim=1)
        return confidence

class EarlyStopping:
    def __init__(self, patience=3, min_lr=1e-6):
        self.patience = patience
        self.min_lr = min_lr
        self.best_loss = None
        self.counter = 0

    def should_stop(self, current_loss, current_lr):
        if self.best_loss is None or current_loss < self.best_loss:
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1
            print(f"Validation loss did not improve. Patience counter: {self.counter}/{self.patience}")

        if self.counter >= self.patience:
            print(f"Early stopping triggered due to patience ({self.patience} epochs of no improvement).")
            return True
        if current_lr < self.min_lr:
            print(f"Early stopping triggered due to learning rate dropping below min_lr: {current_lr:.8f} < {self.min_lr:.8f}")
            return True

        return False

class LFTemplateMatchingDataset(Dataset):
    def __init__(self, base_dir, confidence_threshold=None):
        """
        Initializes the dataset by loading .npy files.
        Args:
            base_dir (str): Path to the base directory containing the filtered data.
            confidence_threshold (float, optional): Confidence threshold for filtering pseudo-labeled data.
        """
        if confidence_threshold is not None:
            threshold_str = f"conf{int(confidence_threshold * 100)}"
            data_dir = os.path.join(base_dir, threshold_str)
        else:
            data_dir = base_dir

        # Load the data from .npy files
        self.images = np.load(os.path.join(data_dir, 'images.npy'), mmap_mode='r')
        self.image_index = np.load(os.path.join(data_dir, 'image_index.npy'), mmap_mode='r')
        self.x_coords = np.load(os.path.join(data_dir, 'x_coords.npy'), mmap_mode='r')
        self.y_coords = np.load(os.path.join(data_dir, 'y_coords.npy'), mmap_mode='r')
        self.class_labels = np.load(os.path.join(data_dir, 'class_labels.npy'), mmap_mode='r')
        self.confidence_scores = np.load(os.path.join(data_dir, 'confidence_scores.npy'), mmap_mode='r')
        self.hcpl_values = np.load(os.path.join(data_dir, 'hcpl_values.npy'), mmap_mode='r')

        self.num_observations = self.images.shape[0]
        print(f"Dataset initialized from directory: {data_dir}")
        print(f"Total number of observations: {self.num_observations}")

    def __len__(self):
        return self.num_observations

    def __getitem__(self, idx):
        # Load the sample by index
        images = self.images[idx]  # Shape should be [17, 31, 33, 33]
        class_label = self.class_labels[idx]
        confidence_score = self.confidence_scores[idx]

        # Convert data to torch tensors
        images_tensor = torch.tensor(images, dtype=torch.float32)
        label_tensor = torch.tensor(class_label, dtype=torch.long)
        confidence_tensor = torch.tensor(confidence_score, dtype=torch.float32)

        return images_tensor, label_tensor, confidence_tensor


def warmup(current_step, warmup_steps, training_steps):
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    else:
        return max(0.0, float(training_steps - current_step) / float(max(1, training_steps - warmup_steps)))

def save_training_plot(train_losses, val_losses, train_accuracies=None, val_accuracies=None, filename='training_plot.png'):
    plt.figure(figsize=(14, 8))

    plt.subplot(2, 1, 1)
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss Over Time')

    # Plot Training and Validation Accuracy if provided - i.e. exclude the unlabled set?
    if train_accuracies is not None and val_accuracies is not None:
        plt.subplot(2, 1, 2)
        plt.plot(train_accuracies, label='Training Accuracy', color='green')
        plt.plot(val_accuracies, label='Validation Accuracy', color='red')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Training and Validation Accuracy Over Time')

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def train_and_evaluate(model, train_loader, val_loader, optimizer, criterion, scheduler, device, num_epochs, patience=3):
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    early_stopping = EarlyStopping(patience=patience, min_lr=1e-6)
    current_lr = optimizer.param_groups[0]['lr']

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        for batch_idx, (inputs, labels, _) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)

        avg_loss = running_loss / len(train_loader)
        accuracy = correct_predictions / total_predictions
        train_losses.append(avg_loss)
        train_accuracies.append(accuracy)

        # Validation
        model.eval()
        val_running_loss = 0.0
        val_correct_predictions = 0
        val_total_predictions = 0

        with torch.no_grad():
            for inputs, labels, _ in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_correct_predictions += (predicted == labels).sum().item()
                val_total_predictions += labels.size(0)

        val_avg_loss = val_running_loss / len(val_loader)
        val_accuracy = val_correct_predictions / val_total_predictions
        val_losses.append(val_avg_loss)
        val_accuracies.append(val_accuracy)

        print(f"Epoch [{epoch+1}/{num_epochs}] complete, Training Loss: {avg_loss:.4f}, "
              f"Training Accuracy: {accuracy:.4f}, Validation Loss: {val_avg_loss:.4f}, "
              f"Validation Accuracy: {val_accuracy:.4f}")

        # Check early stopping criteria
        if early_stopping.should_stop(val_avg_loss, current_lr):
            print("Early stopping triggered")
            break

    return train_losses, train_accuracies, val_losses, val_accuracies

if __name__ == '__main__':
    # Paths to base directories containing pre-filtered data
    base_dir_labeled = 'pre-filteredData/labeled'
    base_dir_pseudo = 'pre-filteredData/prefilteredHCPL'
    confidence_thresholds = [0.9, 0.8, 0.7, 0.6]

    # Create dataset for labeled data
    labeled_dataset = LFTemplateMatchingDataset(os.path.join(base_dir_labeled, 'train'))

    # Create DataLoader for labeled dataset
    batch_size = 8
    labeled_loader = DataLoader(labeled_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # Create datasets and DataLoaders for pseudo-labeled data
    pseudo_loaders = []
    for threshold in confidence_thresholds:
        pseudo_dataset = LFTemplateMatchingDataset(base_dir_pseudo, confidence_threshold=threshold)
        pseudo_loader = DataLoader(pseudo_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        pseudo_loaders.append(pseudo_loader)

    # Example to load a batch and display the shapes
    for images, labels, confidences in labeled_loader:
        print("\nLabeled Dataset - Images shape:", images.shape)
        print("Labeled Dataset - Labels shape:", labels.shape)
        print("Labeled Dataset - Confidences shape:", confidences.shape)
        break

    for i, pseudo_loader in enumerate(pseudo_loaders):
        print(f"\nDisplaying data for confidence threshold: {confidence_thresholds[i]}")
        for images, labels, confidences in pseudo_loader:
            print("\nPseudo-Labeled Dataset - Images shape:", images.shape)
            print("Pseudo-Labeled Dataset - Labels shape:", labels.shape)
            print("Pseudo-Labeled Dataset - Confidences shape:", confidences.shape)
            break
        
# # Main functio
# if __name__ == '__main__':
#     hdf5_file_labeled = 'Xl.h5'
#     hdf5_file_pseudo = 'Xhcpl.h5'
#     confidence_thresholds = [0.9, 0.8, 0.7, 0.6]

#     # Split the labeled dataset into training and validation sets with stratification
#     labeled_dataset = LFTemplateMatchingDataset(hdf5_file_labeled)
#     train_dataset, val_dataset = split_labeled_dataset(labeled_dataset, validation_split=0.2)

#     batch_size = 8
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=10, pin_memory=True)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

#     pseudo_loaders = []
#     for threshold in confidence_thresholds:
#         pseudo_dataset = LFTemplateMatchingDataset(hdf5_file_pseudo, confidence_threshold=threshold, filter_pseudo_labels=True)
#         pseudo_loader = DataLoader(pseudo_dataset, batch_size=batch_size, shuffle=True, num_workers=20, pin_memory=True)
#         pseudo_loaders.append(pseudo_loader)

#     # Example of loading a batch from labeled_loader
#     for images, labels, confidences in train_loader:
#         print("\nLabeled Dataset - Images shape:", images.shape)
#         print("Labeled Dataset - Labels shape:", labels.shape)
#         print("Labeled Dataset - Confidences shape:", confidences.shape)
#         break

#     # Model setup (assuming you have defined Simple3DCNNClassification, EarlyStopping, and warmup logic)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     input_shape = (batch_size, 17, 31, 33, 33)  # Replace with actual input shape if needed
#     model = Simple3DCNNClassification(input_shape=input_shape).to(device)

#     optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
#     criterion = torch.nn.CrossEntropyLoss()

#     # Learning rate scheduler with warmup
#     total_steps = len(train_loader) * 100  # assuming 100 epochs
#     warmup_steps = int(0.1 * total_steps)
#     lr_lambda = lambda current_step: warmup(current_step, warmup_steps, total_steps)
#     scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

#     # Train and evaluate using the labeled dataset
#     train_losses, train_accuracies, val_losses, val_accuracies = train_and_evaluate(
#         model,
#         train_loader,
#         val_loader,
#         optimizer,
#         criterion,
#         scheduler,
#         device,
#         num_epochs=100
#     )

#     # Plot the training and validation loss
#     save_training_plot(train_losses, val_losses)

#     # Optionally, train using pseudo-labeled dataset and compare results
#     print("\nTraining with Pseudo-Labeled Dataset:")
#     for pseudo_loader in pseudo_loaders:
#         train_losses_pseudo, train_accuracies_pseudo, val_losses_pseudo, val_accuracies_pseudo = train_and_evaluate(
#             model,
#             pseudo_loader,
#             val_loader,
#             optimizer,
#             criterion,
#             scheduler,
#             device,
#             num_epochs=100
#         )

#         # Plot training and validation losses with pseudo-labeled data
#         save_training_plot(train_losses_pseudo, val_losses_pseudo, filename='training_plot_pseudo.png')