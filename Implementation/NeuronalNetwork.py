from sklearn.preprocessing import MinMaxScaler
import numpy as np
import cupy as cp
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from Helper.LabelEncoder import encode_labels, LABEL_MAPPING
from Helper.GenerateOutlier import generate_outliers

# Initialize a MinMaxScaler for normalizing the data
min_max_scaler = MinMaxScaler()


class NN_Model(torch.nn.Module):
    """
    A Convolutional Neural Network (CNN) model for classification tasks.
    The model is designed to handle 10x10 input images with 1 channel.
    """
    def __init__(self, N_class=len(LABEL_MAPPING)):
        """
        Initializes the CNN model.

        Parameters:
        - N_class: Number of output classes (default is 4).
        """
        super(NN_Model, self).__init__()
        self.avg_kernel_size = 2  # Kernel size for average pooling
        self.i_size = 10  # Input size (10x10)
        self.num_class = N_class  # Number of output classes
        self.input_space = None
        self.input_size = (self.i_size, self.i_size, 1)  # Input shape: (10, 10, 1)

        # First convolutional layer
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=True),  # Output: 10x10x32
            torch.nn.ReLU(inplace=True),  # ReLU activation
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # Output: 5x5x32
        )

        # Second convolutional layer
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 128, kernel_size=3, stride=1, padding=1, bias=True),  # Output: 5x5x128
            torch.nn.ReLU(inplace=True),  # ReLU activation
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # Output: 2x2x128
        )

        # Fully connected layer
        self.fc = torch.nn.Sequential(
            torch.nn.BatchNorm1d(2 * 2 * 128),  # Batch normalization
            torch.nn.Linear(2 * 2 * 128, self.num_class, bias=True)  # Output: num_class
        )

        # Sigmoid activation (not used in forward pass)
        self.sigmoid = torch.nn.Sigmoid()

    def features(self, input_data):
        """
        Extracts features from the input data using convolutional layers.

        Parameters:
        - input_data: Input tensor of shape (batch_size, 1, 10, 10).

        Returns:
        - x: Feature tensor of shape (batch_size, 128, 2, 2).
        """
        x = self.conv1(input_data)
        x = self.conv2(x)
        return x

    def logits(self, input_data):
        """
        Computes logits (raw predictions) from the feature tensor.

        Parameters:
        - input_data: Feature tensor of shape (batch_size, 128, 2, 2).

        Returns:
        - x: Logits tensor of shape (batch_size, num_class).
        """
        x = input_data.view(input_data.size(0), -1)  # Flatten the tensor
        x = self.fc(x)
        return x


    def forward(self, input_data):
        """
        Forward pass with one-hot encoding option
        Returns:
        - raw logits (for training with CrossEntropyLoss)
        - one-hot encoded predictions (for interpretation)
        """
        features = self.features(input_data)
        logits = self.logits(features)

        # Generate one-hot encoded predictions
        preds = torch.argmax(logits, dim=1)
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, preds.unsqueeze(1), 1)

        return logits, one_hot  # Return both raw logits and one-hot


def fit(X, y, num_epochs=50, batch_size=32, learning_rate=0.001):
    """
    Trains the CNN model on the input data with support for one-hot encoded outputs.

    Parameters:
    - X: Feature matrix (n_samples, n_features)
    - y: Label vector (n_samples,) containing class indices (0-11)
    - num_epochs: Number of training epochs (default: 50)
    - batch_size: Batch size (default: 32)
    - learning_rate: Learning rate (default: 0.001)

    Returns:
    - model: Trained CNN model that outputs (logits, one_hot) in forward pass
    """
    # Convert input data to CuPy array
    X_cp = cp.array(X)

    # Augment data with outliers
    outliers = generate_outliers(X_cp)

    # Create labels - ensure these are integer class indices (0-11)
    labels_inliers = cp.array(encode_labels(y))
    labels_outliers = cp.array([LABEL_MAPPING['Unknown']] * outliers.shape[0])

    # Stack the data and labels
    X = cp.asnumpy(cp.vstack((X_cp, outliers)))
    y = cp.asnumpy(cp.hstack((labels_inliers, labels_outliers)))

    # Normalize and reshape data
    min_max_scaler.fit(X)
    X_norm = min_max_scaler.transform(X)
    X_padded = np.pad(X_norm, ((0, 0), (0, 16)), mode='constant', constant_values=0)
    X_final = X_padded.reshape(X.shape[0], 1, 10, 10)

    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X_final, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    # Create DataLoader
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model with correct number of classes
    model = NN_Model(N_class=len(LABEL_MAPPING)).to(device)
    criterion = nn.CrossEntropyLoss()  # Works with class indices, not one-hot
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            logits, _ = model(inputs)  # We only need logits for training
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")

    print("Training completed.")
    return model


def predict(X, Metadata, clf, batch_size=512):
    """
    Predicts anomaly scores for the input data using the trained CNN model in batches.

    Parameters:
    - X: Feature matrix (n_samples, n_features) to predict anomaly scores for.
    - Metadata: Additional metadata (not used in this function but included for compatibility).
    - clf: Trained CNN model.
    - batch_size: Number of samples to process at once (default: 256).

    Returns:
    - anomaly_scores: Anomaly scores for each sample in X in the original order. Higher scores indicate higher likelihood of being an outlier.
    """
    clf.eval()  # Set model to evaluation mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Normalize all input data at once (scalers are usually lightweight)
    np_known_train_norm = min_max_scaler.transform(X)

    # Pad all data at once (this is just adding zeros, not memory intensive)
    X_padded = np.pad(np_known_train_norm, ((0, 0), (0, 16)), mode='constant', constant_values=0)

    # Initialize array to store results in order
    all_scores = np.zeros(X.shape[0])

    # Process in batches
    with torch.no_grad():  # Disable gradient computation
        for i in range(0, X.shape[0], batch_size):
            # Get the current batch
            batch = X_padded[i:i + batch_size]

            # Reshape batch to match CNN input format (10x10 images)
            X_batch = batch.reshape(-1, 1, 10, 10)  # Shape: (batch_size, channels, 10, 10)

            # Convert to PyTorch tensor and move to device
            X_tensor = torch.tensor(X_batch, dtype=torch.float32).to(device)

            # Forward pass through the model
            logits, outputs = clf(X_tensor)

            # Get predicted probabilities using softmax
            predicted_probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()

            # Store the results in the correct position
            batch_scores = predicted_probs[:, LABEL_MAPPING['Unknown']]
            all_scores[i:i + batch_size] = batch_scores

    return all_scores