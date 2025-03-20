import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import cupy as cp
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from Helper.LabelEncoder import encode_labels, LABEL_MAPPING
from Helper.GenerateOutlier import generate_outliers

# Initialize a MinMaxScaler for normalizing the data
min_max_scaler = MinMaxScaler()

# Hyperparameters
LEARNING_RATE = 0.001
MOMENTUM = 0.9
L2_DECAY = 1e-4
LAMBDA = 1.0  # Weight for Fisher loss
ALPHA = 1.0  # Weight for inter-class distance in Fisher loss
BETA = 1.0  # Weight for MMD loss
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NN_Model(torch.nn.Module):
    """
    A Convolutional Neural Network (CNN) model for classification tasks.
    The model is designed to handle 10x10 input images with 1 channel.
    """
    def __init__(self, N_class=4):
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
        Forward pass through the network.

        Parameters:
        - input_data: Input tensor of shape (batch_size, 1, 10, 10).

        Returns:
        - av: Logits tensor of shape (batch_size, num_class).
        - av: Logits tensor (same as above, for compatibility).
        """
        x = self.features(input_data)
        av = self.logits(x)  # Compute logits
        return av, av  # Return logits directly (no sigmoid applied)


def fisher_loss(pred, centroid):
    """
    Computes the Fisher loss, which consists of within-class scatter (Sw) and between-class scatter (Sb).

    Parameters:
    - pred: Predicted logits tensor of shape (batch_size, num_class).
    - centroid: Centroid tensor of shape (num_class, num_class).

    Returns:
    - Sw: Within-class scatter.
    - Sb: Between-class scatter.
    """
    class_index = torch.argmax(pred, 1).view(-1).type(torch.LongTensor)
    sum_Sw = 0
    for i in range(pred.size(0)):
        sum_Sw += ((pred[i,] - centroid[class_index[i],]) ** 2).sum()
    Sw = sum_Sw / pred.size(0)

    sum_Sb = ((centroid.unsqueeze(1) - centroid.unsqueeze(0)) ** 2).sum()  # Inter-class distance
    Sb = sum_Sb / (centroid.size(0) * (centroid.size(0) - 1))

    return Sw, Sb


def mmd_loss(source_features, target_features):
    """
    Computes the Maximum Mean Discrepancy (MMD) loss between source and target features.

    Parameters:
    - source_features: Feature tensor from the source domain.
    - target_features: Feature tensor from the target domain.

    Returns:
    - mmd_loss: MMD loss value.
    """
    diff = source_features - target_features
    mmd_loss = torch.mean(torch.mm(diff, diff.t()))
    return mmd_loss


def train_sharedcnn(epoch, model, rank_rate, max_threshold, data, label, N_class):
    """
    Trains the CNN model using Fisher loss and MMD loss.

    Parameters:
    - epoch: Current epoch number.
    - model: The CNN model.
    - rank_rate: Rank rate for threshold calculation.
    - max_threshold: Maximum threshold values for each class.
    - data: Training data.
    - label: Training labels.
    - N_class: Number of classes.

    Returns:
    - max_threshold: Updated maximum threshold values.
    """
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=L2_DECAY)
    criterion = nn.CrossEntropyLoss()
    model.train()

    # Convert data to PyTorch tensors
    X_tensor = torch.tensor(data, dtype=torch.float32)
    y_tensor = torch.tensor(label, dtype=torch.long)

    # Create DataLoader
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Initialize centroids
    centroids = torch.zeros(N_class, N_class).to(DEVICE)

    for batch_idx, (data_train, label_train) in enumerate(dataloader):
        data_train, label_train = data_train.to(DEVICE), label_train.to(DEVICE)

        # Add noise to data for MMD loss
        noise = torch.randn_like(data_train) * 0.1
        out_data = data_train + noise

        optimizer.zero_grad()

        # Forward pass
        train_av, train_pred = model(data_train)
        outdata_av, _ = model(out_data)

        # Compute Fisher loss
        Sw, Sb = fisher_loss(train_av, centroids)
        fisherloss = LAMBDA * Sw - ALPHA * Sb

        # Compute MMD loss
        mmd_loss_value = mmd_loss(train_av, outdata_av)

        # Compute cross-entropy loss
        cross_loss = criterion(train_pred, label_train)

        # Total loss
        total_loss = cross_loss + fisherloss + BETA * mmd_loss_value

        # Backward pass and optimization
        total_loss.backward()
        optimizer.step()

        # Update centroids
        with torch.no_grad():
            for cls in range(N_class):
                cls_mask = label_train == cls
                if cls_mask.any():
                    centroids[cls] = train_av[cls_mask].mean(dim=0)

    return max_threshold

def fit(X, y, num_epochs=50, batch_size=32, learning_rate=0.001):
    """
    Trains the CNN model on the input data.

    Parameters:
    - X: Feature matrix (n_samples, n_features).
    - y: Label vector (n_samples,).
    - num_epochs: Number of training epochs (default is 50).
    - batch_size: Batch size for training (default is 32).
    - learning_rate: Learning rate for the optimizer (default is 0.001).

    Returns:
    - model: Trained CNN model.
    """
    # Convert input data to CuPy array
    X_cp = cp.array(X)

    # Augment data with outliers
    outliers = generate_outliers(X_cp)

    # Create labels for inliers and outliers
    labels_inliers = cp.array(encode_labels(y))
    labels_outliers = cp.array(encode_labels(np.full(outliers.shape[0], "Unknown", dtype=object)))

    # Stack the data and labels
    X = cp.asnumpy(cp.vstack((X_cp, outliers)))
    y = cp.asnumpy(cp.hstack((labels_inliers, labels_outliers)))

    # Normalize the data
    min_max_scaler.fit(X)
    X_norm = min_max_scaler.transform(X)


def predict(X, Metadata, clf):
    """
    Predicts anomaly scores for the input data using the trained CNN model.

    Parameters:
    - X: Feature matrix (n_samples, n_features) to predict anomaly scores for.
    - Metadata: Additional metadata (not used in this function but included for compatibility).
    - clf: Trained CNN model.

    Returns:
    - anomaly_scores: Anomaly scores for each sample in X. Higher scores indicate higher likelihood of being an outlier.
    """
    clf.eval()  # Set model to evaluation mode

    with torch.no_grad():  # Disable gradient computation
        # Normalize input data using the same scaler from training
        np_known_train_norm = min_max_scaler.transform(X)

        # Reshape input to match CNN input format (10x10 images)
        X_padded = np.pad(np_known_train_norm, ((0, 0), (0, 16)), mode='constant', constant_values=0)
        X_final = X_padded.reshape(X.shape[0], 1, 10, 10)  # Shape: (samples, channels, 10, 10)

        # Convert to PyTorch tensor
        X_tensor = torch.tensor(X_final, dtype=torch.float32).to(DEVICE)

        # Forward pass through the model
        logits, _ = clf(X_tensor)

        # Get predicted probabilities using softmax
        predicted_probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()

    # Return the probability of the "Unknown" class as anomaly scores
    return predicted_probs[:, LABEL_MAPPING['Unknown']]