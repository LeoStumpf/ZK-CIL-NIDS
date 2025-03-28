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
import math

# Initialize a MinMaxScaler for normalizing the data
min_max_scaler = MinMaxScaler()

# Hyperparameters
num_epochs=50
batch_size=32
learning_rate=0.001
momentum=0.9
l2_decay=0.0001



rank_rate=0.1
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def cal_centroid(label,pred,N_class,last_centroids):
    label_pred = torch.argmax(pred,1).view(-1).type(torch.LongTensor)
    correct = torch.eq(label_pred,label)
    correct_index = torch.nonzero(correct).contiguous().view(-1)
    pred_correct = torch.index_select(pred, 0, correct_index)
    label_correct = torch.index_select(label,0,correct_index)
    # print('label_correct',label_correct)
    # print('pred_correct',pred_correct.size())
    label_correct = label_correct.view(-1,1)
    if label_correct.size(0)>0:
        class_count = torch.zeros(N_class, 1)
        for col in range(N_class):
            # k = torch.ones(label_correct.size())*col
            # print(k)
            kk = torch.tensor([col]).expand_as(label_correct)
            # print(col)
            # print(kk)
            # kk = (torch.tensor([col-1]).unsqueeze(0).expand(label_correct.size(),1))
            class_count[col,] = torch.eq(label_correct, kk.type(torch.LongTensor)).sum(0)
            # 对于每个batch来说，label_correct不一定包含所有类别
            # class_count = torch.zeros(N_class,1).scatter_add(0,label_correct,torch.ones(label_correct.size()))
        positive_class_count = torch.max(class_count, torch.ones(class_count.size()))
        scatter_index = label_correct.expand(label_correct.size(0), N_class)
        centroid = torch.zeros(N_class, N_class).scatter_add(0,scatter_index.type(torch.LongTensor),pred_correct)
        # print('centroid sum',centroid)
        # print('pasitivie_class_count',positive_class_count)
        mean_centroid = centroid / positive_class_count
        # print('mean_centroid',mean_centroid)
        current_centroids = mean_centroid
        for i in range(0, mean_centroid.size(0)):
            if positive_class_count[i] == 1:
                current_centroids[i,] = last_centroids[i,]
                # print('using one class centroids')
                # if torch.equal(mean_centroid[i,],torch.zeros(N_class,1)):
                #     current_centroids[i,] = last_centroids[i,]
                #     print('AAA')
            else:
                current_centroids[i,] = 0.5*last_centroids[i,]+0.5*current_centroids[i,]
    else:
        current_centroids = last_centroids
        # print('all use last_centroids')

    return current_centroids

def cal_threshold(label,pred,centroid,rank_rate):
    label_pred = torch.argmax(pred, 1).view(-1).type(torch.LongTensor)
    correct = torch.eq(label_pred, label)
    correct_index = torch.nonzero(correct).contiguous().view(-1)
    pred_correct = torch.index_select(pred, 0, correct_index)
    dist = []
    threshold = torch.zeros(centroid.size(0))
    if pred_correct.size(0)>0:
        centroid_index = torch.argmax(pred_correct, 1).type(torch.LongTensor)
        for i in range(pred_correct.size(0)):
            dist_to_its_centriod = ((pred_correct[i,] - centroid[centroid_index[i]]) ** 2).sum()
            dist.append(dist_to_its_centriod)
        dist_to_its_centriod = torch.stack(dist)
        for j in range(centroid.size(0)):
            class_j_index = (centroid_index==j).nonzero().view(-1)
            if class_j_index.size(0)>0:
                dist_to_j_centroid = torch.gather(dist_to_its_centriod, 0, class_j_index)
                ascend_dist = torch.sort(dist_to_j_centroid)[0]
                threshold_index = torch.floor(dist_to_j_centroid.size(0) * torch.tensor(rank_rate)).type(
                    torch.LongTensor)
                threshold[j] = ascend_dist[threshold_index]
    # else:
    #     threshold = 0
    return threshold

def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    # source = source.resize_((source.size()[0], source.size()[1], 1, source.size()[2]))
    # target = target.resize_((target.size()[0], target.size()[1], 1, target.size()[2]))
    n_samples = int(source.size()[0])+int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0-total1)**2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)#/len(kernel_val)

def mmd_rbf_noaccelerate(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY -YX)
    return loss

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

    # Reshape to match CNN input size (10x10)
    X_padded = np.pad(X_norm, ((0, 0), (0, 16)), mode='constant', constant_values=0)
    X_final = X_padded.reshape(X.shape[0], 1, 10, 10)  # Shape: (samples, channels, 10, 10)

    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X_final, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    # Create DataLoader
    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model, optimizer, and loss function
    model = NN_Model(len(LABEL_MAPPING)).to(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=l2_decay)
    criterion = nn.CrossEntropyLoss()

    # Init centroids
    last_centroids = torch.zeros(len(LABEL_MAPPING), X_tensor.size(1)).to(device)

    # Training loop
    for epoch in range(num_epochs):
        print("epoch " + str(epoch))
        lambda_fischer = torch.tensor(0.05 * (math.exp(-5 * (epoch / num_epochs))))
        alpha = torch.tensor(0.0001 * (math.exp(-5 * (epoch / num_epochs))))
        beta = torch.tensor(0.01)

        model.train()
        correct = 0
        total_loss = 0
        cross_loss_total = 0
        fisherloss_total = 0
        KL_loss_total = 0

        for data_train, label_train in train_loader:
            data_train, label_train = data_train.to(device), label_train.to(device)

            # Add noise to data
            noise = torch.randn_like(data_train) * 0.1
            out_data = data_train + noise

            # Forward pass
            optimizer.zero_grad()
            train_av, train_pred = model(data_train)
            outdata_av, outdata_pred = model(data_train)

            # Calculate centroids
            centroid = cal_centroid(label_train.cpu(), train_av.cpu(), len(LABEL_MAPPING), last_centroids.cpu())
            last_centroids.data = centroid.cpu()

            # Calculate Fisher loss
            Sw, Sb = fisher_loss(pred=train_av.cpu(), centroid=centroid)
            threshold = cal_threshold(label=label_train.cpu(), pred=train_av.cpu(), centroid=centroid, rank_rate=rank_rate)
            cross_loss = criterion(train_pred, label_train)
            sw_lamda = (lambda_fischer.to(device)) * (Sw.to(device))
            sb_alpha = ((lambda_fischer * alpha).to(device)) * (Sb.to(device))
            fisherloss = sw_lamda - sb_alpha

            # Calculate MMD loss
            KL_loss_nobeta = mmd_rbf_noaccelerate(outdata_av, train_av)
            KL_loss = (beta.to(device)) * KL_loss_nobeta

            # Total loss
            Loss = cross_loss + fisherloss - KL_loss

            # Backward pass and optimization
            Loss.backward()
            optimizer.step()

            # Calculate accuracy
            pred = train_pred.max(1)[1]
            correct += pred.eq(label_train.data.view_as(pred)).cpu().sum()

            # Accumulate losses
            total_loss += Loss.item()
            cross_loss_total += cross_loss.item()
            fisherloss_total += fisherloss.item()
            KL_loss_total += KL_loss.item()

    return model




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
        X_tensor = torch.tensor(X_final, dtype=torch.float32)

        # Move to appropriate device (CPU or GPU)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X_tensor = X_tensor.to(device)

        # Forward pass through the model
        logits, outputs = clf(X_tensor)

        # Get predicted probabilities using softmax
        predicted_probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()

    # Return the probability of the "Unknown" class as anomaly scores
    return predicted_probs[:, LABEL_MAPPING['Unknown']]