import numpy
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
import torch.utils.data as Data

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


class SharedCNN(torch.nn.Module):
    def __init__(self,N_class):
        super(SharedCNN,self).__init__()
        self.sharedNet = cnn(N_class)

    def forward(self, indata, outdata):
        av_indata, pred_indata = self.sharedNet(indata)
        av_outdata, pred_outdata = self.sharedNet(outdata)
        return av_indata, pred_indata, av_outdata, pred_outdata

def cnn(N_class):
    # new dataset CICIDS2017
    model = NN_Model(N_class)
    return model

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

def get_source_loader( data_source_np0, label_sources_np0):
    data_source = torch.from_numpy(data_source_np0)
    label_source = torch.from_numpy(label_sources_np0)
    torch_source_dataset = Data.TensorDataset(data_source, label_source)
    source_loader = Data.DataLoader(
        dataset=torch_source_dataset,  # torch TensorDataset format
        batch_size=BATCH_SIZE,  # mini batch size
        shuffle=True,  # 要不要打乱数据 (打乱比较好)
        num_workers=2,  # 多线程来读数据
    )
    return source_loader

BATCH_SIZE = 512
LEARNING_RATE = 0.001
momentum = 0.9
l2_decay = 5e-4
# train with fisher loss and KL loss
def train_sharedcnn(epoch,model,rank_rate,max_threshold,data,label,N_class, lamda, alpha, beta):
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=momentum, weight_decay=l2_decay)
    correct = 0
    loss = torch.nn.CrossEntropyLoss()
    model.train()
    train_loader = get_source_loader(data,label)
    len_train_dataset = len(train_loader.dataset)
    len_train_loader = len(train_loader)
    iter_train = iter(train_loader)
    num_iter = len_train_loader

    last_centroids = torch.load('CICIDS_centroids.pt')
    for i in range(1, num_iter):
        data_train, label_train = next(iter_train)
        noise = torch.FloatTensor(data_train.size(0),data_train.size(1), data_train.size(2), data_train.size(3)).normal_(0, 1)
        out_data = data_train.float() + noise
        data_train, label_train = Variable(data_train).float().to(device), Variable(label_train).type(
            torch.LongTensor).to(device)
        out_data = Variable(out_data).float().to(device)
        with torch.no_grad():
            last_centroids = Variable(last_centroids)
        optimizer.zero_grad()
        train_av, train_pred, outdata_av, outdata_pred = model(data_train,out_data)
        centroid = cal_centroid(label_train.cpu(), train_av.cpu(), N_class, last_centroids)
        last_centroids.data = centroid
        Sw, Sb = fisher_loss(pred=train_av.cpu(), centroid=centroid)
        threshold = cal_threshold(label=label_train.cpu(), pred=train_av.cpu(), centroid=centroid, rank_rate=rank_rate)
        cross_loss = loss(train_pred, label_train)
        sw_lamda= (lamda.to(device)) * (Sw.to(device))
        sb_alpha = ((lamda * alpha).to(device)) * (Sb.to(device))
        fisherloss = sw_lamda - sb_alpha
        KL_loss_nobeta = mmd_rbf_noaccelerate(outdata_av,train_av)
        KL_loss = (beta.to(device))* KL_loss_nobeta
        Loss = cross_loss + fisherloss - KL_loss
        pred = train_pred.max(1)[1]
        correct += pred.eq(label_train.data.view_as(pred)).cpu().sum()
        Loss.backward()
        optimizer.step()
    for m in range(threshold.size(0)):
        if threshold[m] > max_threshold[m]:
            max_threshold[m] = threshold[m]
    torch.save(last_centroids.data, 'CICIDS_centroids.pt')
    Accuracy = 100. * correct.type(torch.FloatTensor) / len_train_dataset
    print(
        'Train Epoch:{}\tLoss: {:.6f}\tcross_loss: {:.6f}\tfisherloss: {:.6f}\tSw: {:.6f}\tSb: {:.6f}\tSw_lamda: {:.6f}\tSb_alpha: {:.6f}\tMMD: {:.6f}\tKLloss: {:.6f}\tAccuracy: {:.4f}'.format(
            epoch, Loss, cross_loss, fisherloss, Sw, Sb, sw_lamda, sb_alpha, KL_loss_nobeta, KL_loss, Accuracy))
    return max_threshold

def cal_dist_to_centroids(pred,centroid):
    dist = []
    for i in range(centroid.size(0)):
        dist_to_centroid = ((pred-centroid[i,])**2).sum(1)
        dist.append(dist_to_centroid)
    dist_to_centroids = torch.stack(dist,dim=1)
    return dist_to_centroids

def cal_min_dis_to_centroid(pred,centroid):
    all_distances = cal_dist_to_centroids(pred=pred, centroid=centroid)
    dist_to_its_centriod = torch.min(all_distances, dim=1)[0]
    min_dist_class_index = torch.min(all_distances,dim=1)[1]
    return dist_to_its_centriod,min_dist_class_index

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
    y = encode_labels(y)

    # Normalize the data
    min_max_scaler.fit(X)
    X_norm = min_max_scaler.transform(X)

    # Reshape to match CNN input size (10x10)
    X_padded = np.pad(X_norm, ((0, 0), (0, 16)), mode='constant', constant_values=0)
    X_final = X_padded.reshape(X.shape[0], 1, 10, 10)  # Shape: (samples, channels, 10, 10)

    # Convert to PyTorch tensors
    #X_tensor = torch.tensor(X_final, dtype=torch.float32)
    #y_tensor = torch.tensor(y, dtype=torch.long)

    # Create DataLoader
    #dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    #train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model, optimizer, and loss function
    #model = NN_Model(len(LABEL_MAPPING)).to(device)
    model = SharedCNN(len(LABEL_MAPPING)).to(device)
    #model = models.SharedCNN(N_class1).to(device)

    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=l2_decay)
    criterion = nn.CrossEntropyLoss()

    # Init centroids

    centroids_zeros = torch.zeros(len(LABEL_MAPPING), len(LABEL_MAPPING))
    torch.save(centroids_zeros, 'CICIDS_centroids20200601.pt')
    torch.save(centroids_zeros, 'CICIDS_centroids.pt')

    # Training loop
    for epoch in range(num_epochs):
        print("epoch " + str(epoch))
        lambda_fischer = torch.tensor(0.05 * (math.exp(-5 * (epoch / num_epochs))))
        alpha = torch.tensor(0.0001 * (math.exp(-5 * (epoch / num_epochs))))
        beta = torch.tensor(0.01)

        max_threshold = torch.zeros(len(LABEL_MAPPING))
        max_threshold = train_sharedcnn(epoch, model, rank_rate=0.99, max_threshold=max_threshold, data=X_final, label=y, N_class=len(LABEL_MAPPING), lamda=lambda_fischer, alpha=alpha, beta=beta)

    return model



def predict(X, Metadata, model):
    """
    Predicts anomaly scores for the input data using the trained CNN model.

    Parameters:
    - X: Feature matrix (n_samples, n_features) to predict anomaly scores for.
    - Metadata: Additional metadata (not used in this function but included for compatibility).
    - clf: Trained CNN model.

    Returns:
    - anomaly_scores: Anomaly scores for each sample in X. Higher scores indicate higher likelihood of being an outlier.
    """
    # Normalize the data
    min_max_scaler.fit(X)
    X_norm = min_max_scaler.transform(X)

    # Reshape to match CNN input size (10x10)
    X_padded = np.pad(X_norm, ((0, 0), (0, 16)), mode='constant', constant_values=0)
    X_final = X_padded.reshape(X.shape[0], 1, 10, 10)  # Shape: (samples, channels, 10, 10)

    loss = torch.nn.CrossEntropyLoss()
    label = np.zeros(X.shape[0])
    test_loader = get_source_loader(X_final, label)
    len_test_dataset = len(test_loader.dataset)

    last_centroids = torch.zeros(len(LABEL_MAPPING), len(LABEL_MAPPING))

    for data_test,label_test in test_loader:
        data_test, label_test = Variable(data_test).float().to(device),Variable(label_test).type(torch.LongTensor).to(device)
        test_av,test_pred, _, _ = model(data_test,data_test)

        centroid = cal_centroid(label_test.cpu(), test_av.cpu(), len(LABEL_MAPPING), last_centroids)
        last_centroids = centroid

        centroids = torch.load('CICIDS_centroids20200601.pt')

        pred = test_pred.max(1)[1]
        dist_to_its_centriod, min_dist_class_index = cal_min_dis_to_centroid(pred=test_av.cpu(), centroid=centroids)

        weights = test_pred[:,LABEL_MAPPING['Unknown']].cpu() + dist_to_its_centriod


    # Return the probability of the "Unknown" class as anomaly scores
    return weights.detach().numpy()