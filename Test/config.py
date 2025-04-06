import pandas as pd
import numpy as np
import os

IMPLEMENTATIONS = [
    #"OneClassForest",
    "IsolationForest",
    "NeuronalNetworkLoss",
    "NeuronalNetwork",
    "OneClassSVN",
    "EnergyFlowClassifier",
    "DistanceLOF",
    "Random"
]

base_path = "/home/leo/ZK-CIL-NIDS/dataset"
def load_data(dataset_name, train_test):
    path = os.path.join(base_path, dataset_name, f"Dataset{train_test}.npz")

    """Loads X, y, and aux_data from a .npz file."""
    data = np.load(path, allow_pickle=True)
    return data['X'], data['y'], pd.DataFrame({'flow_index': data['Flow_index']})

# Convert execution times to a readable format (hours, minutes, seconds)
def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours}h {minutes}m {seconds}s"