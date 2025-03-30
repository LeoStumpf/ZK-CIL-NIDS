import cuml
import numpy as np
import cupy as cp
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from cuml.neighbors import NearestNeighbors
import gc
import sys
from pympler import asizeof

def print_memory_usage(cluster_dict):
    for key, value in cluster_dict.items():
        size = asizeof.asizeof(value)  # Use pympler's asizeof to capture full object size
        print(f"{key}: {size} bytes")

def fit(X, y):
    """
    Trains a Distance Local Outlier Factor (LOF) model for each unique label in the dataset.
    The models are trained using a combination of RandomForestClassifier and NearestNeighbors.

    Parameters:
    - X: Feature matrix (n_samples, n_features).
    - y: Label vector (n_samples,) indicating the class of each sample.

    Returns:
    - lof_clusters: A dictionary containing LOF models and their associated data for each label.
    - rf_model: Trained RandomForestClassifier model.
    """
    # Dictionary to store LOF models and their associated data for each label
    lof_clusters = {}

    # Encode labels into numeric values
    label_encoder = LabelEncoder()
    numeric_labels = label_encoder.fit_transform(y)

    # Train a RandomForestClassifier to predict labels
    rf_model = RandomForestClassifier(n_estimators=100)
    rf_model.fit(X, numeric_labels)

    # Iterate over each unique label in the dataset
    for label, number in zip(label_encoder.classes_, range(len(label_encoder.classes_))):
        label_count = (y == label).sum()
        print(f"Processing label {label} which appears {label_count} times")

        # Filter the feature matrix to include only samples with the current label
        label_mask = y == label
        X_train = X[label_mask]

        # Set parameters for NearestNeighbors
        #nn_learn = min(label_count - 10, 1000)  # Number of neighbors for training
        nn_learn = 300
        nn_test = 20  # Number of neighbors for testing
        threshold = 0.5  # Threshold for outlier detection
        label = str(label)  # Convert label to string for storage

        # Compute min, max, mean, and std of the training data
        minV = np.min(X_train, axis=0)
        maxV = np.max(X_train, axis=0)
        means = np.mean(X_train, axis=0)
        stds = np.std(X_train, axis=0) + 1e-10  # Add small value to avoid division by zero

        # Scale the training data
        print("Scale the training data")
        X_scaled = (X - means) / stds
        X_train_scaled_np = X_scaled.astype(np.float32)
        del X_train, X_scaled
        gc.collect()

        #X_train_scaled = cp.asarray(X_train_scaled_np)
        #del X_train_scaled_np
        gc.collect()

        print("train model")
        # Fit the NearestNeighbors model to the scaled training data
        X_train_scaled_cp = cp.asarray(X_train_scaled_np)
        nn_model = NearestNeighbors(n_neighbors=nn_learn + 1, algorithm='brute', metric='manhattan')
        nn_model.fit(X_train_scaled_cp)
        distances, indices = nn_model.kneighbors(X_train_scaled_cp)
        del X_train_scaled_cp
        distances = distances.get()
        indices = indices.get()
        cp._default_memory_pool.free_all_blocks()

        del nn_model

        # Initialize arrays to store min and max values for each sample
        min_values = np.zeros((X_train_scaled_np.shape[0], X_train_scaled_np.shape[1]))
        max_values = np.zeros((X_train_scaled_np.shape[0], X_train_scaled_np.shape[1]))

        print("store data")
        # Find breakpoints in distances to identify local neighborhoods
        break_indices = []
        for i in range(X_train_scaled_np.shape[0]):
            start_index = 2
            dist_diff = distances[i, start_index:] - distances[i, start_index - 1:-1]
            percent_increase = dist_diff / (distances[i, start_index - 1:-1] + 1e-10)

            # Find the index where the percentage increase exceeds the threshold
            break_index = (np.argmax(percent_increase > threshold) + start_index
                           if np.any(percent_increase > threshold)
                           else indices.shape[1] - 1)
            break_indices.append(int(break_index))

            # Calculate min and max values across the neighbors
            min_values[i] = np.min(X_train_scaled_np[indices[i, 0:break_index]], axis=0)
            max_values[i] = np.max(X_train_scaled_np[indices[i, 0:break_index]], axis=0)



        if label == "BENIGN":
            #mask = np.array(break_indices)>30
            # match the len of mask its length
            mask = np.zeros(len(break_indices), dtype=bool)

            # Set every fith entry to True
            mask[::5] = True

            X_train_scaled_np = X_train_scaled_np[mask]
            min_values = min_values[mask]
            max_values = max_values[mask]

            print(f"Reduced length = {len(X_train_scaled_np)}")
            ...

        # Free memory
        mean_distances = np.mean(distances, axis=1)
        del indices, distances
        cp._default_memory_pool.free_all_blocks()
        gc.collect()

        # Calculate local reachability density
        mean_distances = np.mean(min_values, axis=1)
        local_reachability_density = mean_distances + 1e-10

        # Fit a NearestNeighbors model for testing
        nn = NearestNeighbors(n_neighbors=nn_test, algorithm='brute', metric='manhattan', handle=None)
        nn.fit(X_train_scaled_np)

        # Store LOF model and associated data
        lof_clusters[number] = {
            'nn_learn': nn_learn,
            'nn_test': nn_test,
            'threshold': threshold,
            'min_values': min_values,
            'max_values': max_values,
            'nn': nn,
            'local_reachability_density': local_reachability_density,
            'min': minV,
            'max': maxV,
            'mean': means,
            'std': stds,
            'label': label
        }

        print("Cycle:", number)
        print_memory_usage(lof_clusters[number])

        # Optional: Print total size of the cluster
        total_size = asizeof.asizeof(lof_clusters[number])
        print(f"Total size of current cluster: {total_size} bytes")

        # Free memory after storing the model
        del min_values, max_values, X_train_scaled_np
        cp._default_memory_pool.free_all_blocks()
        gc.collect()
        print("done. next run")

    return lof_clusters, rf_model


def RowInformationGain(row, threshold_known, cluster_dict):
    """
    Calculates the information gain for a row based on its cluster assignments and certainties.

    Parameters:
    - row: A row from the metadata containing cluster assignments and certainties.
    - threshold_known: Threshold to filter clusters based on certainty.
    - cluster_dict: Dictionary containing cluster information.

    Returns:
    - rowInformationGain: The calculated information gain for the row.
    """
    # Filter clusters and certainties based on the threshold
    filtered_cluster_list = []
    filtered_certainty_list = []
    for cluster, cert in zip(row['clusters'], row['certainty']):
        if cert > threshold_known:
            filtered_cluster_list.append(cluster)
            filtered_certainty_list.append(cert)

    # Count occurrences of each cluster
    cluster_counts = Counter(filtered_cluster_list)

    rowInformationGain = 0
    for cluster in set(filtered_cluster_list):
        sum_certainties = sum(cert for cl, cert in zip(filtered_cluster_list, filtered_certainty_list) if cl == cluster)
        num_samples_new = cluster_counts[cluster]
        num_known_class_A = cluster_dict[cluster]['num_known']
        num_samples = cluster_dict[cluster]['num_samples']

        # Calculate risk and worst-case information gain
        risk = 1 - (num_known_class_A / num_samples)
        p1 = (num_known_class_A + num_samples_new) / (num_samples + num_samples_new)
        p2 = num_samples_new / (num_samples + num_samples_new)
        worst_case_ig = -(p1 * np.log2(p1) + p2 * np.log2(p2))

        rowInformationGain += (risk * worst_case_ig) * sum_certainties

    return rowInformationGain


def cluster_data(data):
    """
    Performs clustering on the input data using HDBSCAN.

    Parameters:
    - data: Input feature matrix (n_samples, n_features).

    Returns:
    - cluster_labels: Cluster assignments for each sample.
    """
    # Normalize the data
    data_min = cp.min(data, axis=0)
    data_max = cp.max(data, axis=0)
    data_range = data_max - data_min
    normalized_data = (data - data_min) / data_range
    normalized_data = cp.nan_to_num(normalized_data, nan=0)

    # Perform clustering using HDBSCAN
    min_samples = 50  # Minimum number of samples per cluster
    print(f"Running HDBSCAN with min_samples={min_samples}")
    clustering_model = cuml.HDBSCAN(min_samples=min_samples, output_type='cupy', metric='euclidean')
    cluster_labels = clustering_model.fit_predict(normalized_data)

    return cluster_labels


def predict(X, Metadata, model):
    """
    Predicts anomaly scores for the input data using the trained LOF models and clustering.

    Parameters:
    - X: Feature matrix (n_samples, n_features) to predict anomaly scores for.
    - Metadata: Additional metadata containing flow information.
    - model: Tuple containing the LOF models and the RandomForestClassifier.

    Returns:
    - classInfoGain: List of information gain scores for each sample.
    """
    lof_list, rf_model = model
    batch_size = 100000  # Process data in batches to manage memory
    threshold_known = 1e-10  # Threshold for known samples

    # Convert input data to CuPy array
    X_cp = cp.array(X)

    # Generate LOF scores for each batch
    for start in range(0, X.shape[0], batch_size):
        end = min(start + batch_size, X.shape[0])
        data_batch = X[start:end]

        # Predict labels using the RandomForestClassifier
        predicted_labels = rf_model.predict(X).astype(int)
        lof_scores = np.zeros((len(lof_list), len(X)))
        novelty_flags = np.zeros((len(lof_list), len(X)))
        labels = [None] * len(lof_list)

        # Calculate LOF scores for each label
        for num in lof_list.keys():
            X_scaled = (X - lof_list[num]["mean"]) / lof_list[num]["std"]
            X_test_scaled = X_scaled.astype(cp.float32)

            distance_test, indices_test = lof_list[num]["nn"].kneighbors(X_test_scaled)
            del distance_test  # Free memory

            min_distance_scores = cp.full(indices_test.shape[0], cp.inf)
            for i in range(indices_test.shape[1]):
                min_distance_scores = cp.minimum(
                    min_distance_scores,
                    cp.max(cp.maximum(
                        -(cp.asarray(X_test_scaled) - cp.asarray(lof_list[num]["min_values"])[indices_test[:, i]]),
                        -(cp.asarray(lof_list[num]["max_values"])[indices_test[:, i]] - cp.asarray(X_test_scaled))
                    ), axis=1)
                )

            lof_scores[num] = cp.asnumpy(min_distance_scores)

        # Assign LOF scores based on predicted labels
        lof_score = lof_scores[predicted_labels, np.arange(lof_scores.shape[1])]

    # Perform clustering on the input data
    cluster_index = cluster_data(X_cp).tolist()
    del(X_cp)

    print("done with clustering")

    # Create a dictionary to store cluster information
    cluster_dict = {}
    for cluster in set(cluster_index):
        inicis = np.where(np.array(cluster_index) == cluster)
        count_below_threshold = (lof_score < threshold_known).sum()
        num_samples = cluster_index.count(cluster)

        cluster_dict[cluster] = {
            "num_samples": num_samples,
            "num_known": count_below_threshold,
        }
    cluster_index = np.array(cluster_index)

    # Group metadata by flow index and calculate information gain
    flow_groups = Metadata.groupby('flow_index').apply(lambda x: x.index.tolist()).reset_index()
    flow_groups.columns = ['flow_index', 'indexes']
    flow_groups['clusters'] = flow_groups['indexes'].apply(lambda idx_list: cluster_index[idx_list].tolist())
    flow_groups['certainty'] = flow_groups['indexes'].apply(lambda idx_list: lof_score[idx_list].tolist())
    flow_groups['classInfoGain'] = flow_groups.apply(lambda row: RowInformationGain(row, threshold_known, cluster_dict), axis=1)

    # Merge information gain scores with metadata
    Metadata = Metadata.merge(flow_groups[['flow_index', 'classInfoGain']], on='flow_index', how='left')
    return Metadata['classInfoGain'].tolist()