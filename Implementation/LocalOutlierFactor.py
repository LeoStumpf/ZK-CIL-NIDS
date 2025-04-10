import numpy as np
from sklearn.neighbors import LocalOutlierFactor


def fit(X, y):
    """
    Trains a Local Outlier Factor (LOF) model for each unique label in the dataset.

    Parameters:
    - X: Feature matrix (n_samples, n_features).
    - y: Label vector (n_samples,) indicating the class of each sample.

    Returns:
    - lof_models: A dictionary where keys are unique labels and values are
                  trained LOF models for each label.
    """
    # Dictionary to store the trained LOF models for each label
    lof_models = {}

    # Iterate over each unique label in the dataset
    for label in np.unique(y):
        # Filter the feature matrix to include only samples with the current label
        X_label = X[y == label]

        # Initialize a LOF model with optimized hyperparameters
        lof = LocalOutlierFactor(
            n_neighbors=20,  # Number of neighbors to consider
            algorithm='auto',  # Automatically choose the best algorithm
            leaf_size=30,  # Leaf size for KDTree or BallTree
            metric='minkowski',  # Distance metric (default is Euclidean)
            p=2,  # Power parameter for Minkowski metric (2 = Euclidean)
            novelty=True,  # Enable novelty detection (required for predict)
            contamination='auto'  # Let the algorithm estimate the contamination
        )

        # Fit the LOF model to the data for the current label
        lof.fit(X_label)

        # Store the trained model in the dictionary with the label as the key
        lof_models[label] = lof

    return lof_models


def predict(X, Metadata, lof_models):
    """
    Predicts anomaly scores for the input data using the trained LOF models.

    Parameters:
    - X: Feature matrix (n_samples, n_features) to predict anomaly scores for.
    - Metadata: Additional metadata (not used in this function but can be extended).
    - lof_models: Dictionary of trained LOF models for each label.

    Returns:
    - sample_scores: Anomaly scores for each sample in X. Higher scores indicate
                     higher likelihood of being an outlier.
    """
    # Initialize a matrix to store anomaly scores for each sample and each model
    sample_scores = np.zeros((X.shape[0], len(lof_models)))

    # Iterate over each trained LOF model
    for num, (label, lof) in enumerate(lof_models.items()):
        # Compute the negative outlier factor (lower is more outlier)
        # We negate it so higher values indicate more anomalous
        sample_scores[:, num] = -lof.score_samples(X)

    # For each sample, find the maximum score across all models
    # The maximum score corresponds to the model that considers the sample most anomalous
    sample_scores = np.max(sample_scores, axis=1)

    return sample_scores