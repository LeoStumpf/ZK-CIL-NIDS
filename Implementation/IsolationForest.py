from sklearn.ensemble import IsolationForest
from joblib import parallel_backend
import numpy as np

def fit(X, y):
    """
    Trains an Isolation Forest model for each unique label in the dataset.
    Each model is trained only on the data corresponding to its label.

    Parameters:
    - X: Feature matrix (n_samples, n_features) containing the training data.
    - y: Label vector (n_samples,) indicating the class of each sample.

    Returns:
    - isolation_forest_models: A dictionary where keys are unique labels and values are
                               trained IsolationForest models for each label.
    """
    # Dictionary to store the trained Isolation Forest models for each label
    isolation_forest_models = {}

    # Iterate over each unique label in the dataset
    for label in np.unique(y):
        # Filter the feature matrix to include only samples with the current label
        X_label = X[y == label]

        # Initialize the IsolationForest model with optimized hyperparameters
        clf = IsolationForest(
            n_estimators=200,  # More trees for better generalization and separation
            contamination=0.01,  # Assumes 1% of the data is anomalous (outliers)
            max_samples=0.8,  # Use 80% of the training data for each tree
            max_features=0.8,  # Use 80% of the features for each split
            n_jobs=-1,  # Use all available CPU cores for parallel processing
            random_state=42  # Set random seed for reproducibility
        )

        # Train the IsolationForest model on the data for the current label
        clf.fit(X_label)

        # Store the trained model in the dictionary with the label as the key
        isolation_forest_models[label] = clf

    return isolation_forest_models

def predict(X, Metadata, model):
    """
    Predicts anomaly scores for the input data using the trained Isolation Forest models.
    Each sample is scored against the model corresponding to its predicted label.

    Parameters:
    - X: Feature matrix (n_samples, n_features) to predict anomaly scores for.
    - Metadata: Additional metadata (not used in this function but included for compatibility).
    - model: Dictionary of trained IsolationForest models for each label.

    Returns:
    - weights: Anomaly scores for each sample in X. Higher scores indicate higher likelihood of being an outlier.
    """
    # Initialize an array to store anomaly scores for each sample
    weights = np.zeros((X.shape[0], len(model)))

    # Iterate over each trained IsolationForestClassifier model
    for num, (label, rf) in enumerate(model.items()):
        # Use threading for parallel processing since the score_samples method is not CPU-bound
        with parallel_backend("threading", n_jobs=4):
            # returns the anomaly score of the input samples. The lower, the more abnormal.
            # thus i invert as we calculate with the higher the abnormal
            weights[:, num] = -rf.score_samples(X)  # Invert scores

    # For each sample, find the minimum outlier probability across all models
    # The minimum probability corresponds to the model that considers the sample most "normal"
    weights = np.min(weights, axis=1)

    return weights
