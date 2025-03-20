import numpy as np
from sklearn.svm import OneClassSVM

def fit(X, y):
    """
    Trains a One-Class SVM model for each unique label in the dataset.

    Parameters:
    - X: Feature matrix (n_samples, n_features).
    - y: Label vector (n_samples,) indicating the class of each sample.

    Returns:
    - one_class_svm_models: A dictionary where keys are unique labels and values are
                            trained OneClassSVM models for each label.
    """
    # Dictionary to store the trained One-Class SVM models for each label
    one_class_svm_models = {}

    # Iterate over each unique label in the dataset
    for label in np.unique(y):
        # Filter the feature matrix to include only samples with the current label
        X_label = X[y == label]

        # Initialize a One-Class SVM model with optimized hyperparameters
        svm = OneClassSVM(
            nu=0.0001,  # Assumes 0.001% of the data is anomalous (outliers)
            kernel="rbf",  # Radial Basis Function kernel for capturing complex boundaries
            gamma="scale",  # Automatically scales the kernel based on the data
            tol=1e-3,  # Tolerance for stopping criteria (convergence)
            shrinking=True  # Uses shrinking heuristic to speed up training
        )

        # Fit the One-Class SVM model to the data for the current label
        svm.fit(X_label)

        # Store the trained model in the dictionary with the label as the key
        one_class_svm_models[label] = svm

    return one_class_svm_models


def predict(X, Metadata, one_class_svm_models):
    """
    Predicts anomaly scores for the input data using the trained One-Class SVM models.

    Parameters:
    - X: Feature matrix (n_samples, n_features) to predict anomaly scores for.
    - Metadata: Additional metadata (not used in this function but can be extended).
    - one_class_svm_models: Dictionary of trained OneClassSVM models for each label.

    Returns:
    - sample_scores: Anomaly scores for each sample in X. Higher scores indicate
                     higher likelihood of being an outlier.
    """
    # Initialize a matrix to store anomaly scores for each sample and each model
    sample_scores = np.zeros((X.shape[0], len(one_class_svm_models)))

    # Iterate over each trained One-Class SVM model
    for num, (label, svm) in enumerate(one_class_svm_models.items()):
        # Compute the anomaly scores for the input data using the current model
        # In One-Class SVM, higher scores indicate more "normal" samples
        sample_scores[:, num] = svm.score_samples(X)

    # For each sample, find the minimum score across all models
    # The minimum score corresponds to the model that considers the sample most "normal"
    sample_scores = np.min(sample_scores, axis=1)

    # Invert the scores so that higher values indicate higher likelihood of being an outlier
    sample_scores = -sample_scores

    return sample_scores