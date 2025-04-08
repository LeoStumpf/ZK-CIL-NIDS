import cupy as cp
import numpy as np
from cuml.ensemble import RandomForestClassifier as cuRFC
from Helper.GenerateOutlier import generate_outliers

def fit(X, y):
    """
    Trains a One-Class Random Forest model for each unique label in the dataset.
    The models are trained on augmented data, which includes generated outliers.

    Parameters:
    - X: Feature matrix (n_samples, n_features).
    - y: Label vector (n_samples,) indicating the class of each sample.

    Returns:
    - one_class_rf_models: A dictionary where keys are unique labels and values are
                           trained RandomForestClassifier models for each label.
    """
    # Dictionary to store the trained One-Class Random Forest models for each label
    one_class_rf_models = {}

    # Iterate over each unique label in the dataset
    for label in np.unique(y):
        # Filter the feature matrix to include only samples with the current label
        X_label = cp.array(X[y == label])

        # Initialize a RandomForestClassifier model with optimal settings
        one_class_rf_models[label] = cuRFC(
            n_estimators=300,  # More trees for better generalization
            max_depth=15,  # Limit tree depth to prevent overfitting
            max_features="sqrt",  # Use sqrt(num_features) for feature selection per split
            max_samples=0.8,  # Use 80% of the training data for each tree
            bootstrap=True,  # Enable bootstrap sampling for aligment
            n_bins=16,  # More bins for better feature split resolution
            n_streams=4,  # Use parallel processing for faster training
            random_state=42  # Set random seed for reproducibility
        )

        # Generate synthetic outliers to augment the training data
        outliers = generate_outliers(X_label)

        # Create labels for inliers (0) and outliers (1)
        labels_inliers = cp.zeros(X_label.shape[0])  # Inliers are labeled as 0
        labels_outliers = cp.ones(outliers.shape[0])  # Outliers are labeled as 1

        # Combine inliers and outliers into a single augmented dataset
        augmented_data = cp.vstack((X_label, outliers))
        labels = cp.hstack((labels_inliers, labels_outliers))

        # Train the RandomForestClassifier on the augmented data
        one_class_rf_models[label].fit(augmented_data, labels)

    return one_class_rf_models


def predict(X, Metadata, model):
    """
    Predicts anomaly scores for the input data using the trained One-Class Random Forest models.

    Parameters:
    - X: Feature matrix (n_samples, n_features) to predict anomaly scores for.
    - Metadata: Additional metadata (not used in this function but can be extended).
    - model: Dictionary of trained RandomForestClassifier models for each label.

    Returns:
    - sample_scores: Anomaly scores for each sample in X. Higher scores indicate
                     higher likelihood of being an outlier.
    """
    # Initialize a matrix to store anomaly scores for each sample and each model
    sample_scores = np.zeros((X.shape[0], len(model)))

    # Iterate over each trained RandomForestClassifier model
    for num, (label, rf) in enumerate(model.items()):
        # Predict the probability of each sample being an outlier (class 1)
        sample_scores[:, num] = rf.predict_proba(X)[:, 1]

    # For each sample, find the minimum outlier probability across all models
    # The minimum probability corresponds to the model that considers the sample most "normal"
    sample_scores = np.min(sample_scores, axis=1)

    return sample_scores