import cupy as cp
import numpy as np
from cuml.ensemble import RandomForestClassifier as cuRFC
from Helper.GenerateOutlier import generate_outliers, _generate_histogram_based


def fit(X, y):
    """
    Trains a One-Class Random Forest model for each unique label using OCRF methodology.
    Now optimized to avoid memory overflow by training trees incrementally.

    Parameters:
    - X: Feature matrix (n_samples, n_features).
    - y: Label vector (n_samples,).
    - L: Number of trees in the forest (default: 300).
    - KRSM: Proportion of features to select per tree (default: 0.5).
    - bins: Number of bins for histogram-based outlier generation (default: 50).
    - N_outlier: Number of outliers to generate per tree (default: 1000).

    Returns:
    - one_class_rf_models: A dictionary where keys are unique labels and values are trained RandomForestClassifier models.
    """

    L = 300
    KRSM = 0.5
    bins = 50
    N_outlier = 1000

    one_class_rf_models = {}
    n_features = X.shape[1]

    for label in np.unique(y):
        X_label = cp.array(X[y == label])

        rf_model = cuRFC(
            n_estimators=L,
            max_depth=15,
            max_features=int(KRSM * n_features),  # Random subspace selection
            max_samples=0.8,
            bootstrap=True,  # Enable bootstrapping
            n_bins=16,
            n_streams=4,
            random_state=42
        )

        # Train trees incrementally to save memory
        for tree_idx in range(L):
            # Bootstrap sample
            bootstrap_idx = cp.random.choice(X_label.shape[0], size=X_label.shape[0], replace=True)
            bootstrap_sample = X_label[bootstrap_idx]

            # Generate outliers only for this tree
            #outliers = _generate_histogram_based(N_outlier, bootstrap_sample, bins=bins)
            outliers = generate_outliers(X_label, target_outliers=N_outlier)

            # Create labels
            labels = cp.hstack((cp.zeros(bootstrap_sample.shape[0]), cp.ones(outliers.shape[0])))

            # Train on current batch and discard outliers to save memory
            rf_model.fit(cp.vstack((bootstrap_sample, outliers)), labels)

        one_class_rf_models[label] = rf_model

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