import numpy as np
from efc import EnergyBasedFlowClassifier

def fit(X, y):
    """
    Trains an Energy-Based Flow Classifier (EFC) model for anomaly detection.
    The model is trained on the specified base class (e.g., 'BENIGN') to learn its distribution.

    Parameters:
    - X: Feature matrix (n_samples, n_features).
    - y: Label vector (n_samples,) indicating the class of each sample.

    Returns:
    - clf: Trained EnergyBasedFlowClassifier model.
    """
    # Initialize the EnergyBasedFlowClassifier with suggested parameters
    clf = EnergyBasedFlowClassifier(
        pseudocounts=0.2,  # Slightly lower pseudocounts for stricter anomaly detection
        cutoff_quantile=0.98,  # High quantile to reduce false positives
        n_bins=30,  # Default number of bins for discretization
        n_jobs=1  # Use a single CPU core (set to -1 to use all cores if needed)
    )

    # Train the EFC model on the specified base class (e.g., 'BENIGN')
    # The base_class parameter defines the class to be modeled as "normal"
    clf.fit(X, y, base_class='BENIGN')

    return clf


def predict(X, Metadata, clf):
    """
    Predicts anomaly scores for the input data using the trained EnergyBasedFlowClassifier model.

    Parameters:
    - X: Feature matrix (n_samples, n_features) to predict anomaly scores for.
    - Metadata: Additional metadata (not used in this function but can be extended).
    - clf: Trained EnergyBasedFlowClassifier model.

    Returns:
    - y_energies: Anomaly scores for each sample in X. Higher scores indicate
                  higher likelihood of being an outlier.
    """
    # Ensure the input data is in the correct format for the EFC model
    X = np.array(X, dtype=np.float64, copy=True)  # Convert to float64
    X = X.copy(order='C')  # Ensure the array is contiguous in memory

    # Debugging: Check if the array is writable and its memory layout
    print("X is writable:", X.flags.writeable)
    print("X flags:", X.flags)

    # Explicitly require the array to be C-contiguous, writable, and aligned
    X = np.require(X, dtype=np.float64, requirements=['C', 'O', 'W'])

    # Predict anomaly scores using the trained EFC model
    # y_pred: Predicted labels (not used here)
    # y_energies: Energy scores, where higher values indicate higher likelihood of being an outlier
    y_pred, y_energies = clf.predict(X, return_energies=True)

    return y_energies