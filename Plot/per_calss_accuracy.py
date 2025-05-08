import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from Helper.LabelEncoder import encode_labels, LABEL_MAPPING

def plot_per_class(y_test, weights):
    # Reverse mapping for decoding predictions
    INV_LABEL_MAPPING = {v: k for k, v in LABEL_MAPPING.items()}

    # Encode ground truth labels
    y_true = np.array([LABEL_MAPPING[label] for label in y_test])

    # Predict class with highest probability
    y_pred = np.argmax(weights, axis=1)

    # Convert to string labels
    y_true_str = np.array([INV_LABEL_MAPPING[i] for i in y_true])
    y_pred_str = np.array([INV_LABEL_MAPPING[i] for i in y_pred])

    # Get list of all classes in the test set
    unique_classes = np.unique(y_true_str)

    # Calculate metrics per class
    results = {}

    for cls in unique_classes:
        y_true_binary = (y_true_str == cls).astype(int)
        y_pred_binary = (y_pred_str == cls).astype(int)

        precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
        recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
        f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
        accuracy = accuracy_score(y_true_binary, y_pred_binary)

        results[cls] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'accuracy': accuracy
        }

    # Print results
    for cls, metrics in results.items():
        print(f"Class: {cls}")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        print()