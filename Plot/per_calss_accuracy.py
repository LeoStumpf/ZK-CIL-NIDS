import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from Helper.LabelEncoder import encode_labels, LABEL_MAPPING

def get_top_10_percent_per_chunk(weights_novel_model, num_chunks=30):
    weights_novel_model = np.array(weights_novel_model)
    n = len(weights_novel_model)
    chunk_size = n // num_chunks
    top_indices = []

    for i in range(num_chunks):
        start = i * chunk_size
        # Ensure last chunk includes any remainder
        end = (i + 1) * chunk_size if i < num_chunks - 1 else n
        chunk = weights_novel_model[start:end]

        # Get threshold for top 10% within the chunk
        if len(chunk) == 0:
            continue  # skip empty chunks (can happen if n < num_chunks)
        threshold = np.percentile(chunk, 90)

        # Get indices in the original array
        chunk_indices = np.arange(start, end)
        top_chunk_indices = chunk_indices[chunk >= threshold]

        top_indices.extend(top_chunk_indices)

    return np.array(top_indices)


# weights_novel_model: the higher the abnormal
def plot_per_class(y_test, weights_rf_model, weights_novel_model):
    # Reverse mapping for decoding predictions
    INV_LABEL_MAPPING = {v: k for k, v in LABEL_MAPPING.items()}

    # Encode ground truth labels
    y_true = np.array([LABEL_MAPPING[label] for label in y_test])

    # Predict class with highest probability from RF model
    y_pred_base = np.argmax(weights_rf_model, axis=1)

    # Convert to string labels
    y_true_str = np.array([INV_LABEL_MAPPING[i] for i in y_true])
    y_pred_base_str = np.array([INV_LABEL_MAPPING[i] for i in y_pred_base])

    # --- Metrics BEFORE novelty adjustment ---
    unique_classes = np.unique(y_true_str)
    results_base = {}

    for cls in unique_classes:
        y_true_binary = (y_true_str == cls).astype(int)
        y_pred_binary = (y_pred_base_str == cls).astype(int)

        results_base[cls] = {
            'precision': precision_score(y_true_binary, y_pred_binary, zero_division=0),
            'recall': recall_score(y_true_binary, y_pred_binary, zero_division=0),
            'f1_score': f1_score(y_true_binary, y_pred_binary, zero_division=0),
            'accuracy': accuracy_score(y_true_binary, y_pred_binary)
        }

    # --- Apply novelty logic ---
    y_pred_adjusted = y_pred_base.copy()
    top_10_percent_indices = get_top_10_percent_per_chunk(weights_novel_model)

    # Adjust predictions for top 10% abnormal
    for idx in top_10_percent_indices:
        if y_true[idx] == LABEL_MAPPING["BENIGN"]:
            y_pred_adjusted[idx] = LABEL_MAPPING['Unknown']  # Novelty FP: Outlier but actually benign
        else:
            y_pred_adjusted[idx] = y_true[idx]  # Novelty TP: Outlier and indeed an attack

    y_pred_adjusted_str = np.array([INV_LABEL_MAPPING[i] for i in y_pred_adjusted])

    # --- Metrics AFTER novelty adjustment ---
    results_adjusted = {}

    for cls in unique_classes:
        y_true_binary = (y_true_str == cls).astype(int)
        y_pred_binary = (y_pred_adjusted_str == cls).astype(int)

        results_adjusted[cls] = {
            'precision': precision_score(y_true_binary, y_pred_binary, zero_division=0),
            'recall': recall_score(y_true_binary, y_pred_binary, zero_division=0),
            'f1_score': f1_score(y_true_binary, y_pred_binary, zero_division=0),
            'accuracy': accuracy_score(y_true_binary, y_pred_binary)
        }

    # --- Print comparison ---
    for cls in unique_classes:
        print(f"Class: {cls}")
        print("  BEFORE adjustment:")
        for metric, value in results_base[cls].items():
            print(f"    {metric}: {value:.4f}")
        print("  AFTER adjustment:")
        for metric, value in results_adjusted[cls].items():
            print(f"    {metric}: {value:.4f}")
        print()
