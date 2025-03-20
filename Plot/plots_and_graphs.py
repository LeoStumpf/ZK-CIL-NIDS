import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc

import os

# Ensure the path is always relative to the script location
save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../04_pics/")


def plot_weights(y, lof_score):
    y_true = np.array(y, dtype=str)

    uncertainties = np.array(lof_score, dtype=np.float32)
    percentiles = np.linspace(0, 100, 99)
    thresholds_ = np.percentile(uncertainties, percentiles)

    # Remove duplicate values to ensure meaningful bins
    thresholds = np.unique(thresholds_).tolist()

    # thresholds = np.linspace(0, 1, 100)  # Define thresholds between 0 and 1

    # Target classes (list of labels considered as "true")
    target_labels = ["BENIGN"]

    # Convert y_true and y_predict to binary (1 if in target_labels, 0 otherwise)
    y_Actual_Postitive = (~np.isin(y_true, target_labels)).astype(int)

    ##DEBUG
    # Create the boolean mask where y_Actual_Positive == 1
    mask = y_Actual_Postitive == 1

    # Use the mask to filter the uncertainties array
    uncertainties_filtered = uncertainties[mask]

    precision_scores = []
    recall_scores = []
    tpr_scores = []
    fpr_scores = []
    true_samples = []
    pos_rates = []

    for threshold in thresholds:
        y_Predicted_Positive = (uncertainties >= threshold)

        # Compute TP, FP, FN based on conditions
        true_positives = (y_Actual_Postitive == 1) & (y_Predicted_Positive == 1)
        true_negatives = (y_Actual_Postitive == 0) & (y_Predicted_Positive == 0)
        false_negatives = (y_Actual_Postitive == 1) & (y_Predicted_Positive == 0)
        false_positives = (y_Actual_Postitive == 0) & (y_Predicted_Positive == 1)

        tp = np.sum(true_positives)
        fp = np.sum(false_positives)
        fn = np.sum(false_negatives)
        tn = np.sum(true_negatives)

        num_positive = np.sum(y_Predicted_Positive)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # True Positive Rate (Recall)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # False Positive Rate
        pos_rate = tp / num_positive

        precision_scores.append(precision)
        recall_scores.append(recall)
        tpr_scores.append(tpr)
        fpr_scores.append(fpr)
        true_samples.append(num_positive)
        pos_rates.append(pos_rate)

        # print(f"precision = {precision}, recall = {recall}")

    precision_scores = np.array(precision_scores)
    recall_scores = np.array(recall_scores)
    tpr_scores = np.array(tpr_scores)
    fpr_scores = np.array(fpr_scores)

    # Sort recall in ascending order to ensure correct AUC computation
    sorted_indices = np.argsort(recall_scores)
    recall_sorted = recall_scores[sorted_indices]
    precision_sorted = precision_scores[sorted_indices]

    # Select points to annotate: first, last, and two middle ones
    annotation_indices = [0, len(thresholds) // 3, 2 * len(thresholds) // 3, -1]

    # Plot Precision-Recall Curve for BENIGN class
    plt.figure()
    plt.plot(recall_scores, precision_scores, marker='o', linestyle='-', label=f'Class known')
    # Annotate selected points
    for idx in annotation_indices:
        plt.annotate(f"{thresholds[idx]:.2f}, {true_samples[idx]}",
                     (recall_scores[idx], precision_scores[idx]),
                     textcoords="offset points",
                     xytext=(5, 5),
                     ha='right',
                     fontsize=10,
                     color='red')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve for known')
    plt.legend(loc='upper left')
    plt.grid()

    # Save the plot
    plt.savefig(os.path.join(save_path, "known_auc.png"), format='png')
    plt.close()

    # Plot ROC Curve (TPR vs FPR)
    plt.figure()
    plt.plot(fpr_scores, tpr_scores, marker='o', linestyle='-', label=f'ROC Curve')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Random Guessing', color='r')  # Diagonal line for random guessing
    for idx in annotation_indices:
        plt.annotate(f"{thresholds[idx]:.2f}, {true_samples[idx]}",
                     (fpr_scores[idx], tpr_scores[idx]),
                     textcoords="offset points",
                     xytext=(5, 5),
                     ha='right',
                     fontsize=10,
                     color='blue')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid()
    plt.savefig(os.path.join(save_path, "roc_curve.png"), format='png')
    plt.close()

    plt.figure(figsize=(10, 5))

    # Use index positions (0, 1, 2, ..., n) for x-axis instead of actual threshold values
    indices = np.arange(len(thresholds))

    # Adjust the bar width to ensure there is space between bars
    bar_width = 0.8

    # Plot the bars using indices for x-axis and threshold values for height
    plt.bar(indices, pos_rates, width=bar_width, color='skyblue', edgecolor='black', label='TP / num_positives')

    # Customize x-ticks to display threshold values at the correct positions
    plt.xticks(indices, [f"{t:.2f}" for t in thresholds], rotation=45)

    # Labels and title
    plt.xlabel('Threshold')
    plt.ylabel('TP / num_positives')
    plt.title('True Positives Rate vs. Threshold')

    # Add gridlines on y-axis for better readability
    plt.grid(axis='y', linestyle='--')

    # Show legend
    plt.legend()

    # Save the plot
    plt.savefig(os.path.join(save_path, "tp_vs_threshold_bar.png"), format='png')
    plt.close()

    # Calculate and print AUC
    auc_class = auc(recall_sorted, precision_sorted)
    print(f"AUC for Class known: {auc_class}")