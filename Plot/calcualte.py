import os
import numpy as np
from .plots import Plot_PrecisionRecall, Plot_TprFpr, Plot_TruePositivesRate
from sklearn.metrics import auc  # Import the auc function from sklearn
import json

def convert_ndarrays_to_lists(obj):
    """Recursively convert ndarrays in a dictionary to lists."""
    if isinstance(obj, dict):
        return {key: convert_ndarrays_to_lists(value) for key, value in obj.items()}
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (list, tuple)):
        return [convert_ndarrays_to_lists(item) for item in obj]
    else:
        return obj

def combine_and_save_dicts(dict1, dict2, filename):
    # Combine the two dictionaries
    combined_dict = {**dict1, **dict2}

    # Convert any ndarrays to lists
    combined_dict = convert_ndarrays_to_lists(combined_dict)

    # Sort the combined dictionary by key alphabetically
    sorted_dict = dict(sorted(combined_dict.items()))

    # Save the sorted dictionary to a file in JSON format
    with open(filename, 'w') as file:
        json.dump(sorted_dict, file, indent=4)

    print(f"Combined and sorted dictionary saved to {filename}")


def load_dict_from_file(filename):
    # Load the dictionary from the file
    with open(filename, 'r') as file:
        loaded_dict = json.load(file)

    return loaded_dict


def plot_weights(y, lof_score, subfolder, Metadata, plot_infos):
    # Ensure the directory exists
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../04_pics/")
    save_path = os.path.join(save_path, subfolder)
    os.makedirs(save_path, exist_ok=True)

    # Plot weights based on Samples
    sample_path = os.path.join(save_path, "samples")
    os.makedirs(sample_path, exist_ok=True)
    plot_infos["flow/samples"] = "samples"
    generate_plots(y, lof_score, sample_path, plot_infos)

    # Plot weights based on Flows
    Metadata["weights"] = Metadata
    Metadata["labels"] = y

    grouped = Metadata.groupby("flow_index").agg({
        "weights": "mean",  # Compute mean of weights
        "labels": "first"  # Take the first label
    }).reset_index()

    sample_path = os.path.join(save_path, "flows")
    os.makedirs(sample_path, exist_ok=True)
    plot_infos["flow/samples"] = "flows"
    generate_plots(grouped["labels"].to_numpy(), grouped["weights"].tolist(), sample_path, plot_infos)



def generate_plots(y, lof_score, save_path, plot_infos):
    scores = calculation(y, lof_score)

    # Plot Precision-Recall Curve
    Plot_PrecisionRecall(scores, save_path, plot_infos)

    # Plot_TprFpr
    Plot_TprFpr(scores, save_path, plot_infos)

    # Plot_TruePositivesRate
    Plot_TruePositivesRate(scores, save_path, plot_infos)

    # Raw measurements file
    path = os.path.join(save_path, f"Measurements_{plot_infos['flow/samples']}_{plot_infos['algorithm_name']}_{plot_infos['dataset_name']}.json")
    combine_and_save_dicts(scores, plot_infos, path)


def calculation(y, lof_score):
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
    output = dict()

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

    output["precision_scores"] = np.array(precision_scores)
    output["recall_scores"] = np.array(recall_scores)
    output["tpr_scores"] = np.array(tpr_scores)
    output["fpr_scores"] = np.array(fpr_scores)

    output["pos_rates"] = pos_rates

    # Calculate the Area Under the Precision/Recall Curve (AUPRC)
    output["auprc"] = auc(output["recall_scores"], output["precision_scores"])

    # Calculate the Area Under the ROC Curve (AUROC)
    output["auroc"] = auc(output["fpr_scores"], output["tpr_scores"])

    # Sort recall in ascending order to ensure correct AUC computation
    sorted_indices = np.argsort(output["recall_scores"])
    output["recall_sorted"] = output["recall_scores"][sorted_indices]
    output["precision_sorted"] = output["precision_scores"][sorted_indices]

    # Select points to annotate: first, last, and two middle ones
    output["annotated_points"] = list()
    annotation_indices = [0, len(thresholds) // 3, 2 * len(thresholds) // 3, -1]
    for idx in annotation_indices:
        output["annotated_points"].append([idx, f"{thresholds[idx]:.2f}, {true_samples[idx]}"])


    return output