import os
import numpy as np
from .plots import Plot_AUPRIN, Plot_TprFpr, Plot_AUPROUT, Plot_TruePositivesRate, Plot_DrawingProbability
from sklearn.metrics import auc, precision_score  # Import the auc function from sklearn
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


def plot_weights(y, lof_score, subfolder, Metadata, plot_infos, list_known):
    # Ensure the directory exists
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../04_pics/")
    save_path = os.path.join(save_path, subfolder)
    os.makedirs(save_path, exist_ok=True)

    # Plot weights based on Samples
    sample_path = os.path.join(save_path, "samples")
    os.makedirs(sample_path, exist_ok=True)
    plot_infos["flow/samples"] = "samples"
    generate_plots(y, lof_score, sample_path, plot_infos, list_known)

    # Plot weights based on Flows
    Metadata = Metadata.copy()
    Metadata["weights"] = lof_score
    Metadata["labels"] = y

    grouped = Metadata.groupby("flow_index").agg({
        "weights": "mean",  # Compute mean of weights
        "labels": "first"  # Take the first label
    }).reset_index()

    sample_path = os.path.join(save_path, "flows")
    os.makedirs(sample_path, exist_ok=True)
    plot_infos["flow/samples"] = "flows"
    generate_plots(grouped["labels"].to_numpy(), grouped["weights"].tolist(), sample_path, plot_infos, list_known)



def generate_plots(y, lof_score, save_path, plot_infos, list_known):
    scores = calculation(y, lof_score, list_known)

    # Plot Precision-Recall Curve
    Plot_AUPRIN(scores, save_path, plot_infos)
    Plot_AUPROUT(scores, save_path, plot_infos)

    # Plot_TprFpr
    Plot_TprFpr(scores, save_path, plot_infos)

    # Plot_TruePositivesRate
    Plot_TruePositivesRate(scores, save_path, plot_infos)

    # Plot_DrawingProbability
    Plot_DrawingProbability(scores, save_path, plot_infos)

    # Raw measurements file
    path = os.path.join(save_path, f"Measurements_{plot_infos['flow/samples']}_{plot_infos['algorithm_name']}_{plot_infos['dataset_name']}.json")
    combine_and_save_dicts(scores, plot_infos, path)


def calculation(y, lof_score, list_known):
    # Store values as NP
    y = np.array(y, dtype=str)
    uncertainties = np.array(lof_score, dtype=np.float32)

    # Split uncertainties into 50 equal-length sections (not sorted)
    n_sections = 50
    section_size = len(uncertainties) // n_sections
    sections = [
        uncertainties[i * section_size:(i + 1) * section_size] if i < n_sections - 1 else uncertainties[i * section_size:]
        for i in range(n_sections)
    ]

    thresholds_sections = list()
    for section in sections:
        # Create Bins with equal amount of things
        percentiles = np.linspace(0, 100, 99)
        thresholds = np.percentile(section, percentiles).tolist()
        #thresholds = np.unique(thresholds).tolist()
        thresholds_sections.append(thresholds)

    # all the known classes are considered as the positive class
    # and the unknown class is considered as the negative class
    # except for AUPROUT
    y_Actual_Postitive = np.isin(y, list_known)  # Mark known classes as actual positives (for ROC and AUPRIN)
    y_Actual_Postitive_AUPROUT = ~y_Actual_Postitive  # Mark unknown classes as actual positives (for AUPROUT)

    # Initialize Scores
    precision_scores = []
    recall_scores = []
    precision_AUPROUT_scores = []
    recall_AUPROUT_scores = []
    tpr_scores = []
    fpr_scores = []
    true_samples = []
    pos_rates = []
    detection_errors = []
    output = dict()

    # For each percentile threshold index
    for threshold_num in range(len(thresholds_sections[0])):
        y_Predicted_Positive_parts = []
        y_Predicted_Positive_AUPROUT_parts = []

        for section_idx, section in enumerate(sections):
            threshold = thresholds_sections[section_idx][threshold_num]

            section_prediction_pos = section < threshold
            section_prediction_auprout = section >= threshold

            y_Predicted_Positive_parts.append(section_prediction_pos)
            y_Predicted_Positive_AUPROUT_parts.append(section_prediction_auprout)

        # Combine predictions from all sections
        y_Predicted_Positive = np.concatenate(y_Predicted_Positive_parts)
        y_Predicted_Positive_AUPROUT = np.concatenate(y_Predicted_Positive_AUPROUT_parts)  # Predict sample as unknown (positive for AUPROUT) if uncertainty is high
        num_positive = np.sum(y_Predicted_Positive)
        thresholds = thresholds_sections[0]

        # Compute TP, FP, FN based on conditions
        tp = np.sum( y_Actual_Postitive &  y_Predicted_Positive) # True Positives: Known correctly predicted as known
        tn = np.sum(~y_Actual_Postitive & ~y_Predicted_Positive) # True Negatives: Unknown correctly predicted as unknown
        fp = np.sum(~y_Actual_Postitive &  y_Predicted_Positive) # False Positives: Unknown incorrectly predicted as known
        fn = np.sum( y_Actual_Postitive & ~y_Predicted_Positive) # False Negatives: Known incorrectly predicted as unknown

        # Calculate scores
        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # True Positive Rate (Recall)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # False Positive Rate
        pos_rate = tp / num_positive if num_positive > 0 else 0.0

        # Compute AUPROUT scores (unknowns as positives)
        tp_AUPROUT = np.sum(y_Actual_Postitive_AUPROUT & y_Predicted_Positive_AUPROUT)  # Unknown correctly predicted as unknown
        fp_AUPROUT = np.sum(~y_Actual_Postitive_AUPROUT & y_Predicted_Positive_AUPROUT)  # Known incorrectly predicted as unknown
        fn_AUPROUT = np.sum(y_Actual_Postitive_AUPROUT & ~y_Predicted_Positive_AUPROUT)  # Unknown incorrectly predicted as known
        precision_AUPROUT = tp_AUPROUT / (tp_AUPROUT + fp_AUPROUT) if (tp_AUPROUT + fp_AUPROUT) > 0 else 1.0
        recall_AUPROUT = tp_AUPROUT / (tp_AUPROUT + fn_AUPROUT) if (tp_AUPROUT + fn_AUPROUT) > 0 else 0.0

        # Calculate detection error when TPR >= 0.95
        detection_error = 0.5 * (1 - tpr) + 0.5 * fpr if tpr >= 0.95 else None

        # Store values
        precision_scores.append(precision)
        recall_scores.append(recall)
        tpr_scores.append(tpr)
        fpr_scores.append(fpr)
        true_samples.append(num_positive)
        pos_rates.append(pos_rate)
        precision_AUPROUT_scores.append(precision_AUPROUT)
        recall_AUPROUT_scores.append(recall_AUPROUT)
        detection_errors.append(detection_error)

    output["precision_scores"] = np.array(precision_scores)
    output["recall_scores"] = np.array(recall_scores)
    output["precision_AUPROUT_scores"] = np.array(precision_AUPROUT_scores)
    output["recall_AUPROUT_scores"] = np.array(recall_AUPROUT_scores)
    output["tpr_scores"] = np.array(tpr_scores)
    output["fpr_scores"] = np.array(fpr_scores)
    output["detection_errors"] = np.array([de for de in detection_errors if de is not None])
    output["detection_error"] = np.mean(output["detection_errors"])
    output["pos_rates"] = pos_rates

    # Convert uncertainties to weights
    # If there are negative values, shift all values by adding the absolute minimum
    min_value = np.min(uncertainties)
    if min_value < 0:
        uncertainties_W = uncertainties + abs(min_value)
    else:
        uncertainties_W = uncertainties

    uncertainties_W += 1e-5 # add a small values to avoid 0

    # Normalize to [0, 1]
    weights = uncertainties_W / np.sum(uncertainties_W)

    # Probability of drawing a negative sample
    negative_indices = ~y_Actual_Postitive # Mark unknown classes as true
    P_negative = np.sum(weights[negative_indices])

    # Calculate probabilities for 1 to max_samples draws
    output["probabilitie_drawing"] = [1 - (1 - P_negative) ** k for k in range(1, 200 + 1)]

    # Calculate the Area Under the Precision/Recall Curve (AUPRC)
    sorted_indices = np.argsort(output["recall_scores"])
    output["AUPRIN"] = auc(output["recall_scores"][sorted_indices], output["precision_scores"][sorted_indices])

    sorted_indices = np.argsort(output["recall_AUPROUT_scores"])
    output["AUPROUT"] = auc(output["recall_AUPROUT_scores"][sorted_indices], output["precision_AUPROUT_scores"][sorted_indices])

    # Calculate the Area Under the ROC Curve (AUROC)
    sorted_indices = np.argsort(output["fpr_scores"])
    output["AUROC"] = auc(output["fpr_scores"][sorted_indices], output["tpr_scores"][sorted_indices])

    # Select points to annotate: first, last, and two middle ones
    output["annotated_points"] = list()
    annotation_indices = [0, len(thresholds) // 3, 2 * len(thresholds) // 3, -1]
    for idx in annotation_indices:
        output["annotated_points"].append([idx, f"{thresholds[idx]:.2f}, {true_samples[idx]}"])

    # Get sorted indices (descending order)
    sorted_indices = np.argsort(-uncertainties.copy())

    # Sort y_Actual_Postitive accordingly
    sorted_y_Actual_Postitive = y_Actual_Postitive[sorted_indices]

    # Find the index of the first False value
    output["indices_drawing"] = int(np.where(sorted_y_Actual_Postitive == False)[0][0] + 1)  # Add 1 for 1-based index

    #store value range
    output["min_max"] = [min(lof_score), max(lof_score)]
    output["thresholds"] = thresholds

    # all the known classes are considered as the positive class
    # and the unknown class is considered as the negative class

    #get number of questions graph data
    y_Predicted_Positive_parts = []
    y_Predicted_Positive_AUPROUT_parts = []
    nr_quesitons = [1,2,5,10,20,50, 100,200,1000, 2000, 5000, 10000, 20000]

    # Track precision results
    total_nr_questions = []
    precision_per_question = []
    recall_per_question = []
    unknown_per_question = []
    tn_per_question = []

    for nr_question in nr_quesitons:
        y_Predicted_Positive_parts = []  # Predicted known

        for section in sections:
            k = min(nr_question, len(section))  # Avoid out-of-bounds

            # Get indices of top-k highest uncertainties (most likely unknown)
            topk_indices = np.argpartition(-section, k - 1)[:k]

            # Create mask: all True (known), then set top-k to False (unknown)
            section_prediction = np.ones_like(section, dtype=bool)
            section_prediction[topk_indices] = False  # Predict as unknown → not positive

            y_Predicted_Positive_parts.append(section_prediction)

        # Merge predictions across all sections
        y_Predicted_Positive = np.concatenate(y_Predicted_Positive_parts)
        total_nr_question = int(np.sum(~y_Predicted_Positive))

        # Calculate precision (how many of predicted known were actually known)
        tp = np.sum(y_Actual_Postitive & y_Predicted_Positive)  # True Positives: Known correctly predicted as known
        tn = np.sum(~y_Actual_Postitive & ~y_Predicted_Positive)  # True Negatives: Unknown correctly predicted as unknown
        fp = np.sum(~y_Actual_Postitive & y_Predicted_Positive)  # False Positives: Unknown incorrectly predicted as known
        fn = np.sum(y_Actual_Postitive & ~y_Predicted_Positive)  # False Negatives: Known incorrectly predicted as unknown

        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        true_unknown = tn /total_nr_question

        total_nr_questions.append(total_nr_question)
        precision_per_question.append(precision)
        recall_per_question.append(recall)
        unknown_per_question.append(true_unknown)
        tn_per_question.append(int(tn))

    output["nr_quesitons"] = nr_quesitons
    output["total_nr_questions"] = total_nr_questions
    output["precision_per_question"] = precision_per_question
    output["recall_per_question"] = recall_per_question
    output["unknown_percent_per_question"] = unknown_per_question
    output["unknown_absolut_per_question"] = tn_per_question

    return output