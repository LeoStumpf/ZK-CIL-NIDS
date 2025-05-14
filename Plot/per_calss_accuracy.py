import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from Helper.LabelEncoder import encode_labels, LABEL_MAPPING
import matplotlib.pyplot as plt
import os

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


def actual_plot(results_base, results_adjusted, results_al):
    # Prepare data
    categories = []
    values_base = []
    values_adjusted = []
    values_al = []

    for key in results_base:
        if key == 'BENIGN':
            categories.append('Precision\nBENIGN')
            values_base.append(results_base[key]['precision'])
            values_adjusted.append(results_adjusted[key]['precision'])
            values_al.append(results_al[key]['precision'])

        categories.append(f"Recall\n{key}")
        values_base.append(results_base[key]['recall'])
        values_adjusted.append(results_adjusted[key]['recall'])
        values_al.append(results_al[key]['recall'])

    # Plotting
    cm_to_inch = 1 / 2.54
    width_cm = 8.89  # single-column width
    height_cm = 12 

    fig, ax = plt.subplots(figsize=(width_cm * cm_to_inch, height_cm * cm_to_inch), facecolor='white')

    y_pos = range(len(categories))
    bar_width = 0.25

    # Offset positions for grouping (base on top)
    y_base = [y - bar_width for y in y_pos]
    y_adjusted = list(y_pos)
    y_al = [y + bar_width for y in y_pos]

    bars_base = ax.barh(y_base, values_base, height=bar_width, color='#848482', label='Closed-Set')
    bars_adjusted = ax.barh(y_adjusted, values_adjusted, height=bar_width, color='#38ACEC', label='Block Unknown')
    bars_al = ax.barh(y_al, values_al, height=bar_width, color='#7FFFD4', label='AL')

    # Add text annotations
    for bar in bars_base:
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{bar.get_width():.4f}", va='center', ha='left')

    for bar in bars_adjusted:
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{bar.get_width():.4f}", va='center', ha='left')

    for bar in bars_al:
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{bar.get_width():.4f}", va='center', ha='left')

    # Final touches
    ax.set_facecolor('white')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(categories)
    ax.invert_yaxis()  # Highest scores on top
    ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax.grid(False)
    ax.set_xlabel('')
    ax.set_title('')

    # Move legend below the plot in a horizontal row
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0.03), ncol=3, frameon=False, fontsize=8)

    # Set font sizes
    plt.rcParams.update({
        'font.size': 8,
        'axes.titlesize': 8,
        'axes.labelsize': 8,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
    })

    # Save and close
    plt.tight_layout()
    plt.savefig(os.path.join("../04_pics/aggregated/special", "NIPS_bar_plot.png"), format='png', dpi=400, bbox_inches='tight',
                facecolor='white')
    plt.close()

# weights_novel_model: the higher the abnormal
def plot_per_class(y_test, weights_rf_model, weights_novel_model, rf_model_classes, weights_rf_model_AL, rf_model_classes_AL):
    # Reverse mapping for decoding predictions
    INV_LABEL_MAPPING = {v: k for k, v in LABEL_MAPPING.items()}

    # Encode ground truth labels
    y_true = np.array([LABEL_MAPPING[label] for label in y_test])

    # Predict class with highest probability from RF model
    y_pred_base = rf_model_classes[np.argmax(weights_rf_model, axis=1)]
    y_pred_al = rf_model_classes_AL[np.argmax(weights_rf_model_AL, axis=1)]

    # Convert to string labels
    y_true_str = np.array([INV_LABEL_MAPPING[i] for i in y_true])
    y_pred_base_str = np.array([INV_LABEL_MAPPING[i] for i in y_pred_base])
    y_pred_al_str = np.array([INV_LABEL_MAPPING[i] for i in y_pred_al])

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

    # --- Metrics AL adjustment ---
    results_al = {}

    for cls in unique_classes:
        y_true_binary = (y_true_str == cls).astype(int)
        y_pred_binary = (y_pred_al_str == cls).astype(int)

        results_al[cls] = {
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
        print("  AFTER AL:")
        for metric, value in results_al[cls].items():
            print(f"    {metric}: {value:.4f}")
        print()

    #actually Plotting the stuff
    actual_plot(results_base, results_adjusted, results_al)
