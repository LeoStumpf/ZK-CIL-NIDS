import os
import numpy as np
import matplotlib.pyplot as plt



# Common plot settings
DPI = 300  # High resolution
plt.style.use('seaborn-v0_8')  # Use a clean and professional style

def Plot_AUPRIN(scores, save_path, metadata):
    FIG_SIZE = (8, 8)  # Square size for equal importance of TPR and FPR

    # get eleements
    recall_scores = scores["recall_scores"]
    precision_scores = scores["precision_scores"]
    annotated_points = scores["annotated_points"]

    # Create plot
    plt.figure(figsize=FIG_SIZE, dpi=DPI)

    # Set background to white
    plt.gca().set_facecolor('white')  # Set plot background to white
    plt.gcf().set_facecolor('white')  # Set figure background to white

    # Plot ROC curve
    colors = plt.cm.tab10.colors  # A qualitative color palette
    plt.plot(recall_scores, precision_scores, marker='o', linestyle='-', label=f'Class known', color=colors[0])

    # Annotate selected points with smarter placement
    for idx, text in annotated_points:
        x, y = recall_scores[idx], precision_scores[idx]
        # Adjust annotation position based on point location
        if x > 0.8 and y > 0.8:  # Top-right corner
            xytext = (-10, -10)  # Move annotation down and left
        elif x < 0.2 and y > 0.8:  # Top-left corner
            xytext = (10, -10)  # Move annotation down and right
        else:
            xytext = (5, 5)  # Default offset
        plt.annotate(text,
                     (x, y),
                     textcoords="offset points",
                     xytext=xytext,
                     ha='right',
                     fontsize=8,
                     color='blue',
                     bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.2'))

    # Add meta information in the top-left corner
    meta_info = (
            r"$\bf{Algorithm:}$ " + f"{metadata['algorithm_name']}\n" +
            r"$\bf{Dataset:}$ " + f"{metadata['dataset_name']}\n" +
            r"$\bf{Execution\ time\ fit:}$ " + f"{metadata['execution_time_fit']}\n" +
            r"$\bf{Execution\ time\ predict:}$ " + f"{metadata['execution_time_predict']}\n" +
            r"$\bf{AUPRIN:}$ " + f"{scores['AUPRIN']:.5f}"
    )

    plt.text(0.02, 0.98, meta_info,
             transform=plt.gca().transAxes,
             fontsize=8,
             verticalalignment='top',
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.5'))

    # Add explanation for annotated points to the legend
    plt.plot([], [], ' ', label="Annotations:\nThreshold / Num of True Positives")  # Empty plot for legend entry


    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve (AUPRIN)')
    plt.legend(loc='lower right', fontsize=8, frameon=True, framealpha=0.8, facecolor='white')

    # Fix axes limits and grid
    plt.xlim(0, 1)
    #plt.ylim(0, 1)
    plt.grid(alpha=0.5, linestyle='--', linewidth=0.5, color='gray')

    # Add a frame around the plot
    for spine in plt.gca().spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(0.5)

    # Save and close
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"Plot_AUPRIN_{metadata['flow/samples']}_{metadata['algorithm_name']}_{metadata['dataset_name']}.png"), format='png', dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()

def Plot_AUPROUT(scores, save_path, metadata):
    FIG_SIZE = (8, 8)  # Square size for equal importance of TPR and FPR

    # get eleements
    recall_scores = scores["recall_AUPROUT_scores"]
    precision_scores = scores["precision_AUPROUT_scores"]
    annotated_points = scores["annotated_points"]

    # Create plot
    plt.figure(figsize=FIG_SIZE, dpi=DPI)

    # Set background to white
    plt.gca().set_facecolor('white')  # Set plot background to white
    plt.gcf().set_facecolor('white')  # Set figure background to white

    # Plot ROC curve
    colors = plt.cm.tab10.colors  # A qualitative color palette
    plt.plot(recall_scores, precision_scores, marker='o', linestyle='-', label=f'Class known', color=colors[0])

    # Annotate selected points with smarter placement
    for idx, text in annotated_points:
        x, y = recall_scores[idx], precision_scores[idx]
        # Adjust annotation position based on point location
        if x > 0.8 and y > 0.8:  # Top-right corner
            xytext = (-10, -10)  # Move annotation down and left
        elif x < 0.2 and y > 0.8:  # Top-left corner
            xytext = (10, -10)  # Move annotation down and right
        else:
            xytext = (5, 5)  # Default offset
        plt.annotate(text,
                     (x, y),
                     textcoords="offset points",
                     xytext=xytext,
                     ha='right',
                     fontsize=8,
                     color='blue',
                     bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.2'))

    # Add meta information in the top-left corner
    meta_info = (
            r"$\bf{Algorithm:}$ " + f"{metadata['algorithm_name']}\n" +
            r"$\bf{Dataset:}$ " + f"{metadata['dataset_name']}\n" +
            r"$\bf{Execution\ time\ fit:}$ " + f"{metadata['execution_time_fit']}\n" +
            r"$\bf{Execution\ time\ predict:}$ " + f"{metadata['execution_time_predict']}\n" +
            r"$\bf{AUPROUT:}$ " + f"{scores['AUPROUT']:.5f}"
    )

    plt.text(0.02, 0.98, meta_info,
             transform=plt.gca().transAxes,
             fontsize=8,
             verticalalignment='top',
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.5'))

    # Add explanation for annotated points to the legend
    plt.plot([], [], ' ', label="Annotations:\nThreshold / Num of True Positives")  # Empty plot for legend entry


    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve (AUPROUT)')
    plt.legend(loc='lower right', fontsize=8, frameon=True, framealpha=0.8, facecolor='white')

    # Fix axes limits and grid
    plt.xlim(0, 1)
    #plt.ylim(0, 1)
    plt.grid(alpha=0.5, linestyle='--', linewidth=0.5, color='gray')

    # Add a frame around the plot
    for spine in plt.gca().spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(0.5)

    # Save and close
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"Plot_AUPROUT_{metadata['flow/samples']}_{metadata['algorithm_name']}_{metadata['dataset_name']}.png"), format='png', dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()



def Plot_TprFpr(scores, save_path, metadata):
    FIG_SIZE = (8, 8)  # Square size for equal importance of TPR and FPR

    # Get elements
    annotated_points = scores["annotated_points"]
    fpr_scores = scores["fpr_scores"]
    tpr_scores = scores["tpr_scores"]

    # Create plot
    plt.figure(figsize=FIG_SIZE, dpi=DPI)

    # Set background to white
    plt.gca().set_facecolor('white')  # Set plot background to white
    plt.gcf().set_facecolor('white')  # Set figure background to white

    # Plot ROC curve
    colors = plt.cm.tab10.colors  # A qualitative color palette
    plt.plot(fpr_scores, tpr_scores, marker='o', linestyle='-', label='ROC Curve', color=colors[0])
    plt.plot([0, 1], [0, 1], linestyle='--', label='Random Guessing', color='r')  # Diagonal line for random guessing

    # Annotate selected points with smarter placement
    for idx, text in annotated_points:
        x, y = fpr_scores[idx], tpr_scores[idx]
        # Adjust annotation position based on point location
        if x > 0.8 and y > 0.8:  # Top-right corner
            xytext = (-10, -10)  # Move annotation down and left
        elif x < 0.2 and y > 0.8:  # Top-left corner
            xytext = (10, -10)  # Move annotation down and right
        else:
            xytext = (5, 5)  # Default offset
        plt.annotate(text,
                     (x, y),
                     textcoords="offset points",
                     xytext=xytext,
                     ha='right',
                     fontsize=8,
                     color='blue',
                     bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.2'))

    # Add meta information in the top-left corner
    meta_info = (
            r"$\bf{Algorithm:}$ " + f"{metadata['algorithm_name']}\n" +
            r"$\bf{Dataset:}$ " + f"{metadata['dataset_name']}\n" +
            r"$\bf{Execution\ time\ fit:}$ " + f"{metadata['execution_time_fit']}\n" +
            r"$\bf{Execution\ time\ predict:}$ " + f"{metadata['execution_time_predict']}\n" +
            r"$\bf{AUROC:}$ " + f"{scores['AUROC']:.5f}"
    )

    plt.text(0.02, 0.98, meta_info,
             transform=plt.gca().transAxes,
             fontsize=8,
             verticalalignment='top',
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.5'))

    # Add explanation for annotated points to the legend
    plt.plot([], [], ' ', label="Annotations:\nThreshold / Num of True Positives")  # Empty plot for legend entry

    # Labels, title, and legend
    plt.xlabel('False Positive Rate', fontsize=10)
    plt.ylabel('True Positive Rate', fontsize=10)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=12, pad=10)
    plt.legend(loc='lower right', fontsize=8, frameon=True, framealpha=0.8, facecolor='white')

    # Fix axes limits and grid
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(alpha=0.5, linestyle='--', linewidth=0.5, color='gray')

    # Add a frame around the plot
    for spine in plt.gca().spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(0.5)

    # Save and close
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"Plot_AUROC_{metadata['flow/samples']}_{metadata['algorithm_name']}_{metadata['dataset_name']}.png"), format='png', dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()

def Plot_TruePositivesRate(scores, save_path, metadata):
    FIG_SIZE = (10, 5)
    # Extract elements
    pos_rates = scores["pos_rates"]
    indices = np.arange(len(pos_rates))

    # Create plot
    plt.figure(figsize=FIG_SIZE, dpi=DPI)

    # Set background to white
    plt.gca().set_facecolor('white')  # Set plot background to white
    plt.gcf().set_facecolor('white')  # Set figure background to white

    # Plot ROC curve
    colors = plt.cm.tab10.colors  # A qualitative color palette
    plt.plot(indices, pos_rates, marker='o', linestyle='-', label='TP / num_positives', color=colors[0])

    # Add meta information in the top-left corner
    meta_info = (
            r"$\bf{Algorithm:}$ " + f"{metadata['algorithm_name']}\n" +
            r"$\bf{Dataset:}$ " + f"{metadata['dataset_name']}\n" +
            r"$\bf{Execution\ time\ fit:}$ " + f"{metadata['execution_time_fit']}\n" +
            r"$\bf{Execution\ time\ predict:}$ " + f"{metadata['execution_time_predict']}\n"
    )

    plt.text(0.02, 0.98, meta_info,
             transform=plt.gca().transAxes,
             fontsize=8,
             verticalalignment='top',
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.5'))

    # Labels and title
    plt.xlabel('Threshold', fontsize=10)
    plt.ylabel('TP / num_positives', fontsize=10)
    plt.title('True Positives Rate vs. Threshold', fontsize=12, pad=10)

    # Ensure y-axis starts at 0, while max is determined dynamically
    plt.ylim(0, max(pos_rates) * 1.1)  # Adds 10% padding for better visibility

    # Add gridlines for better readability
    plt.grid(alpha=0.5, linestyle='--', linewidth=0.5, color='gray')

    # Show legend
    plt.legend(loc='lower right', fontsize=8, frameon=True, framealpha=0.8, facecolor='white')

    # Add a frame around the plot
    for spine in plt.gca().spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(0.5)

    # Save and close
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"Plot_TPR_{metadata['flow/samples']}_{metadata['algorithm_name']}_{metadata['dataset_name']}.png"), format='png', dpi=DPI, bbox_inches='tight',
                facecolor='white')
    plt.close()

def Plot_DrawingProbability(scores, save_path, metadata):
    FIG_SIZE = (10, 5)
    # Extract elements
    probabilitie_drawing = scores["probabilitie_drawing"]
    indices = np.arange(len(probabilitie_drawing))

    # Create plot
    plt.figure(figsize=FIG_SIZE, dpi=DPI)

    # Set background to white
    plt.gca().set_facecolor('white')  # Set plot background to white
    plt.gcf().set_facecolor('white')  # Set figure background to white

    # Plot ROC curve
    colors = plt.cm.tab10.colors  # A qualitative color palette
    plt.plot(indices, probabilitie_drawing, marker='o', linestyle='-', label='TP / num_positives', color=colors[0])

    # Calculate first indices where probabilities exceed thresholds
    thresholds = [0.9, 0.95, 0.99]
    threshold_indices = [
        next((i + 1) for i, p in enumerate(probabilitie_drawing) if p >= 0.9) if any(p >= 0.9 for p in probabilitie_drawing) else f"90% > {len(probabilitie_drawing)}",
        next((i + 1) for i, p in enumerate(probabilitie_drawing) if p >= 0.95) if any(p >= 0.95 for p in probabilitie_drawing) else f"95% > {len(probabilitie_drawing)}",
        next((i + 1) for i, p in enumerate(probabilitie_drawing) if p >= 0.99) if any(p >= 0.99 for p in probabilitie_drawing) else f"99% > {len(probabilitie_drawing)}",
    ]

    # Add meta information in the top-left corner
    meta_info = (
            r"$\bf{Algorithm:}$ " + f"{metadata['algorithm_name']}\n" +
            r"$\bf{Dataset:}$ " + f"{metadata['dataset_name']}\n" +
            r"$\bf{Execution\ time\ fit:}$ " + f"{metadata['execution_time_fit']}\n" +
            r"$\bf{Execution\ time\ predict:}$ " + f"{metadata['execution_time_predict']}\n" +
            r"$\bf{Threshold\ indices:}$ " + "\n" +
            r"$\ \ \ \ 90\%: $" + f" {threshold_indices[0]}\n" +
            r"$\ \ \ \ 95\%: $" + f" {threshold_indices[1]}\n" +
            r"$\ \ \ \ 99\%: $" + f" {threshold_indices[2]}\n"
    )

    plt.text(0.02, 0.98, meta_info,
             transform=plt.gca().transAxes,
             fontsize=8,
             verticalalignment='top',
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.5'))

    # Labels and title
    plt.xlabel('Num Samples', fontsize=10)
    plt.ylabel('Probability Novel Sample', fontsize=10)
    plt.title('Probability to Draw Novel Sample', fontsize=12, pad=10)

    # Ensure y-axis starts at 0, while max is determined dynamically
    plt.ylim(0, max(probabilitie_drawing) * 1.1)  # Adds 10% padding for better visibility

    # Add gridlines for better readability
    plt.grid(alpha=0.5, linestyle='--', linewidth=0.5, color='gray')

    # Show legend
    plt.legend(loc='lower right', fontsize=8, frameon=True, framealpha=0.8, facecolor='white')

    # Add a frame around the plot
    for spine in plt.gca().spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(0.5)

    # Save and close
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"Plot_Probabilitis_{metadata['flow/samples']}_{metadata['algorithm_name']}_{metadata['dataset_name']}.png"), format='png', dpi=DPI,
                bbox_inches='tight',
                facecolor='white')
    plt.close()