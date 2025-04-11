import os
import numpy as np
import matplotlib.pyplot as plt


def format_plot_general1(fig, ax, metadata):
    # Set background to white
    ax.set_facecolor('white')
    fig.set_facecolor('white')

    # Add meta information in the top-left corner
    meta_info = (
            r"$\bf{Dataset:}$ " + f"{metadata['dataset_name']}\n" +
            r"$\bf{flow/samples:}$ " + f"{metadata['flow/samples']}\n"
    )

    ax.text(0.02, 0.98, meta_info,
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.5'))
    return fig, ax


def format_plot_general2(fig, ax, metadata):
    ax.legend(loc='lower right', fontsize=8, frameon=True, framealpha=0.8, facecolor='white')
    ax.grid(alpha=0.5, linestyle='--', linewidth=0.5, color='gray')

    # Add a frame around the plot
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(0.5)

    # Save and close
    fig.tight_layout()
    fig.savefig(os.path.join(metadata['save_path'], f"Plot_{metadata['Plot_name']}_{metadata['flow/samples']}_{metadata['dataset_name']}.png"), format='png', dpi=metadata['DPI'], bbox_inches='tight',
                facecolor='white')
    plt.close(fig)


def format_plot_TprFpr(fig, ax, metadata):
    # Apply general formatting 1
    fig, ax = format_plot_general1(fig, ax, metadata)

    # Diagonal line for random guessing
    ax.plot([0, 1], [0, 1], linestyle='--', label='Random (theory)', color='r')

    # Labels, title, and legend
    ax.set_xlabel('False Positive Rate', fontsize=10)
    ax.set_ylabel('True Positive Rate', fontsize=10)
    ax.set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=12, pad=10)

    # Fix axes limits and grid
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Apply general formatting 2
    metadata['Plot_name'] = "AUROC"
    format_plot_general2(fig, ax, metadata)


def format_plot_AUPRIN(fig, ax, metadata):
    # Apply general formatting 1
    fig, ax = format_plot_general1(fig, ax, metadata)

    # Labels, title, and legend
    ax.set_xlabel('Recall', fontsize=10)
    ax.set_ylabel('Precision', fontsize=10)
    ax.set_title('Precision-Recall Curve (AUPRIN)', fontsize=12, pad=10)

    # Fix axes limits and grid
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Apply general formatting 2
    metadata['Plot_name'] = "AUPRIN"
    format_plot_general2(fig, ax, metadata)


def format_plot_AUPROUT(fig, ax, metadata):
    # Apply general formatting 1
    fig, ax = format_plot_general1(fig, ax, metadata)

    # Labels, title, and legend
    ax.set_xlabel('Recall', fontsize=10)
    ax.set_ylabel('Precision', fontsize=10)
    ax.set_title('Precision-Recall Curve (AUPROUT)', fontsize=12, pad=10)

    # Fix axes limits and grid
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Apply general formatting 2
    metadata['Plot_name'] = "AUPROUT"
    format_plot_general2(fig, ax, metadata)


def format_plot_TruePositivesRate(fig, ax, metadata):
    # Apply general formatting 1
    fig, ax = format_plot_general1(fig, ax, metadata)

    # Labels, title, and legend
    ax.set_xlabel('Threshold', fontsize=10)
    ax.set_ylabel('TP / num_positives', fontsize=10)
    ax.set_title('True Positives Rate vs. Threshold', fontsize=12, pad=10)

    # Fix axes limits and grid
    # ax.set_xlim(0, 1)
    #ax.set_ylim(0, max(pos_rates) * 1.1)

    # Apply general formatting 2
    metadata['Plot_name'] = "TPR"
    format_plot_general2(fig, ax, metadata)


def format_plot_DrawingProbability(fig, ax, metadata):
    # Apply general formatting 1
    fig, ax = format_plot_general1(fig, ax, metadata)

    # Labels, title, and legend
    ax.set_xlabel('Num Samples', fontsize=10)
    ax.set_ylabel('Probability Novel Sample', fontsize=10)
    ax.set_title('Probability to Draw Novel Sample', fontsize=12, pad=10)

    # Fix axes limits and grid
    # ax.set_xlim(0, 1)
    #ax.set_ylim(0, max(probabilitie_drawing) * 1.1)

    # Apply general formatting 2
    metadata['Plot_name'] = "Probabilitis"
    format_plot_general2(fig, ax, metadata)