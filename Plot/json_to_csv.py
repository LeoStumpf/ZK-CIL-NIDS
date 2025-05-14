import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plots_arregates
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import TwoSlopeNorm


THRESHOLDS = [0.90, 0.95, 0.99]

# Define datasets and implementations of interest
TARGET_DATASETS = ["TrainDay0_TestDay1234"]
IMPLEMENTATIONS = [
    "OneClassForest",
    "OneClassForestwoBootstrap",
    "IsolationForest",
    "NeuronalNetworkLoss",
    "NeuronalNetwork",
    "OneClassSVN",
    "EnergyFlowClassifier",
    "DistanceLOF",
    "LocalOutlierFactor",
    "Random"
]

Attack_datasets = {
    'TrainDay0_DDos': 'DDOS',
    'TrainDay0_Dos': 'DOS',
    'TrainDay0_FTPPatator': 'FTPPatator',
    'TrainDay0_Heartbleed': 'Heartbleed',
    'TrainDay0_Infiltration': 'Infiltration',
    'TrainDay0_PortScan': 'PortScan',
    'TrainDay0_SSHPatator': 'SSHPatator',
    'TrainDay0_TestBotnet': 'Botnet',
    'TrainDay0_Web': 'Web'
}



# Map each algorithm to a fixed color
tab_colors = plt.cm.tab10.colors
algorithm_colors = {name: tab_colors[i % 10] for i, name in enumerate(IMPLEMENTATIONS)}

BASE_DIR = "../04_pics"
PARAMS = [
    #Name
    "algorithm_name",
    "dataset_name",
    "flow/samples",

    #Execution Time
    "execution_time_fit",
    "execution_time_predict",

    #Parametre
    "AUPRIN",
    "AUPROUT",
    "AUROC",
    "indices_drawing"
    ]

for threshold in THRESHOLDS:
    PARAMS.append(f'first_above_{int(threshold * 100)}')

# List of additional list fields you want to extract
LIST_FIELDS = [
    "precision_scores",
    "precision_AUPROUT_scores",
    # Add more list-based fields here if needed
]

OUTPUT_FILE = "output.csv"
LATEX_FILE = "output_tables.tex"

def sanitize_latex(value):
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value).replace("_", "\\_") if value is not None else "---"

def generate_index_drawing(df_all):
    # Initialize new columns
    for threshold in THRESHOLDS:
        df_all[f'first_above_{int(threshold * 100)}'] = None

    # Fill in the index values
    for idx, row in df_all.iterrows():
        probabilities = row["probabilitie_drawing"]
        list_length = len(probabilities)

        for threshold in THRESHOLDS:
            index = next((i + 1 for i, v in enumerate(probabilities) if v > threshold), list_length + 1)
            df_all.at[idx, f'first_above_{int(threshold * 100)}'] = index

    return df_all

def generate_plots_specific3(df_input):
    base_plot_dir = "../04_pics/aggregated/special"
    os.makedirs(base_plot_dir, exist_ok=True)

    df_samples = df_input.copy()
    df_samples = df_samples[df_samples['flow/samples'] == "samples"]
    df_samples['first_above_99'] = pd.to_numeric(df_samples['first_above_99'], errors='coerce')

    df_samples = df_samples[df_samples['dataset_name'].isin(Attack_datasets.keys())]
    df_samples.loc[:, 'dataset_name'] = df_samples['dataset_name'].map(Attack_datasets)

    colors = ["green", "orange", "red"]
    cmap = LinearSegmentedColormap.from_list("custom_gradient", colors, N=256)

    # Pivot the dataframe
    heatmap_data = df_samples.pivot_table(
        index='dataset_name',
        columns='algorithm_name',
        values='first_above_99',
        aggfunc='min'  # or 'mean' if there are multiple values and you prefer average
    )

    # Plot heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".0f",
        cmap=cmap,
        cbar_kws={'label': 'First Above 99'},
        vmin=0, vmax=200  # important for consistent coloring across heatmaps
    )
    #plt.title('Heatmap of First Above 99 Values')
    #plt.xlabel('Algorithm')
    #plt.ylabel('Dataset')
    plt.xlabel('')
    plt.ylabel('')
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(os.path.join(base_plot_dir, "Plot_heatmap.png"),
                format='png', dpi=100, bbox_inches='tight', facecolor='white')
    plt.close()

    # Extract the baseline values (where algorithm is 'NeuronalNetwork' or 'ran
    baseline = df_samples[df_samples['algorithm_name'] == "Random"][['dataset_name', 'first_above_99']]
    baseline = baseline.rename(columns={'first_above_99': 'baseline_value'})

    # Merge baseline values back into the full dataframe
    df_samples = df_samples.merge(baseline, on='dataset_name', how='left')

    # Compute the difference to baseline
    df_samples['difference_to_baseline'] =  df_samples['baseline_value'] - df_samples['first_above_99']

    # Pivot to get matrix for heatmap
    heatmap_diff = df_samples.pivot_table(
        index='dataset_name',
        columns='algorithm_name',
        values='difference_to_baseline',
        aggfunc='mean'  # or min, or first, based on your use case
    )

    # Plotting
    plt.figure(figsize=(12, 8))
    norm = TwoSlopeNorm(vmin=-10, vcenter=0, vmax=20)
    sns.heatmap(
        heatmap_diff,
        annot=True,
        fmt=".1f",
        cmap="RdYlGn",  # Red → Yellow → Green
        norm=norm,
        center=0,  # 0 difference is orange/yellow
        cbar_kws={'label': 'Difference to Baseline'}
    )
    #plt.title(f'Heatmap of Difference to random')
    #plt.xlabel('Algorithm')
    #plt.ylabel('Dataset')
    plt.xlabel('')
    plt.ylabel('')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(base_plot_dir, "Plot_heatmap_relative.png"),
                format='png', dpi=100, bbox_inches='tight', facecolor='white')
    plt.close()


def generate_plots_specific2(df_input):
    base_plot_dir = "../04_pics/aggregated/special"
    os.makedirs(base_plot_dir, exist_ok=True)

    def prepare_and_plot(ax, df, title):
        df = df[df['dataset_name'].isin(Attack_datasets.keys())]
        df.loc[:, 'dataset_name'] = df['dataset_name'].map(Attack_datasets)

        # Prepare complete grid
        all_datasets = sorted(df['dataset_name'].unique())
        all_algorithms = sorted(df['algorithm_name'].unique())
        full_index = pd.MultiIndex.from_product([all_datasets, all_algorithms], names=['dataset_name', 'algorithm_name'])
        df_full = df.set_index(['dataset_name', 'algorithm_name']).reindex(full_index).reset_index()

        df_full['first_above_99_for_plot'] = df_full['first_above_99'].fillna(0).astype(float)
        df_full['is_missing'] = df_full['first_above_99'].isna()

        # Plot settings
        num_algos = len(all_algorithms)
        num_datasets = len(all_datasets)
        bar_width = 0.8 / num_algos
        x = np.arange(num_datasets)

        for i, algo in enumerate(all_algorithms):
            subset = df_full[df_full['algorithm_name'] == algo]
            offsets = x - 0.4 + i * bar_width + bar_width / 2
            heights = subset['first_above_99_for_plot']

            ax.bar(offsets, heights, width=bar_width, label=algo, color=algorithm_colors[algo])

            for j, (value, missing) in enumerate(zip(subset['first_above_99'], subset['is_missing'])):
                label = 'NA' if missing else int(value)
                if not missing and value > 200:
                    label = ">"
                y = subset.iloc[j]['first_above_99_for_plot']
                ax.text(offsets[j], y + 0.3, str(label), ha='center', va='bottom', fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels(all_datasets, rotation=0, fontsize=12)
        ax.set_ylim(0, 210)
        ax.set_ylabel(f'Number {title} for Probability >99%', fontsize=14)
        ax.set_title(f'Number {title} for Probability >99%', fontsize=14)

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(25, 10), dpi=100, sharex=True)

    df_samples = df_input.copy()
    df_samples = df_samples[df_samples['flow/samples'] == "samples"]
    prepare_and_plot(ax1, df_samples, title="Samples")

    df_flows = df_input.copy()
    df_flows = df_flows[df_flows['flow/samples'] == "flows"]
    prepare_and_plot(ax2, df_flows, title="Flows")

    # Custom legend below both plots
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels,
               loc='lower center',
               ncol=5,
               frameon=False,
               fontsize=12)

    # Adjust layout
    fig.tight_layout()
    fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.1)
    plt.savefig(os.path.join(base_plot_dir, "Plot_samples_vs_flows_attack_drawf.png"),
                format='png', dpi=100, bbox_inches='tight', facecolor='white')
    plt.close()


def generate_plots_specific(df_all):
    DPI = 100
    # Ensure the main plot folder exists
    base_plot_dir = "../04_pics/aggregated/special"
    os.makedirs(base_plot_dir, exist_ok=True)

    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(20, 16))

    general_settig = {'flow/samples': 'samples', 'DPI': DPI}
    plots = [
        {'dataset_name': 'TrainDay0_TestDay1234', 'ax': axes[0, 0], "metric": "AUROC"},
        {'dataset_name': 'TrainDay01_TestDay234', 'ax': axes[0, 1], "metric": "AUROC"},
        {'dataset_name': 'TrainDay012_TestDay34', 'ax': axes[0, 2], "metric": "AUROC"},
        {'dataset_name': 'TrainDay0123_TestDay4', 'ax': axes[0, 3], "metric": "AUROC"},

        {'dataset_name': 'TrainDay0_TestDay1234', 'ax': axes[1, 0], "metric": "AUPRIN"},
        {'dataset_name': 'TrainDay01_TestDay234', 'ax': axes[1, 1], "metric": "AUPRIN"},
        {'dataset_name': 'TrainDay012_TestDay34', 'ax': axes[1, 2], "metric": "AUPRIN"},
        {'dataset_name': 'TrainDay0123_TestDay4', 'ax': axes[1, 3], "metric": "AUPRIN"},

        {'dataset_name': 'TrainDay0_TestDay1234', 'ax': axes[2, 0], "metric": "AUPROUT"},
        {'dataset_name': 'TrainDay01_TestDay234', 'ax': axes[2, 1], "metric": "AUPROUT"},
        {'dataset_name': 'TrainDay012_TestDay34', 'ax': axes[2, 2], "metric": "AUPROUT"},
        {'dataset_name': 'TrainDay0123_TestDay4', 'ax': axes[2, 3], "metric": "AUPROUT"}
    ]

    fig.set_facecolor('white')
    for plot in plots:
        plot_infos = general_settig | plot
        ax = plot["ax"]
        ax.set_facecolor('white')

        filtered = df_all[(df_all['dataset_name'] == plot["dataset_name"]) & (df_all['flow/samples'] == general_settig["flow/samples"])]
        filtered = filtered.sort_values(by="algorithm_name")
        for _, row in filtered.iterrows():
            algorithm = row["algorithm_name"]
            if plot["metric"] == "AUROC":
                fpr_scores = row["fpr_scores"]
                tpr_scores = row["tpr_scores"]
                ax.plot(fpr_scores, tpr_scores,
                label=algorithm,
                marker='o',
                linestyle='-',
                color=algorithm_colors[algorithm],
                markersize=3)

            if plot["metric"] == "AUPRIN":
                precision_scores = row["precision_scores"]
                recall_scores = row["recall_scores"]
                ax.plot(recall_scores, precision_scores,
                label=algorithm,
                marker='o',
                linestyle='-',
                color=algorithm_colors[algorithm],
                markersize=3)

            if plot["metric"] == "AUPROUT":
                precision_AUPROUT_scores = row["precision_AUPROUT_scores"]
                recall_AUPROUT_scores = row["recall_AUPROUT_scores"]
                ax.plot(recall_AUPROUT_scores, precision_AUPROUT_scores,
                label=algorithm,
                marker='o',
                linestyle='-',
                color=algorithm_colors[algorithm],
                markersize=3)

        ax.set_xlabel('False Positive Rate', fontsize=10)
        ax.set_ylabel('True Positive Rate', fontsize=10)

        # Square axe
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal', adjustable='box')

    fig.subplots_adjust(hspace=0.4)
    #fig.subplots_adjust(vspace=0.4)

    #set titles
    axes[0, 0].set_title("TrainDay0_TestDay1234", pad=10, fontsize=14)
    axes[0, 1].set_title("TrainDay01_TestDay234", pad=10, fontsize=14)
    axes[0, 2].set_title("TrainDay012_TestDay34", pad=10, fontsize=14)
    axes[0, 3].set_title("TrainDay0123_TestDay4", pad=10, fontsize=14)

    pos = axes[0, 0].get_position()
    y_center = (pos.y0 + pos.y1) / 2
    fig.text(pos.x0 - 0.07, y_center, "AUROC", va='center', ha='center', fontsize=14, rotation=90)

    pos = axes[1, 0].get_position()
    y_center = (pos.y0 + pos.y1) / 2
    fig.text(pos.x0 - 0.07, y_center, "AUPRIN", va='center', ha='center', fontsize=14, rotation=90)

    pos = axes[2, 0].get_position()
    y_center = (pos.y0 + pos.y1) / 2
    fig.text(pos.x0 - 0.07, y_center, "AUPROUT", va='center', ha='center', fontsize=14, rotation=90)

    # Create one shared legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels,
               loc='lower center',
               ncol=5,
               bbox_to_anchor=(0.5, -0.01),
               frameon=False,
               fontsize=12)

    # Adjust layout
    fig.tight_layout()
    fig.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.05)

    # Save and close
    fig.savefig(os.path.join(base_plot_dir, f"Plot_{general_settig['flow/samples']}_days.png"), format='png', dpi=general_settig['DPI'],
                bbox_inches='tight',
                facecolor='white')
    plt.close(fig)



def generate_plots(df_all):
    # Ensure the main plot folder exists
    base_plot_dir = "../04_pics/aggregated"
    os.makedirs(base_plot_dir, exist_ok=True)

    # Group by "flow/samples" and then by "dataset_name"
    for flow_sample_type, df_flow_group in df_all.groupby("flow/samples"):
        subfolder_path = os.path.join(base_plot_dir, flow_sample_type)
        os.makedirs(subfolder_path, exist_ok=True)

        for dataset_name, df_dataset_group in df_flow_group.groupby("dataset_name"):
            #Plot_AUPRIN, Plot_TprFpr, Plot_AUPROUT, Plot_TruePositivesRate, Plot_DrawingProbability
            DPI = 300
            fig_AUPRIN, ax_AUPRIN = plt.subplots(figsize=(8, 8), dpi=DPI)
            fig_TprFpr, ax_TprFpr = plt.subplots(figsize=(8, 8), dpi=DPI)
            fig_AUPROUT, ax_AUPROUT = plt.subplots(figsize=(8, 8), dpi=DPI)
            fig_TruePositivesRate, ax_TruePositivesRate = plt.subplots(figsize=(10, 5), dpi=DPI)
            fig_DrawingProbability, ax_DrawingProbability = plt.subplots(figsize=(10, 5), dpi=DPI)

            df_dataset_group = df_dataset_group.sort_values(by="algorithm_name")
            for _, row in df_dataset_group.iterrows():
                algorithm = row["algorithm_name"]
                fpr_scores = row["fpr_scores"]
                tpr_scores = row["tpr_scores"]
                ax_TprFpr.plot(fpr_scores, tpr_scores, label=algorithm, marker='o', linestyle='-', color=algorithm_colors[algorithm])

                precision_scores = row["precision_scores"]
                recall_scores = row["recall_scores"]
                ax_AUPRIN.plot(recall_scores, precision_scores, label=algorithm, marker='o', linestyle='-', color=algorithm_colors[algorithm])

                precision_AUPROUT_scores = row["precision_AUPROUT_scores"]
                recall_AUPROUT_scores = row["recall_AUPROUT_scores"]
                ax_AUPROUT.plot(recall_AUPROUT_scores, precision_AUPROUT_scores, label=algorithm, marker='o', linestyle='-', color=algorithm_colors[algorithm])

                pos_rates = row["pos_rates"]
                indices = np.arange(len(pos_rates))
                ax_TruePositivesRate.plot(indices, pos_rates, label=algorithm, marker='o', linestyle='-', color=algorithm_colors[algorithm])

                probabilitie_drawing = row["probabilitie_drawing"]
                indices = np.arange(len(probabilitie_drawing))
                ax_DrawingProbability.plot(indices, probabilitie_drawing, label=algorithm, marker='o', linestyle='-', color=algorithm_colors[algorithm])

            plot_infos = {'dataset_name': dataset_name, 'flow/samples':flow_sample_type, 'save_path': subfolder_path, 'DPI': DPI}
            plots_arregates.format_plot_TprFpr(fig_TprFpr, ax_TprFpr, plot_infos)
            plots_arregates.format_plot_AUPRIN(fig_AUPRIN, ax_AUPRIN, plot_infos)
            plots_arregates.format_plot_AUPROUT(fig_AUPROUT, ax_AUPROUT, plot_infos)
            plots_arregates.format_plot_TruePositivesRate(fig_TruePositivesRate, ax_TruePositivesRate, plot_infos)
            plots_arregates.format_plot_DrawingProbability(fig_DrawingProbability, ax_DrawingProbability, plot_infos)


def print_table_lines(df_all, dataset, flow_samples, impl):
    # Filter the DataFrame
    filtered = df_all[(df_all['dataset_name'] == dataset) & (df_all['flow/samples'] == flow_samples) & (df_all['algorithm_name'] == impl)]

    if not filtered.empty:
        # Get the first row as a dict
        row = filtered.iloc[0].to_dict()
    else:
        # Return dict with all column names and "--" as values
        row = {col: '-' for col in df_all.columns}
        row["algorithm_name"] = impl

    line = " & ".join([
        sanitize_latex(row["algorithm_name"]),
        sanitize_latex(row["execution_time_fit"]),
        sanitize_latex(row["execution_time_predict"]),
        sanitize_latex(row["AUPRIN"]),
        sanitize_latex(row["AUPROUT"]),
        sanitize_latex(row["AUROC"]),
        sanitize_latex(row["indices_drawing"]),
        sanitize_latex(row["first_above_90"]),
        sanitize_latex(row["first_above_95"]),
        sanitize_latex(row["first_above_99"])
    ]) + " \\\\\n"
    f.write(line)

if __name__ == "__main__":

    # Find all JSON files recursively
    json_files = []
    for root, _, files in os.walk(BASE_DIR):
        for file in files:
            if file.endswith(".json"):
                json_files.append(os.path.join(root, file))

    # Process each JSON file and extract the parameters
    df_dicts = []
    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                content = json.load(f)

            all_fields = list(content.keys())

            # Extract only the last two parts of the file path
            short_filename = "_".join(json_file.split(os.sep)[-2:])  # Get last two parts

            entry = {"filename": short_filename}
            for key in all_fields:
                entry[key] = content.get(key, None)

            df_dicts.append(entry)
        except Exception as e:
            print(f"Error reading {json_file}: {e}")

    # Convert to DataFrame
    df_all = pd.DataFrame(df_dicts)

    # Generate DF drawn index
    df_all = generate_index_drawing(df_all)

    #generate_plots(df_all)
    #generate_plots_specific(df_all)
    #generate_plots_specific2(df_all)
    generate_plots_specific3(df_all)

    # Write data to CSV
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        df_all.to_csv(f, columns=PARAMS, index=False, encoding="utf-8")
    print(f"CSV file saved: {OUTPUT_FILE}")

    # ------------------ LaTeX TABLE OUTPUT ------------------ #

    # Create LaTeX tables
    with open(LATEX_FILE, "w", encoding="utf-8") as f:
        f.write("\\documentclass{article}\n\\usepackage{booktabs}\n\\begin{document}\n")

        for flow_samples, group_fs in df_all.groupby('flow/samples'):
            #print(f"Processing flow samples:{flow_samples}")

            for dataset, group_ds in group_fs.groupby("dataset_name"):
                #print(f"Processing dataset_name:{dataset}")

                f.write(f"\\section*{{Results for dataset: {dataset}, {flow_samples}}}\n")

                # Create LaTeX table
                f.write("\\begin{table}[h!]\n\\centering\n")
                f.write(f"\\caption{{Results for dataset {dataset}, flow samples: {flow_samples}}}\n")
                f.write("\\begin{tabular}{lrrrrrrrrrr}\n")
                f.write("\\toprule\n")
                f.write("Algorithm & Fit Time & Predict Time & AUPRIN & AUPROUT & AUROC & i\_drawn & $\geq 0.9\%$ & $\geq 0.95\%$ & $\geq 0.99\%$ \\\\\n")
                f.write("\\midrule\n")

                for impl in IMPLEMENTATIONS:
                    print_table_lines(df_all, dataset, flow_samples, impl)

                f.write("\\bottomrule\n")
                f.write("\\end{tabular}\n\\end{table}\n\n")


        #write specific tabe Sample score results for day datasets
        f.write("\\begin{table*}[t]\n\\centering\n")
        f.write(f"\\caption{{Sample score results for day datasets}}\n")
        f.write("\\begin{tabular}{lrrrrrrrrrr}\n")
        f.write("\\toprule\n")
        for dataset in ["TrainDay0_TestDay1234", "TrainDay01_TestDay234", "TrainDay012_TestDay34", "TrainDay0123_TestDay4"]:
            f.write(f"{sanitize_latex(dataset)} & Fit Time & Predict Time & AUPRIN & AUPROUT & AUROC & i\_drawn & $\geq 0.9\%$ & $\geq 0.95\%$ & $\geq 0.99\%$ \\\\\n")
            f.write("\\midrule\n")

            # Sort the dataframe
            sorted_df = df_all[(df_all['dataset_name'] == dataset) & (df_all['flow/samples'] == flow_samples)].sort_values(
                by='first_above_99',
                key=lambda x: x.where(pd.notna(x), np.inf)  # Put NaNs at the bottom
            )
            sorted_IMPLEMENTATIONS = sorted_df['algorithm_name'].tolist()

            #for impl in IMPLEMENTATIONS:
            for impl in sorted_IMPLEMENTATIONS:
                print_table_lines(df_all, dataset, "samples", impl)
            f.write("\\midrule\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n\\end{table*}\n\n")

        f.write("\\end{document}\n")

        print(f"LaTeX tables saved: {LATEX_FILE}")


