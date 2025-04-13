import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plots_arregates

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
            for impl in IMPLEMENTATIONS:
                print_table_lines(df_all, dataset, "samples", impl)
            f.write("\\midrule\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n\\end{table*}\n\n")

        f.write("\\end{document}\n")

        print(f"LaTeX tables saved: {LATEX_FILE}")


