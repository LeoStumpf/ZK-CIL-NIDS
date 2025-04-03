import os
import json
import csv

if __name__ == "__main__":
    # Define the directory and parameters to extract
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
        ]
    THRESHOLDS = [0.90, 0.95, 0.99]
    OUTPUT_FILE = "output.csv"

    # Find all JSON files recursively
    json_files = []
    for root, _, files in os.walk(BASE_DIR):
        for file in files:
            if file.endswith(".json"):
                json_files.append(os.path.join(root, file))

    # Process each JSON file and extract the parameters
    data = []
    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                content = json.load(f)

            # Extract only the last two parts of the file path
            short_filename = "_".join(json_file.split(os.sep)[-2:])  # Get last two parts

            # Extract parameters (default to None if missing)
            row = [short_filename] + [content.get(param, None) for param in PARAMS]
            data.append(row)

            # Extract probability list and compute threshold indices
            probabilities = content.get("probabilitie_drawing", [])
            list_length = len(probabilities)

            for threshold in THRESHOLDS:
                index = next((i + 1 for i, v in enumerate(probabilities) if v > threshold), list_length + 1)
                row.append(index)
        except Exception as e:
            print(f"Error reading {json_file}: {e}")

    # Write data to CSV
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["filename"] + PARAMS + [f"first_above_{t}" for t in THRESHOLDS]
        writer.writerow(header)  # Write header
        writer.writerows(data)  # Write data rows

    print(f"CSV file saved: {OUTPUT_FILE}")
