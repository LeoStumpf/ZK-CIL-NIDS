import numpy as np
import pandas as pd

from Plot.plots_and_graphs import plot_weights
from Implementation.EnergyFlowClassifier import predict, fit


def load_data(filename):
    """Loads X, y, and aux_data from a .npz file."""
    data = np.load(filename, allow_pickle=True)
    return data['X'], data['y'], pd.DataFrame({'flow_index': data['Flow_index']})



if __name__ == '__main__':
    # export_data()
    X, y, _ = load_data("0_monday.npz")

    # train detector
    #lof_list, rf_model = generate_model(X, y)
    #model = fit(X[:20, :], y[:20])
    model = fit(X, y)

    del X, y

    # Load data day2
    X, y, Metadata = load_data("1_tuesday.npz")



    # plot
    weights = predict(X, Metadata, model)

    plot_weights(y, weights)

    Metadata["weights"] = weights
    Metadata["labels"] = y

    grouped = Metadata.groupby("flow_index").agg({
        "weights": "mean",  # Compute mean of weights
        "labels": "first"  # Take the first label
    }).reset_index()

    plot_weights(grouped["labels"].to_numpy(), grouped["weights"].tolist())