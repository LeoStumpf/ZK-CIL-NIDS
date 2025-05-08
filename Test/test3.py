# Import necessary modules and functions
from config import IMPLEMENTATIONS, load_data, format_time
from Plot import plot_weights
import unittest
import time
import gc
import os
import cupy as cp
import torch
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from Helper.LabelEncoder import encode_labels, LABEL_MAPPING
import numpy as np
from Plot.per_calss_accuracy import plot_per_class

# Define log file path
LOG_FILE = "test_results.log"

# Ensure log file exists
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w") as f:
        f.write("Timestamp, Implementation, Dataset, Status, Fit Time (s), Predict Time (s), Error\n")

dataset_path = "/home/leo/ZK-CIL-NIDS/dataset"

class TestPredictFitFunctions(unittest.TestCase):
    def log_result(self, implementation, dataset, status, fit_time=None, predict_time=None, error=""):
        """ Logs test results to a file. """
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(LOG_FILE, "a") as f:
            f.write(f"{timestamp}, {implementation}, {dataset}, {status}, {fit_time or ''}, {predict_time or ''}, {error}\n")
            print(f"{timestamp}, {implementation}, {dataset}, {status}, {fit_time or ''}, {predict_time or ''}, {error}\n")

    day_datasets = [
        #"TrainDay0_TestDay1234",
        #"TrainDay01_TestDay234",
        "TrainDay012_TestDay34",
        #"TrainDay0123_TestDay4",
        #"TrainDay0_DDos",
        #"TrainDay0_Dos",
        #"TrainDay0_FTPPatator",
        #"TrainDay0_Heartbleed",
        #"TrainDay0_Infiltration",
        #"TrainDay0_SSHPatator",
        #"TrainDay0_TestBotnet",
        #"TrainDay0_PortScan",
        #"TrainDay0_Web"
    ]

    def test_complete_days(self):
        module_name = "supervised_RF"
        module_path = "Implementation." + module_name
        module_path_novelty = "Implementation." + "OneClassSVN"

        with self.subTest(module=module_name):
            try:
                module = __import__(module_path_novelty, fromlist=["predict", "fit"])
                predict = getattr(module, "predict")
                fit = getattr(module, "fit")
            except Exception as e:
                print(f"Skipping {module_name} due to import error: {e}")
                #continue  # Skip this implementation if it fails to load

            for test_dataset in self.day_datasets:
                with self.subTest(dataset=test_dataset):
                    try:
                        print(f"Running {module_name} on {test_dataset}")

                        # Load training data
                        X_train, y_train, _ = load_data(dataset_path, test_dataset, "Train")
                        list_known = list(set(y_train))


                        start_time_fit = time.time()
                        # Fit RF-model
                        numeric_labels = encode_labels(y_train)
                        rf_model = RandomForestClassifier(n_estimators=100)
                        rf_model.fit(X_train, numeric_labels)

                        # Fit Novelty-model
                        novelty_model = fit(X_train, y_train)
                        end_time_fit = time.time()
                        execution_time_fit = end_time_fit - start_time_fit

                        # Free training data
                        del X_train, y_train, _
                        cp._default_memory_pool.free_all_blocks()
                        gc.collect()

                        # Load test data
                        X_test, y_test, Metadata = load_data(dataset_path, test_dataset, "Test")

                        # Predict
                        start_time_predict = time.time()
                        weights_rf_model = rf_model.predict_proba(X_test)
                        weights_novel_model = predict(X_test, Metadata, novelty_model)

                        #weights = predict(X_test, Metadata, model)
                        end_time_predict = time.time()
                        execution_time_predict = end_time_predict - start_time_predict

                        # Plot results
                        plot_infos = {
                            "algorithm_name": module_name,
                            "dataset_name": test_dataset,
                            "execution_time_fit": format_time(execution_time_fit),
                            "execution_time_predict": format_time(execution_time_predict)
                        }
                        #numeric_labels = np.where(y_test == "BENIGN", 0, 1)
                        #list_known = ["BENIGN"]
                        #weights_for_plot = 1 - weights[:,LABEL_MAPPING["BENIGN"]]
                        #plot_weights(y_test, weights_for_plot, module_name, Metadata, plot_infos, list_known)

                        plot_per_class(y_test, weights_rf_model, weights_novel_model)

                        # Log success
                        self.log_result(module_name, test_dataset, "PASSED", execution_time_fit, execution_time_predict)

                        # Free model memory
                        try:
                            rf_model.to("cpu")
                        except:
                            pass
                        del rf_model, X_test, y_test, Metadata, weights_rf_model

                        torch.cuda.empty_cache()
                        gc.collect()
                        cp._default_memory_pool.free_all_blocks()
                        torch.cuda.synchronize()

                    except Exception as e:
                        self.log_result(module_name, test_dataset, "FAILED", error=str(e))
                        self.fail(f"Failure in {module_name} - {test_dataset}: {e}")

