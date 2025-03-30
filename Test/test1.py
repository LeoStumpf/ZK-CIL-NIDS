# Import necessary modules and functions
from config import IMPLEMENTATIONS, load_data, format_time
from Plot import plot_weights
import unittest
import time
import gc
import cupy as cp

# Define a test class for testing prediction and fitting functions
class TestPredictFitFunctions(unittest.TestCase):

    # List of datasets for different day combinations
    day_datasets = [
        "TrainDay0_TestDay1234",
        "TrainDay01_TestDay234",
        "TrainDay012_TestDay34",
        "TrainDay0123_TestDay4",
        "TrainDay0_DDos",
        "TrainDay0_Dos",
        "TrainDay0_FTPPatator",
        "TrainDay0_Heartbleed",
        "TrainDay0_Infiltration",
        "TrainDay0_SSHPatator",
        "TrainDay0_SSHPatator",
        "TrainDay0_Web"
    ]

    # Test method to check the complete workflow for each dataset
    def test_complete_days(self):

        # Iterate over each implementation module
        for module_name in IMPLEMENTATIONS:
            print(f"Run Operation {module_name}")
            # Construct the module path dynamically
            module_path = "Implementation." + module_name

            # Use subTest to isolate each module's test
            with self.subTest(module=module_path):
                # Dynamically import the module and get the 'predict' and 'fit' functions
                module = __import__(module_path, fromlist=["predict", "fit"])
                predict = getattr(module, "predict")
                fit = getattr(module, "fit")

                # Iterate over each dataset in the list
                for test_dataset in self.day_datasets:
                    print(f"Run Dataset {test_dataset}")
                    # Load training data: features (X_train), labels (y_train), and metadata
                    X_train, y_train, _ = load_data(test_dataset, "Train")
                    list_known = list(set(y_train))

                    # Train the model using the training data
                    start_time_fit = time.time()  # Start timer for fit
                    print("Fit")
                    model = fit(X_train, y_train)
                    #model = fit(X_train[:2000, :], y_train[:2000])
                    end_time_fit = time.time()  # End timer for fit
                    execution_time_fit = end_time_fit - start_time_fit  # Calculate fit time

                    # Clear training data from CPU and GPU memory
                    del X_train, y_train
                    cp._default_memory_pool.free_all_blocks()  # Free CUDA memory
                    gc.collect()  # Force garbage collection

                    # Load testing data: features (X_test), labels (y_test), and metadata
                    X_test, y_test, Metadata = load_data(test_dataset, "Test")

                    # Use the trained model to make predictions on the test data
                    start_time_predict = time.time()  # Start timer for predict
                    print("Predict")
                    weights = predict(X_test, Metadata, model)
                    #weights = predict(X_test[:200, :], Metadata[:200], model)
                    end_time_predict = time.time()  # End timer for predict
                    execution_time_predict = end_time_predict - start_time_predict  # Calculate predict time

                    # Plot the results (weights) for visualization
                    plot_infos = {
                        "algorithm_name": module_name,
                        "dataset_name": test_dataset,
                        "execution_time_fit": format_time(execution_time_fit),
                        "execution_time_predict": format_time(execution_time_predict)
                    }

                    plot_weights(y_test, weights, module_name, Metadata, plot_infos, list_known)
                    #plot_weights(y_test[:200], weights, module_name, Metadata[:200], plot_infos, list_known)

                    # Cleanup after each dataset
                    del X_test, y_test, Metadata, weights, model
                    cp._default_memory_pool.free_all_blocks()  # Clear CUDA cache
                    gc.collect()  # Collect garbage

# Entry point to run the tests
if __name__ == "__main__":
    unittest.main()