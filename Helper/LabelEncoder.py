import numpy as np

# Define label mapping
LABEL_MAPPING = {
    'BENIGN': 0,
    'DoS': 1,
    'DDoS': 2,
    'Web': 3,
    'Infiltration': 4,
    'Botnet': 5,
    'Portscan': 6,
    'FTP-Patator': 7,
    'SSH-Patator': 8,
    'Heartbleed': 9,
    'Unknown': 10
}

def encode_labels(y):
    """
    Encodes string labels into integers based on a predefined mapping.

    Parameters:
    - y (numpy array): Array of string labels.

    Returns:
    - numpy array of encoded labels.
    """
    return np.array([LABEL_MAPPING.get(label, -1) for label in y])  # -1 for unknown labels (if any)

# Example usage:
# y = np.array(["BENIGN", "DoS", "Botnet", "Unknown"])
# y_encoded = encode_labels(y)
# print(y_encoded)  # Output: [0 1 5 -1]  (-1 if "Unknown" is not in LABEL_MAPPING)
