import cupy as cp
import numpy as np
from cuml.neighbors import NearestNeighbors

eps = 0.8


def generate_outliers(data):
    batch_size = 100

    n_samples, n_features = cp.shape(data)
    num_outliers = max(2 * n_samples, 1000)
    outliers = []

    # Use NearestNeighbors to ensure generated points are outside eps distance
    nn = NearestNeighbors(n_neighbors=1, metric='euclidean').fit(data)

    while len(outliers) < num_outliers:
        # Generate batch of random directions
        random_points = data[cp.random.randint(0, n_samples, size=batch_size)]
        random_directions = cp.random.randn(batch_size, n_features)
        random_directions = random_directions / cp.linalg.norm(random_directions, axis=1)[:, cp.newaxis]  # Normalize directions

        # Generate candidate outlier points at exactly `eps` distance away
        candidates = random_points + 1.2 * eps * random_directions

        # Ensure the candidate points are within the [0, 1] range for all dimensions
        candidates = cp.clip(candidates, 0, 1)

        # Check distances using NearestNeighbors
        distances, _ = nn.kneighbors(candidates)

        # Filter candidates that are outside eps distance
        valid_candidates = candidates[distances[:, 0] > eps]

        outliers.extend(valid_candidates)

        if len(outliers) > num_outliers:
            outliers = outliers[:num_outliers]

    return cp.array(outliers)
