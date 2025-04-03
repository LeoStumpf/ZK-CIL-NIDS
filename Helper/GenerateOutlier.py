import cupy as cp
import numpy as np
from cuml.neighbors import NearestNeighbors
import faiss
import torch

eps = 0.8

def _generate_random(samples_per_method, n_features):
    return cp.random.uniform(0, 1, (samples_per_method, n_features))


def _generate_mirrored(samples_per_method, data):
    idx = cp.random.choice(len(data), samples_per_method)
    return 1 - data[idx]  # Geometric mirroring


def _generate_histogram_based(samples_per_method, data, bins=50):
    """
    Generates outliers using complementary histograms.

    Parameters:
    - samples_per_method: Number of outliers to generate
    - data: Input data (cupy array)
    - bins: Number of bins for histogram estimation

    Returns:
    - cupy array of generated outliers
    """
    n_samples, n_features = data.shape
    hist_outliers = []

    for f in range(n_features):
        hist, bin_edges = cp.histogram(data[:, f], bins=bins, density=True)
        hist = hist / cp.sum(hist)  # Normalize histogram
        complementary_hist = 1 - hist  # Complementary histogram
        complementary_hist /= cp.sum(complementary_hist)  # Re-normalize

        sampled_bins = cp.random.choice(len(bin_edges) - 1, size=samples_per_method, p=complementary_hist)

        # Sample uniformly from the selected bin range
        outlier_feature_values = bin_edges[sampled_bins] + cp.random.uniform(0, 1, samples_per_method) * (
                bin_edges[sampled_bins + 1] - bin_edges[sampled_bins]
        )
        hist_outliers.append(outlier_feature_values)

    return cp.column_stack(hist_outliers)


def _generate_directional(samples_per_method, n_features, data_tensor, device, n_samples, batch_size, multiplier=1.2):
    """Process one safe batch"""
    # Calculate distances in chunks
    chunk_size = min(1024, batch_size)  # Process 1024 distance comparisons at a time

    idx = torch.randint(0, n_samples, (batch_size,), device=device)
    base_points = data_tensor[idx]
    directions = torch.randn(batch_size, n_features, device=device)
    directions = directions / torch.norm(directions, dim=1, keepdim=True)
    candidates = base_points + multiplier * directions
    candidates = torch.clamp(candidates, 0, 1)

    # Process distance calculations in chunks
    mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
    for i in range(0, n_samples, chunk_size):
        chunk = data_tensor[i:i + chunk_size]
        dists = torch.cdist(candidates, chunk)
        min_dists = torch.min(dists, dim=1).values if i == 0 else torch.minimum(
            min_dists,
            torch.min(dists, dim=1).values
        )

    mask = min_dists > 1.0  # Using fixed eps=1.0 for memory safety
    return cp.asarray(candidates[mask].cpu().numpy())


def generate_outliers(data, target_outliers=None):
    """
    Generates outliers scaled to input data size with balanced methods.

    Parameters:
    - data: Input data (numpy or cupy array)
    - min_samples: Minimum samples per method (default: 1000)
    - batch_size: Processing batch size (default: 1000)
    - target_ratio: Desired outlier/inlier ratio (default: 0.5 = 50% outliers)

    Returns:
    - cupy array of combined outliers
    """
    min_samples = 1000
    batch_size = 1000
    target_ratio = 0.5

    # Convert to PyTorch tensor on GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_tensor = torch.from_numpy(cp.asnumpy(data)).float().to(device)
    n_samples, n_features = data_tensor.shape

    # Calculate target samples based on input size and ratio
    if target_outliers is None:
        target_outliers = max(min_samples, int(n_samples * target_ratio))
        target_outliers = int(min(target_outliers, 1e6)) # limit number of outliers due to memory constraints
    samples_per_method = max(min_samples, int(target_outliers / 3))  # Split across 3 methods

    # Generate outliers using all three methods
    dir_outliers = _generate_directional(
        int(min(samples_per_method, target_outliers)),
        n_features, data_tensor, device, n_samples, batch_size
    )

    remaining = target_outliers - len(dir_outliers)
    hist_outliers = _generate_histogram_based(
        int(min(samples_per_method, max(0, remaining))),
        data,
        bins=50
    )

    remaining = remaining - len(hist_outliers)
    mirrored_outliers = _generate_mirrored(
        int(min(samples_per_method/2, max(0, remaining))),
        data
    )

    remaining = remaining - len(mirrored_outliers)
    rand_outliers = _generate_random(
        int(max(0, remaining)),
        n_features
    )

    # Combine and trim to exact target
    combined = cp.vstack([dir_outliers, rand_outliers, hist_outliers, mirrored_outliers])
    return combined[:target_outliers]  # Ensure exact count

