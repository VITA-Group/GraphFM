from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Literal, Tuple

import numpy as np
import torch

from .graphon import StepGraphon
from .sampling import SamplingMode, graphon_to_weighted_adjacency


OrderingMethod = Literal["degree", "spectral", "usvt"]


# ============================================================================
# CPU implementations (for reference / small graphs)
# ============================================================================


def _get_node_order_degree_cpu(a: np.ndarray) -> np.ndarray:
    """Order nodes by degree (ascending) - CPU version."""
    deg = a.sum(axis=1)
    return np.argsort(deg)


def _get_node_order_spectral_cpu(a: np.ndarray) -> np.ndarray:
    """Order nodes by Fiedler vector - CPU version."""
    n = a.shape[0]
    if n <= 2:
        return np.arange(n)

    deg = a.sum(axis=1)
    deg = np.maximum(deg, 1e-10)
    d_inv_sqrt = 1.0 / np.sqrt(deg)
    l_norm = np.eye(n) - (d_inv_sqrt[:, None] * a * d_inv_sqrt[None, :])

    _, eigenvectors = np.linalg.eigh(l_norm)
    fiedler_vector = eigenvectors[:, 1]

    return np.argsort(fiedler_vector)


def _accumulate_bins_cpu(
    a_sorted: np.ndarray,
    bins: int,
    accum: np.ndarray,
    counts: np.ndarray,
) -> None:
    """Accumulate edge weights into bins - CPU version (vectorized)."""
    n = a_sorted.shape[0]
    bin_idx = (np.arange(n) * bins) // n
    # Vectorized accumulation using np.add.at
    bi = bin_idx[:, None].repeat(n, axis=1)  # (n, n)
    bj = bin_idx[None, :].repeat(n, axis=0)  # (n, n)
    np.add.at(accum, (bi.ravel(), bj.ravel()), a_sorted.ravel())
    np.add.at(counts, (bi.ravel(), bj.ravel()), 1.0)


# ============================================================================
# GPU implementations
# ============================================================================


def _get_node_orders_degree_gpu(
    adjacencies_t: torch.Tensor,
) -> torch.Tensor:
    """Batch compute node orders by degree - GPU version.

    Args:
        adjacencies_t: (B, n, n) batch of adjacency matrices

    Returns:
        (B, n) tensor of node orderings
    """
    deg = adjacencies_t.sum(dim=2)  # (B, n)
    return torch.argsort(deg, dim=1)  # (B, n)


def _get_node_orders_spectral_gpu(
    adjacencies_t: torch.Tensor,
) -> torch.Tensor:
    """Batch compute node orders by Fiedler vector - GPU version.

    Args:
        adjacencies_t: (B, n, n) batch of adjacency matrices

    Returns:
        (B, n) tensor of node orderings
    """
    B, n, _ = adjacencies_t.shape
    if n <= 2:
        return torch.arange(n, device=adjacencies_t.device).unsqueeze(0).expand(B, -1)

    # Compute degrees
    deg = adjacencies_t.sum(dim=2)  # (B, n)
    deg = torch.clamp(deg, min=1e-10)

    # D^{-1/2}
    d_inv_sqrt = 1.0 / torch.sqrt(deg)  # (B, n)

    # Normalized Laplacian: L = I - D^{-1/2} A D^{-1/2}
    # (B, n, 1) * (B, n, n) * (B, 1, n) = (B, n, n)
    eye = torch.eye(n, device=adjacencies_t.device, dtype=adjacencies_t.dtype)
    l_norm = eye - (d_inv_sqrt.unsqueeze(2) * adjacencies_t * d_inv_sqrt.unsqueeze(1))

    # Batch eigendecomposition
    _, eigenvectors = torch.linalg.eigh(l_norm)  # (B, n, n)

    # Fiedler vector (second column)
    fiedler_vectors = eigenvectors[:, :, 1]  # (B, n)

    return torch.argsort(fiedler_vectors, dim=1)  # (B, n)


def _reorder_adjacencies_gpu(
    adjacencies_t: torch.Tensor,
    orders: torch.Tensor,
) -> torch.Tensor:
    """Reorder adjacency matrices according to given orderings - GPU version.

    Args:
        adjacencies_t: (B, n, n) batch of adjacency matrices
        orders: (B, n) batch of node orderings

    Returns:
        (B, n, n) reordered adjacency matrices
    """
    B, n, _ = adjacencies_t.shape

    # Advanced indexing for batch reordering
    batch_idx = torch.arange(B, device=adjacencies_t.device).view(B, 1, 1)
    row_idx = orders.unsqueeze(2).expand(B, n, n)
    col_idx = orders.unsqueeze(1).expand(B, n, n)

    return adjacencies_t[batch_idx, row_idx, col_idx]


def _accumulate_bins_gpu(
    a_sorted_batch: torch.Tensor,
    bins: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Accumulate edge weights into bins - GPU batch version.

    Args:
        a_sorted_batch: (B, n, n) batch of sorted adjacency matrices
        bins: Number of bins
        device: Torch device

    Returns:
        (accum, counts) both of shape (bins, bins)
    """
    B, n, _ = a_sorted_batch.shape

    # Compute bin indices for this size
    bin_idx = (torch.arange(n, device=device) * bins) // n  # (n,)

    # Create bin index matrices
    bi = bin_idx.view(1, n, 1).expand(B, n, n)  # (B, n, n)
    bj = bin_idx.view(1, 1, n).expand(B, n, n)  # (B, n, n)

    # Flatten for scatter_add
    flat_idx = bi * bins + bj  # (B, n, n) -> linear index into (bins, bins)

    # Accumulate using scatter_add
    accum = torch.zeros(bins * bins, device=device, dtype=torch.float64)
    counts = torch.zeros(bins * bins, device=device, dtype=torch.float64)

    accum.scatter_add_(0, flat_idx.reshape(-1), a_sorted_batch.reshape(-1).double())
    counts.scatter_add_(0, flat_idx.reshape(-1), torch.ones(B * n * n, device=device, dtype=torch.float64))

    return accum.view(bins, bins), counts.view(bins, bins)


# ============================================================================
# USVT (Universal Singular Value Thresholding) implementations
# ============================================================================


def _downsample_to_bins(matrix: np.ndarray, bins: int) -> np.ndarray:
    """Downsample a matrix to bins x bins by averaging blocks."""
    n = matrix.shape[0]
    result = np.zeros((bins, bins), dtype=np.float64)

    for i in range(bins):
        for j in range(bins):
            i_start = (i * n) // bins
            i_end = ((i + 1) * n) // bins
            j_start = (j * n) // bins
            j_end = ((j + 1) * n) // bins
            result[i, j] = matrix[i_start:i_end, j_start:j_end].mean()

    return result


def _estimate_usvt_gpu(
    adjacencies: List[np.ndarray],
    bins: int,
    threshold: float = 2.02,
    device: str = "cuda",
) -> StepGraphon:
    """Estimate graphon using Universal Singular Value Thresholding - GPU version.

    Reference:
    Chatterjee, Sourav. "Matrix estimation by universal singular value thresholding."
    The Annals of Statistics 43.1 (2015): 177-214.

    Args:
        adjacencies: List of adjacency matrices (can have different sizes).
        bins: Number of bins for the step graphon.
        threshold: Threshold multiplier for singular values (default 2.02).
        device: Torch device.

    Returns:
        Estimated StepGraphon.
    """
    dev = torch.device(device)

    # Find max size and pad all matrices to that size
    max_n = max(a.shape[0] for a in adjacencies)

    padded = []
    for a in adjacencies:
        n = a.shape[0]
        if n < max_n:
            pad_a = np.zeros((max_n, max_n), dtype=a.dtype)
            pad_a[:n, :n] = a
            padded.append(pad_a)
        else:
            padded.append(a)

    # Stack and move to device
    adj_batch = np.stack(padded)
    adj_t = torch.from_numpy(adj_batch).to(dev, dtype=torch.float32)

    # Average adjacency matrices
    avg_adj = torch.mean(adj_t, dim=0)  # (max_n, max_n)

    # SVD
    u, s, vh = torch.linalg.svd(avg_adj, full_matrices=False)

    # Threshold singular values
    singular_threshold = threshold * (max_n ** 0.5)
    s_thresholded = torch.where(s < singular_threshold, torch.zeros_like(s), s)

    # Reconstruct
    graphon = u @ torch.diag(s_thresholded) @ vh

    # Clip to [0, 1]
    graphon = torch.clamp(graphon, 0.0, 1.0)

    # Ensure symmetry
    graphon = 0.5 * (graphon + graphon.T)

    # Downsample to bins x bins for StepGraphon compatibility
    graphon_np = graphon.cpu().numpy()
    step_matrix = _downsample_to_bins(graphon_np, bins)

    return StepGraphon(bins=bins, matrix=step_matrix)


def _estimate_usvt_cpu(
    adjacencies: List[np.ndarray],
    bins: int,
    threshold: float = 2.02,
) -> StepGraphon:
    """Estimate graphon using Universal Singular Value Thresholding - CPU version.

    Reference:
    Chatterjee, Sourav. "Matrix estimation by universal singular value thresholding."
    The Annals of Statistics 43.1 (2015): 177-214.

    Args:
        adjacencies: List of adjacency matrices (can have different sizes).
        bins: Number of bins for the step graphon.
        threshold: Threshold multiplier for singular values (default 2.02).

    Returns:
        Estimated StepGraphon.
    """
    # Find max size and pad all matrices to that size
    max_n = max(a.shape[0] for a in adjacencies)

    padded = []
    for a in adjacencies:
        n = a.shape[0]
        if n < max_n:
            pad_a = np.zeros((max_n, max_n), dtype=a.dtype)
            pad_a[:n, :n] = a
            padded.append(pad_a)
        else:
            padded.append(a)

    # Average adjacency matrices
    avg_adj = np.mean(np.stack(padded), axis=0)

    # SVD
    u, s, vh = np.linalg.svd(avg_adj, full_matrices=False)

    # Threshold singular values
    singular_threshold = threshold * (max_n ** 0.5)
    s_thresholded = np.where(s < singular_threshold, 0.0, s)

    # Reconstruct
    graphon = u @ np.diag(s_thresholded) @ vh

    # Clip to [0, 1] and ensure symmetry
    graphon = np.clip(graphon, 0.0, 1.0)
    graphon = 0.5 * (graphon + graphon.T)

    # Downsample to bins x bins for StepGraphon compatibility
    step_matrix = _downsample_to_bins(graphon, bins)

    return StepGraphon(bins=bins, matrix=step_matrix)


def estimate_step_graphon_gpu(
    adjacencies: List[np.ndarray],
    bins: int,
    method: OrderingMethod = "spectral",
    device: str = "cuda",
) -> StepGraphon:
    """Estimate a step graphon from observed graphs - GPU accelerated.

    Args:
        adjacencies: List of adjacency matrices (can have different sizes).
        bins: Number of bins for the step graphon.
        method: Node ordering method ("degree", "spectral", or "usvt").
        device: Torch device ("cuda" or "cpu").

    Returns:
        Estimated StepGraphon.
    """
    # Handle USVT separately (different algorithm)
    if method == "usvt":
        return _estimate_usvt_gpu(adjacencies, bins, device=device)

    dev = torch.device(device)

    # Group adjacencies by size for batch processing
    by_size: Dict[int, List[np.ndarray]] = defaultdict(list)
    for a in adjacencies:
        by_size[a.shape[0]].append(a)

    total_accum = torch.zeros(bins, bins, device=dev, dtype=torch.float64)
    total_counts = torch.zeros(bins, bins, device=dev, dtype=torch.float64)

    order_fn = _get_node_orders_degree_gpu if method == "degree" else _get_node_orders_spectral_gpu

    for _, adj_list in by_size.items():
        # Stack into batch tensor
        adj_batch = np.stack(adj_list)
        adj_t = torch.from_numpy(adj_batch).to(dev, dtype=torch.float32)

        # Compute orderings
        orders = order_fn(adj_t)

        # Reorder adjacencies
        adj_sorted = _reorder_adjacencies_gpu(adj_t, orders)

        # Accumulate into bins
        accum, counts = _accumulate_bins_gpu(adj_sorted, bins, dev)
        total_accum += accum
        total_counts += counts

    # Compute final matrix
    mat = torch.zeros_like(total_accum)
    mask = total_counts > 0
    mat[mask] = total_accum[mask] / total_counts[mask]
    mat = 0.5 * (mat + mat.T)

    return StepGraphon(bins=bins, matrix=mat.cpu().numpy())


# ============================================================================
# Unified API
# ============================================================================


def estimate_step_graphon(
    adjacencies: List[np.ndarray],
    bins: int,
    method: OrderingMethod = "spectral",
    device: str = "cpu",
) -> StepGraphon:
    """Estimate a step graphon from observed graphs.

    Args:
        adjacencies: List of adjacency matrices (can have different sizes).
        bins: Number of bins for the step graphon.
        method: Node ordering method:
            - "degree": Order by node degree (fast, simple)
            - "spectral": Order by Fiedler vector (better for complex structures)
            - "usvt": Universal Singular Value Thresholding (direct estimation)
        device: Device for computation ("cpu" or "cuda").

    Returns:
        Estimated StepGraphon.
    """
    # Handle USVT separately (different algorithm)
    if method == "usvt":
        if device != "cpu":
            return _estimate_usvt_gpu(adjacencies, bins, device=device)
        return _estimate_usvt_cpu(adjacencies, bins)

    if device != "cpu":
        return estimate_step_graphon_gpu(adjacencies, bins, method, device)

    # CPU fallback
    accum = np.zeros((bins, bins), dtype=np.float64)
    counts = np.zeros((bins, bins), dtype=np.float64)

    order_fn = _get_node_order_degree_cpu if method == "degree" else _get_node_order_spectral_cpu

    for a in adjacencies:
        order = order_fn(a)
        a_sorted = a[order][:, order]
        _accumulate_bins_cpu(a_sorted, bins, accum, counts)

    mat = np.divide(accum, counts, out=np.zeros_like(accum), where=counts > 0)
    mat = 0.5 * (mat + mat.T)
    return StepGraphon(bins=bins, matrix=mat)


def synthesize_from_step(
    step_graphon: StepGraphon,
    n: int,
    rng: np.random.Generator,
    sampling_mode: SamplingMode = "uniform_value",
) -> np.ndarray:
    """Synthesize a graph from a step graphon.

    Args:
        step_graphon: The step graphon to sample from.
        n: Number of nodes in the output graph.
        rng: Random number generator.
        sampling_mode: Sampling mode for edge generation.

    Returns:
        Adjacency matrix of shape (n, n).
    """
    return graphon_to_weighted_adjacency(step_graphon, n, rng=rng, sampling_mode=sampling_mode)
