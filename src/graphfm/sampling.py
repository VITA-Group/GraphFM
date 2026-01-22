from __future__ import annotations

from typing import Literal

import numpy as np

from .graphon import Graphon


SamplingMode = Literal["uniform_value", "bin_value", "uniform_bernoulli"]


def graphon_to_weighted_adjacency(
    graphon: Graphon,
    n: int,
    rng: np.random.Generator,
    sampling_mode: SamplingMode = "uniform_value",
) -> np.ndarray:
    """Sample n node positions and compute adjacency matrix from graphon.

    Args:
        graphon: The graphon function W(u, v).
        n: Number of nodes.
        rng: Random generator (required for uniform modes).
        sampling_mode: Sampling strategy:
            - "uniform_value": u_i ~ Uniform(0,1), A_ij = W(u_i, u_j)
            - "bin_value": u_i = (i+1)/n, A_ij = W(u_i, u_j)
            - "uniform_bernoulli": u_i ~ Uniform(0,1), A_ij ~ Bernoulli(W(u_i, u_j))
    """
    # Node position sampling
    if sampling_mode == "bin_value":
        u = (np.arange(n, dtype=np.float64) + 1.0) / float(n)
    else:  # uniform_value or uniform_bernoulli
        u = rng.uniform(0.0, 1.0, size=n)

    uu, vv = np.meshgrid(u, u, indexing="ij")
    w = graphon(uu, vv).astype(np.float64)

    # Edge weight computation
    if sampling_mode == "uniform_bernoulli":
        a = (rng.random(size=w.shape) < w).astype(np.float64)
    else:  # uniform_value or bin_value
        a = w

    a = 0.5 * (a + a.T)
    np.fill_diagonal(a, 0.0)
    return a


def normalize_shift_operator(a: np.ndarray) -> np.ndarray:
    """Definition 2 uses n * Delta = A; return Delta."""
    n = a.shape[0]
    return a / float(n)
