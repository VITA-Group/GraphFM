from __future__ import annotations

from typing import Optional

import numpy as np

from .graphon import Graphon


def graphon_to_weighted_adjacency(
    graphon: Graphon,
    n: int,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Sample n node positions and set A_ij = W(u_i, u_j) directly (no Bernoulli sampling).

    Args:
        graphon: The graphon function W(u, v).
        n: Number of nodes.
        rng: Random generator. If provided, sample u_i ~ Uniform(0,1).
             If None, use deterministic u_i = (i+1)/n.
    """
    if rng is not None:
        u = rng.uniform(0.0, 1.0, size=n)
    else:
        u = (np.arange(n, dtype=np.float64) + 1.0) / float(n)
    uu = u[:, None]
    vv = u[None, :]
    a = graphon(uu, vv).astype(np.float64)
    a = 0.5 * (a + a.T)
    np.fill_diagonal(a, 0.0)
    return a


def normalize_shift_operator(a: np.ndarray) -> np.ndarray:
    """Definition 2 uses n * Delta = A; return Delta."""
    n = a.shape[0]
    return a / float(n)
