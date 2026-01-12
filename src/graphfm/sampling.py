from __future__ import annotations

import numpy as np

from .graphon import Graphon


def graphon_to_weighted_adjacency(graphon: Graphon, n: int) -> np.ndarray:
    """Use u_i = (i+1)/n and set A_ij = W(u_i, u_j) directly (no Bernoulli sampling)."""
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
