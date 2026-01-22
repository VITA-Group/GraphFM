from __future__ import annotations

from typing import List, Optional

import numpy as np

from .graphon import StepGraphon
from .sampling import graphon_to_weighted_adjacency


def estimate_step_graphon(adjacencies: List[np.ndarray], bins: int) -> StepGraphon:
    accum = np.zeros((bins, bins), dtype=np.float64)
    counts = np.zeros((bins, bins), dtype=np.float64)
    for a in adjacencies:
        n = a.shape[0]
        deg = a.sum(axis=1)
        order = np.argsort(deg)
        a_sorted = a[order][:, order]
        bin_idx = (np.arange(n) * bins) // n
        for i in range(n):
            bi = bin_idx[i]
            for j in range(n):
                bj = bin_idx[j]
                accum[bi, bj] += a_sorted[i, j]
                counts[bi, bj] += 1.0
    mat = np.divide(accum, counts, out=np.zeros_like(accum), where=counts > 0)
    mat = 0.5 * (mat + mat.T)
    return StepGraphon(bins=bins, matrix=mat)


def synthesize_from_step(
    step_graphon: StepGraphon,
    n: int,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    return graphon_to_weighted_adjacency(step_graphon, n, rng=rng)
