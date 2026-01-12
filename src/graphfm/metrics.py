from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np


def sliced_wasserstein(
    x: np.ndarray,
    y: np.ndarray,
    projections: int = 50,
    rng: np.random.Generator | None = None,
) -> float:
    if rng is None:
        rng = np.random.default_rng(0)
    dim = x.shape[1]
    dirs = rng.normal(size=(projections, dim))
    dirs = dirs / (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12)
    x_proj = x @ dirs.T
    y_proj = y @ dirs.T
    x_sorted = np.sort(x_proj, axis=0)
    y_sorted = np.sort(y_proj, axis=0)
    if x_sorted.shape[0] != y_sorted.shape[0]:
        n = min(x_sorted.shape[0], y_sorted.shape[0])
        x_sorted = x_sorted[:n]
        y_sorted = y_sorted[:n]
    return float(np.mean(np.abs(x_sorted - y_sorted)))


def discrepancy_set(
    token_sets_a: List[np.ndarray],
    token_sets_b: List[np.ndarray],
    samples_per_graph: int,
    projections: int = 50,
    rng: np.random.Generator | None = None,
) -> float:
    if rng is None:
        rng = np.random.default_rng(0)
    tokens_a = []
    for tokens in token_sets_a:
        idx = rng.choice(tokens.shape[0], size=min(samples_per_graph, tokens.shape[0]), replace=False)
        tokens_a.append(tokens[idx])
    tokens_b = []
    for tokens in token_sets_b:
        idx = rng.choice(tokens.shape[0], size=min(samples_per_graph, tokens.shape[0]), replace=False)
        tokens_b.append(tokens[idx])
    a = np.concatenate(tokens_a, axis=0)
    b = np.concatenate(tokens_b, axis=0)
    if a.shape[0] != b.shape[0]:
        n = min(a.shape[0], b.shape[0])
        idx_a = rng.choice(a.shape[0], size=n, replace=False)
        idx_b = rng.choice(b.shape[0], size=n, replace=False)
        a = a[idx_a]
        b = b[idx_b]
    return sliced_wasserstein(a, b, projections=projections, rng=rng)


@dataclass(frozen=True)
class EigengapStats:
    min_gap: float
    gap_k: float


def eigengap_stats(eigenvalues: np.ndarray, k: int) -> EigengapStats:
    evals = np.sort(eigenvalues)
    gaps = np.diff(evals)
    k = min(k, len(evals) - 1)
    if k <= 0:
        return EigengapStats(min_gap=float("nan"), gap_k=float("nan"))
    min_gap = float(np.min(gaps[:k]))
    gap_k = float(gaps[k - 1])
    return EigengapStats(min_gap=min_gap, gap_k=gap_k)
