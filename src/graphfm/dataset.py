from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from .graphon import FourierGraphon, Graphon, StepGraphon
from .sampling import graphon_to_weighted_adjacency, normalize_shift_operator
from .pe import PEConfig, compute_pe


@dataclass
class GraphSample:
    adjacency: np.ndarray
    delta: np.ndarray
    label: int
    tokens: np.ndarray


def sample_graphs(
    graphons: Sequence[Graphon],
    sizes: Sequence[int],
    per_class: int,
    pe_cfg: PEConfig,
    rng: np.random.Generator,
) -> List[GraphSample]:
    samples: List[GraphSample] = []
    for c, w in enumerate(graphons):
        for n in sizes:
            for _ in range(per_class):
                a = graphon_to_weighted_adjacency(w, n)
                delta = normalize_shift_operator(a)
                tokens = compute_pe(delta, pe_cfg)
                samples.append(GraphSample(adjacency=a, delta=delta, label=c, tokens=tokens))
    rng.shuffle(samples)
    return samples


def size_allocation_path(
    sizes_small: Sequence[int],
    sizes_large: Sequence[int],
    total_budget: int,
    lambda_mix: float,
) -> Dict[int, int]:
    sizes_small = list(sizes_small)
    sizes_large = list(sizes_large)
    budget_small = int(round(total_budget * (1.0 - lambda_mix)))
    budget_large = total_budget - budget_small
    alloc: Dict[int, int] = {}

    def allocate(sizes: Sequence[int], budget: int) -> None:
        if budget <= 0:
            return
        size_cycle = list(sizes)
        idx = 0
        while budget > 0:
            n = size_cycle[idx % len(size_cycle)]
            if budget - n < 0:
                break
            alloc[n] = alloc.get(n, 0) + 1
            budget -= n
            idx += 1

    allocate(sizes_small, budget_small)
    allocate(sizes_large, budget_large)
    return alloc


def sample_with_allocation(
    graphons: Sequence[Graphon],
    allocation: Dict[int, int],
    pe_cfg: PEConfig,
    rng: np.random.Generator,
) -> List[GraphSample]:
    samples: List[GraphSample] = []
    for c, w in enumerate(graphons):
        for n, count in allocation.items():
            for _ in range(count):
                a = graphon_to_weighted_adjacency(w, n)
                delta = normalize_shift_operator(a)
                tokens = compute_pe(delta, pe_cfg)
                samples.append(GraphSample(adjacency=a, delta=delta, label=c, tokens=tokens))
    rng.shuffle(samples)
    return samples
