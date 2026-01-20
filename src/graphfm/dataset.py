from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .graphon import FourierGraphon, Graphon, StepGraphon
from .sampling import graphon_to_weighted_adjacency, normalize_shift_operator
from .pe import PEConfig, compute_pe


@dataclass
class GraphSample:
    adjacency: np.ndarray
    delta: np.ndarray
    label: int
    tokens: Optional[np.ndarray]


def apply_pe(samples: List[GraphSample], pe_cfg: PEConfig) -> None:
    for sample in samples:
        sample.tokens = compute_pe(sample.delta, pe_cfg)


def sample_graphs(
    graphons: Sequence[Graphon],
    sizes: Sequence[int],
    per_class: int,
    pe_cfg: PEConfig,
    rng: np.random.Generator,
) -> List[GraphSample]:
    samples = sample_graphs_raw(graphons, sizes, per_class, rng)
    apply_pe(samples, pe_cfg)
    return samples


def sample_graphs_raw(
    graphons: Sequence[Graphon],
    sizes: Sequence[int],
    per_class: int,
    rng: np.random.Generator,
) -> List[GraphSample]:
    samples: List[GraphSample] = []
    for c, w in enumerate(graphons):
        for n in sizes:
            for _ in range(per_class):
                a = graphon_to_weighted_adjacency(w, n)
                delta = normalize_shift_operator(a)
                samples.append(GraphSample(adjacency=a, delta=delta, label=c, tokens=None))
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
    samples = sample_with_allocation_raw(graphons, allocation, rng)
    apply_pe(samples, pe_cfg)
    return samples


def sample_with_allocation_raw(
    graphons: Sequence[Graphon],
    allocation: Dict[int, int],
    rng: np.random.Generator,
) -> List[GraphSample]:
    samples: List[GraphSample] = []
    for c, w in enumerate(graphons):
        for n, count in allocation.items():
            for _ in range(count):
                a = graphon_to_weighted_adjacency(w, n)
                delta = normalize_shift_operator(a)
                samples.append(GraphSample(adjacency=a, delta=delta, label=c, tokens=None))
    rng.shuffle(samples)
    return samples


def _pack_samples(samples: List[GraphSample]) -> Dict[str, np.ndarray]:
    return {
        "adjacency": np.array([s.adjacency for s in samples], dtype=object),
        "delta": np.array([s.delta for s in samples], dtype=object),
        "label": np.array([s.label for s in samples], dtype=np.int64),
    }


def _unpack_samples(data: np.lib.npyio.NpzFile, prefix: str) -> List[GraphSample]:
    adjacency = data[f"{prefix}_adjacency"]
    delta = data[f"{prefix}_delta"]
    labels = data[f"{prefix}_label"]
    tokens = data[f"{prefix}_tokens"] if f"{prefix}_tokens" in data else None
    samples: List[GraphSample] = []
    for idx in range(len(labels)):
        samples.append(
            GraphSample(
                adjacency=adjacency[idx],
                delta=delta[idx],
                label=int(labels[idx]),
                tokens=tokens[idx] if tokens is not None else None,
            )
        )
    return samples


def save_dataset(
    path: Path,
    train_samples: List[GraphSample],
    val_samples: List[GraphSample],
    test_samples: List[GraphSample],
) -> None:
    payload: Dict[str, np.ndarray] = {}
    for prefix, samples in (
        ("train", train_samples),
        ("val", val_samples),
        ("test", test_samples),
    ):
        packed = _pack_samples(samples)
        for key, value in packed.items():
            payload[f"{prefix}_{key}"] = value
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def load_dataset(path: Path) -> Tuple[List[GraphSample], List[GraphSample], List[GraphSample]]:
    data = np.load(path, allow_pickle=True)
    train_samples = _unpack_samples(data, "train")
    val_samples = _unpack_samples(data, "val")
    test_samples = _unpack_samples(data, "test")
    return train_samples, val_samples, test_samples
