from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm

from .graphon import (
    ControlledFourierGraphon,
    FourierGraphon,
    Graphon,
    StepGraphon,
    make_controlled_fourier_graphons,
    make_fourier_graphons,
)
from .sampling import (
    SamplingMode,
    graphon_to_weighted_adjacency,
    normalize_shift_operator,
)
from .pe import PEConfig, compute_pe


@dataclass
class GraphSample:
    adjacency: np.ndarray
    delta: np.ndarray
    label: int
    tokens: Optional[np.ndarray]


@dataclass
class DatasetConfig:
    num_classes: int = 4
    rho: float = 0.5
    num_terms: int = 5
    coeff_scale: float = 0.2
    train_sizes: Sequence[int] = (64, 128, 256, 512)
    test_sizes: Sequence[int] = (64, 128, 256, 512, 768, 1024)
    per_class_train: int = 3
    per_class_test: int = 2
    total_budget: int = 10_000
    seed: int = 0
    lambda_mix: float = 0.0
    sampling_mode: SamplingMode = "uniform_value"
    graphon_type: str = "fourier"  # "fourier" or "controlled_fourier"


def apply_pe(samples: List[GraphSample], pe_cfg: PEConfig) -> None:
    for sample in samples:
        sample.tokens = compute_pe(sample.delta, pe_cfg)


def sample_graphs(
    graphons: Sequence[Graphon],
    sizes: Sequence[int],
    per_class: int,
    pe_cfg: PEConfig,
    rng: np.random.Generator,
    sampling_mode: SamplingMode = "uniform_value",
) -> List[GraphSample]:
    samples = sample_graphs_raw(graphons, sizes, per_class, rng, sampling_mode=sampling_mode)
    apply_pe(samples, pe_cfg)
    return samples


def sample_graphs_raw(
    graphons: Sequence[Graphon],
    sizes: Sequence[int],
    per_class: int,
    rng: np.random.Generator,
    show_progress: bool = True,
    sampling_mode: SamplingMode = "uniform_value",
) -> List[GraphSample]:
    samples: List[GraphSample] = []
    total = len(graphons) * len(sizes) * per_class
    pbar = tqdm(total=total, desc="Sampling graphs", disable=not show_progress)
    for c, w in enumerate(graphons):
        for n in sizes:
            for _ in range(per_class):
                a = graphon_to_weighted_adjacency(w, n, rng=rng, sampling_mode=sampling_mode)
                delta = normalize_shift_operator(a)
                samples.append(GraphSample(adjacency=a, delta=delta, label=c, tokens=None))
                pbar.update(1)
    pbar.close()
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
    sampling_mode: SamplingMode = "uniform_value",
) -> List[GraphSample]:
    samples = sample_with_allocation_raw(graphons, allocation, rng, sampling_mode=sampling_mode)
    apply_pe(samples, pe_cfg)
    return samples


def sample_with_allocation_raw(
    graphons: Sequence[Graphon],
    allocation: Dict[int, int],
    rng: np.random.Generator,
    show_progress: bool = True,
    sampling_mode: SamplingMode = "uniform_value",
) -> List[GraphSample]:
    samples: List[GraphSample] = []
    total = len(graphons) * sum(allocation.values())
    pbar = tqdm(total=total, desc="Sampling graphs", disable=not show_progress)
    for c, w in enumerate(graphons):
        for n, count in allocation.items():
            for _ in range(count):
                a = graphon_to_weighted_adjacency(w, n, rng=rng, sampling_mode=sampling_mode)
                delta = normalize_shift_operator(a)
                samples.append(GraphSample(adjacency=a, delta=delta, label=c, tokens=None))
                pbar.update(1)
    pbar.close()
    rng.shuffle(samples)
    return samples


def _pack_samples(samples: List[GraphSample], desc: str = "Packing") -> Dict[str, np.ndarray]:
    """Pack samples by size for efficient storage (no pickle needed)."""
    from collections import defaultdict

    by_size: Dict[int, List[Tuple[int, GraphSample]]] = defaultdict(list)
    for idx, s in enumerate(samples):
        n = s.adjacency.shape[0]
        by_size[n].append((idx, s))

    payload: Dict[str, np.ndarray] = {}
    order = []
    size_list = []

    for n in tqdm(sorted(by_size.keys()), desc=desc):
        group = by_size[n]
        indices = [g[0] for g in group]
        order.extend(indices)
        size_list.extend([n] * len(group))

        # Regular 3D arrays - no pickle needed
        payload[f"adj_{n}"] = np.stack([g[1].adjacency for g in group]).astype(np.float32)
        payload[f"delta_{n}"] = np.stack([g[1].delta for g in group]).astype(np.float32)
        payload[f"label_{n}"] = np.array([g[1].label for g in group], dtype=np.int64)
        # Handle tokens if present
        if group[0][1].tokens is not None:
            payload[f"tokens_{n}"] = np.stack([g[1].tokens for g in group]).astype(np.float32)

    payload["order"] = np.array(order, dtype=np.int64)
    payload["sizes"] = np.array(size_list, dtype=np.int64)
    return payload


def _unpack_samples(data: np.lib.npyio.NpzFile, prefix: str) -> List[GraphSample]:
    """Fast unpacking: direct array slicing, no pickle."""
    order = data[f"{prefix}_order"]
    sizes = data[f"{prefix}_sizes"]

    # Preload all size groups into memory
    unique_sizes = np.unique(sizes)
    size_data: Dict[int, Dict] = {}
    for n in unique_sizes:
        n = int(n)
        has_tokens = f"{prefix}_tokens_{n}" in data
        size_data[n] = {
            "adj": data[f"{prefix}_adj_{n}"],
            "delta": data[f"{prefix}_delta_{n}"],
            "label": data[f"{prefix}_label_{n}"],
            "tokens": data[f"{prefix}_tokens_{n}"] if has_tokens else None,
            "idx": 0,
        }

    # Rebuild in original order
    samples: List[Optional[GraphSample]] = [None] * len(order)
    for pos in range(len(order)):
        n = int(sizes[pos])
        d = size_data[n]
        i = d["idx"]
        samples[order[pos]] = GraphSample(
            adjacency=d["adj"][i],
            delta=d["delta"][i],
            label=int(d["label"][i]),
            tokens=d["tokens"][i] if d["tokens"] is not None else None,
        )
        d["idx"] += 1

    return samples  # type: ignore


def save_dataset(
    path: Path,
    train_samples: List[GraphSample],
    val_samples: List[GraphSample],
    test_samples: List[GraphSample],
    compress: bool = True,
) -> None:
    payload: Dict[str, np.ndarray] = {}
    for prefix, samples in (
        ("train", train_samples),
        ("val", val_samples),
        ("test", test_samples),
    ):
        packed = _pack_samples(samples, desc=f"Packing {prefix}")
        for key, value in packed.items():
            payload[f"{prefix}_{key}"] = value
    path.parent.mkdir(parents=True, exist_ok=True)
    print("Saving..." if not compress else "Compressing and saving...")
    if compress:
        np.savez_compressed(path, **payload)
    else:
        np.savez(path, **payload)


def load_dataset(path: Path) -> Tuple[List[GraphSample], List[GraphSample], List[GraphSample]]:
    data = np.load(path)
    train_samples = _unpack_samples(data, "train")
    val_samples = _unpack_samples(data, "val")
    test_samples = _unpack_samples(data, "test")
    return train_samples, val_samples, test_samples


# --- Dataset generation functions (moved from experiments.py) ---


def _split_train_val(samples: List[GraphSample], val_frac: float, rng: np.random.Generator):
    rng.shuffle(samples)
    cut = int(round(len(samples) * (1.0 - val_frac)))
    return samples[:cut], samples[cut:]


def _cache_key(params: Dict) -> str:
    payload = json.dumps(params, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]


def _fmt_float(value: float) -> str:
    return f"{value:.3f}".replace(".", "p")


def _dataset_counts(config: DatasetConfig) -> Tuple[int, int, int]:
    allocation = size_allocation_path(
        sizes_small=(config.train_sizes[0], config.train_sizes[1]),
        sizes_large=(config.train_sizes[2], config.train_sizes[3]),
        total_budget=config.total_budget,
        lambda_mix=config.lambda_mix,
    )
    total_train = config.num_classes * sum(allocation.values())
    train_cut = int(round(total_train * (1.0 - 0.2)))
    val_count = total_train - train_cut
    test_count = config.num_classes * len(config.test_sizes) * config.per_class_test
    return train_cut, val_count, test_count


def _dataset_cache_path(cache_dir: Path, config: DatasetConfig) -> Path:
    train_count, val_count, test_count = _dataset_counts(config)
    total = train_count + val_count + test_count
    test_ratio = test_count / max(total, 1)
    params = {
        "config": config.__dict__,
    }
    key = _cache_key(params)
    name = (
        f"dataset_ns{total}_lm{_fmt_float(config.lambda_mix)}_tr{_fmt_float(test_ratio)}"
        f"_seed{config.seed}_{key}.npz"
    )
    return cache_dir / name


def generate_samples(
    config: DatasetConfig,
    rng: np.random.Generator,
) -> Tuple[List[GraphSample], List[GraphSample], List[GraphSample]]:
    """Generate train/val/test samples from graphons.

    This function was previously named _generate_size_shift_samples.
    """
    if config.graphon_type == "controlled_fourier":
        graphons = make_controlled_fourier_graphons(
            num_classes=config.num_classes,
            rho=config.rho,
            num_terms=config.num_terms,
            rng=rng,
        )
    else:  # default "fourier"
        graphons = make_fourier_graphons(
            num_classes=config.num_classes,
            rho=config.rho,
            num_terms=config.num_terms,
            coeff_scale=config.coeff_scale,
            rng=rng,
        )
    allocation = size_allocation_path(
        sizes_small=(config.train_sizes[0], config.train_sizes[1]),
        sizes_large=(config.train_sizes[2], config.train_sizes[3]),
        total_budget=config.total_budget,
        lambda_mix=config.lambda_mix,
    )
    train_samples = sample_with_allocation_raw(
        graphons, allocation, rng, sampling_mode=config.sampling_mode
    )
    train_samples, val_samples = _split_train_val(train_samples, 0.2, rng)
    test_samples = sample_graphs_raw(
        graphons, config.test_sizes, config.per_class_test, rng,
        sampling_mode=config.sampling_mode
    )
    return train_samples, val_samples, test_samples


def generate_dataset(
    cache_dir: Path,
    config: DatasetConfig,
    overwrite: bool = False,
) -> Path:
    """Generate and cache a dataset.

    This function was previously named generate_size_shift_dataset.
    """
    cache_path = _dataset_cache_path(cache_dir, config)
    if cache_path.exists() and not overwrite:
        return cache_path
    rng = np.random.default_rng(config.seed)
    train_samples, val_samples, test_samples = generate_samples(
        config=config,
        rng=rng,
    )
    save_dataset(cache_path, train_samples, val_samples, test_samples)
    return cache_path


def generate_pe_sweep_dataset(
    cache_dir: Path,
    config: DatasetConfig,
    overwrite: bool = False,
) -> Path:
    return generate_datasets(cache_dir=cache_dir, config=config, overwrite=overwrite)


def generate_datasets(
    cache_dir: Path,
    config: DatasetConfig,
    overwrite: bool = False,
) -> Path:
    cache_path = _dataset_cache_path(cache_dir, config)
    if cache_path.exists() and not overwrite:
        return cache_path
    rng = np.random.default_rng(config.seed)
    train_samples, val_samples, test_samples = generate_samples(
        config=config,
        rng=rng,
    )
    save_dataset(cache_path, train_samples, val_samples, test_samples)
    return cache_path


def _resplit_cache_path(
    cache_dir: Path,
    dataset_name: str,
    gap_ratio: float,
    val_ratio: float,
    seed: int,
    num_samples: int,
) -> Path:
    """Generate cache path for resplit indices."""
    params = {
        "dataset": dataset_name,
        "gap_ratio": gap_ratio,
        "val_ratio": val_ratio,
        "seed": seed,
        "num_samples": num_samples,
    }
    key = _cache_key(params)
    gap_tag = f"{gap_ratio:.1f}".replace(".", "p")
    name = f"resplit_{dataset_name}_gap{gap_tag}_seed{seed}_{key}.npz"
    return cache_dir / name


def _compute_resplit_indices(
    samples: List[GraphSample],
    gap_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute train/val/test indices for size-gap resplit.

    Finds split where:
    - Train and test sizes are non-overlapping (max train < min test)
    - median(test_sizes) ≈ gap_ratio × median(train_sizes)

    Returns:
        (train_indices, val_indices, test_indices) as numpy arrays
    """
    if not samples:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    rng = np.random.default_rng(seed)

    # Get sizes and sort samples by size
    sizes = np.array([s.adjacency.shape[0] for s in samples])
    sorted_indices = np.argsort(sizes)
    sorted_sizes = sizes[sorted_indices]

    # Find best split point that minimizes |median(test)/median(train) - gap_ratio|
    n = len(samples)
    best_split_idx = None
    best_ratio_diff = float('inf')

    for i in range(1, n):
        # Check non-overlap: train_max < test_min
        train_max = sorted_sizes[i - 1]
        test_min = sorted_sizes[i]
        if train_max >= test_min:
            continue  # Skip overlapping sizes

        # Compute median ratio
        train_median = np.median(sorted_sizes[:i])
        test_median = np.median(sorted_sizes[i:])
        actual_ratio = test_median / train_median
        ratio_diff = abs(actual_ratio - gap_ratio)

        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_split_idx = i

    # Log achieved ratio if found
    if best_split_idx is not None:
        split_idx = best_split_idx
        train_median = np.median(sorted_sizes[:split_idx])
        test_median = np.median(sorted_sizes[split_idx:])
        print(
            f"Resplit: train median={train_median:.1f}, test median={test_median:.1f}, "
            f"ratio={test_median/train_median:.2f} (target={gap_ratio})",
            flush=True,
        )
    else:
        split_idx = n // 2
        print(
            f"Warning: Could not find non-overlapping split. "
            f"Using 50/50 split. Size range: {sorted_sizes[0]}-{sorted_sizes[-1]}",
            flush=True,
        )

    # Split into train+val and test (using original indices)
    train_val_orig_indices = sorted_indices[:split_idx].tolist()
    test_orig_indices = sorted_indices[split_idx:].tolist()

    # Stratified split of train+val by label
    by_label: Dict[int, List[int]] = {}
    for orig_idx in train_val_orig_indices:
        label = samples[orig_idx].label
        if label not in by_label:
            by_label[label] = []
        by_label[label].append(orig_idx)

    train_indices = []
    val_indices = []

    for label, class_indices in by_label.items():
        rng.shuffle(class_indices)
        n_val = max(1, int(len(class_indices) * val_ratio))
        n_train = len(class_indices) - n_val
        train_indices.extend(class_indices[:n_train])
        val_indices.extend(class_indices[n_train:])

    # Shuffle final indices
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    rng.shuffle(test_orig_indices)

    return (
        np.array(train_indices, dtype=np.int64),
        np.array(val_indices, dtype=np.int64),
        np.array(test_orig_indices, dtype=np.int64),
    )


def resplit_by_size_gap(
    samples: List[GraphSample],
    gap_ratio: float = 2.0,
    val_ratio: float = 0.1,
    seed: int = 0,
    cache_dir: Optional[Path] = None,
    dataset_name: Optional[str] = None,
) -> Tuple[List[GraphSample], List[GraphSample], List[GraphSample]]:
    """Resplit samples to ensure size gap between train and test.

    Sorts all samples by size, then splits so that:
    - Train contains smaller graphs
    - Test contains larger graphs
    - max(train_sizes) < min(test_sizes) (non-overlapping)
    - median(test_sizes) ≈ gap_ratio × median(train_sizes)

    Args:
        samples: All graph samples
        gap_ratio: Target ratio between median test size and median train size
        val_ratio: Fraction of train for validation
        seed: Random seed
        cache_dir: Optional cache directory for storing resplit indices
        dataset_name: Dataset name for cache key (required if cache_dir is set)

    Returns:
        (train_samples, val_samples, test_samples)
    """
    if not samples:
        return [], [], []

    # Try to load from cache
    cache_path = None
    if cache_dir is not None and dataset_name is not None:
        cache_path = _resplit_cache_path(
            cache_dir, dataset_name, gap_ratio, val_ratio, seed, len(samples)
        )
        if cache_path.exists():
            print(f"Loading resplit indices from cache: {cache_path}")
            data = np.load(cache_path)
            train_indices = data["train_indices"]
            val_indices = data["val_indices"]
            test_indices = data["test_indices"]
            return (
                [samples[i] for i in train_indices],
                [samples[i] for i in val_indices],
                [samples[i] for i in test_indices],
            )

    # Compute resplit indices
    train_indices, val_indices, test_indices = _compute_resplit_indices(
        samples, gap_ratio, val_ratio, seed
    )

    # Save to cache
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving resplit indices to cache: {cache_path}")
        np.savez(
            cache_path,
            train_indices=train_indices,
            val_indices=val_indices,
            test_indices=test_indices,
        )

    return (
        [samples[i] for i in train_indices],
        [samples[i] for i in val_indices],
        [samples[i] for i in test_indices],
    )


def load_real_dataset(
    name: str,
    root: Path,
    seed: int = 0,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> Tuple[List[GraphSample], List[GraphSample], List[GraphSample], int]:
    """Load a TUDataset and convert to GraphSamples.
    
    Args:
        name: Name of TUDataset (e.g., "IMDB-BINARY", "MUTAG")
        root: Root directory for dataset storage
        seed: Random seed for splitting
        val_ratio: Fraction of data for validation
        test_ratio: Fraction of data for testing

    Returns:
        Tuple of (train_samples, val_samples, test_samples, num_classes)
    """
    try:
        from torch_geometric.datasets import TUDataset
        from torch_geometric.utils import to_dense_adj
    except ImportError:
        raise ImportError("torch_geometric is required for real datasets. Please install it.")

    import torch
    
    # Load dataset
    dataset = TUDataset(root=str(root), name=name)
    num_classes = dataset.num_classes
    
    samples: List[GraphSample] = []
    
    for data in tqdm(dataset, desc=f"Loading {name}"):
        # Convert to dense adjacency (N, N)
        # to_dense_adj returns (batch_size, N, N), here batch_size=1
        adj = to_dense_adj(data.edge_index, max_num_nodes=data.num_nodes)[0].numpy()
        
        # Ensure it's symmetric (usually is for TUDataset, but good to be safe)
        # Real datasets might be directed, but GraphFM assumes undirected for now (PEs etc)
        # We symmetrize by OR-ing edges (if binary) or averaging (if weighted)
        # TUDatasets are typically unweighted binary.
        adj = np.maximum(adj, adj.T)
        
        # Compute normalized shift operator
        delta = normalize_shift_operator(adj)
        
        label = int(data.y.item()) if data.y is not None else 0
        
        samples.append(GraphSample(
            adjacency=adj,
            delta=delta,
            label=label,
            tokens=None
        ))

    # Stratified split
    rng = np.random.default_rng(seed)
    
    # Group by label
    by_label: Dict[int, List[GraphSample]] = {}
    for s in samples:
        if s.label not in by_label:
            by_label[s.label] = []
        by_label[s.label].append(s)
        
    train_samples = []
    val_samples = []
    test_samples = []
    
    for label, class_samples in by_label.items():
        rng.shuffle(class_samples)
        n = len(class_samples)
        n_val = int(n * val_ratio)
        n_test = int(n * test_ratio)
        n_train = n - n_val - n_test
        
        # Ensure at least one train sample if possible, unless dataset is tiny
        if n_train <= 0 and n > 0:
             # Fallback: simple split
             train_samples.extend(class_samples)
             continue

        train_samples.extend(class_samples[:n_train])
        val_samples.extend(class_samples[n_train:n_train + n_val])
        test_samples.extend(class_samples[n_train + n_val:])
    
    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    rng.shuffle(test_samples)
    
    return train_samples, val_samples, test_samples, num_classes
