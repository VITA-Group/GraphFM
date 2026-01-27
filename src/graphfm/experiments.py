from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .dataset import (
    DatasetConfig,
    GraphSample,
    _dataset_cache_path,
    generate_dataset,
    generate_datasets,
    generate_pe_sweep_dataset,
    generate_samples,
    load_dataset,
)
from .graphon import (
    ControlledFourierGraphon,
    graphon_l2_distance,
    make_controlled_fourier_graphons,
    perturb_controlled_graphon_monotonic,
    perturb_graphon_coeffs,
)
from .merge import OrderingMethod, estimate_step_graphon, synthesize_from_step
from .metrics import (
    discrepancy_set,
    discrepancy_set_all,
    discrepancy_set_proportional,
)
from .pe import PEConfig, compute_pe_batch, eigh_batch_for_learnable
from .sampling import SamplingMode, graphon_to_weighted_adjacency, normalize_shift_operator
from .train import TrainConfig, evaluate_classifier, evaluate_classifier_by_size, train_classifier


def _format_size_stats(samples: List[GraphSample]) -> str:
    from collections import Counter

    counts = Counter(s.delta.shape[0] for s in samples)
    parts = ", ".join(f"{size}:{count}" for size, count in sorted(counts.items()))
    return f"n={len(samples)} sizes={{ {parts} }}"


def _compute_tokens_gpu(
    samples: List[GraphSample],
    pe_cfg: PEConfig,
    device: str = "cuda",
) -> List[np.ndarray]:
    """Compute tokens for samples using GPU batch processing.

    Args:
        samples: list of GraphSample (tokens field is ignored)
        pe_cfg: PE configuration
        device: torch device

    Returns:
        List of token arrays (numpy), one per sample in original order
    """
    from collections import defaultdict

    # Group by size
    by_size: Dict[int, List[Tuple[int, GraphSample]]] = defaultdict(list)
    for i, s in enumerate(samples):
        by_size[s.delta.shape[0]].append((i, s))

    tokens_out: List[Optional[np.ndarray]] = [None] * len(samples)
    dev = torch.device(device)

    for n, group in by_size.items():
        indices = [g[0] for g in group]
        deltas = np.stack([g[1].delta for g in group])
        deltas_t = torch.from_numpy(deltas).to(dev, dtype=torch.float32)

        tokens_t = compute_pe_batch(deltas_t, pe_cfg)  # (B, n, k)
        tokens_np = tokens_t.cpu().numpy()

        for j, idx in enumerate(indices):
            tokens_out[idx] = tokens_np[j]

    return tokens_out  # type: ignore


def _compute_tokens_learnable(
    samples: List[GraphSample],
    pe_cfg: PEConfig,
    learnable_pe: torch.nn.Module,
    device: str = "cuda",
) -> List[np.ndarray]:
    """Compute tokens using a trained learnable PE module.

    Args:
        samples: list of GraphSample
        pe_cfg: PE configuration
        learnable_pe: trained StableExpressivePE module
        device: torch device

    Returns:
        List of token arrays (numpy), one per sample in original order
    """
    from collections import defaultdict

    # Group by size
    by_size: Dict[int, List[Tuple[int, GraphSample]]] = defaultdict(list)
    for i, s in enumerate(samples):
        by_size[s.delta.shape[0]].append((i, s))

    tokens_out: List[Optional[np.ndarray]] = [None] * len(samples)
    dev = torch.device(device)
    learnable_pe.eval()

    with torch.no_grad():
        for n, group in by_size.items():
            indices = [g[0] for g in group]
            deltas = np.stack([g[1].delta for g in group])
            deltas_t = torch.from_numpy(deltas).to(dev, dtype=torch.float32)

            Lambda, V = eigh_batch_for_learnable(deltas_t, pe_cfg.k)
            tokens_list = learnable_pe(Lambda, V)  # List of [n, m] tensors

            for j, idx in enumerate(indices):
                tokens_out[idx] = tokens_list[j].cpu().numpy()

    return tokens_out  # type: ignore


def _attach_tokens_to_samples(
    samples: List[GraphSample],
    pe_cfg: PEConfig,
    device: str = "cuda",
) -> None:
    """Compute tokens once and attach to samples in-place."""
    tokens_list = _compute_tokens_gpu(samples, pe_cfg, device)
    for sample, tokens in zip(samples, tokens_list):
        sample.tokens = tokens


def _compute_eigengap_stats_gpu(
    samples: List[GraphSample],
    k: int,
    device: str = "cuda",
) -> Tuple[float, float]:
    """Compute eigengap statistics using GPU batch processing.

    Args:
        samples: list of GraphSample
        k: number of eigenvalues to consider for gaps
        device: torch device

    Returns:
        (avg_min_gap, avg_gap_k) averaged over all samples
    """
    from collections import defaultdict

    # Group by size
    by_size: Dict[int, List[GraphSample]] = defaultdict(list)
    for s in samples:
        by_size[s.delta.shape[0]].append(s)

    all_min_gaps = []
    all_gap_ks = []
    dev = torch.device(device)

    for n, group in by_size.items():
        deltas = np.stack([s.delta for s in group])
        deltas_t = torch.from_numpy(deltas).to(dev, dtype=torch.float32)

        # Batch eigenvalue computation
        evals = torch.linalg.eigvalsh(deltas_t)  # (B, n)
        evals_sorted, _ = torch.sort(evals, dim=1)
        gaps = torch.diff(evals_sorted, dim=1)  # (B, n-1)

        k_eff = min(k, n - 1)
        if k_eff > 0:
            min_gaps = gaps[:, :k_eff].min(dim=1).values  # (B,)
            gap_ks = gaps[:, k_eff - 1]  # (B,)
            all_min_gaps.append(min_gaps.cpu().numpy())
            all_gap_ks.append(gap_ks.cpu().numpy())

    if not all_min_gaps:
        return float("nan"), float("nan")

    avg_min_gap = float(np.mean(np.concatenate(all_min_gaps)))
    avg_gap_k = float(np.mean(np.concatenate(all_gap_ks)))
    return avg_min_gap, avg_gap_k


def run_size_shift(
    out_dir: Path,
    pe_cfg: PEConfig,
    train_cfg: TrainConfig,
    config: DatasetConfig,
    merging_method: Optional[OrderingMethod] = None,
    discrepancy_mode: str = "proportional",
    cache_dir: Optional[Path] = None,
    output_suffix: Optional[str] = None,
) -> Dict:
    rng = np.random.default_rng(config.seed)
    if cache_dir is not None:
        cache_path = _dataset_cache_path(cache_dir, config)
        if not cache_path.exists():
            raise SystemExit(
                f"Dataset cache not found: {cache_path}. Run scripts/generate_dataset.py first."
            )
        train_samples, val_samples, test_samples = load_dataset(cache_path)
        print(
            "Loaded dataset size stats:",
            "train", _format_size_stats(train_samples),
            "val", _format_size_stats(val_samples),
            "test", _format_size_stats(test_samples),
        )
    else:
        train_samples, val_samples, test_samples = generate_samples(
            config=config,
            rng=rng,
        )
    # Handle learnable PE differently - no pre-computed tokens
    is_learnable_pe = pe_cfg.kind == "spe_learnable"
    learnable_pe = None

    if not is_learnable_pe:
        # Pre-compute tokens once for all samples
        _attach_tokens_to_samples(train_samples, pe_cfg, train_cfg.device)
        _attach_tokens_to_samples(val_samples, pe_cfg, train_cfg.device)
        _attach_tokens_to_samples(test_samples, pe_cfg, train_cfg.device)

    if merging_method is not None:
        merged = []
        for c in range(config.num_classes):
            class_graphs = [s.adjacency for s in train_samples if s.label == c]
            step = estimate_step_graphon(
                class_graphs, bins=16, method=merging_method, device=train_cfg.device
            )
            a = synthesize_from_step(
                step, n=max(config.train_sizes), rng=rng, sampling_mode=config.sampling_mode
            )
            delta = normalize_shift_operator(a)
            merged.append(GraphSample(adjacency=a, delta=delta, label=c, tokens=None))
        if not is_learnable_pe:
            _attach_tokens_to_samples(merged, pe_cfg, train_cfg.device)
        train_samples = train_samples + merged
        from collections import Counter
        merged_counts = Counter(s.delta.shape[0] for s in merged)
        total_counts = Counter(s.delta.shape[0] for s in train_samples)
        sizes = sorted(total_counts.keys())
        parts = [
            f"{size}:{merged_counts.get(size, 0)}/{total_counts.get(size, 0)}"
            for size in sizes
        ]
        print("Merged graphs by size:", " ".join(parts))

    # Train and evaluate
    if is_learnable_pe:
        # Learnable PE: train jointly with model
        model, learnable_pe = train_classifier(train_samples, val_samples, config.num_classes, train_cfg, pe_cfg)
        train_error = evaluate_classifier(model, train_samples, train_cfg, pe_cfg, learnable_pe=learnable_pe)
        test_error = evaluate_classifier(model, test_samples, train_cfg, pe_cfg, learnable_pe=learnable_pe)
        test_error_by_size = evaluate_classifier_by_size(model, test_samples, train_cfg, pe_cfg, learnable_pe=learnable_pe)
    else:
        # Pre-computed tokens: pass pe_cfg=None
        model = train_classifier(train_samples, val_samples, config.num_classes, train_cfg, pe_cfg=None)
        train_error = evaluate_classifier(model, train_samples, train_cfg, pe_cfg=None)
        test_error = evaluate_classifier(model, test_samples, train_cfg, pe_cfg=None)
        test_error_by_size = evaluate_classifier_by_size(model, test_samples, train_cfg, pe_cfg=None)

    # Compute ID/OOD errors
    train_sizes_set = set(config.train_sizes)

    id_errors = [err for sz, err in test_error_by_size.items() if sz in train_sizes_set]
    ood_errors = [err for sz, err in test_error_by_size.items() if sz not in train_sizes_set]

    id_error = float(np.mean(id_errors)) if id_errors else 0.0
    ood_error = float(np.mean(ood_errors)) if ood_errors else 0.0

    # Compute tokens for discrepancy calculation
    if is_learnable_pe and learnable_pe is not None:
        # For learnable PE, compute tokens using the trained module
        train_tokens = _compute_tokens_learnable(train_samples, pe_cfg, learnable_pe, train_cfg.device)
        test_tokens = _compute_tokens_learnable(test_samples, pe_cfg, learnable_pe, train_cfg.device)
    else:
        # Use pre-computed tokens
        train_tokens = [s.tokens for s in train_samples]
        test_tokens = [s.tokens for s in test_samples]

    # Compute eigengap stats using GPU batch processing
    avg_min_gap, avg_gap_k = _compute_eigengap_stats_gpu(test_samples, pe_cfg.k, train_cfg.device)

    if discrepancy_mode == "proportional":
        discrepancy = discrepancy_set_proportional(
            train_tokens,
            test_tokens,
            total_samples=128 * len(train_tokens),
            projections=50,
            rng=rng,
        )
    elif discrepancy_mode == "uniform":
        discrepancy = discrepancy_set(
            train_tokens, test_tokens, samples_per_graph=128, projections=50, rng=rng
        )
    elif discrepancy_mode == "all":
        discrepancy = discrepancy_set_all(
            train_tokens, test_tokens, projections=50, rng=rng
        )
    else:
        raise ValueError(f"Unknown discrepancy_mode: {discrepancy_mode}")

    result = {
        "train_error": train_error,
        "test_error": test_error,
        "id_error": id_error,
        "ood_error": ood_error,
        "test_error_by_size": {str(k): v for k, v in test_error_by_size.items()},
        "discrepancy_set": discrepancy,
        "discrepancy_mode": discrepancy_mode,
        "eigengap_min": avg_min_gap,
        "eigengap_k": avg_gap_k,
        "lambda_mix": config.lambda_mix,
        "merging_method": merging_method,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    lambda_tag = f"{config.lambda_mix:.2f}".replace(".", "p")
    merge_tag = f"_merge_{merging_method}" if merging_method else ""
    suffix_tag = f"_{output_suffix}" if output_suffix else ""
    out_name = f"size_shift_lambda_{lambda_tag}{merge_tag}{suffix_tag}.json"
    (out_dir / out_name).write_text(json.dumps(result, indent=2))
    return result


def run_pe_sweep(
    out_dir: Path,
    pe_grid: Sequence[PEConfig],
    train_cfg: TrainConfig,
    config: DatasetConfig,
    discrepancy_mode: str,
    cache_dir: Optional[Path] = None,
) -> Dict:
    rng = np.random.default_rng(config.seed)
    if cache_dir is not None:
        cache_path = _dataset_cache_path(cache_dir, config)
        if not cache_path.exists():
            raise SystemExit(
                f"Dataset cache not found: {cache_path}. Run scripts/generate_dataset.py first."
            )
        train_samples, val_samples, test_samples = load_dataset(cache_path)
        print(
            "Loaded dataset size stats:",
            "train", _format_size_stats(train_samples),
            "val", _format_size_stats(val_samples),
            "test", _format_size_stats(test_samples),
        )
    else:
        train_samples, val_samples, test_samples = generate_samples(
            config=config,
            rng=rng,
        )

    results = []
    train_sizes_set = set(config.train_sizes)
    for pe_cfg in pe_grid:
        is_learnable_pe = pe_cfg.kind == "spe_learnable"
        learnable_pe = None

        if is_learnable_pe:
            # Learnable PE: train jointly with model
            model, learnable_pe = train_classifier(train_samples, val_samples, config.num_classes, train_cfg, pe_cfg)
            test_error = evaluate_classifier(model, test_samples, train_cfg, pe_cfg, learnable_pe=learnable_pe)
            test_error_by_size = evaluate_classifier_by_size(model, test_samples, train_cfg, pe_cfg, learnable_pe=learnable_pe)
            # Compute tokens using learned PE for discrepancy
            train_tokens = _compute_tokens_learnable(train_samples, pe_cfg, learnable_pe, train_cfg.device)
            test_tokens = _compute_tokens_learnable(test_samples, pe_cfg, learnable_pe, train_cfg.device)
        else:
            # Pre-compute tokens once for this PE config
            _attach_tokens_to_samples(train_samples, pe_cfg, train_cfg.device)
            _attach_tokens_to_samples(val_samples, pe_cfg, train_cfg.device)
            _attach_tokens_to_samples(test_samples, pe_cfg, train_cfg.device)

            # Train and evaluate using pre-computed tokens (pe_cfg=None)
            model = train_classifier(train_samples, val_samples, config.num_classes, train_cfg, pe_cfg=None)
            test_error = evaluate_classifier(model, test_samples, train_cfg, pe_cfg=None)
            test_error_by_size = evaluate_classifier_by_size(model, test_samples, train_cfg, pe_cfg=None)
            # Use pre-computed tokens for discrepancy calculation
            train_tokens = [s.tokens for s in train_samples]
            test_tokens = [s.tokens for s in test_samples]

        # Compute ID/OOD errors
        id_errors = [err for sz, err in test_error_by_size.items() if sz in train_sizes_set]
        ood_errors = [err for sz, err in test_error_by_size.items() if sz not in train_sizes_set]
        id_error = float(np.mean(id_errors)) if id_errors else 0.0
        ood_error = float(np.mean(ood_errors)) if ood_errors else 0.0

        if discrepancy_mode == "proportional":
            discrepancy = discrepancy_set_proportional(
                train_tokens,
                test_tokens,
                total_samples=128 * len(train_samples),
                projections=50,
                rng=rng,
            )
        elif discrepancy_mode == "uniform":
            discrepancy = discrepancy_set(
                train_tokens, test_tokens, samples_per_graph=128, projections=50, rng=rng
            )
        elif discrepancy_mode == "all":
            discrepancy = discrepancy_set_all(
                train_tokens, test_tokens, projections=50, rng=rng
            )
        else:
            raise ValueError(f"Unknown discrepancy_mode: {discrepancy_mode}")

        # Compute eigengap stats using GPU batch processing
        avg_min_gap, avg_gap_k = _compute_eigengap_stats_gpu(test_samples, pe_cfg.k, train_cfg.device)
        results.append(
            {
                "pe": pe_cfg.__dict__,
                "test_error": test_error,
                "id_error": id_error,
                "ood_error": ood_error,
                "test_error_by_size": {str(k): v for k, v in test_error_by_size.items()},
                "discrepancy_set": discrepancy,
                "discrepancy_mode": discrepancy_mode,
                "eigengap_min": avg_min_gap,
                "eigengap_k": avg_gap_k,
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "pe_sweep.json").write_text(json.dumps(results, indent=2))
    return {"results": results}


def run_merge_graph(
    out_dir: Path,
    pe_cfg: PEConfig,
    train_cfg: TrainConfig,
    config: DatasetConfig,
    merging_method: OrderingMethod = "spectral",
    merging_ratio: float = 0.5,
    merging_size: float = 2.0,
    discrepancy_mode: str = "proportional",
    cache_dir: Optional[Path] = None,
) -> Dict:
    """Run merge_graph experiment with controllable merging ratio and sizes.

    Args:
        out_dir: Output directory for results
        pe_cfg: Positional encoding configuration
        train_cfg: Training configuration
        config: Dataset configuration
        merging_method: Node ordering method ("degree" or "spectral")
        merging_ratio: Ratio of merged graphs to original graphs per class
        merging_size: Size multiplier for merged graphs (e.g., 1.5, 2.0, 3.0)
        discrepancy_mode: Discrepancy calculation mode
        cache_dir: Optional cache directory for dataset

    Returns:
        Dictionary with experiment results
    """
    rng = np.random.default_rng(config.seed)

    # Load or generate base dataset
    if cache_dir is not None:
        cache_path = _dataset_cache_path(cache_dir, config)
        if not cache_path.exists():
            raise SystemExit(
                f"Dataset cache not found: {cache_path}. Run scripts/generate_dataset.py first."
            )
        train_samples, val_samples, test_samples = load_dataset(cache_path)
        print(
            "Loaded dataset size stats:",
            "train", _format_size_stats(train_samples),
            "val", _format_size_stats(val_samples),
            "test", _format_size_stats(test_samples),
        )
    else:
        train_samples, val_samples, test_samples = generate_samples(
            config=config,
            rng=rng,
        )

    # Pre-compute tokens for base samples
    _attach_tokens_to_samples(train_samples, pe_cfg, train_cfg.device)
    _attach_tokens_to_samples(val_samples, pe_cfg, train_cfg.device)
    _attach_tokens_to_samples(test_samples, pe_cfg, train_cfg.device)

    # Compute merged sizes
    merged_sizes = tuple(int(s * merging_size) for s in config.train_sizes)

    # Generate merged graphs for each class
    merged = []
    num_original_per_class = {}
    num_merged_per_class = {}

    for c in range(config.num_classes):
        class_graphs = [s.adjacency for s in train_samples if s.label == c]
        num_original_per_class[c] = len(class_graphs)
        num_merged = int(len(class_graphs) * merging_ratio)
        num_merged_per_class[c] = num_merged

        if num_merged == 0:
            continue

        # Estimate step graphon from class graphs
        step = estimate_step_graphon(
            class_graphs, bins=16, method=merging_method, device=train_cfg.device
        )

        # Generate merged graphs at scaled sizes (round-robin)
        for i in range(num_merged):
            target_size = merged_sizes[i % len(merged_sizes)]
            a = synthesize_from_step(
                step, n=target_size, rng=rng, sampling_mode=config.sampling_mode
            )
            delta = normalize_shift_operator(a)
            merged.append(GraphSample(adjacency=a, delta=delta, label=c, tokens=None))

    # Attach tokens to merged samples
    if merged:
        _attach_tokens_to_samples(merged, pe_cfg, train_cfg.device)

    # Combine original and merged training samples
    train_samples_augmented = train_samples + merged

    # Log merging statistics
    from collections import Counter
    merged_counts = Counter(s.delta.shape[0] for s in merged)
    total_counts = Counter(s.delta.shape[0] for s in train_samples_augmented)
    sizes = sorted(total_counts.keys())
    parts = [
        f"{size}:{merged_counts.get(size, 0)}/{total_counts.get(size, 0)}"
        for size in sizes
    ]
    print(f"Merged graphs (ratio={merging_ratio}, size_mult={merging_size}):", " ".join(parts))

    # Train and evaluate using pre-computed tokens (pe_cfg=None)
    model = train_classifier(
        train_samples_augmented, val_samples, config.num_classes, train_cfg, pe_cfg=None
    )
    train_error = evaluate_classifier(model, train_samples_augmented, train_cfg, pe_cfg=None)
    test_error = evaluate_classifier(model, test_samples, train_cfg, pe_cfg=None)

    # Compute ID/OOD errors
    test_error_by_size = evaluate_classifier_by_size(model, test_samples, train_cfg, pe_cfg=None)
    train_sizes_set = set(config.train_sizes)

    id_errors = [err for sz, err in test_error_by_size.items() if sz in train_sizes_set]
    ood_errors = [err for sz, err in test_error_by_size.items() if sz not in train_sizes_set]

    id_error = float(np.mean(id_errors)) if id_errors else 0.0
    ood_error = float(np.mean(ood_errors)) if ood_errors else 0.0

    # Use pre-computed tokens for discrepancy calculation
    train_tokens = [s.tokens for s in train_samples_augmented]
    test_tokens = [s.tokens for s in test_samples]

    # Compute eigengap stats using GPU batch processing
    avg_min_gap, avg_gap_k = _compute_eigengap_stats_gpu(test_samples, pe_cfg.k, train_cfg.device)

    if discrepancy_mode == "proportional":
        discrepancy = discrepancy_set_proportional(
            train_tokens,
            test_tokens,
            total_samples=128 * len(train_tokens),
            projections=50,
            rng=rng,
        )
    elif discrepancy_mode == "uniform":
        discrepancy = discrepancy_set(
            train_tokens, test_tokens, samples_per_graph=128, projections=50, rng=rng
        )
    elif discrepancy_mode == "all":
        discrepancy = discrepancy_set_all(
            train_tokens, test_tokens, projections=50, rng=rng
        )
    else:
        raise ValueError(f"Unknown discrepancy_mode: {discrepancy_mode}")

    result = {
        "train_error": train_error,
        "test_error": test_error,
        "id_error": id_error,
        "ood_error": ood_error,
        "test_error_by_size": {str(k): v for k, v in test_error_by_size.items()},
        "discrepancy_set": discrepancy,
        "discrepancy_mode": discrepancy_mode,
        "eigengap_min": avg_min_gap,
        "eigengap_k": avg_gap_k,
        "lambda_mix": config.lambda_mix,
        "merging_method": merging_method,
        "merging_ratio": merging_ratio,
        "merging_size": merging_size,
        "num_original_train": len(train_samples),
        "num_merged": len(merged),
        "merged_sizes": list(merged_sizes),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    # Format: merge_graph_method_{method}_ratio_{ratio}_size_{size}.json
    ratio_tag = f"{merging_ratio:.2f}".replace(".", "p")
    size_tag = f"{merging_size:.1f}".replace(".", "p")
    lambda_mix_tag = f"{config.lambda_mix:.2f}".replace(".", "p")
    out_name = f"merge_graph_method_{merging_method}_ratio_{ratio_tag}_size_{size_tag}_lambda_{lambda_mix_tag}.json"
    (out_dir / out_name).write_text(json.dumps(result, indent=2))
    return result


def _generate_perturbed_test_samples(
    original_graphons: List[ControlledFourierGraphon],
    perturbed_graphons: List[ControlledFourierGraphon],
    perturb_ratio: float,
    test_sizes: Sequence[int],
    per_class_test: int,
    sampling_mode: SamplingMode,
    rng: np.random.Generator,
) -> Tuple[List[GraphSample], List[GraphSample], List[GraphSample]]:
    """Generate test samples with probabilistic mixing of original and perturbed graphons.

    Args:
        original_graphons: Original graphons (one per class)
        perturbed_graphons: Perturbed graphons (one per class)
        perturb_ratio: Probability of using perturbed graphon for each sample
        test_sizes: Sizes of test graphs
        per_class_test: Number of samples per class per size
        sampling_mode: Graph sampling mode
        rng: Random generator

    Returns:
        Tuple of (all_samples, id_samples, ood_samples)
        - all_samples: All test samples
        - id_samples: Samples from original graphons
        - ood_samples: Samples from perturbed graphons
    """
    all_samples: List[GraphSample] = []
    id_samples: List[GraphSample] = []
    ood_samples: List[GraphSample] = []

    for c, (orig_g, pert_g) in enumerate(zip(original_graphons, perturbed_graphons)):
        for n in test_sizes:
            for _ in range(per_class_test):
                # Decide whether to use perturbed graphon
                use_perturbed = rng.random() < perturb_ratio
                graphon = pert_g if use_perturbed else orig_g

                a = graphon_to_weighted_adjacency(graphon, n, rng=rng, sampling_mode=sampling_mode)
                delta = normalize_shift_operator(a)
                sample = GraphSample(adjacency=a, delta=delta, label=c, tokens=None)

                all_samples.append(sample)
                if use_perturbed:
                    ood_samples.append(sample)
                else:
                    id_samples.append(sample)

    rng.shuffle(all_samples)
    return all_samples, id_samples, ood_samples


def run_perturb_graphon(
    out_dir: Path,
    pe_cfg: PEConfig,
    train_cfg: TrainConfig,
    config: DatasetConfig,
    perturb_levels: Sequence[float] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    perturb_ratio: float = 0.5,
    max_l2_distance: float = 0.1,
    discrepancy_mode: str = "proportional",
    cache_dir: Optional[Path] = None,
) -> Dict:
    """Run perturb_graphon experiment testing generalization under graphon perturbation.

    Training is done once with fixed lambda_mix. Then we evaluate on test sets with
    varying perturbation levels, where a configurable ratio of test graphs come from
    perturbed graphons.

    Args:
        out_dir: Output directory for results
        pe_cfg: Positional encoding configuration
        train_cfg: Training configuration
        config: Dataset configuration (must use graphon_type="controlled_fourier")
        perturb_levels: Perturbation levels to evaluate (each in [0, 1])
        perturb_ratio: Fraction of test graphs from perturbed graphons
        max_l2_distance: Maximum L2 distance at perturb_level=1.0
        discrepancy_mode: Discrepancy calculation mode
        cache_dir: Optional cache directory for training dataset

    Returns:
        Dictionary with summary of all results
    """
    if config.graphon_type != "controlled_fourier":
        raise ValueError(
            "perturb_graphon experiment requires graphon_type='controlled_fourier' "
            f"but got '{config.graphon_type}'"
        )

    rng = np.random.default_rng(config.seed)

    # Load or generate training data
    if cache_dir is not None:
        cache_path = _dataset_cache_path(cache_dir, config)
        if not cache_path.exists():
            raise SystemExit(
                f"Dataset cache not found: {cache_path}. Run scripts/generate_dataset.py first."
            )
        train_samples, val_samples, _ = load_dataset(cache_path)
        print(
            "Loaded training dataset:",
            "train", _format_size_stats(train_samples),
            "val", _format_size_stats(val_samples),
        )
    else:
        train_samples, val_samples, _ = generate_samples(config=config, rng=rng)

    # Create base graphons (use same seed as dataset generation for consistency)
    graphon_rng = np.random.default_rng(config.seed)
    original_graphons = make_controlled_fourier_graphons(
        num_classes=config.num_classes,
        rho=config.rho,
        num_terms=config.num_terms,
        rng=graphon_rng,
    )

    # Pre-compute tokens for training samples
    _attach_tokens_to_samples(train_samples, pe_cfg, train_cfg.device)
    _attach_tokens_to_samples(val_samples, pe_cfg, train_cfg.device)

    # Train model once
    print("Training model on fixed training distribution...")
    model = train_classifier(train_samples, val_samples, config.num_classes, train_cfg, pe_cfg=None)
    train_error = evaluate_classifier(model, train_samples, train_cfg, pe_cfg=None)
    print(f"Training error: {train_error:.4f}")

    # Evaluate at each perturbation level
    results = []
    train_sizes_set = set(config.train_sizes)

    for perturb_level in perturb_levels:
        print(f"\n{'='*60}")
        print(f"Perturbation level: {perturb_level}")
        print(f"{'='*60}")

        # Create perturbed graphons with monotonic perturbation (fixed direction per class)
        perturbed_graphons = []
        l2_distances = {}

        for c, orig_g in enumerate(original_graphons):
            pert_g = perturb_controlled_graphon_monotonic(
                orig_g,
                perturbation_level=perturb_level,
                max_l2_distance=max_l2_distance,
                direction_seed=config.seed + c,  # Fixed direction per class
            )
            perturbed_graphons.append(pert_g)

            # Compute and log L2 distance
            l2_dist = graphon_l2_distance(orig_g, pert_g)
            l2_distances[str(c)] = l2_dist
            print(f"  Class {c}: L2 distance = {l2_dist:.6f}")

        avg_l2_distance = np.mean(list(l2_distances.values()))
        print(f"  Average L2 distance: {avg_l2_distance:.6f}")

        # Generate test samples with probabilistic mixing
        test_rng = np.random.default_rng(config.seed + int(perturb_level * 1000))
        test_samples, id_test_samples, ood_test_samples = _generate_perturbed_test_samples(
            original_graphons=original_graphons,
            perturbed_graphons=perturbed_graphons,
            perturb_ratio=perturb_ratio,
            test_sizes=config.test_sizes,
            per_class_test=config.per_class_test,
            sampling_mode=config.sampling_mode,
            rng=test_rng,
        )

        print(f"  Test samples: {len(test_samples)} total, {len(id_test_samples)} ID, {len(ood_test_samples)} OOD")

        # Compute tokens for test samples
        _attach_tokens_to_samples(test_samples, pe_cfg, train_cfg.device)
        if id_test_samples:
            _attach_tokens_to_samples(id_test_samples, pe_cfg, train_cfg.device)
        if ood_test_samples:
            _attach_tokens_to_samples(ood_test_samples, pe_cfg, train_cfg.device)

        # Evaluate
        test_error = evaluate_classifier(model, test_samples, train_cfg, pe_cfg=None)
        id_error = evaluate_classifier(model, id_test_samples, train_cfg, pe_cfg=None) if id_test_samples else 0.0
        ood_error = evaluate_classifier(model, ood_test_samples, train_cfg, pe_cfg=None) if ood_test_samples else 0.0

        # Error by size
        test_error_by_size = evaluate_classifier_by_size(model, test_samples, train_cfg, pe_cfg=None)

        print(f"  Test error: {test_error:.4f} (ID: {id_error:.4f}, OOD: {ood_error:.4f})")

        # Compute discrepancy
        train_tokens = [s.tokens for s in train_samples]
        test_tokens = [s.tokens for s in test_samples]

        if discrepancy_mode == "proportional":
            discrepancy = discrepancy_set_proportional(
                train_tokens,
                test_tokens,
                total_samples=128 * len(train_tokens),
                projections=50,
                rng=rng,
            )
        elif discrepancy_mode == "uniform":
            discrepancy = discrepancy_set(
                train_tokens, test_tokens, samples_per_graph=128, projections=50, rng=rng
            )
        elif discrepancy_mode == "all":
            discrepancy = discrepancy_set_all(
                train_tokens, test_tokens, projections=50, rng=rng
            )
        else:
            raise ValueError(f"Unknown discrepancy_mode: {discrepancy_mode}")

        # Build result for this level
        level_result = {
            "perturb_level": perturb_level,
            "perturb_ratio": perturb_ratio,
            "max_l2_distance": max_l2_distance,
            "l2_distances": l2_distances,
            "avg_l2_distance": avg_l2_distance,
            "train_error": train_error,
            "test_error": test_error,
            "id_error": id_error,
            "ood_error": ood_error,
            "test_error_by_size": {str(k): v for k, v in test_error_by_size.items()},
            "discrepancy_set": discrepancy,
            "lambda_mix": config.lambda_mix,
        }
        results.append(level_result)

        # Save per-level result
        out_dir.mkdir(parents=True, exist_ok=True)
        level_tag = f"{perturb_level:.2f}".replace(".", "p")
        lambda_tag = f"{config.lambda_mix:.2f}".replace(".", "p")
        out_name = f"perturb_graphon_level_{level_tag}_lambda_{lambda_tag}.json"
        (out_dir / out_name).write_text(json.dumps(level_result, indent=2))

    # Save summary
    summary = {
        "experiment": "perturb_graphon",
        "config": {
            "lambda_mix": config.lambda_mix,
            "perturb_levels": list(perturb_levels),
            "perturb_ratio": perturb_ratio,
            "max_l2_distance": max_l2_distance,
            "num_classes": config.num_classes,
            "train_sizes": list(config.train_sizes),
            "test_sizes": list(config.test_sizes),
            "pe_kind": pe_cfg.kind,
            "pe_k": pe_cfg.k,
        },
        "results": results,
    }
    (out_dir / "perturb_graphon_summary.json").write_text(json.dumps(summary, indent=2))

    return summary
