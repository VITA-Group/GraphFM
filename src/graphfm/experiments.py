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
from .graphon import perturb_graphon_coeffs
from .merge import estimate_step_graphon, synthesize_from_step
from .metrics import (
    discrepancy_set,
    discrepancy_set_all,
    discrepancy_set_proportional,
)
from .pe import PEConfig, compute_pe_batch
from .sampling import normalize_shift_operator
from .train import TrainConfig, evaluate_classifier, evaluate_classifier_by_size, train_classifier


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
    use_merging: bool = False,
    discrepancy_mode: str = "proportional",
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
    else:
        train_samples, val_samples, test_samples = generate_samples(
            config=config,
            rng=rng,
        )
    # Pre-compute tokens once for all samples
    _attach_tokens_to_samples(train_samples, pe_cfg, train_cfg.device)
    _attach_tokens_to_samples(val_samples, pe_cfg, train_cfg.device)
    _attach_tokens_to_samples(test_samples, pe_cfg, train_cfg.device)

    if use_merging:
        merged = []
        for c in range(config.num_classes):
            class_graphs = [s.adjacency for s in train_samples if s.label == c]
            step = estimate_step_graphon(class_graphs, bins=16)
            a = synthesize_from_step(step, n=max(config.train_sizes), rng=rng)
            delta = normalize_shift_operator(a)
            merged.append(GraphSample(adjacency=a, delta=delta, label=c, tokens=None))
        _attach_tokens_to_samples(merged, pe_cfg, train_cfg.device)
        train_samples = train_samples + merged

    # Train and evaluate using pre-computed tokens (pe_cfg=None)
    model = train_classifier(train_samples, val_samples, config.num_classes, train_cfg, pe_cfg=None)
    train_error = evaluate_classifier(model, train_samples, train_cfg, pe_cfg=None)
    test_error = evaluate_classifier(model, test_samples, train_cfg, pe_cfg=None)

    # Compute ID/OOD errors
    test_error_by_size = evaluate_classifier_by_size(model, test_samples, train_cfg, pe_cfg=None)
    train_sizes_set = set(config.train_sizes)

    id_errors = [err for sz, err in test_error_by_size.items() if sz in train_sizes_set]
    ood_errors = [err for sz, err in test_error_by_size.items() if sz not in train_sizes_set]

    id_error = float(np.mean(id_errors)) if id_errors else 0.0
    ood_error = float(np.mean(ood_errors)) if ood_errors else 0.0

    # Use pre-computed tokens for discrepancy calculation
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
        "use_merging": use_merging,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    lambda_tag = f"{config.lambda_mix:.2f}".replace(".", "p")
    merge_tag = "_merge" if use_merging else ""
    out_name = f"size_shift_lambda_{lambda_tag}{merge_tag}.json"
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
    else:
        train_samples, val_samples, test_samples = generate_samples(
            config=config,
            rng=rng,
        )

    results = []
    train_sizes_set = set(config.train_sizes)
    for pe_cfg in pe_grid:
        # Pre-compute tokens once for this PE config
        _attach_tokens_to_samples(train_samples, pe_cfg, train_cfg.device)
        _attach_tokens_to_samples(val_samples, pe_cfg, train_cfg.device)
        _attach_tokens_to_samples(test_samples, pe_cfg, train_cfg.device)

        # Train and evaluate using pre-computed tokens (pe_cfg=None)
        model = train_classifier(train_samples, val_samples, config.num_classes, train_cfg, pe_cfg=None)
        test_error = evaluate_classifier(model, test_samples, train_cfg, pe_cfg=None)

        # Compute ID/OOD errors
        test_error_by_size = evaluate_classifier_by_size(model, test_samples, train_cfg, pe_cfg=None)
        id_errors = [err for sz, err in test_error_by_size.items() if sz in train_sizes_set]
        ood_errors = [err for sz, err in test_error_by_size.items() if sz not in train_sizes_set]
        id_error = float(np.mean(id_errors)) if id_errors else 0.0
        ood_error = float(np.mean(ood_errors)) if ood_errors else 0.0

        # Use pre-computed tokens for discrepancy calculation
        train_tokens = [s.tokens for s in train_samples]
        test_tokens = [s.tokens for s in test_samples]

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
