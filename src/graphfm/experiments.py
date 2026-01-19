from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from .dataset import GraphSample, sample_with_allocation, sample_graphs, size_allocation_path
from .graphon import make_fourier_graphons, perturb_graphon_coeffs
from .merge import estimate_step_graphon, synthesize_from_step
from .metrics import (
    discrepancy_set,
    discrepancy_set_all,
    discrepancy_set_proportional,
    eigengap_stats,
)
from .pe import PEConfig, compute_pe
from .sampling import normalize_shift_operator
from .train import TrainConfig, evaluate_classifier, train_classifier


@dataclass
class ExperimentConfig:
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


def _split_train_val(samples: List[GraphSample], val_frac: float, rng: np.random.Generator):
    rng.shuffle(samples)
    cut = int(round(len(samples) * (1.0 - val_frac)))
    return samples[:cut], samples[cut:]


def run_size_shift(
    out_dir: Path,
    pe_cfg: PEConfig,
    train_cfg: TrainConfig,
    config: ExperimentConfig,
    use_merging: bool = False,
    lambda_mix: float = 0.0,
    discrepancy_mode: str = "uniform",
) -> Dict:
    rng = np.random.default_rng(config.seed)
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
        lambda_mix=lambda_mix,
    )
    train_samples = sample_with_allocation(graphons, allocation, pe_cfg, rng)
    train_samples, val_samples = _split_train_val(train_samples, 0.2, rng)

    if use_merging:
        merged = []
        for c in range(config.num_classes):
            class_graphs = [s.adjacency for s in train_samples if s.label == c]
            step = estimate_step_graphon(class_graphs, bins=16)
            a = synthesize_from_step(step, n=max(config.train_sizes))
            delta = normalize_shift_operator(a)
            tokens = compute_pe(delta, pe_cfg)
            merged.append(GraphSample(adjacency=a, delta=delta, label=c, tokens=tokens))
        train_samples = train_samples + merged

    model = train_classifier(train_samples, val_samples, config.num_classes, train_cfg)
    train_error = evaluate_classifier(model, train_samples, train_cfg)
    test_samples = sample_graphs(graphons, config.test_sizes, config.per_class_test, pe_cfg, rng)
    test_error = evaluate_classifier(model, test_samples, train_cfg)

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
    eig_stats = []
    for s in test_samples:
        evals = np.linalg.eigvalsh(s.delta)
        eig_stats.append(eigengap_stats(evals, pe_cfg.k))
    avg_min_gap = float(np.mean([e.min_gap for e in eig_stats]))
    avg_gap_k = float(np.mean([e.gap_k for e in eig_stats]))

    result = {
        "train_error": train_error,
        "test_error": test_error,
        "discrepancy_set": discrepancy,
        "discrepancy_mode": discrepancy_mode,
        "eigengap_min": avg_min_gap,
        "eigengap_k": avg_gap_k,
        "lambda_mix": lambda_mix,
        "use_merging": use_merging,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    lambda_tag = f"{lambda_mix:.2f}".replace(".", "p")
    merge_tag = "_merge" if use_merging else ""
    out_name = f"size_shift_lambda_{lambda_tag}{merge_tag}.json"
    (out_dir / out_name).write_text(json.dumps(result, indent=2))
    return result


def run_pe_sweep(
    out_dir: Path,
    pe_grid: Sequence[PEConfig],
    train_cfg: TrainConfig,
    config: ExperimentConfig,
    discrepancy_mode: str,
) -> Dict:
    rng = np.random.default_rng(config.seed)
    graphons = make_fourier_graphons(
        num_classes=config.num_classes,
        rho=config.rho,
        num_terms=config.num_terms,
        coeff_scale=config.coeff_scale,
        rng=rng,
    )
    train_samples = sample_graphs(
        graphons, config.train_sizes, config.per_class_train, pe_grid[0], rng
    )
    train_samples, val_samples = _split_train_val(train_samples, 0.2, rng)
    test_samples = sample_graphs(graphons, config.test_sizes, config.per_class_test, pe_grid[0], rng)

    results = []
    for pe_cfg in pe_grid:
        for s in train_samples + val_samples + test_samples:
            s.tokens = compute_pe(s.delta, pe_cfg)
        model = train_classifier(train_samples, val_samples, config.num_classes, train_cfg)
        test_error = evaluate_classifier(model, test_samples, train_cfg)
        if discrepancy_mode == "proportional":
            discrepancy = discrepancy_set_proportional(
                [s.tokens for s in train_samples],
                [s.tokens for s in test_samples],
                total_samples=128 * len(train_samples),
                projections=50,
                rng=rng,
            )
        elif discrepancy_mode == "uniform":
            discrepancy = discrepancy_set(
                [s.tokens for s in train_samples],
                [s.tokens for s in test_samples],
                samples_per_graph=128,
                projections=50,
                rng=rng,
            )
        elif discrepancy_mode == "all":
            discrepancy = discrepancy_set_all(
                [s.tokens for s in train_samples],
                [s.tokens for s in test_samples],
                projections=50,
                rng=rng,
            )
        else:
            raise ValueError(f"Unknown discrepancy_mode: {discrepancy_mode}")
        eig_stats = []
        for s in test_samples:
            evals = np.linalg.eigvalsh(s.delta)
            eig_stats.append(eigengap_stats(evals, pe_cfg.k))
        avg_min_gap = float(np.mean([e.min_gap for e in eig_stats]))
        avg_gap_k = float(np.mean([e.gap_k for e in eig_stats]))
        results.append(
            {
                "pe": pe_cfg.__dict__,
                "test_error": test_error,
                "discrepancy_set": discrepancy,
                "discrepancy_mode": discrepancy_mode,
                "eigengap_min": avg_min_gap,
                "eigengap_k": avg_gap_k,
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "pe_sweep.json").write_text(json.dumps(results, indent=2))
    return {"results": results}
