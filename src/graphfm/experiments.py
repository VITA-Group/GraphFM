from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .dataset import (
    GraphSample,
    apply_pe,
    load_dataset,
    sample_with_allocation_raw,
    sample_graphs_raw,
    save_dataset,
    size_allocation_path,
)
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




def _generate_size_shift_samples(
    config: DatasetConfig,
    rng: np.random.Generator,
) -> Tuple[List[GraphSample], List[GraphSample], List[GraphSample]]:
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
    train_samples = sample_with_allocation_raw(graphons, allocation, rng)
    train_samples, val_samples = _split_train_val(train_samples, 0.2, rng)
    test_samples = sample_graphs_raw(
        graphons, config.test_sizes, config.per_class_test, rng
    )
    return train_samples, val_samples, test_samples




def generate_size_shift_dataset(
    cache_dir: Path,
    config: DatasetConfig,
    overwrite: bool = False,
) -> Path:
    cache_path = _dataset_cache_path(cache_dir, config)
    if cache_path.exists() and not overwrite:
        return cache_path
    rng = np.random.default_rng(config.seed)
    train_samples, val_samples, test_samples = _generate_size_shift_samples(
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
    train_samples, val_samples, test_samples = _generate_size_shift_samples(
        config=config,
        rng=rng,
    )
    save_dataset(cache_path, train_samples, val_samples, test_samples)
    return cache_path

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
        train_samples, val_samples, test_samples = _generate_size_shift_samples(
            config=config,
            rng=rng,
        )
    apply_pe(train_samples, pe_cfg)
    apply_pe(val_samples, pe_cfg)
    apply_pe(test_samples, pe_cfg)

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
        train_samples, val_samples, test_samples = _generate_size_shift_samples(
            config=config,
            rng=rng,
        )

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
