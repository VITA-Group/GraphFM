from __future__ import annotations

import argparse
from pathlib import Path

from graphfm.config import load_config
from graphfm.dataset import DatasetConfig, generate_datasets


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate and cache dataset for experiments"
    )
    parser.add_argument("--cache_dir", type=str, required=True)
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--lambda_mix", type=float, default=None)
    parser.add_argument(
        "--sampling_mode",
        choices=["uniform_value", "bin_value", "uniform_bernoulli"],
        default="uniform_value",
        help="Sampling mode for graph generation",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)

    if args.config:
        # Load dataset config from YAML file
        dataset_cfg, _, _ = load_config(Path(args.config))
        # Allow command-line args to override config values
        lambda_mix = args.lambda_mix if args.lambda_mix is not None else dataset_cfg.lambda_mix
        sampling_mode = args.sampling_mode if args.sampling_mode is not None else dataset_cfg.sampling_mode
        dataset_cfg = DatasetConfig(
            num_classes=dataset_cfg.num_classes,
            rho=dataset_cfg.rho,
            num_terms=dataset_cfg.num_terms,
            coeff_scale=dataset_cfg.coeff_scale,
            train_sizes=dataset_cfg.train_sizes,
            test_sizes=dataset_cfg.test_sizes,
            per_class_train=dataset_cfg.per_class_train,
            per_class_test=dataset_cfg.per_class_test,
            total_budget=dataset_cfg.total_budget,
            seed=dataset_cfg.seed,
            lambda_mix=lambda_mix,
            sampling_mode=sampling_mode,
        )
    else:
        # Use defaults with optional overrides
        dataset_cfg = DatasetConfig(
            lambda_mix=args.lambda_mix if args.lambda_mix is not None else 0.0,
            sampling_mode=args.sampling_mode if args.sampling_mode is not None else "uniform_value",
        )

    cache_path = generate_datasets(
        cache_dir=cache_dir,
        config=dataset_cfg,
        overwrite=args.overwrite,
    )
    print(f"Saved dataset cache: {cache_path}")


if __name__ == "__main__":
    main()
