from __future__ import annotations

import argparse
from pathlib import Path

from graphfm.config import load_config, merge_config_with_args
from graphfm.dataset import DatasetConfig
from graphfm.experiments import run_merge_graph, run_pe_sweep, run_perturb_graphon, run_size_shift
from graphfm.pe import PEConfig
from graphfm.train import TrainConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphon_type", type=str, default="controlled_fourier")
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--experiment", choices=["size_shift", "pe_sweep", "merge_graph", "perturb_graphon"], required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--pe_kind", type=str, default=None)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--m", type=int, default=None)
    parser.add_argument("--lambda_mix", type=float, default=None)
    parser.add_argument(
        "--sampling_mode",
        choices=["uniform_value", "bin_value", "uniform_bernoulli"],
        default="uniform_value",
        help="Sampling mode for graph generation",
    )
    parser.add_argument(
        "--merging_method",
        choices=["degree", "spectral", "usvt"],
        default=None,
        help="Graphon estimation method for merging (None = no merging)",
    )
    parser.add_argument(
        "--merging_ratio",
        type=float,
        default=0.5,
        help="Ratio of merged graphs to original graphs per class",
    )
    parser.add_argument(
        "--merging_size",
        type=float,
        choices=[1.5, 2.0, 3.0],
        default=2.0,
        help="Size multiplier for merged graphs",
    )
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--cache_dir", type=str)
    parser.add_argument(
        "--discrepancy_mode",
        choices=["uniform", "proportional", "all"],
        default="uniform",
    )
    # perturb_graphon experiment arguments
    parser.add_argument(
        "--perturb_levels",
        type=float,
        nargs="+",
        default=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        help="Perturbation levels to evaluate (each in [0, 1])",
    )
    parser.add_argument(
        "--perturb_ratio",
        type=float,
        default=0.5,
        help="Fraction of test graphs from perturbed graphons",
    )
    parser.add_argument(
        "--max_l2_distance",
        type=float,
        default=0.1,
        help="Maximum L2 distance at perturb_level=1.0",
    )
    args = parser.parse_args()

    # Load config from file or use defaults
    if args.config:
        exp_cfg, train_cfg, pe_cfg = load_config(Path(args.config))
        exp_cfg, train_cfg, pe_cfg = merge_config_with_args(
            exp_cfg, train_cfg, pe_cfg, args
        )
    else:
        # Fallback to command-line args with defaults
        exp_cfg = DatasetConfig(
            lambda_mix=args.lambda_mix if args.lambda_mix is not None else 0.0,
            sampling_mode=args.sampling_mode if args.sampling_mode is not None else "uniform_value",
        )
        train_cfg = TrainConfig(
            model=args.model if args.model is not None else "deepsets",
            device=args.device if args.device is not None else "cpu",
        )
        pe_cfg = PEConfig(
            kind=args.pe_kind if args.pe_kind is not None else "eig",
            k=args.k if args.k is not None else 16,
            m=args.m if args.m is not None else 16,
        )

    out_dir = Path(args.output)

    if args.experiment == "size_shift":
        run_size_shift(
            out_dir=out_dir,
            pe_cfg=pe_cfg,
            train_cfg=train_cfg,
            config=exp_cfg,
            merging_method=args.merging_method,
            discrepancy_mode=args.discrepancy_mode,
            cache_dir=Path(args.cache_dir) if args.cache_dir else None,
        )
        return

    if args.experiment == "pe_sweep":
        pe_grid = [
            PEConfig(kind="eig", k=k) for k in [8, 16, 32, 64]
        ] + [
            PEConfig(kind="proj", k=k, m=m) for k in [8, 16, 32] for m in [8, 16, 32]
        ]
        run_pe_sweep(
            out_dir=out_dir,
            pe_grid=pe_grid,
            train_cfg=train_cfg,
            config=exp_cfg,
            discrepancy_mode=args.discrepancy_mode,
            cache_dir=Path(args.cache_dir) if args.cache_dir else None,
        )
        return

    if args.experiment == "merge_graph":
        run_merge_graph(
            out_dir=out_dir,
            pe_cfg=pe_cfg,
            train_cfg=train_cfg,
            config=exp_cfg,
            merging_method=args.merging_method if args.merging_method else "spectral",
            merging_ratio=args.merging_ratio,
            merging_size=args.merging_size,
            discrepancy_mode=args.discrepancy_mode,
            cache_dir=Path(args.cache_dir) if args.cache_dir else None,
        )
        return

    if args.experiment == "perturb_graphon":
        # Ensure graphon_type is controlled_fourier for this experiment
        exp_cfg.graphon_type = "controlled_fourier"
        run_perturb_graphon(
            out_dir=out_dir,
            pe_cfg=pe_cfg,
            train_cfg=train_cfg,
            config=exp_cfg,
            perturb_levels=args.perturb_levels,
            perturb_ratio=args.perturb_ratio,
            max_l2_distance=args.max_l2_distance,
            discrepancy_mode=args.discrepancy_mode,
            cache_dir=Path(args.cache_dir) if args.cache_dir else None,
        )
        return


if __name__ == "__main__":
    main()
