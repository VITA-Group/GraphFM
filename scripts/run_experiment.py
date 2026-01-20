from __future__ import annotations

import argparse
from pathlib import Path

from graphfm.config import load_config, merge_config_with_args
from graphfm.experiments import DatasetConfig, run_pe_sweep, run_size_shift
from graphfm.pe import PEConfig
from graphfm.train import TrainConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--experiment", choices=["size_shift", "pe_sweep"], required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--pe_kind", type=str, default=None)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--m", type=int, default=None)
    parser.add_argument("--lambda_mix", type=float, default=None)
    parser.add_argument("--use_merging", action="store_true")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--cache_dir", type=str)
    parser.add_argument(
        "--discrepancy_mode",
        choices=["uniform", "proportional", "all"],
        default="uniform",
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
            lambda_mix=args.lambda_mix if args.lambda_mix is not None else 0.0
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
            use_merging=args.use_merging,
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


if __name__ == "__main__":
    main()
