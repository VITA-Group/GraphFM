from __future__ import annotations

import argparse
from pathlib import Path
import sys

# ROOT = Path(__file__).resolve().parents[1]
# SRC = ROOT / "src"
# if str(SRC) not in sys.path:
#     sys.path.insert(0, str(SRC))

from graphfm.experiments import ExperimentConfig, run_pe_sweep, run_size_shift
from graphfm.pe import PEConfig
from graphfm.train import TrainConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=["size_shift", "pe_sweep"], required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--pe_kind", type=str, default="eig")
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--m", type=int, default=16)
    parser.add_argument("--lambda_mix", type=float, default=0.0)
    parser.add_argument("--use_merging", action="store_true")
    parser.add_argument("--model", type=str, default="deepsets")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--discrepancy_mode",
        choices=["uniform", "proportional", "all"],
        default="uniform",
    )
    args = parser.parse_args()

    out_dir = Path(args.output)
    exp_cfg = ExperimentConfig()
    train_cfg = TrainConfig(model=args.model, device=args.device)

    if args.experiment == "size_shift":
        pe_cfg = PEConfig(kind=args.pe_kind, k=args.k, m=args.m)
        run_size_shift(
            out_dir=out_dir,
            pe_cfg=pe_cfg,
            train_cfg=train_cfg,
            config=exp_cfg,
            use_merging=args.use_merging,
            lambda_mix=args.lambda_mix,
            discrepancy_mode=args.discrepancy_mode,
        )
        return

    if args.experiment == "pe_sweep":
        pe_grid = [
            PEConfig(kind="eig", k=k) for k in [8, 16, 32, 64]
        ] + [
            PEConfig(kind="proj", k=k, m=m) for k in [8, 16, 32] for m in [8, 16, 32]
        ]
        run_pe_sweep(out_dir=out_dir, pe_grid=pe_grid, train_cfg=train_cfg, config=exp_cfg, discrepancy_mode=args.discrepancy_mode)
        return


if __name__ == "__main__":
    main()
