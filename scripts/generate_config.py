from __future__ import annotations

import argparse
from pathlib import Path

from graphfm.config import generate_config_filename, save_config
from graphfm.experiments import DatasetConfig
from graphfm.pe import PEConfig
from graphfm.train import TrainConfig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a config file with descriptive filename"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="configs",
        help="Output directory for config file",
    )

    # Experiment config
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--rho", type=float, default=0.5)
    parser.add_argument("--num_terms", type=int, default=5)
    parser.add_argument("--coeff_scale", type=float, default=0.2)
    parser.add_argument(
        "--train_sizes",
        type=int,
        nargs="+",
        default=[64, 128, 256, 512],
    )
    parser.add_argument(
        "--test_sizes",
        type=int,
        nargs="+",
        default=[64, 128, 256, 512, 768, 1024],
    )
    parser.add_argument("--per_class_train", type=int, default=3)
    parser.add_argument("--per_class_test", type=int, default=2)
    parser.add_argument("--total_budget", type=int, default=10000)
    parser.add_argument("--lambda_mix", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)

    # Train config
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--model", type=str, default="deepsets")
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--degree_bins", type=int, default=32)
    parser.add_argument("--device", type=str, default="cpu")

    # PE config
    parser.add_argument("--pe_kind", type=str, default="eig")
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--m", type=int, default=16)
    parser.add_argument("--spe_alpha", type=float, default=10.0)
    parser.add_argument("--spe_tau", type=float, default=0.0)
    parser.add_argument("--pe_seed", type=int, default=0)

    args = parser.parse_args()

    exp_cfg = DatasetConfig(
        num_classes=args.num_classes,
        rho=args.rho,
        num_terms=args.num_terms,
        coeff_scale=args.coeff_scale,
        train_sizes=tuple(args.train_sizes),
        test_sizes=tuple(args.test_sizes),
        per_class_train=args.per_class_train,
        per_class_test=args.per_class_test,
        total_budget=args.total_budget,
        lambda_mix=args.lambda_mix,
        seed=args.seed,
    )

    train_cfg = TrainConfig(
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        model=args.model,
        hidden=args.hidden,
        degree_bins=args.degree_bins,
        device=args.device,
    )

    pe_cfg = PEConfig(
        kind=args.pe_kind,
        k=args.k,
        m=args.m,
        spe_alpha=args.spe_alpha,
        spe_tau=args.spe_tau,
        seed=args.pe_seed,
    )

    filename = generate_config_filename(exp_cfg, train_cfg, pe_cfg)
    output_path = Path(args.output_dir) / filename

    save_config(output_path, exp_cfg, train_cfg, pe_cfg)
    print(f"Config saved to: {output_path}")


if __name__ == "__main__":
    main()
