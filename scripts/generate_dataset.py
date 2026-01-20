from __future__ import annotations

import argparse
from pathlib import Path

from graphfm.experiments import ExperimentConfig, generate_datasets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", type=str, required=True)
    parser.add_argument("--lambda_mix", type=float, default=0.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    exp_cfg = ExperimentConfig(lambda_mix=args.lambda_mix)
    cache_path = generate_datasets(
        cache_dir=cache_dir,
        config=exp_cfg,
        overwrite=args.overwrite,
    )
    print(f"Saved dataset cache: {cache_path}")


if __name__ == "__main__":
    main()
