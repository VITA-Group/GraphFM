from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "matplotlib is required for plotting. Install it and retry."
    ) from exc


def load_results(input_dir: Path) -> List[Tuple[float, float, float, float, float, float]]:
    rows: List[Tuple[float, float, float, float, float, float]] = []
    for path in sorted(input_dir.glob("size_shift_lambda_*.json*")):
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if "lambda_mix" not in data:
            continue
        rows.append(
            (
                float(data["lambda_mix"]),
                float(data.get("discrepancy_set", float("nan"))),
                float(data.get("train_error", float("nan"))),
                float(data.get("test_error", float("nan"))),
                float(data.get("id_error", float("nan"))),
                float(data.get("ood_error", float("nan"))),
            )
        )
    rows.sort(key=lambda x: x[0])
    return rows


def _series(
    rows: List[Tuple[float, float, float, float, float, float]]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lambdas = np.array([r[0] for r in rows])
    discrepancy = np.array([r[1] for r in rows])
    train_error = np.array([r[2] for r in rows])
    test_error = np.array([r[3] for r in rows])
    id_error = np.array([r[4] for r in rows])
    ood_error = np.array([r[5] for r in rows])
    return lambdas, discrepancy, train_error, test_error, id_error, ood_error


def _global_limits(values: List[np.ndarray]) -> Tuple[Optional[float], Optional[float]]:
    if not values:
        return None, None
    stacked = np.concatenate([v[~np.isnan(v)] for v in values if v.size])
    if stacked.size == 0:
        return None, None
    return float(np.min(stacked)), float(np.max(stacked))


def plot_single(rows: List[Tuple[float, float, float, float, float, float]], output_path: Path) -> None:
    lambdas, discrepancy, train_error, test_error, id_error, ood_error = _series(rows)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(lambdas, discrepancy, "o-", color="#1f77b4", label="discrepancy_set")
    ax1.set_xlabel("lambda_mix")
    ax1.set_ylabel("discrepancy_set", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")

    ax2 = ax1.twinx()
    ax2.plot(lambdas, train_error, "d-.", color="#2ca02c", label="train_error")
    ax2.plot(lambdas, test_error, "s--", color="#d62728", label="test_error")
    ax2.plot(lambdas, id_error, "^:", color="#9467bd", label="id_error")
    ax2.plot(lambdas, ood_error, "v:", color="#ff7f0e", label="ood_error")
    ax2.set_ylabel("error", color="#333333")
    ax2.tick_params(axis="y", labelcolor="#333333")
    ax2.legend(loc="upper right")

    ax1.set_title("Size-shift: discrepancy and errors vs lambda_mix")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)


def plot_compare(
    rows_a: List[Tuple[float, float, float, float, float, float]],
    rows_b: List[Tuple[float, float, float, float, float, float]],
    output_path: Path,
    label_a: str,
    label_b: str,
) -> None:
    lambdas_a, discrepancy_a, train_error_a, test_error_a, id_error_a, ood_error_a = _series(rows_a)
    lambdas_b, discrepancy_b, train_error_b, test_error_b, id_error_b, ood_error_b = _series(rows_b)

    disc_min, disc_max = _global_limits([discrepancy_a, discrepancy_b])
    err_min, err_max = _global_limits(
        [train_error_a, test_error_a, id_error_a, ood_error_a,
         train_error_b, test_error_b, id_error_b, ood_error_b]
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=False)

    ax1 = axes[0]
    ax1.plot(lambdas_a, discrepancy_a, "o-", color="#1f77b4", label="discrepancy_set")
    ax1.set_xlabel("lambda_mix")
    ax1.set_ylabel("discrepancy_set", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    if disc_min is not None and disc_max is not None:
        ax1.set_ylim(disc_min, disc_max)
    ax1.set_title(label_a)

    ax1b = ax1.twinx()
    ax1b.plot(lambdas_a, train_error_a, "d-.", color="#2ca02c", label="train_error")
    ax1b.plot(lambdas_a, test_error_a, "s--", color="#d62728", label="test_error")
    ax1b.plot(lambdas_a, id_error_a - train_error_a, "^:", color="#9467bd", label="id_gap")
    ax1b.plot(lambdas_a, ood_error_a - train_error_a, "v:", color="#ff7f0e", label="ood_gap")
    # ax1b.plot(lambdas_a, test_error_a - train_error_a, "h--", color="#17becf", label="gap")

    ax1b.set_ylabel("error", color="#333333")
    ax1b.tick_params(axis="y", labelcolor="#333333")
    ax1b.legend(loc="upper right", fontsize=8)
    if err_min is not None and err_max is not None:
        ax1b.set_ylim(err_min, err_max)

    ax2 = axes[1]
    ax2.plot(lambdas_b, discrepancy_b, "o-", color="#1f77b4", label="discrepancy_set")
    ax2.set_xlabel("lambda_mix")
    ax2.set_ylabel("discrepancy_set", color="#1f77b4")
    ax2.tick_params(axis="y", labelcolor="#1f77b4")
    if disc_min is not None and disc_max is not None:
        ax2.set_ylim(disc_min, disc_max)
    ax2.set_title(label_b)

    ax2b = ax2.twinx()
    ax2b.plot(lambdas_b, train_error_b, "d-.", color="#2ca02c", label="train_error")
    ax2b.plot(lambdas_b, test_error_b, "s--", color="#d62728", label="test_error")
    ax2b.plot(lambdas_b, id_error_b - train_error_b, "^:", color="#9467bd", label="id_gap")
    ax2b.plot(lambdas_b, ood_error_b, "v:", color="#ff7f0e", label="ood_gap")
    # ax2b.plot(lambdas_b, test_error_b - train_error_b, "h--", color="#17becf", label="gap")

    ax2b.set_ylabel("error", color="#333333")
    ax2b.tick_params(axis="y", labelcolor="#333333")
    ax2b.legend(loc="upper right", fontsize=8)
    if err_min is not None and err_max is not None:
        ax2b.set_ylim(err_min, err_max)

    fig.suptitle("Size-shift: discrepancy and errors vs lambda_mix")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str)
    parser.add_argument("--input_dir_a", type=str)
    parser.add_argument("--input_dir_b", type=str)
    parser.add_argument("--label_a", type=str, default="A")
    parser.add_argument("--label_b", type=str, default="B")
    parser.add_argument("--output", type=str, default="size_shift_lambda_plot.png")
    args = parser.parse_args()

    if args.input_dir:
        rows = load_results(Path(args.input_dir))
        if not rows:
            print(rows)
            raise SystemExit("No size_shift_lambda_*.json files found.")
        plot_single(rows, Path(args.output))
        return

    if not args.input_dir_a or not args.input_dir_b:
        raise SystemExit("Provide --input_dir for single plot or both --input_dir_a/--input_dir_b for comparison.")

    rows_a = load_results(Path(args.input_dir_a))
    rows_b = load_results(Path(args.input_dir_b))
    if not rows_a or not rows_b:
        raise SystemExit("No size_shift_lambda_*.json files found in one of the input dirs.")
    plot_compare(rows_a, rows_b, Path(args.output), args.label_a, args.label_b)


if __name__ == "__main__":
    main()
