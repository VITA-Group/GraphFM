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
    for path in sorted(input_dir.glob("*.json*")):
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

    label_fs = 22
    tick_fs = 20
    legend_fs = 20

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(lambdas, discrepancy, "o-", color="#1f77b4", label="Discrepancy")
    ax1.set_xlabel(r"$\lambda$", fontsize=label_fs)
    ax1.set_ylabel("Discrepancy", color="#1f77b4", fontsize=label_fs)
    ax1.tick_params(axis="y", labelcolor="#1f77b4", labelsize=tick_fs)
    ax1.tick_params(axis="x", labelsize=tick_fs)

    ax2 = ax1.twinx()
    ax2.plot(lambdas, train_error, "d-.", color="#2ca02c", label="Train Error")
    ax2.plot(lambdas, test_error, "s--", color="#d62728", label="Test Error")
    # ax2.plot(lambdas, id_error, "^:", color="#9467bd", label="ID Error")
    # ax2.plot(lambdas, ood_error, "v:", color="#ff7f0e", label="OOD Error")
    ax2.set_ylabel("Error", color="#333333", fontsize=label_fs)
    ax2.tick_params(axis="y", labelcolor="#333333", labelsize=tick_fs)
    ax2.legend(loc="upper right", fontsize=legend_fs)
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

    label_fs = 22
    tick_fs = 20
    legend_fs = 20

    err_min, err_max = _global_limits([test_error_a, test_error_b])

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        lambdas_a,
        test_error_a,
        marker="s",
        linestyle="--",
        color="#d62728",
        label="w/o Merging",
        linewidth=2,
    )
    ax.plot(
        lambdas_b,
        test_error_b,
        marker="o",
        linestyle="-",
        color="#1f77b4",
        label="w/ Merging",
        linewidth=2,
    )

    ax.set_xlabel(r"$\lambda$", fontsize=label_fs)
    ax.set_ylabel("Test Error", fontsize=label_fs)
    ax.tick_params(axis="x", labelsize=tick_fs)
    ax.tick_params(axis="y", labelsize=tick_fs)
    ax.legend(loc="upper right", fontsize=legend_fs)
    if err_min is not None and err_max is not None:
        padding = 0.05 * (err_max - err_min) if err_max > err_min else 0.02
        ax.set_ylim(err_min - padding, err_max + padding)

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
