from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "matplotlib is required for plotting. Install it and retry."
    ) from exc


def load_results(input_dir: Path) -> List[Tuple[float, float, float]]:
    rows: List[Tuple[float, float, float]] = []
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
                float(data.get("test_error", float("nan"))),
            )
        )
    rows.sort(key=lambda x: x[0])
    return rows


def plot(rows: List[Tuple[float, float, float]], output_path: Path) -> None:
    lambdas = np.array([r[0] for r in rows])
    discrepancy = np.array([r[1] for r in rows])
    test_error = np.array([r[2] for r in rows])

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(lambdas, discrepancy, "o-", color="#1f77b4", label="discrepancy_set")
    ax1.set_xlabel("lambda_mix")
    ax1.set_ylabel("discrepancy_set", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")

    ax2 = ax1.twinx()
    ax2.plot(lambdas, test_error, "s--", color="#d62728", label="test_error")
    ax2.set_ylabel("test_error", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")

    ax1.set_title("Size-shift: discrepancy and test error vs lambda_mix")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output", type=str, default="size_shift_lambda_plot.png")
    args = parser.parse_args()

    rows = load_results(Path(args.input_dir))
    if not rows:
        raise SystemExit("No size_shift_lambda_*.json files found.")
    plot(rows, Path(args.output))


if __name__ == "__main__":
    main()
