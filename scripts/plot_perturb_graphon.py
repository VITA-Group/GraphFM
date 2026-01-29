from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise SystemExit(
        "matplotlib is required for plotting. Install it and retry."
    ) from exc


def load_results(input_dir: Path) -> List[Dict]:
    """Load perturb_graphon results from JSON files."""
    results: List[Dict] = []
    for path in sorted(input_dir.glob("perturb_graphon_level_*.json")):
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if "perturb_level" not in data:
            continue
        results.append(data)
    # Sort by perturb_level
    results.sort(key=lambda x: x["perturb_level"])
    return results


def load_summary(input_dir: Path) -> Dict:
    """Load summary file if available."""
    summary_path = input_dir / "perturb_graphon_summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text())
    return {}


def plot_perturb_graphon(results: List[Dict], output_path: Path, title: str = "") -> None:
    """Plot perturb_graphon results.

    X-axis: perturb_level (with L2 distance in brackets)
    Y-axis: test error (with ID/OOD breakdown)
    """
    if not results:
        raise SystemExit("No perturb_graphon results found.")

    # Extract data
    perturb_levels = [r["perturb_level"] for r in results]
    avg_l2_distances = [r["avg_l2_distance"] for r in results]
    test_errors = [r["test_error"] for r in results]
    id_errors = [r["id_error"] for r in results]
    ood_errors = [r["ood_error"] for r in results]
    train_errors = [r["train_error"] for r in results]

    # Create x-axis labels: "level (L2=0.xxx)"
    x_labels = [
        f"{level:.1f}\n(L2={l2:.3f})"
        for level, l2 in zip(perturb_levels, avg_l2_distances)
    ]

    label_fs = 22
    tick_fs = 20
    legend_fs = 20

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(perturb_levels))
    width = 0.6

    # Plot lines
    ax.plot(x, test_errors, "s-", color="#d62728", label="Test Error", linewidth=2, markersize=8)
    ax.plot(x, id_errors, "^--", color="#2ca02c", label="ID Error (Original)", linewidth=1.5, markersize=7)
    ax.plot(x, ood_errors, "v--", color="#ff7f0e", label="OOD Error (Perturbed)", linewidth=1.5, markersize=7)
    ax.axhline(
        y=train_errors[0],
        color="#1f77b4",
        linestyle=":",
        label=f"Train Error ({train_errors[0]:.3f})",
        linewidth=1.5,
    )

    ax.set_xlabel("Perturbation Level (avg L2 distance)", fontsize=label_fs)
    ax.set_ylabel("Error", fontsize=label_fs)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=tick_fs)
    ax.tick_params(axis="y", labelsize=tick_fs)
    ax.legend(loc="upper left", fontsize=legend_fs)
    ax.grid(True, alpha=0.3)

    # Set y-axis limits with some padding
    all_errors = test_errors + id_errors + ood_errors + train_errors
    y_min = max(0, min(all_errors) - 0.05)
    y_max = min(1, max(all_errors) + 0.05)
    ax.set_ylim(y_min, y_max)

    ax.tick_params(axis="x", labelsize=tick_fs)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    print(f"Saved plot to {output_path}")


def plot_multi_lambda(
    results_by_lambda: Dict[float, List[Dict]],
    output_path: Path,
) -> None:
    """Plot results for multiple lambda_mix values on the same plot."""
    label_fs = 22
    tick_fs = 20
    legend_fs = 20

    fig, ax = plt.subplots(figsize=(12, 7))

    colors = plt.cm.viridis(np.linspace(0, 1, len(results_by_lambda)))

    for (lambda_mix, results), color in zip(sorted(results_by_lambda.items()), colors):
        perturb_levels = [r["perturb_level"] for r in results]
        avg_l2_distances = [r["avg_l2_distance"] for r in results]
        test_errors = [r["test_error"] for r in results]

        # Create x-axis values
        x = np.arange(len(perturb_levels))

        ax.plot(
            x,
            test_errors,
            "o-",
            color=color,
            label=rf"$\lambda$={lambda_mix:.1f}",
            linewidth=2,
            markersize=6,
        )

    # Use first result set for x-axis labels
    first_results = list(results_by_lambda.values())[0]
    perturb_levels = [r["perturb_level"] for r in first_results]
    avg_l2_distances = [r["avg_l2_distance"] for r in first_results]
    x_labels = [
        f"{level:.1f}\n(L2={l2:.3f})"
        for level, l2 in zip(perturb_levels, avg_l2_distances)
    ]

    x = np.arange(len(perturb_levels))
    ax.set_xlabel("Perturbation Level (avg L2 distance)", fontsize=label_fs)
    ax.set_ylabel("Test Error", fontsize=label_fs)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=tick_fs)
    ax.tick_params(axis="y", labelsize=tick_fs)
    ax.legend(loc="upper left", fontsize=legend_fs)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="x", labelsize=tick_fs)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    print(f"Saved multi-lambda plot to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot perturb_graphon experiment results")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing result JSON files")
    parser.add_argument("--output", type=str, default=None, help="Output plot path")
    parser.add_argument("--title", type=str, default="", help="Custom plot title")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)

    # Try to load summary first
    summary = load_summary(input_dir)

    # Load per-level results
    results = load_results(input_dir)

    if not results and summary and "results" in summary:
        # Use results from summary if per-level files not found
        results = summary["results"]

    if not results:
        # Try to find multiple lambda_mix runs
        results_by_lambda: Dict[float, List[Dict]] = {}
        for path in input_dir.glob("perturb_graphon_level_*_lambda_*.json"):
            try:
                data = json.loads(path.read_text())
                lambda_mix = data.get("lambda_mix", 0.0)
                if lambda_mix not in results_by_lambda:
                    results_by_lambda[lambda_mix] = []
                results_by_lambda[lambda_mix].append(data)
            except json.JSONDecodeError:
                continue

        if results_by_lambda:
            # Sort each lambda's results by perturb_level
            for lm in results_by_lambda:
                results_by_lambda[lm].sort(key=lambda x: x["perturb_level"])

            if len(results_by_lambda) > 1:
                # Multiple lambda values - create comparison plot
                output_path = Path(args.output) if args.output else input_dir / "perturb_graphon_multi_lambda.png"
                plot_multi_lambda(results_by_lambda, output_path)

            # Also create individual plots
            for lambda_mix, res in results_by_lambda.items():
                lambda_tag = f"{lambda_mix:.2f}".replace(".", "p")
                output_path = input_dir / f"perturb_graphon_lambda_{lambda_tag}.png"
                plot_perturb_graphon(res, output_path, title=f"lambda_mix={lambda_mix}")
            return

        raise SystemExit(f"No perturb_graphon results found in {input_dir}")

    # Single lambda_mix case
    output_path = Path(args.output) if args.output else input_dir / "perturb_graphon_plot.png"
    plot_perturb_graphon(results, output_path, title=args.title)


if __name__ == "__main__":
    main()
