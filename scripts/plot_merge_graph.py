from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
except ImportError as exc:
    raise SystemExit(
        "matplotlib is required for plotting. Install it and retry."
    ) from exc


def load_merge_results(input_dir: Path) -> List[Dict]:
    """Load merge_graph experiment results from JSON files."""
    results = []
    for path in sorted(input_dir.glob("merge_graph_*.json")):
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if "merging_ratio" not in data or "merging_size" not in data:
            continue
        results.append(data)
    return results


def build_grid(
    results: List[Dict],
    metric: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a 2D grid of metric values indexed by ratio and size."""
    ratios = sorted(set(r["merging_ratio"] for r in results))
    sizes = sorted(set(r["merging_size"] for r in results))

    grid = np.full((len(ratios), len(sizes)), np.nan)
    ratio_idx = {r: i for i, r in enumerate(ratios)}
    size_idx = {s: i for i, s in enumerate(sizes)}

    for r in results:
        i = ratio_idx[r["merging_ratio"]]
        j = size_idx[r["merging_size"]]
        grid[i, j] = r.get(metric, np.nan)

    return np.array(ratios), np.array(sizes), grid


def plot_heatmaps(results: List[Dict], output_path: Path) -> None:
    """Create a 2x3 grid of heatmaps for different metrics."""
    metrics = [
        ("test_error", "Test Error"),
        ("train_error", "Train Error"),
        ("id_error", "ID Error"),
        ("ood_error", "OOD Error"),
        ("discrepancy_set", "Discrepancy"),
        ("num_merged", "Num Merged"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()

    for ax, (metric, title) in zip(axes, metrics):
        ratios, sizes, grid = build_grid(results, metric)

        if np.all(np.isnan(grid)):
            ax.set_title(f"{title}\n(no data)")
            ax.axis("off")
            continue

        im = ax.imshow(grid, aspect="auto", cmap="RdYlGn_r", origin="lower")
        ax.set_xticks(range(len(sizes)))
        ax.set_xticklabels([f"{s:.1f}" for s in sizes])
        ax.set_yticks(range(len(ratios)))
        ax.set_yticklabels([f"{r:.2f}" for r in ratios])
        ax.set_xlabel("Merging Size")
        ax.set_ylabel("Merging Ratio")
        ax.set_title(title)

        # Annotate cells with values
        for i in range(len(ratios)):
            for j in range(len(sizes)):
                val = grid[i, j]
                if not np.isnan(val):
                    text_color = "white" if val > (np.nanmin(grid) + np.nanmax(grid)) / 2 else "black"
                    if metric == "num_merged":
                        ax.text(j, i, f"{int(val)}", ha="center", va="center", color=text_color, fontsize=9)
                    else:
                        ax.text(j, i, f"{val:.3f}", ha="center", va="center", color=text_color, fontsize=9)

        fig.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle("Merge Graph Experiment Results", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    print(f"Saved heatmap plot to {output_path}")


def plot_line_by_ratio(results: List[Dict], output_path: Path) -> None:
    """Create line plots showing metrics vs merging_size, one line per ratio."""
    metrics = [
        ("test_error", "Test Error"),
        ("ood_error", "OOD Error"),
        ("discrepancy_set", "Discrepancy"),
    ]

    ratios = sorted(set(r["merging_ratio"] for r in results))
    sizes = sorted(set(r["merging_size"] for r in results))
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(ratios)))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, (metric, title) in zip(axes, metrics):
        for ratio, color in zip(ratios, colors):
            ratio_results = [r for r in results if r["merging_ratio"] == ratio]
            ratio_results.sort(key=lambda x: x["merging_size"])
            x = [r["merging_size"] for r in ratio_results]
            y = [r.get(metric, np.nan) for r in ratio_results]
            ax.plot(x, y, "o-", color=color, label=f"ratio={ratio:.2f}", linewidth=2, markersize=8)

        ax.set_xlabel("Merging Size")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Merge Graph: Metrics vs Merging Size", fontsize=14)
    fig.tight_layout()
    output_line = output_path.with_name(output_path.stem + "_lines" + output_path.suffix)
    fig.savefig(output_line, dpi=200)
    print(f"Saved line plot to {output_line}")


def plot_line_by_size(results: List[Dict], output_path: Path) -> None:
    """Create line plots showing metrics vs merging_ratio, one line per size."""
    metrics = [
        ("test_error", "Test Error"),
        ("ood_error", "OOD Error"),
        ("discrepancy_set", "Discrepancy"),
    ]

    ratios = sorted(set(r["merging_ratio"] for r in results))
    sizes = sorted(set(r["merging_size"] for r in results))
    colors = plt.cm.plasma(np.linspace(0, 0.8, len(sizes)))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, (metric, title) in zip(axes, metrics):
        for size, color in zip(sizes, colors):
            size_results = [r for r in results if r["merging_size"] == size]
            size_results.sort(key=lambda x: x["merging_ratio"])
            x = [r["merging_ratio"] for r in size_results]
            y = [r.get(metric, np.nan) for r in size_results]
            ax.plot(x, y, "s-", color=color, label=f"size={size:.1f}", linewidth=2, markersize=8)

        ax.set_xlabel("Merging Ratio")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Merge Graph: Metrics vs Merging Ratio", fontsize=14)
    fig.tight_layout()
    output_line = output_path.with_name(output_path.stem + "_by_ratio" + output_path.suffix)
    fig.savefig(output_line, dpi=200)
    print(f"Saved line plot to {output_line}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot merge_graph experiment results")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory with merge_graph JSON results")
    parser.add_argument("--output", type=str, default="merge_graph_plot.png", help="Output file path")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_path = Path(args.output)

    results = load_merge_results(input_dir)
    if not results:
        raise SystemExit(f"No merge_graph_*.json files found in {input_dir}")

    print(f"Found {len(results)} merge_graph result files")

    # Generate all three plot types
    plot_heatmaps(results, output_path)
    plot_line_by_ratio(results, output_path)
    plot_line_by_size(results, output_path)


if __name__ == "__main__":
    main()
