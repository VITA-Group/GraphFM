from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise SystemExit(
        "matplotlib is required for plotting. Install it and retry."
    ) from exc


def load_results(input_dir: Path) -> List[Dict]:
    """Load PE sweep results from JSON files."""
    results: List[Dict] = []

    # Try loading individual files first
    for path in sorted(input_dir.glob("pe_sweep_*.json")):
        if path.name == "pe_sweep.json":
            continue  # Skip combined file
        try:
            data = json.loads(path.read_text())
            if "pe" in data:
                results.append(data)
        except json.JSONDecodeError:
            continue

    # If no individual files, try combined file
    if not results:
        combined_path = input_dir / "pe_sweep.json"
        if combined_path.exists():
            try:
                data = json.loads(combined_path.read_text())
                if isinstance(data, list):
                    results = data
            except json.JSONDecodeError:
                pass

    return results


def plot_eig_pe(results: List[Dict], output_path: Path) -> None:
    """Plot eigenvalue PE results: k vs error."""
    # Filter eig results
    eig_results = [r for r in results if r["pe"]["kind"] == "eig"]
    if not eig_results:
        print("No eig PE results found.")
        return

    # Sort by k
    eig_results.sort(key=lambda x: x["pe"]["k"])

    k_values = [r["pe"]["k"] for r in eig_results]
    test_errors = [r["test_error"] for r in eig_results]
    id_errors = [r["id_error"] for r in eig_results]
    ood_errors = [r["ood_error"] for r in eig_results]
    discrepancies = [r["discrepancy_set"] for r in eig_results]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot errors on left y-axis
    ax1.plot(k_values, test_errors, "s-", color="#d62728", label="test_error", linewidth=2, markersize=8)
    ax1.plot(k_values, id_errors, "^--", color="#2ca02c", label="id_error", linewidth=1.5, markersize=7)
    ax1.plot(k_values, ood_errors, "v--", color="#ff7f0e", label="ood_error", linewidth=1.5, markersize=7)

    ax1.set_xlabel("k (number of eigenvalues)", fontsize=12)
    ax1.set_ylabel("Error", fontsize=12, color="#333333")
    ax1.set_xticks(k_values)
    ax1.tick_params(axis="y", labelcolor="#333333")
    ax1.legend(loc="upper left", fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot discrepancy on right y-axis
    ax2 = ax1.twinx()
    ax2.plot(k_values, discrepancies, "o:", color="#1f77b4", label="discrepancy", linewidth=1.5, markersize=6)
    ax2.set_ylabel("Discrepancy", fontsize=12, color="#1f77b4")
    ax2.tick_params(axis="y", labelcolor="#1f77b4")
    ax2.legend(loc="upper right", fontsize=10)

    ax1.set_title("Eigenvalue PE: Error vs k", fontsize=14)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    print(f"Saved eig PE plot to {output_path}")


def plot_proj_pe(results: List[Dict], output_path: Path) -> None:
    """Plot projection PE results: k x m heatmap."""
    # Filter proj results
    proj_results = [r for r in results if r["pe"]["kind"] == "proj"]
    if not proj_results:
        print("No proj PE results found.")
        return

    # Extract unique k and m values
    k_values = sorted(set(r["pe"]["k"] for r in proj_results))
    m_values = sorted(set(r["pe"]["m"] for r in proj_results))

    # Build heatmap matrices
    test_error_matrix = np.full((len(k_values), len(m_values)), np.nan)
    ood_error_matrix = np.full((len(k_values), len(m_values)), np.nan)

    for r in proj_results:
        k_idx = k_values.index(r["pe"]["k"])
        m_idx = m_values.index(r["pe"]["m"])
        test_error_matrix[k_idx, m_idx] = r["test_error"]
        ood_error_matrix[k_idx, m_idx] = r["ood_error"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Test error heatmap
    ax1 = axes[0]
    im1 = ax1.imshow(test_error_matrix, cmap="RdYlGn_r", aspect="auto")
    ax1.set_xticks(range(len(m_values)))
    ax1.set_xticklabels(m_values)
    ax1.set_yticks(range(len(k_values)))
    ax1.set_yticklabels(k_values)
    ax1.set_xlabel("m (readout dim)", fontsize=12)
    ax1.set_ylabel("k (eigenvalues)", fontsize=12)
    ax1.set_title("Test Error", fontsize=14)

    # Add text annotations
    for i in range(len(k_values)):
        for j in range(len(m_values)):
            val = test_error_matrix[i, j]
            if not np.isnan(val):
                ax1.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=9,
                         color="white" if val > 0.5 else "black")

    plt.colorbar(im1, ax=ax1)

    # OOD error heatmap
    ax2 = axes[1]
    im2 = ax2.imshow(ood_error_matrix, cmap="RdYlGn_r", aspect="auto")
    ax2.set_xticks(range(len(m_values)))
    ax2.set_xticklabels(m_values)
    ax2.set_yticks(range(len(k_values)))
    ax2.set_yticklabels(k_values)
    ax2.set_xlabel("m (readout dim)", fontsize=12)
    ax2.set_ylabel("k (eigenvalues)", fontsize=12)
    ax2.set_title("OOD Error", fontsize=14)

    # Add text annotations
    for i in range(len(k_values)):
        for j in range(len(m_values)):
            val = ood_error_matrix[i, j]
            if not np.isnan(val):
                ax2.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=9,
                         color="white" if val > 0.5 else "black")

    plt.colorbar(im2, ax=ax2)

    fig.suptitle("Projection PE: k x m Heatmap", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    print(f"Saved proj PE plot to {output_path}")


def plot_proj_pe_lines(results: List[Dict], output_path: Path) -> None:
    """Plot projection PE results as line charts (one line per k)."""
    # Filter proj results
    proj_results = [r for r in results if r["pe"]["kind"] == "proj"]
    if not proj_results:
        print("No proj PE results found.")
        return

    # Group by k
    k_values = sorted(set(r["pe"]["k"] for r in proj_results))
    m_values = sorted(set(r["pe"]["m"] for r in proj_results))

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.viridis(np.linspace(0, 1, len(k_values)))

    for k, color in zip(k_values, colors):
        k_results = [r for r in proj_results if r["pe"]["k"] == k]
        k_results.sort(key=lambda x: x["pe"]["m"])

        m_vals = [r["pe"]["m"] for r in k_results]
        test_errors = [r["test_error"] for r in k_results]

        ax.plot(m_vals, test_errors, "o-", color=color, label=f"k={k}", linewidth=2, markersize=8)

    ax.set_xlabel("m (readout dim)", fontsize=12)
    ax.set_ylabel("Test Error", fontsize=12)
    ax.set_xticks(m_values)
    ax.legend(title="k", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_title("Projection PE: Test Error vs m (by k)", fontsize=14)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    print(f"Saved proj PE lines plot to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot PE sweep experiment results")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing result JSON files")
    parser.add_argument("--output", type=str, default=None, help="Output plot path (optional)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    results = load_results(input_dir)

    if not results:
        raise SystemExit(f"No PE sweep results found in {input_dir}")

    # Separate eig and proj results
    eig_results = [r for r in results if r["pe"]["kind"] == "eig"]
    proj_results = [r for r in results if r["pe"]["kind"] == "proj"]

    # Plot eig PE
    if eig_results:
        eig_output = input_dir / "pe_sweep_eig.png"
        plot_eig_pe(results, eig_output)

    # Plot proj PE
    if proj_results:
        proj_heatmap_output = input_dir / "pe_sweep_proj_heatmap.png"
        plot_proj_pe(results, proj_heatmap_output)

        proj_lines_output = input_dir / "pe_sweep_proj_lines.png"
        plot_proj_pe_lines(results, proj_lines_output)

    print(f"Done. Output files in {input_dir}/")


if __name__ == "__main__":
    main()
