# GraphFM Pipeline

A Graph Foundation Model pipeline for studying graph neural network generalization through graphons and positional encoding strategies. Implements the Section 4 experimental plan from `Generalizable_GraphFM_theory.pdf`.

**Key deviation from paper**: Uses graphon values W(u_i, u_j) directly as weighted adjacency matrix rather than Bernoulli edge sampling (where u_i ~ Uniform(0,1)).

## Installation

Using [uv](https://github.com/astral-sh/uv) (recommended):
```bash
uv sync
```

Or traditional venv:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Quick Start

```bash
# 1. Generate dataset cache (using config file)
python scripts/generate_dataset.py \
  --config configs/budget10k_deepsets_h128_ep50_params50k_eig_k16.yaml \
  --cache_dir ./.cache \
  --lambda_mix 0.3

# 2. Run experiment with same config
python scripts/run_experiment.py \
  --config configs/budget10k_deepsets_h128_ep50_params50k_eig_k16.yaml \
  --experiment size_shift \
  --output runs/size_shift \
  --lambda_mix 0.3 \
  --device cuda \
  --cache_dir ./.cache

# 3. Visualize results
bash plot.sh runs/size_shift/
```

## Configuration

### Config File Format

Config files use YAML format with three sections:

```yaml
dataset:
  num_classes: 4          # Number of graphon classes
  train_sizes: [64, 128, 256, 512]
  test_sizes: [64, 128, 256, 512, 768, 1024]
  total_budget: 10000     # Total node budget for training
  lambda_mix: 0.0         # Size allocation (0=small, 1=large)

train:
  epochs: 50
  model: deepsets         # deepsets | gin | degree
  hidden: 128
  device: cpu

pe:
  kind: eig               # eig | proj | spe
  k: 16                   # Number of eigenvalues
  m: 16                   # Readout dimension (for proj/spe)
```

### Preset Configurations

| Config File | Scale | Model | Budget | PE |
|-------------|-------|-------|--------|-----|
| `budget10k_deepsets_h128_ep50_params50k_eig_k16.yaml` | Toy | DeepSets | 10K | eig k=16 |
| `budget100k_deepsets_h256_ep100_params200k_eig_k32.yaml` | Small | DeepSets | 100K | eig k=32 |
| `budget1m_gin_h512_ep200_params2m_eig_k64.yaml` | Medium | GIN | 1M | eig k=64 |
| `budget10m_gin_h768_ep500_params10m_proj_k128_m64.yaml` | FM | GIN | 10M | proj k=128 |

### Generate New Config

```bash
python scripts/generate_config.py \
  --output_dir configs \
  --total_budget 1000000 \
  --model deepsets \
  --hidden 512 \
  --epochs 200 \
  --pe_kind eig \
  --k 64
```

Filename format: `budget{N}_{model}_h{H}_ep{E}_params{P}_{pe}_k{K}[_m{M}].yaml`

## Dataset Generation

Datasets can be pre-generated and cached for faster experiment runs.

### Using Config File (Recommended)

Use the same config file for dataset generation and experiments to ensure parameters match:

```bash
python scripts/generate_dataset.py \
  --config configs/budget10k_deepsets_h128_ep50_params50k_eig_k16.yaml \
  --cache_dir ./.cache \
  --lambda_mix 0.3
```

The `--lambda_mix` flag overrides the config value if provided.

### Using Defaults

```bash
python scripts/generate_dataset.py \
  --cache_dir ./.cache \
  --lambda_mix 0.3
```

### Batch Generation

Generate datasets for multiple lambda values:

```bash
# Default: lambda 0.0 to 1.0 in 0.1 increments
bash generate_data.sh ./.cache

# Custom lambda values
bash generate_data.sh ./.cache 0.0 0.5 1.0
```

### Cache Mechanism

- Datasets are cached in `{cache_dir}/` with hashed parameter names
- Cache is automatically used when `--cache_dir` is provided to experiments
- Use `--overwrite` flag in generate_dataset.py to regenerate existing cache

## Experiments

### Size Shift Experiment

**Purpose**: Test model generalization across different graph sizes.

The `lambda_mix` parameter controls training size allocation:
- `lambda_mix=0.0`: Allocate budget to smaller graphs (more small graphs, fewer large)
- `lambda_mix=1.0`: Allocate budget to larger graphs (fewer small graphs, more large)
- `lambda_mix=0.5`: Balanced allocation

```bash
# Single run
python scripts/run_experiment.py \
  --config configs/budget10k_deepsets_h128_ep50_params50k_eig_k16.yaml \
  --experiment size_shift \
  --output runs/size_shift \
  --lambda_mix 0.3 \
  --device cuda \
  --cache_dir ./.cache

# Multi-GPU sweep across lambda values
bash train.sh size
```

Output: `{output}/size_shift_lambda_{value}.json`
```json
{
  "train_error": 0.05,
  "test_error": 0.12,
  "discrepancy_set": 0.034,
  "lambda_mix": 0.3,
  "merging_method": null
}
```

### Merge Graph Experiment

**Purpose**: Study graphon-based data augmentation with controllable merging ratio and merged graph sizes.

The experiment estimates a step-graphon from training graphs per class, then synthesizes new graphs at scaled sizes to augment the training set.

**Parameters**:
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `merging_method` | str | "spectral" | Node ordering: "degree" or "spectral" |
| `merging_ratio` | float | 0.5 | Ratio of merged graphs to original graphs per class |
| `merging_size` | float | 2.0 | Size multiplier for merged graphs (1.5, 2.0, or 3.0) |

```bash
# Single run
python scripts/run_experiment.py \
  --config configs/budget10k_deepsets_h128_ep50_params50k_eig_k16.yaml \
  --experiment merge_graph \
  --output runs/merge_graph \
  --merging_method spectral \
  --merging_ratio 0.5 \
  --merging_size 2.0 \
  --device cuda \
  --cache_dir ./.cache

# Multi-GPU sweep: 3 ratios × 3 sizes = 9 configurations
bash train.sh merge
```

Output: `{output}/merge_graph_method_{method}_ratio_{ratio}_size_{size}.json`
```json
{
  "train_error": 0.05,
  "test_error": 0.10,
  "id_error": 0.08,
  "ood_error": 0.12,
  "discrepancy_set": 0.028,
  "merging_method": "spectral",
  "merging_ratio": 0.5,
  "merging_size": 2.0,
  "num_original_train": 200,
  "num_merged": 100,
  "merged_sizes": [128, 256, 512, 1024]
}
```

### PE Sweep Experiment

**Purpose**: Grid search over positional encoding configurations.

Compares different PE types and parameters:
- **eig**: Eigenvector-based PE, uses top-k eigenvalues
- **proj**: Projected PE with random projection matrix
- **spe**: Spectral PE variant with alpha/tau parameters

```bash
python scripts/run_experiment.py \
  --experiment pe_sweep \
  --output runs/pe_sweep \
  --device cuda \
  --cache_dir ./.cache
```

Default grid searches:
- eig: k ∈ {8, 16, 32, 64}
- proj: k ∈ {8, 16, 32}, m ∈ {8, 16, 32}

## Multi-GPU Training

The `train.sh` script supports parallel execution across multiple GPUs:

```bash
bash train.sh           # Default: runs size_shift experiment
bash train.sh size      # Runs size_shift with lambda 0.0 to 1.0 (6 configs)
bash train.sh merge     # Runs merge_graph with ratio/size sweep (9 configs)
```

Configuration variables in `train.sh`:
- `DEVICES`: GPU indices for round-robin assignment
- `CONFIG`: Path to YAML config file
- `CACHE_DIR`: Dataset cache directory

## Visualization

```bash
# Auto-detect experiment type from directory contents
bash plot.sh runs/size_shift/
bash plot.sh runs/merge_graph/

# Force specific plot type
bash plot.sh size runs/size_shift/
bash plot.sh merge runs/merge_graph/

# Compare two size_shift directories
bash plot.sh runs/dirA/ runs/dirB/
```

### Size Shift Plots

Single plot showing discrepancy and errors vs lambda_mix.

Output: `{dir}/single_plot.png`

### Merge Graph Plots

Three visualization types:
- **Heatmap** (`merge_graph_heatmap.png`): 2×3 grid showing metrics across ratio/size combinations
- **Lines by ratio** (`merge_graph_heatmap_lines.png`): Metrics vs merging_size, one line per ratio
- **Lines by size** (`merge_graph_heatmap_by_ratio.png`): Metrics vs merging_ratio, one line per size

## Components

- **Graphons**: Fourier family with controllable spectra; step-graphon estimator for merging
- **Graph sampling**: For size n, sample node locations u_i ~ Uniform(0,1) and set A_ij = W(u_i, u_j) (symmetric, weighted)
- **PE tokens**: eig-PE, proj-PE, and SPE variant
- **Models**: DeepSets (default), Degree histogram + MLP baseline, optional GIN
- **Metrics**: Test error, sliced-W1 discrepancy between train/test token measures, eigengap proxies
- **Data curation**: Fixed budget size-allocation path + optional graphon-guided merging
