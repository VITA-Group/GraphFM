# GraphFM Pipeline

A Graph Foundation Model pipeline for studying graph neural network generalization through graphons and positional encoding strategies. Implements the Section 4 experimental plan from `Generalizable_GraphFM_theory.pdf`.

**Key deviation from paper**: Uses graphon values at i/n directly as weighted adjacency matrix rather than Bernoulli edge sampling.

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
  --model gin \
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
bash train.sh
```

Output: `{output}/size_shift_lambda_{value}.json`
```json
{
  "train_error": 0.05,
  "test_error": 0.12,
  "discrepancy_set": 0.034,
  "lambda_mix": 0.3,
  "use_merging": false
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

## Visualization

```bash
# Single directory
bash plot.sh runs/size_shift/

# Compare two experiment directories
bash plot.sh runs/dirA/ runs/dirB/ output_dir/ compare.png
```

## Components

- **Graphons**: Fourier family with controllable spectra; step-graphon estimator for merging
- **Graph sampling**: For size n, set node locations u_i = (i+1)/n and A_ij = W(u_i, u_j) (symmetric, weighted)
- **PE tokens**: eig-PE, proj-PE, and SPE variant
- **Models**: DeepSets (default), Degree histogram + MLP baseline, optional GIN
- **Metrics**: Test error, sliced-W1 discrepancy between train/test token measures, eigengap proxies
- **Data curation**: Fixed budget size-allocation path + optional graphon-guided merging
