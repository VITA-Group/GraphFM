# GraphFM pipeline

This repo implements a minimal, runnable pipeline for the Section 4 experimental plan in
`Generalizable_GraphFM_theory.pdf`. The only deviation is the graphon-to-graph sampling rule:
**we use the graphon values at i/n directly as a weighted adjacency matrix** rather than
Bernoulli edge sampling.
## Installation
```bash
uv sync
```
or
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Experiments
```bash

python scripts/run_experiment.py --experiment size_shift --output runs/size_shift --device cuda
python scripts/run_experiment.py --experiment pe_sweep --output runs/pe_sweep --device cuda
```

## Components

- Graphons: Fourier family with controllable spectra; step-graphon estimator for merging.
- Graph sampling (modified): for size n, set node locations u_i = (i+1)/n and
  A_ij = W(u_i, u_j) (symmetric, weighted).
- PE tokens: eig-PE, proj-PE, and a simple SPE variant.
- Models: DeepSets (default), Degree histogram + MLP baseline, optional GIN.
- Metrics: test error, sliced-W1 discrepancy between train/test token measures,
  eigengap proxies for stability diagnostics.
- Data curation: fixed budget size-allocation path + optional graphon-guided merging.

<!-- ## Notes

- The weighted adjacency matrix is used as graphon sampling instead of Bernoulli sampling listed in current pdf. -->


