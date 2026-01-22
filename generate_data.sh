#!/usr/bin/env bash
set -euo pipefail

cache_dir="${1:-./.cache}"
shift || true

if [[ "$#" -gt 0 ]]; then
  lambdas=("$@")
else
  lambdas=(0.0 0.2 0.4 0.6 0.8 1.0)
fi

for lm in "${lambdas[@]}"; do
  python scripts/generate_dataset.py \
    --cache_dir "${cache_dir}" \
    --config configs/budget100k_deepsets_h256_ep100_params200k_eig_k32.yaml \
    --lambda_mix "${lm}" \
    --sampling_mode bin_value
done
