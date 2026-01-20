#!/usr/bin/env bash
set -euo pipefail

cache_dir="${1:-./.cache}"
shift || true

if [[ "$#" -gt 0 ]]; then
  lambdas=("$@")
else
  lambdas=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
fi

for lm in "${lambdas[@]}"; do
  python scripts/generate_dataset.py \
    --cache_dir "${cache_dir}" \
    --lambda_mix "${lm}"
done
