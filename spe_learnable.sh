#!/usr/bin/env bash
# Train spe_learnable with different k and m values
# Usage: bash train_spe_learnable.sh

set -euo pipefail

DEVICES=(0 1 3 4 5 6)
CONFIG="configs/budget100k_deepsets_h256_ep100_params200k_eig_k32.yaml"
CACHE_DIR="./.cache"
SAMPLING_MODE="uniform_value"
DISCREPANCY_MODE="proportional"
GRAPHON_TYPE="controlled_fourier"
LAMBDA_MIX=0.8

# k and m values to sweep
K_VALUES=(16 32 64)
M_VALUES=(16 32)

OUTPUT_DIR="runs/spe_learnable_sweep"
mkdir -p "${OUTPUT_DIR}"

IDX=0
for K in "${K_VALUES[@]}"; do
  for M in "${M_VALUES[@]}"; do
    TAG="k${K}_m${M}"
    LOG="${OUTPUT_DIR}/spe_learnable_${TAG}.log"
    DEVICE="cuda:${DEVICES[$((IDX % ${#DEVICES[@]}))]}"

    echo "Starting k=${K}, m=${M} on ${DEVICE}..."

    python scripts/run_experiment.py \
      --experiment size_shift \
      --output "${OUTPUT_DIR}" \
      --pe_kind spe_learnable \
      --k "${K}" \
      --m "${M}" \
      --lambda_mix "${LAMBDA_MIX}" \
      --device "${DEVICE}" \
      --cache_dir "${CACHE_DIR}" \
      --config "${CONFIG}" \
      --sampling_mode "${SAMPLING_MODE}" \
      --discrepancy_mode "${DISCREPANCY_MODE}" \
      --graphon_type "${GRAPHON_TYPE}" \
      --epochs 200 \
      --output_suffix "${TAG}" \
      > "${LOG}" 2>&1 &

    IDX=$((IDX + 1))
  done
done

echo "Started ${IDX} jobs in background. Output directory: ${OUTPUT_DIR}"
echo "Use 'jobs' or 'ps aux | grep run_experiment' to monitor progress."
echo "Logs: ${OUTPUT_DIR}/spe_learnable_*.log"
