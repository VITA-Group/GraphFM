#!/usr/bin/env bash
# Usage:
#   bash train.sh           # Run size_shift experiment (default)
#   bash train.sh size      # Run size_shift experiment
#   bash train.sh merge     # Run merge_graph experiment with ratio/size sweep
#   bash train.sh perturb   # Run perturb_graphon experiment

set -euo pipefail

MODE="${1:-size}"
DEVICES=(4 5 6 0 1 2 3)
CONFIG="configs/budget100k_deepsets_h256_ep100_params200k_eig_k32.yaml"
CACHE_DIR="./.cache"
SAMPLING_MODE="uniform_value"
DISCREPANCY_MODE="proportional"
GRAPHON_TYPE="controlled_fourier"

if [[ "${MODE}" == "size" ]]; then
  # Size shift experiment: sweep lambda_mix from 0 to 1
  OUTPUT_DIR="runs/budget100k_deepsets/size_shift"
  mkdir -p "${OUTPUT_DIR}"

  IDX=0
  for LAMBDA in $(seq 0 0.2 1); do
    TAG=$(printf "%.2f" "${LAMBDA}" | tr '.' 'p')
    LOG="${OUTPUT_DIR}/size_shift_lambda_${TAG}.log"
    DEVICE="cuda:${DEVICES[$((IDX % ${#DEVICES[@]}))]}"
    python scripts/run_experiment.py \
      --experiment size_shift \
      --output "${OUTPUT_DIR}" \
      --lambda_mix "${LAMBDA}" \
      --device "${DEVICE}" \
      --cache_dir "${CACHE_DIR}" \
      --config "${CONFIG}" \
      --sampling_mode "${SAMPLING_MODE}" \
      --discrepancy_mode "${DISCREPANCY_MODE}" \
      --graphon_type "${GRAPHON_TYPE}" \
      > "${LOG}" 2>&1 &
    IDX=$((IDX + 1))
  done

elif [[ "${MODE}" == "merge" ]]; then
  # Merge graph experiment: sweep merging_ratio and merging_size
  OUTPUT_DIR="runs/budget100k_deepsets/merge_graph_usvt"
  mkdir -p "${OUTPUT_DIR}"

  RATIOS=(0.01)
  SIZES=(2.0)
  LAMBDA_MIXS=(1.0 0.8 0.6 0.4 0.2 0.0)  # Focus on small graphs by default

  IDX=0
  for RATIO in "${RATIOS[@]}"; do
    for SIZE in "${SIZES[@]}"; do
      for LAMBDA_MIX in "${LAMBDA_MIXS[@]}"; do
      RATIO_TAG=$(printf "%.2f" "${RATIO}" | tr '.' 'p')
      SIZE_TAG=$(printf "%.1f" "${SIZE}" | tr '.' 'p')
      LAMBDA_MIX_TAG=$(printf "%.2f" "${LAMBDA_MIX}" | tr '.' 'p')
      LOG="${OUTPUT_DIR}/merge_graph_ratio_${RATIO_TAG}_size_${SIZE_TAG}_lambda_${LAMBDA_MIX_TAG}.log"
      DEVICE="cuda:${DEVICES[$((IDX % ${#DEVICES[@]}))]}"
      python scripts/run_experiment.py \
        --experiment merge_graph \
        --output "${OUTPUT_DIR}" \
        --lambda_mix "${LAMBDA_MIX}" \
        --merging_method usvt \
        --merging_ratio "${RATIO}" \
        --merging_size "${SIZE}" \
        --device "${DEVICE}" \
        --cache_dir "${CACHE_DIR}" \
        --config "${CONFIG}" \
        --sampling_mode "${SAMPLING_MODE}" \
        --discrepancy_mode "${DISCREPANCY_MODE}" \
        --graphon_type "${GRAPHON_TYPE}" \
        > "${LOG}" 2>&1 &
        IDX=$((IDX + 1))
      done
    done
  done

elif [[ "${MODE}" == "perturb" ]]; then
  # Perturb graphon experiment: sweep lambda_mix with perturbation evaluation
  OUTPUT_DIR="runs/budget100k_deepsets/perturb_graphon"
  mkdir -p "${OUTPUT_DIR}"

  PERTURB_LEVELS="0.0 0.2 0.4 0.6 0.8 1.0"
  PERTURB_RATIO=0.5
  MAX_L2_DISTANCE=1

  IDX=0
  LAMBDA=0.2
  # for LAMBDA in 0.2; do
  # for LAMBDA in $(seq 0 0.2 1); do
    TAG=$(printf "%.2f" "${LAMBDA}" | tr '.' 'p')
    LOG="${OUTPUT_DIR}/perturb_graphon_lambda_${TAG}.log"
    DEVICE="cuda:${DEVICES[$((IDX % ${#DEVICES[@]}))]}"
    python scripts/run_experiment.py \
      --experiment perturb_graphon \
      --output "${OUTPUT_DIR}" \
      --lambda_mix "${LAMBDA}" \
      --perturb_levels ${PERTURB_LEVELS} \
      --perturb_ratio "${PERTURB_RATIO}" \
      --max_l2_distance "${MAX_L2_DISTANCE}" \
      --device "${DEVICE}" \
      --cache_dir "${CACHE_DIR}" \
      --config "${CONFIG}" \
      --sampling_mode "${SAMPLING_MODE}" \
      --discrepancy_mode "${DISCREPANCY_MODE}" \
      --graphon_type "${GRAPHON_TYPE}" \
      > "${LOG}" 2>&1 &
    IDX=$((IDX + 1))
  done

else
  echo "Unknown mode: ${MODE}"
  echo "Usage: bash train.sh [size|merge|perturb]"
  exit 1
fi

echo "Started ${IDX} jobs in background. Output directory: ${OUTPUT_DIR}"
echo "Use 'jobs' or 'ps aux | grep run_experiment' to monitor progress."
