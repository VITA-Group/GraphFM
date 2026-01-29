#!/usr/bin/env bash

DATASET="IMDB-BINARY" # IMDB-BINARY, REDDIT-BINARY, COLLAB
mkdir -p runs/debug_real_merge/merge_${DATASET}

DEVICE_LIST=(0 1 2 3 4 5)
MERGING_RATIOS=(0.00 0.01 0.02 0.03 0.04 0.05)

for idx in "${!MERGING_RATIOS[@]}"; do
  MERGING_RATIO="${MERGING_RATIOS[$idx]}"
  DEVICE="${DEVICE_LIST[$((idx % ${#DEVICE_LIST[@]}))]}"
  echo "Starting merge ratio ${MERGING_RATIO} on cuda:${DEVICE}..."
  python scripts/run_experiment.py \
    --experiment real_merge \
    --dataset_name "${DATASET}" \
    --output runs/debug_real_merge/merge_${DATASET} \
    --merging_method usvt \
    --merging_ratio ${MERGING_RATIO} \
    --merging_size 2.0 \
    --cache_dir ./.cache \
    --resplit_gap 2.0 \
    --epochs 100 \
    --device "cuda:${DEVICE}" \
    --pe_kind eig \
    --k 16 > runs/debug_real_merge/merge_${DATASET}/merge_usvt_${MERGING_RATIO}.log 2>&1 &
done

wait
echo "All experiments finished. Check logs in runs/debug_real_merge/"