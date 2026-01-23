OUTPUT_DIR="runs/budget100k_deepsets_controlled_fourier_uniform_value/size_shift_degree"
DEVICES=(4 5 6 0 1 2 3)

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
    --cache_dir ./.cache \
    --config configs/budget100k_deepsets_h256_ep100_params200k_eig_k32.yaml \
    --sampling_mode uniform_value \
    --discrepancy_mode proportional \
    --graphon_type controlled_fourier \
    --merging_method degree \
    > "${LOG}" 2>&1 &
  IDX=$((IDX + 1))
done
