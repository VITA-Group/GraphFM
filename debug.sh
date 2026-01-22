OUTPUT_DIR="runs/budget100k_deepsets/debug"
LAMBDA=0.0
TAG=$(printf "%.2f" "${LAMBDA}" | tr '.' 'p')
LOG="${OUTPUT_DIR}/size_shift_lambda_${TAG}.log"
DEVICE="cuda:6"

python scripts/run_experiment.py \
--experiment size_shift \
--output "${OUTPUT_DIR}" \
--lambda_mix "${LAMBDA}" \
--device "${DEVICE}" \
--cache_dir ./.cache \
--config configs/budget100k_deepsets_h256_ep100_params200k_eig_k32.yaml \
--discrepancy_mode proportional