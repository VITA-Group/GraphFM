OUTPUT_DIR="runs/budget100k_deepsets/debug"
LAMBDA=0.0
TAG=$(printf "%.2f" "${LAMBDA}" | tr '.' 'p')
LOG="${OUTPUT_DIR}/size_shift_lambda_${TAG}.log"
DEVICE="cuda:6"

python scripts/run_experiment.py \
   --experiment size_shift \
   --output runs/debug_spe_learnable \
   --pe_kind spe_learnable \
   --k 16 \
   --m 16 \
   --lambda_mix 0.5 \
   --device cuda:6 \
   --config configs/budget100k_deepsets_h256_ep100_params200k_eig_k32.yaml \
   --discrepancy_mode proportional