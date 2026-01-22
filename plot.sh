#!/usr/bin/env bash

#  - 单目录：
#     bash plot.sh runs/size_shift_all_merge/
#   - 双目录对比：
#     bash plot.sh runs/dirA/ runs/dirB/ runs/ compare.png

set -euo pipefail

dir_a="${1}"
dir_b="${2}"
# out="${3:-runs/size_shift_all_merge/size_shift_lambda_plot.png}"

if [[ -n "${dir_b}" ]]; then
  out="${dir_b}/compare_$(basename "${dir_a}")_$(basename "${dir_b}").png"
  python scripts/plot_size_shift.py \
    --input_dir_a "${dir_a}" \
    --input_dir_b "${dir_b}" \
    --label_a "$(basename "${dir_a}")" \
    --label_b "$(basename "${dir_b}")" \
    --output "${out}"
else
  out="${dir_a}/single_plot.png"
  python scripts/plot_size_shift.py --input_dir "${dir_a}" --output "${out}"
fi
