#!/usr/bin/env bash
# Usage:
#   bash plot.sh <dir>                           # Auto-detect experiment type
#   bash plot.sh <dir_a> <dir_b>                 # Compare two size_shift dirs
#   bash plot.sh size <dir>                      # Force size_shift plot
#   bash plot.sh merge <dir>                     # Force merge_graph plot
#   bash plot.sh perturb <dir>                   # Force perturb_graphon plot

set -euo pipefail

# Parse arguments
if [[ $# -lt 1 ]]; then
  echo "Usage:"
  echo "  bash plot.sh <dir>              # Auto-detect experiment type"
  echo "  bash plot.sh <dir_a> <dir_b>    # Compare two size_shift dirs"
  echo "  bash plot.sh size <dir>         # Force size_shift plot"
  echo "  bash plot.sh merge <dir>        # Force merge_graph plot"
  exit 1
fi

# Check if first arg is a mode
if [[ "${1}" == "size" || "${1}" == "merge" || "${1}" == "perturb" ]]; then
  MODE="${1}"
  DIR="${2}"
else
  # Auto-detect mode based on directory contents
  DIR="${1}"
  DIR_B="${2:-none}"

  if [[ "${DIR_B}" != "none" ]]; then
    # Comparison mode - always use size_shift
    MODE="compare"
  elif ls "${DIR}"/perturb_graphon_*.json 1>/dev/null 2>&1; then
    MODE="perturb"
  elif ls "${DIR}"/merge_graph_*.json 1>/dev/null 2>&1; then
    MODE="merge"
  elif ls "${DIR}"/size_shift_*.json 1>/dev/null 2>&1; then
    MODE="size"
  else
    echo "Error: No experiment results found in ${DIR}"
    echo "Looking for: size_shift_*.json, merge_graph_*.json, or perturb_graphon_*.json"
    exit 1
  fi
fi

case "${MODE}" in
  size)
    OUT="${DIR}/single_plot.png"
    echo "Plotting size_shift results from ${DIR}"
    python scripts/plot_size_shift.py --input_dir "${DIR}" --output "${OUT}"
    echo "Output: ${OUT}"
    ;;

  merge)
    OUT="${DIR}/merge_graph_heatmap.png"
    echo "Plotting merge_graph results from ${DIR}"
    python scripts/plot_merge_graph.py --input_dir "${DIR}" --output "${OUT}"
    echo "Output files:"
    echo "  - ${DIR}/merge_graph_heatmap.png"
    echo "  - ${DIR}/merge_graph_heatmap_lines.png"
    echo "  - ${DIR}/merge_graph_heatmap_by_ratio.png"
    ;;

  perturb)
    echo "Plotting perturb_graphon results from ${DIR}"
    python scripts/plot_perturb_graphon.py --input_dir "${DIR}"
    echo "Output files saved to ${DIR}/"
    ;;

  compare)
    DIR_A="${1}"
    DIR_B="${2}"
    OUT="${DIR_B}/compare_$(basename "${DIR_A}")_$(basename "${DIR_B}").png"
    echo "Comparing size_shift results:"
    echo "  A: ${DIR_A}"
    echo "  B: ${DIR_B}"
    python scripts/plot_size_shift.py \
      --input_dir_a "${DIR_A}" \
      --input_dir_b "${DIR_B}" \
      --label_a "$(basename "${DIR_A}")" \
      --label_b "$(basename "${DIR_B}")" \
      --output "${OUT}"
    echo "Output: ${OUT}"
    ;;

  *)
    echo "Unknown mode: ${MODE}"
    exit 1
    ;;
esac
