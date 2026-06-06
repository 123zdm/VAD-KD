#!/usr/bin/env bash
# Generate motivation figures comparing MSE vs BW2 student checkpoints.
#
# Usage (after student training finishes):
#   ./run_visualize_motivation.sh
#
# Or with custom paths:
#   TEACHER=/path/to/checkpoint-best.pth \
#   MSE_STUDENT=/path/to/mse-student.pth \
#   BW2_STUDENT=/path/to/bw2-student.pth \
#   OUT_DIR=output/avenue/viz_motivation_v2 \
#   ./run_visualize_motivation.sh

set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-/root/autodl-tmp/users/zhang-dong-mei/envs/hstforu/bin/python}"
TEACHER="${TEACHER:-output/avenue/author_teacher_mse_v2/checkpoint-best.pth}"
MSE_STUDENT="${MSE_STUDENT:-output/avenue/author_teacher_mse_v2/checkpoint-best-student.pth}"
BW2_STUDENT="${BW2_STUDENT:-output/avenue/author_teacher_bw2mse_a30_v2/checkpoint-best-student.pth}"
OUT_DIR="${OUT_DIR:-output/avenue/viz_motivation_v2}"
MAX_FRAMES="${MAX_FRAMES:-800}"

"${PYTHON}" util/visualize_distill_motivation.py \
  --dataset avenue \
  --teacher_checkpoint "${TEACHER}" \
  --student_checkpoints \
    "mse=${MSE_STUDENT}" \
    "bw2=${BW2_STUDENT}" \
  --output_dir "${OUT_DIR}" \
  --max_frames "${MAX_FRAMES}"

echo "Done. See ${OUT_DIR}/fig*.png"
