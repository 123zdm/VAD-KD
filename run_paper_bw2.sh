#!/usr/bin/env bash
# Baseline-aligned BW2+MSE student training from a teacher checkpoint.
#
# Usage:
#   TEACHER=/path/to/checkpoint-best.pth ./run_paper_bw2.sh author_teacher_bw2mse_a30_v2
#   ./run_paper_bw2.sh author_teacher_bw2mse_a30_v2 inference

set -euo pipefail

ROOT="/root/autodl-tmp/users/zhang-dong-mei/zhangdongmei/aed-mae"
PYTHON="/root/autodl-tmp/users/zhang-dong-mei/envs/hstforu/bin/python"
DATASET="${DATASET:-avenue}"
EXP_NAME="${1:-author_teacher_bw2mse_a30_v2}"
RUN_TYPE="${2:-train}"
LOSS="${LOSS:-bw2_mse}"
TS_BW2_ALPHA="${TS_BW2_ALPHA:-0.3}"
TEACHER="${TEACHER:-/root/autodl-tmp/users/zhang-dong-mei/zhangdongmei/aed-mae_search/output/avenue/checkpoint-best.pth}"

cd "${ROOT}"
CMD=(
  "${PYTHON}" main.py
  --dataset "${DATASET}"
  --ts_loss_type "${LOSS}"
  --ts_abnormal_strategy all
  --experiment_name "${EXP_NAME}"
  --run_type "${RUN_TYPE}"
)
if [[ "${LOSS}" == "bw2_mse" ]]; then
  CMD+=(--ts_bw2_alpha "${TS_BW2_ALPHA}")
fi
if [[ "${RUN_TYPE}" == "train" ]]; then
  CMD+=(--student_only --teacher_checkpoint "${TEACHER}")
fi

echo "Baseline-aligned run: exp=${EXP_NAME}, loss=${LOSS}, run_type=${RUN_TYPE}"
"${CMD[@]}"
