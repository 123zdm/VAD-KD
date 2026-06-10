#!/usr/bin/env bash
# Stage-2 only (ep100-139): load paper_baseline teacher, train student with Official fusion.
# Scoring: 0.4*teacher_recon + 0.3*ts_gap (NOT Paper Eq.6, no cls head).
#
#   ./run_paper_teacher_official_skip.sh train
#   ./run_paper_teacher_official_skip.sh inference
#
# Env:
#   TEACHER_CKPT=.../paper_baseline/checkpoint-best.pth
#   EXP_NAME=paper_teacher_official_skip

set -euo pipefail

ROOT="/root/autodl-tmp/users/zhang-dong-mei/zhangdongmei/aed-mae"
PYTHON="${PYTHON:-/root/autodl-tmp/users/zhang-dong-mei/envs/hstforu/bin/python}"
DATASET="${DATASET:-avenue}"
EXP_NAME="${EXP_NAME:-paper_teacher_official_skip}"
TEACHER_CKPT="${TEACHER_CKPT:-${ROOT}/output/avenue/paper_baseline/checkpoint-best.pth}"
RUN_TYPE="${1:-train}"
OUT_DIR="${ROOT}/output/${DATASET}/${EXP_NAME}"

cd "${ROOT}"

COMMON=(
  --dataset "${DATASET}"
  --experiment_name "${EXP_NAME}"
  --run_type "${RUN_TYPE}"
  --ts_loss_type mse
  --student_only
  --teacher_checkpoint "${TEACHER_CKPT}"
  --epochs 140
  --start_TS_epoch 100
)

if [[ "${RUN_TYPE}" == "inference" ]]; then
  "${PYTHON}" main.py "${COMMON[@]}" \
    --teacher_checkpoint "${OUT_DIR}/checkpoint-best.pth" \
    --student_checkpoint "${OUT_DIR}/checkpoint-best-student.pth"
else
  "${PYTHON}" main.py "${COMMON[@]}"
fi

echo "Done. Mode=${RUN_TYPE}  output=${OUT_DIR}"
