#!/usr/bin/env bash
# Student-only inference: encoder + student decoder, no teacher decoder forward.
# Score = student reconstruction error + filt() (same post-processing as official test).
#
# Usage:
#   ./run_infer_student_only.sh
#   TEACHER=... STUDENT=... ./run_infer_student_only.sh
#   ./run_infer_student_only.sh output/avenue/r0_mse_skip

set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-/root/autodl-tmp/users/zhang-dong-mei/envs/hstforu/bin/python}"
DATASET="${DATASET:-avenue}"
EXP_DIR="${1:-output/avenue/r0_mse_skip}"
TEACHER="${TEACHER:-${EXP_DIR}/checkpoint-best.pth}"
STUDENT="${STUDENT:-${EXP_DIR}/checkpoint-best-student.pth}"
EXP_NAME="${EXP_NAME:-$(basename "${EXP_DIR}")_student_infer}"

"${PYTHON}" main.py \
  --dataset "${DATASET}" \
  --run_type inference \
  --experiment_name "${EXP_NAME}" \
  --student_infer_only \
  --teacher_checkpoint "${TEACHER}" \
  --student_checkpoint "${STUDENT}" \
  --batch_size 100
