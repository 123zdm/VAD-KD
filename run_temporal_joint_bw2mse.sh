#!/usr/bin/env bash
# Clip-level Temporal Joint BW² + per-frame BW2+MSE student training.
#
# Combines per-frame BW2+MSE with clip-level joint BW².
#
# Usage:
#   TEACHER=/path/to/checkpoint-best.pth ./run_temporal_joint_bw2mse.sh
#   CLIP_LEN=8 TS_JOINT_LAMBDA=0.5 TS_BW2_ALPHA=0.3 BATCH_SIZE=6 ./run_temporal_joint_bw2mse.sh t3_joint_bw2mse_k8
#
# Recommended grid:
#   TS_BW2_ALPHA: 0.2 / 0.3 / 0.5   (per-frame BW2+MSE mix)
#   TS_JOINT_LAMBDA: 0.3 / 0.5 / 0.7 (clip-level joint weight)
#   CLIP_LEN: 8 / 16

set -euo pipefail

ROOT="/root/autodl-tmp/users/zhang-dong-mei/zhangdongmei/aed-mae"
PYTHON="/root/autodl-tmp/users/zhang-dong-mei/envs/hstforu/bin/python"
DATASET="${DATASET:-avenue}"
EXP_NAME="${1:-t3_joint_bw2mse_k8}"
RUN_TYPE="${2:-train}"
LOSS="${LOSS:-temporal_joint_bw2mse}"
CLIP_LEN="${CLIP_LEN:-8}"
CLIP_STRIDE="${CLIP_STRIDE:-1}"
TS_JOINT_LAMBDA="${TS_JOINT_LAMBDA:-0.5}"
TS_JOINT_RANK="${TS_JOINT_RANK:-32}"
TS_BW2_ALPHA="${TS_BW2_ALPHA:-0.3}"
BATCH_SIZE="${BATCH_SIZE:-6}"
TEACHER="${TEACHER:-/root/autodl-tmp/users/zhang-dong-mei/zhangdongmei/aed-mae_search/output/avenue/checkpoint-best.pth}"

cd "${ROOT}"
CMD=(
  "${PYTHON}" main.py
  --dataset "${DATASET}"
  --ts_loss_type "${LOSS}"
  --ts_bw2_alpha "${TS_BW2_ALPHA}"
  --no-ts_bw2_normalize
  --clip_len "${CLIP_LEN}"
  --clip_stride "${CLIP_STRIDE}"
  --ts_joint_lambda "${TS_JOINT_LAMBDA}"
  --ts_joint_rank "${TS_JOINT_RANK}"
  --batch_size "${BATCH_SIZE}"
  --experiment_name "${EXP_NAME}"
  --run_type "${RUN_TYPE}"
)

if [[ "${RUN_TYPE}" == "train" ]]; then
  CMD+=(--student_only --teacher_checkpoint "${TEACHER}")
fi

echo "Temporal joint BW2+MSE: exp=${EXP_NAME}, loss=${LOSS}, clip_len=${CLIP_LEN}, alpha=${TS_BW2_ALPHA}, joint_lambda=${TS_JOINT_LAMBDA}"
"${CMD[@]}"
