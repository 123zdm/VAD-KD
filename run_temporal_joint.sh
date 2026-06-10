#!/usr/bin/env bash
# Clip-level Temporal Joint BW² student training (separate from run_paper_bw2.sh).
#
# Usage:
#   TEACHER=/path/to/checkpoint-best.pth ./run_temporal_joint.sh t2_joint_mse
#   LOSS=temporal_joint CLIP_LEN=8 ./run_temporal_joint.sh t2_joint_only
#   LOSS=temporal_joint_mse CLIP_LEN=8 ./run_temporal_joint.sh t2_joint_mse
#   LOSS=temporal_joint_bw2mse CLIP_LEN=8 ./run_temporal_joint_bw2mse.sh t3_joint_bw2mse_k8
# Note: default BATCH_SIZE=12; for temporal_joint* try BATCH_SIZE=4~8 if OOM.

set -euo pipefail

ROOT="/root/autodl-tmp/users/zhang-dong-mei/zhangdongmei/aed-mae"
PYTHON="/root/autodl-tmp/users/zhang-dong-mei/envs/hstforu/bin/python"
DATASET="${DATASET:-avenue}"
EXP_NAME="${1:-t2_joint_mse}"
RUN_TYPE="${2:-train}"
LOSS="${LOSS:-temporal_joint_mse}"
CLIP_LEN="${CLIP_LEN:-8}"
CLIP_STRIDE="${CLIP_STRIDE:-1}"
TS_JOINT_LAMBDA="${TS_JOINT_LAMBDA:-0.5}"
TS_JOINT_RANK="${TS_JOINT_RANK:-32}"
BATCH_SIZE="${BATCH_SIZE:-12}"
TEACHER="${TEACHER:-/root/autodl-tmp/users/zhang-dong-mei/zhangdongmei/aed-mae_search/output/avenue/checkpoint-best.pth}"

cd "${ROOT}"
CMD=(
  "${PYTHON}" main.py
  --dataset "${DATASET}"
  --ts_loss_type "${LOSS}"
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

echo "Temporal joint run: exp=${EXP_NAME}, loss=${LOSS}, clip_len=${CLIP_LEN}"
"${CMD[@]}"
