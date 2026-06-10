#!/usr/bin/env bash
# Route-A: 4 experiments only (3 single-module ablations + 1 full).
#
#   ./run_route_a.sh map    # 只开 map（全流程，map 需要 Stage-1）
#   ./run_route_a.sh fg     # 只开 fg（student_only，从 teacher 训 Stage-2）
#   ./run_route_a.sh attn   # 只开 attn（student_only）
#   ./run_route_a.sh all    # map + fg + attn 全开（全流程）
#
# Env: TEACHER_CKPT=.../checkpoint-best.pth  (fg/attn/all 的 Stage-2 起点；all 的 full 模式 Stage-1 仍从零训)

set -euo pipefail

ROOT="/root/autodl-tmp/users/zhang-dong-mei/zhangdongmei/aed-mae"
PYTHON="${PYTHON:-/root/autodl-tmp/users/zhang-dong-mei/envs/hstforu/bin/python}"
DATASET="${DATASET:-avenue}"
MODE="${1:-all}"
TEACHER_CKPT="${TEACHER_CKPT:-${ROOT}/output/avenue/r0_mse_skip/checkpoint-best.pth}"

cd "${ROOT}"

COMMON=(
  --dataset "${DATASET}"
  --ts_loss_type mse
  --run_type train
)

case "${MODE}" in
  map)
    # 仅像素 map 监督；必须全流程（Stage-1 epoch 0-99）
    "${PYTHON}" main.py "${COMMON[@]}" \
      --experiment_name spa_a_map \
      --use_anomaly_map_loss
    ;;
  fg)
    # 仅前景解耦蒸馏；只训 Stage-2
    "${PYTHON}" main.py "${COMMON[@]}" \
      --experiment_name spa_a_fg \
      --student_only \
      --teacher_checkpoint "${TEACHER_CKPT}" \
      --use_fg_gated_distill
    ;;
  attn|attn_v2)
    # 仅 patch-attention 打分；只训 Stage-2（v2: 推理用 sigmoid(frame_logit)，与 BCE 训练一致）
    "${PYTHON}" main.py "${COMMON[@]}" \
      --experiment_name spa_a_attn_v2 \
      --student_only \
      --teacher_checkpoint "${TEACHER_CKPT}" \
      --use_patch_attn_score
    ;;
  all)
    # 三模块全开；全流程
    "${PYTHON}" main.py "${COMMON[@]}" \
      --experiment_name spa_a_all \
      --use_anomaly_map_loss \
      --use_fg_gated_distill \
      --use_patch_attn_score
    ;;
  *)
    echo "Usage: $0 {map|fg|attn|all}"
    exit 1
    ;;
esac

echo "Done. Mode=${MODE}"
