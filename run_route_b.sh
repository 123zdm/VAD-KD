#!/usr/bin/env bash
# Route-B: targeted fixes for hard Avenue videos (FP/FN).
#
#   ./run_route_b.sh peak          # 推理：r0_mse_skip + 时序滑窗 max（不重训，~10min）
#   ./run_route_b.sh topk          # 训练：Top-K patch 打分（治漏检）
#   ./run_route_b.sh hardneg       # 训练：难负样本（治误报）
#   ./run_route_b.sh topk_peak     # 训练：Top-K + 评估时时序 peak
#   ./run_route_b.sh topk_hn       # 训练：Top-K + 难负样本
#   ./run_route_b.sh all           # peak 推理 + topk + hardneg 三个实验
#
# Env:
#   TEACHER_CKPT=.../r0_mse_skip/checkpoint-best.pth
#   STUDENT_CKPT=.../r0_mse_skip/checkpoint-best-student.pth  (peak 推理用)

set -euo pipefail

ROOT="/root/autodl-tmp/users/zhang-dong-mei/zhangdongmei/aed-mae"
PYTHON="${PYTHON:-/root/autodl-tmp/users/zhang-dong-mei/envs/hstforu/bin/python}"
DATASET="${DATASET:-avenue}"
MODE="${1:-all}"
TEACHER_CKPT="${TEACHER_CKPT:-${ROOT}/output/avenue/r0_mse_skip/checkpoint-best.pth}"
STUDENT_CKPT="${STUDENT_CKPT:-${ROOT}/output/avenue/r0_mse_skip/checkpoint-best-student.pth}"
PEAK_WIN="${PEAK_WIN:-9}"

cd "${ROOT}"

COMMON=(
  --dataset "${DATASET}"
  --ts_loss_type mse
)

run_peak_infer() {
  "${PYTHON}" main.py "${COMMON[@]}" \
    --run_type inference \
    --experiment_name "spa_b_peak${PEAK_WIN}" \
    --teacher_checkpoint "${TEACHER_CKPT}" \
    --student_checkpoint "${STUDENT_CKPT}" \
    --temporal_peak_window "${PEAK_WIN}"
}

run_topk_train() {
  local extra=("$@")
  "${PYTHON}" main.py "${COMMON[@]}" \
    --run_type train \
    --experiment_name spa_b_topk8 \
    --student_only \
    --teacher_checkpoint "${TEACHER_CKPT}" \
    --use_topk_patch_score \
    --topk_patch_k 8 \
    "${extra[@]}"
}

run_hardneg_train() {
  local extra=("$@")
  "${PYTHON}" main.py "${COMMON[@]}" \
    --run_type train \
    --experiment_name spa_b_hn \
    --student_only \
    --teacher_checkpoint "${TEACHER_CKPT}" \
    --use_hard_normal_mining \
    "${extra[@]}"
}

case "${MODE}" in
  peak)
    run_peak_infer
    ;;
  topk)
    run_topk_train
    ;;
  hardneg|hn)
    run_hardneg_train
    ;;
  topk_peak)
    "${PYTHON}" main.py "${COMMON[@]}" \
      --run_type train \
      --experiment_name "spa_b_topk8_peak${PEAK_WIN}" \
      --student_only \
      --teacher_checkpoint "${TEACHER_CKPT}" \
      --use_topk_patch_score --topk_patch_k 8 \
      --temporal_peak_window "${PEAK_WIN}"
    ;;
  topk_hn|topk_hardneg)
    "${PYTHON}" main.py "${COMMON[@]}" \
      --run_type train \
      --experiment_name spa_b_topk8_hn \
      --student_only \
      --teacher_checkpoint "${TEACHER_CKPT}" \
      --use_topk_patch_score --topk_patch_k 8 \
      --use_hard_normal_mining
    ;;
  all)
    run_peak_infer
    run_topk_train
    run_hardneg_train
    ;;
  *)
    echo "Usage: $0 {peak|topk|hardneg|topk_peak|topk_hn|all}"
    exit 1
    ;;
esac

echo "Done. Mode=${MODE}"
