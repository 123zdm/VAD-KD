#!/usr/bin/env bash
# Submit Route-B experiments via gpuq.
#
#   ./submit_route_b_gpuq.sh              # list
#   ./submit_route_b_gpuq.sh peak         # 快速：r0_mse_skip 推理 + 时序 peak（推荐先跑）
#   ./submit_route_b_gpuq.sh topk         # Top-K patch 训练
#   ./submit_route_b_gpuq.sh hardneg      # 难负样本训练
#   ./submit_route_b_gpuq.sh all          # peak + topk + hardneg 三个 job
#
#   GPU_ID=1 ./submit_route_b_gpuq.sh peak
# Monitor: gpuq -t <job_id>

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEACHER_CKPT="${TEACHER_CKPT:-${SCRIPT_DIR}/output/avenue/r0_mse_skip/checkpoint-best.pth}"
STUDENT_CKPT="${STUDENT_CKPT:-${SCRIPT_DIR}/output/avenue/r0_mse_skip/checkpoint-best-student.pth}"
GPUS="${GPUS:-1}"
GPU_ID="${GPU_ID:-}"
LABEL_PREFIX="${LABEL_PREFIX:-zdm-spb}"
EXP="${1:-help}"

ALL_EXPS=(peak topk hardneg)

declare -A EXP_DESC
EXP_DESC[peak]="推理：r0_mse_skip + temporal_peak_window=9，不重训"
EXP_DESC[topk]="训练：Top-K patch 打分 (spa_b_topk8)"
EXP_DESC[hardneg]="训练：难负样本 (spa_b_hn)"
EXP_DESC[topk_peak]="训练：Top-K + peak 评估 (spa_b_topk8_peak9)"
EXP_DESC[topk_hn]="训练：Top-K + 难负样本 (spa_b_topk8_hn)"
EXP_DESC[all]="peak + topk + hardneg 三个 job"

print_catalog() {
  echo "Route-B experiments (teacher: ${TEACHER_CKPT}):"
  echo ""
  printf "  %-10s → %s\n" "peak" "${EXP_DESC[peak]}"
  printf "  %-10s → output/avenue/spa_b_topk8\n" "topk"
  printf "  %-10s → output/avenue/spa_b_hn\n" "hardneg"
  printf "  %-10s → %s\n" "topk_peak" "${EXP_DESC[topk_peak]}"
  printf "  %-10s → %s\n" "topk_hn" "${EXP_DESC[topk_hn]}"
  echo ""
  echo "  GPU_ID=1 ./submit_route_b_gpuq.sh peak     # 推荐先跑，~10 分钟"
  echo "  ./submit_route_b_gpuq.sh all               # 三个 job 并行"
}

submit_one() {
  local name="$1"
  local label="${LABEL_PREFIX}-${name}"
  local gpu_args=(-G "${GPUS}")
  if [[ -n "${GPU_ID}" ]]; then
    gpu_args+=(-g "${GPU_ID}")
  fi
  local job_id
  job_id=$(gpuq "${gpu_args[@]}" -L "${label}" bash -lc \
    "cd \"${SCRIPT_DIR}\" && TEACHER_CKPT=\"${TEACHER_CKPT}\" STUDENT_CKPT=\"${STUDENT_CKPT}\" ./run_route_b.sh ${name}")
  echo "Queued ${name}: job ${job_id}  |  ${EXP_DESC[$name]}"
  if [[ -n "${GPU_ID}" ]]; then
    echo "  pinned GPU: ${GPU_ID}"
  fi
  echo "  monitor: gpuq -t ${job_id}"
}

if [[ -z "${EXP}" || "${EXP}" == "help" ]]; then
  print_catalog
  exit 0
fi

case "${EXP}" in
  all)
    for e in "${ALL_EXPS[@]}"; do
      submit_one "${e}"
    done
    ;;
  peak|topk|hardneg|hn|topk_peak|topk_hn|topk_hardneg)
    name="${EXP}"
    [[ "${name}" == "hn" ]] && name="hardneg"
    [[ "${name}" == "topk_hardneg" ]] && name="topk_hn"
    submit_one "${name}"
    ;;
  *)
    echo "Unknown: ${EXP}" >&2
    print_catalog
    exit 1
    ;;
esac
