#!/usr/bin/env bash
# Submit 4 Route-A experiments via gpuq.
#
#   ./submit_route_a_gpuq.sh              # list
#   ./submit_route_a_gpuq.sh all          # 4 jobs parallel (default)
#   ./submit_route_a_gpuq.sh all serial   # 4 jobs one GPU serially
#   ./submit_route_a_gpuq.sh map          # single job
#
# Monitor: gpuq -t <job_id>

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEACHER_CKPT="${TEACHER_CKPT:-${SCRIPT_DIR}/output/avenue/r0_mse_skip/checkpoint-best.pth}"
GPUS="${GPUS:-1}"
GPU_ID="${GPU_ID:-}"   # optional: pin to one card, e.g. GPU_ID=1 ./submit_route_a_gpuq.sh attn
LABEL_PREFIX="${LABEL_PREFIX:-zdm-spa}"
MODE="${2:-parallel}"
EXP="${1:-all}"

ALL_EXPS=(map fg attn all)

declare -A EXP_DESC
EXP_DESC[map]="仅 map：Stage-1 像素异常图 BCE，全流程"
EXP_DESC[fg]="仅 fg：Stage-2 背景蒸馏，student_only"
EXP_DESC[attn]="仅 attn v2：patch-attn 打分（修复 train/infer 一致），→ spa_a_attn_v2"
EXP_DESC[attn_v2]="同 attn"
EXP_DESC[all]="全开：map+fg+attn，全流程"

print_catalog() {
  echo "Route-A — 4 experiments (teacher: ${TEACHER_CKPT}):"
  echo ""
  for e in "${ALL_EXPS[@]}"; do
    local out="${e}"
    [[ "${e}" == "attn" ]] && out="attn_v2"
    printf "  %-6s → output/avenue/spa_a_%-6s  %s\n" "${e}" "${out}" "${EXP_DESC[$e]}"
  done
  echo ""
  echo "  ./submit_route_a_gpuq.sh all parallel   # 4 jobs"
  echo "  ./submit_route_a_gpuq.sh map            # one job"
}

submit_one() {
  local name="$1"
  local label="${LABEL_PREFIX}-${name}"
  local job_id
  local gpu_args=(-G "${GPUS}")
  if [[ -n "${GPU_ID}" ]]; then
    gpu_args+=(-g "${GPU_ID}")
  fi
  job_id=$(gpuq "${gpu_args[@]}" -L "${label}" bash -lc \
    "cd \"${SCRIPT_DIR}\" && TEACHER_CKPT=\"${TEACHER_CKPT}\" ./run_route_a.sh ${name}")
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
    if [[ "${MODE}" == "serial" ]]; then
      label="${LABEL_PREFIX}-serial-all"
      cmd="cd \"${SCRIPT_DIR}\""
      for e in "${ALL_EXPS[@]}"; do
        cmd+=" && TEACHER_CKPT=\"${TEACHER_CKPT}\" ./run_route_a.sh ${e}"
      done
      job_id=$(gpuq -G "${GPUS}" -L "${label}" bash -lc "${cmd}")
      echo "Queued serial all-4: job ${job_id}  (gpuq -t ${job_id})"
    else
      for e in "${ALL_EXPS[@]}"; do
        submit_one "${e}"
      done
    fi
    ;;
  map|fg|attn|attn_v2)
    submit_one "${EXP%%_v2}"
    ;;
  *)
    echo "Unknown: ${EXP}  (use map|fg|attn|attn_v2|all)" >&2
    print_catalog
    exit 1
    ;;
esac
