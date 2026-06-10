#!/usr/bin/env bash
# Queue Stage-2 student training from paper_baseline teacher (Official fusion).
#
#   ./submit_paper_teacher_official_skip_gpuq.sh
#   GPU_ID=1 ./submit_paper_teacher_official_skip_gpuq.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPUS="${GPUS:-1}"
GPU_ID="${GPU_ID:-1}"
LABEL_PREFIX="${LABEL_PREFIX:-zdm-paper}"
EXP_NAME="${EXP_NAME:-paper_teacher_official_skip}"

gpu_args=(-G "${GPUS}")
if [[ -n "${GPU_ID}" ]]; then
  gpu_args+=(-g "${GPU_ID}")
fi

job_id=$(gpuq "${gpu_args[@]}" -L "${LABEL_PREFIX}-official-skip" bash -lc \
  "cd \"${SCRIPT_DIR}\" && EXP_NAME=\"${EXP_NAME}\" ./run_paper_teacher_official_skip.sh train")

echo "Queued ${EXP_NAME}: job ${job_id}"
echo "  teacher: output/avenue/paper_baseline/checkpoint-best.pth"
echo "  fusion:  Official (no Eq.6, no cls)"
echo "  epochs:  100-139"
echo "  output:  ${SCRIPT_DIR}/output/avenue/${EXP_NAME}/"
echo "  monitor: gpuq -t ${job_id}"
