#!/usr/bin/env bash
# Submit paper baseline (full teacher+student training) via gpuq.
#
#   ./submit_paper_baseline_gpuq.sh
#   GPU_ID=1 ./submit_paper_baseline_gpuq.sh
#   gpuq -t <job_id>

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPUS="${GPUS:-1}"
GPU_ID="${GPU_ID:-}"
LABEL_PREFIX="${LABEL_PREFIX:-zdm-paper}"
EXP_NAME="${EXP_NAME:-paper_baseline}"

gpu_args=(-G "${GPUS}")
if [[ -n "${GPU_ID}" ]]; then
  gpu_args+=(-g "${GPU_ID}")
fi

job_id=$(gpuq "${gpu_args[@]}" -L "${LABEL_PREFIX}-baseline" bash -lc \
  "cd \"${SCRIPT_DIR}\" && EXP_NAME=\"${EXP_NAME}\" ./run_paper_baseline.sh train")

echo "Queued paper_baseline: job ${job_id}"
echo "  output: ${SCRIPT_DIR}/output/avenue/${EXP_NAME}/"
echo "  logs:   log_train.txt, log_test.txt"
echo "  ckpt:   checkpoint-best.pth (teacher), checkpoint-best-student.pth"
[[ -n "${GPU_ID}" ]] && echo "  GPU:    ${GPU_ID}"
echo "  monitor: gpuq -t ${job_id}"
