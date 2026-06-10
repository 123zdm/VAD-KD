#!/usr/bin/env bash
# Queue paper_baseline_adamw + grad_masking_v1 via gpuq.
#
#   ./submit_paper_baseline_adamw_grad_mask_gpuq.sh
#   GPU_ID=1 ./submit_paper_baseline_adamw_grad_mask_gpuq.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPUS="${GPUS:-1}"
GPU_ID="${GPU_ID:-1}"
LABEL_PREFIX="${LABEL_PREFIX:-zdm-paper}"
EXP_NAME="${EXP_NAME:-paper_baseline_adamw_grad_mask}"

gpu_args=(-G "${GPUS}")
if [[ -n "${GPU_ID}" ]]; then
  gpu_args+=(-g "${GPU_ID}")
fi

job_id=$(gpuq "${gpu_args[@]}" -L "${LABEL_PREFIX}-adamw-grad-mask" bash -lc \
  "cd \"${SCRIPT_DIR}\" && EXP_NAME=\"${EXP_NAME}\" ./run_paper_baseline_adamw_grad_mask.sh train")

echo "Queued paper_baseline_adamw_grad_mask: job ${job_id}"
echo "  optimizer: AdamW, weight_decay=0.05"
echo "  masking: grad_masking_v1"
echo "  output: ${SCRIPT_DIR}/output/avenue/${EXP_NAME}/"
echo "  monitor: gpuq -t ${job_id}"
