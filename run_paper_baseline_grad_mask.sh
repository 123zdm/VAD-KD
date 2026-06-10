#!/usr/bin/env bash
# Paper baseline + grad_masking_v1 (train: mask low-gradient patches; test still uses mask_ratio).
#
# Same as run_paper_baseline.sh except --masking_method grad_masking_v1.
#
#   ./run_paper_baseline_grad_mask.sh              # train → output/avenue/paper_baseline_grad_mask
#   ./run_paper_baseline_grad_mask.sh inference

set -euo pipefail

ROOT="/root/autodl-tmp/users/zhang-dong-mei/zhangdongmei/aed-mae"
PYTHON="${PYTHON:-/root/autodl-tmp/users/zhang-dong-mei/envs/hstforu/bin/python}"
DATASET="${DATASET:-avenue}"
EXP_NAME="${EXP_NAME:-paper_baseline_grad_mask}"
RUN_TYPE="${1:-train}"
OUT_DIR="${ROOT}/output/${DATASET}/${EXP_NAME}"

cd "${ROOT}"

COMMON=(
  --dataset "${DATASET}"
  --paper_baseline
  --masking_method grad_masking_v1
  --experiment_name "${EXP_NAME}"
  --run_type "${RUN_TYPE}"
  --ts_loss_type mse
)

if [[ "${RUN_TYPE}" == "inference" ]]; then
  "${PYTHON}" main.py "${COMMON[@]}" \
    --teacher_checkpoint "${OUT_DIR}/checkpoint-best.pth" \
    --student_checkpoint "${OUT_DIR}/checkpoint-best-student.pth"
else
  "${PYTHON}" main.py "${COMMON[@]}"
fi

echo "Done. Mode=${RUN_TYPE}  masking=grad_masking_v1  output=${OUT_DIR}"
