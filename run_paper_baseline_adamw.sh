#!/usr/bin/env bash
# Paper baseline with _baseline_defaults optimizer: AdamW + weight_decay=0.05.
# All other settings match --paper_baseline (Eq.6, cls head, 25% abnormal, etc.).
#
#   ./run_paper_baseline_adamw.sh              # train → output/avenue/paper_baseline_adamw
#   ./run_paper_baseline_adamw.sh inference

set -euo pipefail

ROOT="/root/autodl-tmp/users/zhang-dong-mei/zhangdongmei/aed-mae"
PYTHON="${PYTHON:-/root/autodl-tmp/users/zhang-dong-mei/envs/hstforu/bin/python}"
DATASET="${DATASET:-avenue}"
EXP_NAME="${EXP_NAME:-paper_baseline_adamw}"
RUN_TYPE="${1:-train}"
OUT_DIR="${ROOT}/output/${DATASET}/${EXP_NAME}"

cd "${ROOT}"

COMMON=(
  --dataset "${DATASET}"
  --paper_baseline
  --optimizer adamw
  --weight_decay 0.05
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

echo "Done. Mode=${RUN_TYPE}  output=${OUT_DIR}"
