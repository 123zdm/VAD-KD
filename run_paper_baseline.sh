#!/usr/bin/env bash
# CVPR 2024 paper baseline: full two-stage training (teacher 100ep + student 40ep).
#
# Model (Avenue, patch 16): encoder 3×CvT/256/4h, teacher decoder 3×128/4h,
# student decoder 1×128/4h — see model_factory.mae_cvt_patch16.
#
#   ./run_paper_baseline.sh              # train → output/avenue/paper_baseline
#   ./run_paper_baseline.sh inference    # eval best ckpts with paper Eq.(6)
#
# Paper settings: Adam lr=1e-4, batch=100, 25% synthetic abnormal,
# grad-weighted recon, cls BCE, Eq.(6) α=0.4 β=0.3 γ=0.3.

set -euo pipefail

ROOT="/root/autodl-tmp/users/zhang-dong-mei/zhangdongmei/aed-mae"
PYTHON="${PYTHON:-/root/autodl-tmp/users/zhang-dong-mei/envs/hstforu/bin/python}"
DATASET="${DATASET:-avenue}"
EXP_NAME="${EXP_NAME:-paper_baseline}"
RUN_TYPE="${1:-train}"
OUT_DIR="${ROOT}/output/${DATASET}/${EXP_NAME}"

cd "${ROOT}"

COMMON=(
  --dataset "${DATASET}"
  --paper_baseline
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
