#!/usr/bin/env bash
# Train Stage-2 (student) only from an existing teacher checkpoint.
#
# Usage:
#   ./run_student_from_teacher.sh bw2mse_a30_all
#   ./run_student_from_teacher.sh bw2mse_a30_all /path/to/checkpoint-best.pth
#
# Env overrides:
#   LOSS=bw2_mse|bw2|mse          (default: bw2_mse)
#   TS_BW2_ALPHA=0.3              hybrid weight: alpha*BW2 + (1-alpha)*MSE
#
# Recommended alpha grid:
#   0.2  -> mostly MSE, safe baseline+
#   0.3  -> default starting point
#   0.5  -> balanced ablation

set -euo pipefail

ROOT="/root/autodl-tmp/users/zhang-dong-mei/zhangdongmei/aed-mae"
PYTHON="/root/autodl-tmp/users/zhang-dong-mei/envs/hstforu/bin/python"
DATASET="${DATASET:-avenue}"
LOSS="${LOSS:-bw2_mse}"
TS_BW2_ALPHA="${TS_BW2_ALPHA:-0.3}"
TEACHER_CKPT="${2:-${ROOT}/output/avenue/author_teacher_bw2mse_a30_v2/checkpoint-best.pth}"

if [[ -n "${1:-}" ]]; then
  EXP_NAME="${1}"
else
  ALPHA_TAG=$(python3 - <<PY
alpha = float("${TS_BW2_ALPHA}")
print(f"{int(round(alpha * 100)):02d}")
PY
)
  if [[ "${LOSS}" == "bw2_mse" ]]; then
    EXP_NAME="bw2mse_a${ALPHA_TAG}"
  elif [[ "${LOSS}" == "bw2" ]]; then
    EXP_NAME="bw2"
  else
    EXP_NAME="mse"
  fi
fi

cd "${ROOT}"
CMD=(
  "${PYTHON}" main.py
  --dataset "${DATASET}"
  --ts_loss_type "${LOSS}"
  --experiment_name "${EXP_NAME}"
  --student_only
  --teacher_checkpoint "${TEACHER_CKPT}"
  --run_type train
)
if [[ "${LOSS}" == "bw2_mse" ]]; then
  CMD+=(--ts_bw2_alpha "${TS_BW2_ALPHA}")
fi
echo "Running: LOSS=${LOSS}, alpha=${TS_BW2_ALPHA}, exp=${EXP_NAME}"
"${CMD[@]}"
