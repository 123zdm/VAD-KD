#!/usr/bin/env bash
# Asymmetric distillation ablations on AED-MAE
set -euo pipefail

ROOT="/root/autodl-tmp/users/zhang-dong-mei/zhangdongmei/aed-mae"
PYTHON="/root/autodl-tmp/users/zhang-dong-mei/envs/hstforu/bin/python"
DATASET="${1:-avenue}"
STRATEGY="${2:-all}"
LOSS="${3:-mse}"
TS_BW2_ALPHA="${TS_BW2_ALPHA:-0.3}"

cd "${ROOT}"
CMD=(
  "${PYTHON}" main.py
  --dataset "${DATASET}"
  --ts_abnormal_strategy "${STRATEGY}"
  --ts_loss_type "${LOSS}"
  --run_type train
)
if [[ "${LOSS}" == "bw2_mse" ]]; then
  CMD+=(--ts_bw2_alpha "${TS_BW2_ALPHA}")
fi
"${CMD[@]}"
