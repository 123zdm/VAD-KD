#!/usr/bin/env bash
# Run E0/E1/E2 asymmetric distillation ablations sequentially.
#
# Usage:
#   ./run_asym_distill_all.sh [dataset]
#   ./run_asym_distill_all.sh avenue
#   ./run_asym_distill_all.sh shanghai

set -euo pipefail

ROOT="/root/autodl-tmp/users/zhang-dong-mei/zhangdongmei/aed-mae"
PYTHON="/root/autodl-tmp/users/zhang-dong-mei/envs/hstforu/bin/python"
DATASET="${1:-avenue}"

cd "${ROOT}"

for STRATEGY in all skip margin; do
  echo "=========================================="
  echo "Dataset=${DATASET}  Strategy=${STRATEGY}"
  echo "=========================================="
  "${PYTHON}" main.py \
    --dataset "${DATASET}" \
    --ts_abnormal_strategy "${STRATEGY}" \
    --run_type train
done

echo "All done: ${DATASET} (e0_all, e1_skip, e2_margin)"
