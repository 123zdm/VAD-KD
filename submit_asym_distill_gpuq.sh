#!/usr/bin/env bash
# Submit asymmetric distillation jobs via gpuq.
#
# Usage:
#   ./submit_asym_distill_gpuq.sh avenue all bw2          # bw2_all on avenue
#   ./submit_asym_distill_gpuq.sh avenue all mse         # mse_all
#   ./submit_asym_distill_gpuq.sh avenue skip bw2        # bw2_skip
#   ./submit_asym_distill_gpuq.sh avenue ALL serial      # legacy: mse all/skip/margin serial
#
# Tail output:  gpuq -t <job_id>
# Full output:  gpuq -c <job_id>

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET="${1:-avenue}"
STRATEGY="${2:-all}"     # all | skip | margin
LOSS="${3:-mse}"         # mse | bw2 | bw2_mse
MODE="${4:-single}"      # single | parallel (for multiple strategies, legacy)
GPUS="${GPUS:-1}"
LABEL_PREFIX="${LABEL_PREFIX:-zdm-asym}"

submit_one() {
  local strategy="$1"
  local loss="$2"
  local label="${LABEL_PREFIX}-${DATASET}-${loss}-${strategy}"
  local job_id
  job_id=$(gpuq -G "${GPUS}" -L "${label}" bash -lc \
    "cd \"${SCRIPT_DIR}\" && ./run_asym_distill.sh \"${DATASET}\" \"${strategy}\" \"${loss}\"")
  echo "Queued ${loss}/${strategy}: job id ${job_id}  (gpuq -t ${job_id})"
}

if [[ "${STRATEGY}" == "ALL" && "${MODE}" == "serial" ]]; then
  LABEL="${LABEL_PREFIX}-${DATASET}-mse-e0-e1-e2-serial"
  JOB_ID=$(gpuq -G "${GPUS}" -L "${LABEL}" bash -lc \
    "cd \"${SCRIPT_DIR}\" && ./run_asym_distill_all.sh \"${DATASET}\"")
  echo "Queued serial mse all/skip/margin: job id ${JOB_ID}"
  echo "Tail output: gpuq -t ${JOB_ID}"
elif [[ "${STRATEGY}" == "ALL" && "${MODE}" == "parallel" ]]; then
  submit_one all mse
  submit_one skip mse
  submit_one margin mse
  echo "Queued 3 parallel mse jobs on dataset=${DATASET}"
else
  submit_one "${STRATEGY}" "${LOSS}"
  echo "Queued single job: dataset=${DATASET}, loss=${LOSS}, strategy=${STRATEGY}"
fi
