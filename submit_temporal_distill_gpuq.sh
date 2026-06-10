#!/usr/bin/env bash
# Submit temporal-distillation experiments via gpuq (task spooler).
#
# Teacher (fixed):
#   /root/autodl-tmp/users/zhang-dong-mei/zhangdongmei/aed-mae_search/output/avenue/checkpoint-best.pth
#
# Usage:
#   ./submit_temporal_distill_gpuq.sh                    # list experiments
#   ./submit_temporal_distill_gpuq.sh CORE parallel      # P0+P1+P2, 4 jobs in parallel (recommended)
#   ./submit_temporal_distill_gpuq.sh CORE serial        # same 4 jobs, one GPU serially
#   ./submit_temporal_distill_gpuq.sh P0                 # main experiment only
#   ./submit_temporal_distill_gpuq.sh P1                   # temporal ablations (2 jobs)
#   ./submit_temporal_distill_gpuq.sh P2                   # single-frame control
#   ./submit_temporal_distill_gpuq.sh P3 parallel          # lambda/K sweep (3 jobs)
#   ./submit_temporal_distill_gpuq.sh t3_joint_bw2mse_k8   # one experiment by name
#
# Monitor:
#   gpuq -t <job_id>    # tail output
#   gpuq -c <job_id>    # full output

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEACHER="${TEACHER:-/root/autodl-tmp/users/zhang-dong-mei/zhangdongmei/aed-mae_search/output/avenue/checkpoint-best.pth}"
GPUS="${GPUS:-1}"
LABEL_PREFIX="${LABEL_PREFIX:-zdm-vad}"
MODE="${2:-single}"   # single | parallel | serial
EXP="${1:-}"

declare -A EXP_DESC
EXP_DESC[t3_joint_bw2mse_k8]="P0 主实验: clip8 + 帧级bw2_mse(α=0.3) + 时序joint(λ=0.5)"
EXP_DESC[t2_joint_mse_k8]="P1-B 时序ablation: clip8 + 帧级MSE + 时序joint(λ=0.5)"
EXP_DESC[t2_joint_only_k8]="P1-A 时序ablation: clip8 + 纯时序joint(无帧级MSE)"
EXP_DESC[r3_bw2mse_a30]="P2 单帧对照: clip1 + bw2_mse(α=0.3) (clip实验要超越的上限)"
EXP_DESC[t3_joint_bw2mse_k8_l03]="P3 扫参: joint_lambda=0.3 (其余同P0)"
EXP_DESC[t3_joint_bw2mse_k8_l07]="P3 扫参: joint_lambda=0.7 (其余同P0)"
EXP_DESC[t3_joint_bw2mse_k16]="P3 扫参: clip_len=16 batch=4 (其余同P0)"

CORE_EXPS=(t3_joint_bw2mse_k8 t2_joint_mse_k8 t2_joint_only_k8 r3_bw2mse_a30)
P1_EXPS=(t2_joint_mse_k8 t2_joint_only_k8)
P3_EXPS=(t3_joint_bw2mse_k8_l03 t3_joint_bw2mse_k8_l07 t3_joint_bw2mse_k16)

print_catalog() {
  echo "Available experiments (teacher: ${TEACHER}):"
  echo ""
  for name in "${CORE_EXPS[@]}" "${P3_EXPS[@]}"; do
    printf "  %-28s %s\n" "${name}" "${EXP_DESC[$name]}"
  done
  echo ""
  echo "Submit groups:"
  echo "  ./submit_temporal_distill_gpuq.sh CORE parallel   # 4 core jobs (recommended first)"
  echo "  ./submit_temporal_distill_gpuq.sh P0               # main experiment only"
  echo "  ./submit_temporal_distill_gpuq.sh P3 parallel      # after P0 shows improvement"
}

run_exp() {
  local name="$1"
  local cmd=""
  case "${name}" in
    t3_joint_bw2mse_k8)
      cmd="TEACHER=\"${TEACHER}\" CLIP_LEN=8 BATCH_SIZE=6 TS_BW2_ALPHA=0.3 TS_JOINT_LAMBDA=0.5 ./run_temporal_joint_bw2mse.sh t3_joint_bw2mse_k8"
      ;;
    t2_joint_mse_k8)
      cmd="TEACHER=\"${TEACHER}\" LOSS=temporal_joint_mse CLIP_LEN=8 BATCH_SIZE=6 TS_JOINT_LAMBDA=0.5 ./run_temporal_joint.sh t2_joint_mse_k8"
      ;;
    t2_joint_only_k8)
      cmd="TEACHER=\"${TEACHER}\" LOSS=temporal_joint CLIP_LEN=8 BATCH_SIZE=6 ./run_temporal_joint.sh t2_joint_only_k8"
      ;;
    r3_bw2mse_a30)
      cmd="/root/autodl-tmp/users/zhang-dong-mei/envs/hstforu/bin/python main.py \
  --dataset avenue --student_only --teacher_checkpoint \"${TEACHER}\" \
  --ts_loss_type bw2_mse --ts_bw2_alpha 0.3 --no-ts_bw2_normalize \
  --clip_len 1 --batch_size 100 \
  --experiment_name r3_bw2mse_a30"
      ;;
    t3_joint_bw2mse_k8_l03)
      cmd="TEACHER=\"${TEACHER}\" CLIP_LEN=8 BATCH_SIZE=6 TS_BW2_ALPHA=0.3 TS_JOINT_LAMBDA=0.3 ./run_temporal_joint_bw2mse.sh t3_joint_bw2mse_k8_l03"
      ;;
    t3_joint_bw2mse_k8_l07)
      cmd="TEACHER=\"${TEACHER}\" CLIP_LEN=8 BATCH_SIZE=6 TS_BW2_ALPHA=0.3 TS_JOINT_LAMBDA=0.7 ./run_temporal_joint_bw2mse.sh t3_joint_bw2mse_k8_l07"
      ;;
    t3_joint_bw2mse_k16)
      cmd="TEACHER=\"${TEACHER}\" CLIP_LEN=16 BATCH_SIZE=4 TS_BW2_ALPHA=0.3 TS_JOINT_LAMBDA=0.5 ./run_temporal_joint_bw2mse.sh t3_joint_bw2mse_k16"
      ;;
    *)
      echo "Unknown experiment: ${name}" >&2
      exit 1
      ;;
  esac
  echo "${cmd}"
}

submit_one() {
  local name="$1"
  local label="${LABEL_PREFIX}-${name}"
  local inner_cmd
  inner_cmd="$(run_exp "${name}")"
  local job_id
  job_id=$(gpuq -G "${GPUS}" -L "${label}" bash -lc "cd \"${SCRIPT_DIR}\" && ${inner_cmd}")
  echo "Queued ${name}: job ${job_id}  |  ${EXP_DESC[$name]}"
  echo "  monitor: gpuq -t ${job_id}"
}

submit_group() {
  local -n arr=$1
  local mode="$2"
  if [[ "${mode}" == "serial" ]]; then
    local label="${LABEL_PREFIX}-serial-$1"
    local serial_cmd="cd \"${SCRIPT_DIR}\""
    for name in "${arr[@]}"; do
      serial_cmd+=" && $(run_exp "${name}")"
    done
    local job_id
    job_id=$(gpuq -G "${GPUS}" -L "${label}" bash -lc "${serial_cmd}")
    echo "Queued serial group ($1): job ${job_id}  (gpuq -t ${job_id})"
  else
    for name in "${arr[@]}"; do
      submit_one "${name}"
    done
  fi
}

if [[ -z "${EXP}" ]]; then
  print_catalog
  exit 0
fi

if [[ ! -f "${TEACHER}" ]]; then
  echo "Teacher checkpoint not found: ${TEACHER}" >&2
  exit 1
fi

case "${EXP}" in
  CORE)
    submit_group CORE_EXPS "${MODE}"
    ;;
  P0)
    submit_one t3_joint_bw2mse_k8
    ;;
  P1)
    submit_group P1_EXPS "${MODE:-parallel}"
    ;;
  P2)
    submit_one r3_bw2mse_a30
    ;;
  P3)
    submit_group P3_EXPS "${MODE:-parallel}"
    ;;
  t3_joint_bw2mse_k8|t2_joint_mse_k8|t2_joint_only_k8|r3_bw2mse_a30|\
  t3_joint_bw2mse_k8_l03|t3_joint_bw2mse_k8_l07|t3_joint_bw2mse_k16)
    submit_one "${EXP}"
    ;;
  *)
    echo "Unknown group/experiment: ${EXP}" >&2
    print_catalog
    exit 1
    ;;
esac
