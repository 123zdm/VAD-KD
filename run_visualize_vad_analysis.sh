#!/usr/bin/env bash
# Pixel heatmaps + per-video score curves (teacher vs student analysis for paper figures).
#
# Usage:
#   ./run_visualize_vad_analysis.sh              # full run (slow: all test frames)
#   SMOKE=1 ./run_visualize_vad_analysis.sh       # quick smoke test
#
# Custom checkpoints:
#   TEACHER=/path/to/checkpoint-best.pth \
#   MSE_STUDENT=/path/to/mse-student.pth \
#   BW2_STUDENT=/path/to/bw2-student.pth \
#   OUT_DIR=output/avenue/viz_vad_analysis \
#   ./run_visualize_vad_analysis.sh

set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-/root/autodl-tmp/users/zhang-dong-mei/envs/hstforu/bin/python}"
TEACHER="${TEACHER:-../aed-mae_search/output/avenue/checkpoint-best.pth}"
MSE_STUDENT="${MSE_STUDENT:-output/avenue/r0_mse_skip/checkpoint-best-student.pth}"
BW2_STUDENT="${BW2_STUDENT:-output/avenue/author_teacher_bw2mse_a30_v2/checkpoint-best-student.pth}"
OUT_DIR="${OUT_DIR:-output/avenue/viz_vad_analysis}"
VIDEO_IDS="${VIDEO_IDS:-01,06,12}"
MAX_FRAMES="${MAX_FRAMES:-0}"
MAX_CURVE_VIDEOS="${MAX_CURVE_VIDEOS:-0}"
USE_FILT="${USE_FILT:-0}"

EXTRA=()
if [[ "${SMOKE:-0}" == "1" ]]; then
  OUT_DIR="${OUT_DIR}_smoke"
  MAX_FRAMES=600
  MAX_CURVE_VIDEOS=2
  VIDEO_IDS="01,06"
  EXTRA+=(--n_abnormal_frames 1 --n_normal_frames 1)
fi
if [[ "${USE_FILT}" == "1" ]]; then
  EXTRA+=(--use_filt)
fi

"${PYTHON}" util/visualize_vad_analysis.py \
  --dataset avenue \
  --teacher_checkpoint "${TEACHER}" \
  --student_checkpoints \
    "mse_skip=${MSE_STUDENT}" \
    "bw2mse=${BW2_STUDENT}" \
  --output_dir "${OUT_DIR}" \
  --video_ids "${VIDEO_IDS}" \
  --max_frames "${MAX_FRAMES}" \
  --max_curve_videos "${MAX_CURVE_VIDEOS}" \
  "${EXTRA[@]}"

echo "Done. See ${OUT_DIR}/pixel/ and ${OUT_DIR}/curves/"
