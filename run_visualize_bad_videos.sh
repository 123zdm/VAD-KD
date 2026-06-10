#!/usr/bin/env bash
# Diagnose worst Avenue videos: FN/FP + pixel heatmaps.
#
# Usage:
#   ./run_visualize_bad_videos.sh
#   VIDEO_IDS=17,20 ./run_visualize_bad_videos.sh

set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-/root/autodl-tmp/users/zhang-dong-mei/envs/hstforu/bin/python}"
TEACHER="${TEACHER:-../aed-mae_search/output/avenue_cls_head/checkpoint-best.pth}"
STUDENT="${STUDENT:-../aed-mae_search/output/avenue_cls_head/checkpoint-best-student.pth}"
OUT_DIR="${OUT_DIR:-output/avenue/bad_video_diagnosis}"
VIDEO_IDS="${VIDEO_IDS:-17,20,16,01}"

"${PYTHON}" util/visualize_bad_videos.py \
  --teacher_checkpoint "${TEACHER}" \
  --student_checkpoint "${STUDENT}" \
  --output_dir "${OUT_DIR}" \
  --video_ids "${VIDEO_IDS}" \
  --top_k_frames 3

echo "Done. See ${OUT_DIR}/curves/ and ${OUT_DIR}/pixel/"
