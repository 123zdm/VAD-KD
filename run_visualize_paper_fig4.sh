#!/usr/bin/env bash
# Paper-style anomaly score curve only (no frame thumbnails).
#
#   ./run_visualize_paper_fig4.sh
#   VIDEO_ID=04 SCORE_MODE=official ./run_visualize_paper_fig4.sh
#   STUDENT=../aed-mae_search/output/avenue_cls_head/checkpoint-best-student.pth ./run_visualize_paper_fig4.sh

set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-/root/autodl-tmp/users/zhang-dong-mei/envs/hstforu/bin/python}"
STUDENT="${STUDENT:-../aed-mae_search/output/avenue_cls_head/checkpoint-best-student.pth}"
VIDEO_ID="${VIDEO_ID:-04}"
SCORE_MODE="${SCORE_MODE:-official}"
OUT_DIR="${OUT_DIR:-output/avenue/viz_score_${SCORE_MODE}_v${VIDEO_ID}}"

"${PYTHON}" -u util/visualize_paper_fig4.py \
  --student_checkpoint "${STUDENT}" \
  --video_id "${VIDEO_ID}" \
  --score_mode "${SCORE_MODE}" \
  --output_dir "${OUT_DIR}"

echo "Done. See ${OUT_DIR}/score_${SCORE_MODE}_video_${VIDEO_ID}.png"
