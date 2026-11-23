#!/usr/bin/env bash
# newsgpt_ai_news_human.csv 평가 실행
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DEVICE="${DEVICE:-cuda:0}"
INFER_STRUCTURE_WEIGHT="${INFER_STRUCTURE_WEIGHT:-0.7}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"

CSV="${CSV:-/home/dsail/2026-AINews-Detection/notebook/real-world/cnet_all_cleaned.csv}" 

# ── 체크포인트 자동 탐색 (최신 .pth) ─────────────────────────────
RUN_DIR="${RUN_DIR:-/home/dsail/2026-AINews-Detection/model/results/2026-06-14/RAID-news-14-57-38-401239}"
CKPT="${CKPT:-RAID-news_30_8_0.7_20260614-145738.pth}"

# RUN_DIR="${RUN_DIR:-/home/dsail/2026-AINews-Detection/model/results/2026-06-17/kor_aigen-17-00-00-713071}"
# CKPT="${CKPT:-kor_aigen_15_8_0.7_epoch15_20260617-170003.pth}"

# RUN_DIR="${RUN_DIR:-/home/dsail/2026-AINews-Detection/model/results/2026-06-24/kor_aigen-12-18-37-880725}"
# CKPT="${CKPT:-kor_aigen_150_8_0.7_20260624-121837.pth}"

echo "CSV        : $CSV"
echo "RUN_DIR    : $RUN_DIR"
echo "CHECKPOINT : $CKPT"
echo "DEVICE     : $DEVICE"
echo "infer_w    : $INFER_STRUCTURE_WEIGHT"
echo ""

EVAL_ARGS=(
  --csv                   "$CSV"
  --run_dir               "$RUN_DIR"
  --checkpoint            "$CKPT"
  --DEVICE                "$DEVICE"
  --infer_structure_weight "$INFER_STRUCTURE_WEIGHT"
)
if [[ "$MAX_SAMPLES" -gt 0 ]]; then
  EVAL_ARGS+=(--max_samples "$MAX_SAMPLES")
fi

python src/eval/eval_realworld.py "${EVAL_ARGS[@]}"


# RUN_DIR=results/2026-06-01/RAID-news-22-28-26-322680 \
# CKPT=RAID-news_5_8_8.0_20260601-222826.pth \
# INFER_STRUCTURE_WEIGHT=0.7 \
# bash run_eval_newsgpt.sh