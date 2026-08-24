#!/usr/bin/env bash
# RoBERTa 모델을 multi_domains_xsum_train.json 으로 평가
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# ── 환경 변수로 오버라이드 가능 ───────────────────────────────────────────────
DEVICE="${DEVICE:-cuda:0}"
BATCH_SIZE="${BATCH_SIZE:-8}"
EMBED_BATCH_SIZE="${EMBED_BATCH_SIZE:-8}"
INFER_STRUCTURE_WEIGHT="${INFER_STRUCTURE_WEIGHT:-0.7}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"

DATA_JSON="${DATA_JSON:-/home/dsail/2026-AINews-Detection/datasets/multi_domains_xsum_test.json}"
RUN_DIR="${RUN_DIR:-/home/dsail/2026-AINews-Detection/model/results/2026-05-31/RAID-news-20-58-38-342807}"
CKPT="${CKPT:-RAID-news_30_8_0.7_20260531-205838.pth}"

OUTPUT_DIR="${OUTPUT_DIR:-${RUN_DIR}/eval_detectRL}"

# ── 경로 확인 ─────────────────────────────────────────────────────────────────
if [[ ! -f "$DATA_JSON" ]]; then
  echo "error: DATA_JSON not found: $DATA_JSON" >&2
  exit 1
fi
if [[ ! -d "$RUN_DIR" ]]; then
  echo "error: RUN_DIR not found: $RUN_DIR" >&2
  exit 1
fi
if [[ ! -f "${RUN_DIR}/${CKPT}" ]]; then
  echo "error: checkpoint not found: ${RUN_DIR}/${CKPT}" >&2
  exit 1
fi

echo "======================================================"
echo " RoBERTa  ×  DetectRL evaluation"
echo "======================================================"
echo "  data_json  : $DATA_JSON"
echo "  run_dir    : $RUN_DIR"
echo "  checkpoint : $CKPT"
echo "  device     : $DEVICE"
echo "  batch_size : $BATCH_SIZE  (embed: $EMBED_BATCH_SIZE)"
echo "  infer_w    : $INFER_STRUCTURE_WEIGHT"
echo "  max_samples: ${MAX_SAMPLES} (0 = all)"
echo "  output_dir : $OUTPUT_DIR"
echo "======================================================"
echo ""

EVAL_ARGS=(
  --data_json              "$DATA_JSON"
  --run_dir                "$RUN_DIR"
  --checkpoint             "$CKPT"
  --DEVICE                 "$DEVICE"
  --batch_size             "$BATCH_SIZE"
  --embed_batch_size       "$EMBED_BATCH_SIZE"
  --infer_structure_weight "$INFER_STRUCTURE_WEIGHT"
  --output_dir             "$OUTPUT_DIR"
)
if [[ "$MAX_SAMPLES" -gt 0 ]]; then
  EVAL_ARGS+=(--max_samples "$MAX_SAMPLES")
fi

python eval/eval_detectRL.py "${EVAL_ARGS[@]}"


# ── 사용 예시 ─────────────────────────────────────────────────────────────────
# # 기본 실행
# bash model/run_train_eval_detectRL.sh
#
# # test split 평가
# DATA_JSON=/home/dsail/2026-AINews-Detection/datasets/multi_domains_xsum_test.json \
# bash model/run_train_eval_detectRL.sh
#
# # GPU / 배치 크기 변경
# DEVICE=cuda:1 BATCH_SIZE=64 bash model/run_train_eval_detectRL.sh
#
# # 다른 체크포인트 지정
# RUN_DIR=results/2026-05-31/RAID-news-20-58-38-342807 \
# CKPT=RAID-news_30_8_0.7_epoch10_20260531-205838.pth \
# bash model/run_train_eval_detectRL.sh
#
# # 빠른 테스트 (첫 200개 샘플)
# MAX_SAMPLES=200 bash model/run_train_eval_detectRL.sh
