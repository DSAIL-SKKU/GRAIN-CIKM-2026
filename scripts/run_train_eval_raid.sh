#!/usr/bin/env bash
# RAID-news 데이터로 학습 후 ai_gen_news CSV 평가까지 한 번에 실행
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

DEVICE="${DEVICE:-cuda:0}"
REG="${REG:-0.7}"
EPOCHS="${EPOCHS:-30}"
PATIENCE="${PATIENCE:-5}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
INFER_STRUCTURE_WEIGHT="${INFER_STRUCTURE_WEIGHT:-0.7}"
SAVE_EVERY_EPOCHS="${SAVE_EVERY_EPOCHS:-5}"

DETECT_LLM="${DETECT_LLM:-all}"
RAID_DOMAIN="${RAID_DOMAIN:-news}"

RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_ROOT="${LOG_ROOT:-$ROOT/results/run_logs}"
mkdir -p "$LOG_ROOT"
SESSION_LOG="$LOG_ROOT/raid_pipeline_${RUN_STAMP}.log"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "$SESSION_LOG"
}

log "session start: $RUN_STAMP"
log "DEVICE=$DEVICE REG=$REG EPOCHS=$EPOCHS INFER_STRUCTURE_WEIGHT=$INFER_STRUCTURE_WEIGHT"

echo "========== [1/2] train_raid.py =========="
python src/train_raid.py \
  --dataset RAID-news \
  --finetune \
  --article_level \
  --DEVICE "$DEVICE" \
  --reg "$REG" \
  --epochs "$EPOCHS" \
  --patience "$PATIENCE" \
  --batch_size "$BATCH_SIZE" \
  --checkpoint_criterion "auc" \
  --infer_structure_weight "$INFER_STRUCTURE_WEIGHT" \
  --save_every_epochs "$SAVE_EVERY_EPOCHS" \
  --detectLLM "$DETECT_LLM" \
  --raid_domain "$RAID_DOMAIN" \
  2>&1 | tee -a "$SESSION_LOG"
train_status=${PIPESTATUS[0]}
if [[ "$train_status" -ne 0 ]]; then
  log "train_raid.py failed (exit=$train_status)"
  exit "$train_status"
fi

# train_raid.py가 저장한 run_dir: results/날짜/RAID-news-시간 (가장 최근 .pth)
RUN_DIR="$(find "$ROOT/results" -name '*.pth' -printf '%T@ %h\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)"
if [[ -z "$RUN_DIR" || ! -d "$RUN_DIR" ]]; then
  log "error: no checkpoint under $ROOT/results (training may have failed)"
  exit 1
fi
CKPT="$(find "$RUN_DIR" -maxdepth 1 -name '*.pth' -printf '%T@ %f\n' | sort -n | tail -1 | cut -d' ' -f2-)"
log "train done: run_dir=$RUN_DIR checkpoint=$CKPT"
log "train_history: $RUN_DIR/train_history.json"
 

EVAL_OUT="${RUN_DIR}/eval_ai_gen_news/run_${RUN_STAMP}"
mkdir -p "$EVAL_OUT"

log ""
log "========== [2/2] eval.py =========="
log "run_dir: $RUN_DIR"
log "checkpoint: $CKPT"
log "eval_output_dir: $EVAL_OUT"

EVAL_ARGS=(
  --DEVICE "$DEVICE"
  --run_dir "$RUN_DIR"
  --checkpoint "$CKPT"
  --infer_structure_weight "$INFER_STRUCTURE_WEIGHT"
  --output_dir "$EVAL_OUT"
)
if [[ "$MAX_SAMPLES" -gt 0 ]]; then
  EVAL_ARGS+=(--max_samples "$MAX_SAMPLES")
fi

python  src/eval/eval.py "${EVAL_ARGS[@]}" 2>&1 | tee -a "$SESSION_LOG"
eval_status=${PIPESTATUS[0]}
if [[ "$eval_status" -ne 0 ]]; then
  log "eval.py failed (exit=$eval_status)"
  exit "$eval_status"
fi

log ""
log "done."
log "checkpoint: $RUN_DIR/$CKPT"
log "eval outputs: $EVAL_OUT"
log "session log: $SESSION_LOG"
# python eval_infer_reg_sweep.py "${EVAL_ARGS[@]}"

