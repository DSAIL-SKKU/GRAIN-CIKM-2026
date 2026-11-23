#!/usr/bin/env bash
# MultiDomainXSum 학습 후 eval.py(ai_gen_news CSV)까지 실행
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DEVICE="${DEVICE:-cuda:0}"
EPOCHS="${EPOCHS:-10}"
PATIENCE="${PATIENCE:-5}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"

# REG, INFER_STRUCTURE_WEIGHT default value = 0.7 (it must same both REG and INFER_STRUCTURE_WEIGHT)
REG="${REG:-0.7}"
INFER_STRUCTURE_WEIGHT="${INFER_STRUCTURE_WEIGHT:-0.7}"

TRAIN_JSON="${TRAIN_JSON:-/home/dsail/2026-AINews-Dataset/benchmark/DetectRL/Benchmark/Benchmark_Data/Multi_Domain/multi_domains_xsum_train.json}"
TEST_JSON="${TEST_JSON:-/home/dsail/2026-AINews-Dataset/benchmark/DetectRL/Benchmark/Benchmark_Data/Multi_Domain/multi_domains_xsum_test.json}"

require_file() {
  local path="$1"
  local label="$2"
  if [[ ! -f "$path" ]]; then
    echo "error: ${label} not found: ${path}" >&2
    exit 1
  fi
}

require_file "$TRAIN_JSON" "train JSON"
require_file "$TEST_JSON" "test JSON"

echo "DEVICE=$DEVICE REG=$REG EPOCHS=$EPOCHS PATIENCE=$PATIENCE"
echo "BATCH_SIZE=$BATCH_SIZE INFER_STRUCTURE_WEIGHT=$INFER_STRUCTURE_WEIGHT"
echo "TRAIN_JSON=$TRAIN_JSON"
echo "TEST_JSON=$TEST_JSON"

echo "========== [1/2] train.py =========="
python src/train.py \
  --dataset MultiDomainXSum \
  --finetune \
  --article_level \
  --DEVICE "$DEVICE" \
  --reg "$REG" \
  --epochs "$EPOCHS" \
  --patience "$PATIENCE" \
  --batch_size "$BATCH_SIZE" \
  --infer_structure_weight "$INFER_STRUCTURE_WEIGHT" \
  --xsum_train_path "$TRAIN_JSON" \
  --xsum_test_path "$TEST_JSON"

RUN_DIR="$(
  find "$ROOT/results" -type d -name 'MultiDomainXSum-*' -printf '%T@ %p\n' 2>/dev/null \
    | sort -n | tail -1 | cut -d' ' -f2-
)"
if [[ -z "${RUN_DIR}" || ! -d "${RUN_DIR}" ]]; then
  echo "error: no MultiDomainXSum run dir under $ROOT/results" >&2
  exit 1
fi

echo ""
echo "========== [2/2] eval.py =========="
echo "run_dir: $RUN_DIR"

EVAL_ARGS=(
  --DEVICE "$DEVICE"
  --run_dir "$RUN_DIR"
  --infer_structure_weight "$INFER_STRUCTURE_WEIGHT"
)
if [[ "$MAX_SAMPLES" -gt 0 ]]; then
  EVAL_ARGS+=(--max_samples "$MAX_SAMPLES")
fi

python src/eval/eval.py "${EVAL_ARGS[@]}"

echo ""
echo "done. run_dir: $RUN_DIR"