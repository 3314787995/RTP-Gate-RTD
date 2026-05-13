#!/usr/bin/env bash
set -euo pipefail

cd /home/cike/hs
source /home/cike/hs/paper2_layer_pruning/env.sh
source "$ROOT/envs/layer-prune-p100/bin/activate"

TOOLS="$ROOT/experiments/day2_gemma_core_eval/tools"
RESULTS="$ROOT/results/day2_gemma_core_eval"
REPORTS="$ROOT/reports/day2_gemma_core_eval"
MODELS="$ROOT/models/day2_gemma_core_eval"

mkdir -p "$RESULTS" "$REPORTS" "$ROOT/logs/day2_gemma_core_eval"

run_eval() {
  local label="$1"
  local model_path="$2"
  local strategy="$3"
  local layers="$4"
  local task="$5"
  local prompt="$6"
  local samples="$7"
  local max_new="$8"

  echo "===== RUN ${label} ${task} samples=${samples} ====="
  python "$TOOLS/run_gemma_transformers_eval.py" \
    --model-label "$label" \
    --model-path "$model_path" \
    --task "$task" \
    --prompt-key "$prompt" \
    --samples "$samples" \
    --split test \
    --output-dir "$RESULTS" \
    --prune-strategy "$strategy" \
    --pruned-layers "$layers" \
    --max-new-tokens "$max_new" \
    --syntax-fallback
}

run_all_tasks() {
  local label="$1"
  local model_path="$2"
  local strategy="$3"
  local layers="$4"

  run_eval "$label" "$model_path" "$strategy" "$layers" gsm8k GEMMA_INSTRUCT 20 512
  run_eval "$label" "$model_path" "$strategy" "$layers" humeval GEMMA_HUMEVAL 5 512
  run_eval "$label" "$model_path" "$strategy" "$layers" xsum GEMMA_XSUM 20 96
}

run_all_tasks gemma_base google/gemma-2-2b-it baseline ""
run_all_tasks gemma_reverse_6 "$MODELS/gemma_reverse_6" reverse 20,21,22,23,24,25
run_all_tasks gemma_bi_6 "$MODELS/gemma_bi_6" bi 2,11,20,21,23,24

python "$TOOLS/summarize_day2_results.py" \
  --results-dir "$RESULTS" \
  --reports-dir "$REPORTS"
