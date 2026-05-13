#!/usr/bin/env bash
set -euo pipefail

cd /home/cike/hs
source /home/cike/hs/paper2_layer_pruning/env.sh
source "$ROOT/envs/layer-prune-p100/bin/activate"

TOOLS="$ROOT/experiments/day2_gemma_core_eval/tools"
RESULTS="$ROOT/results/day2_gemma_diagnostics"
REPORTS="$ROOT/reports/day2_gemma_diagnostics"
LOGS="$ROOT/logs/day2_gemma_diagnostics"
MODELS="$ROOT/models/day2_gemma_core_eval"

mkdir -p "$RESULTS" "$REPORTS" "$LOGS"

run_diag() {
  local diagnostic="$1"
  local label="$2"
  local model_path="$3"
  local task="$4"
  local prompt="$5"
  local samples="$6"
  local max_new="$7"
  local runtime_skip="$8"

  echo "===== DIAG ${diagnostic} ${label} ${task} samples=${samples} runtime_skip=${runtime_skip:-none} ====="
  python "$TOOLS/run_gemma_diagnostic_eval.py" \
    --diagnostic-name "$diagnostic" \
    --model-label "$label" \
    --model-path "$model_path" \
    --task "$task" \
    --prompt-key "$prompt" \
    --samples "$samples" \
    --split test \
    --output-dir "$RESULTS" \
    --max-new-tokens "$max_new" \
    --temperature 0 \
    --runtime-skip-layers "$runtime_skip"
}

# 1. XSUM deterministic: compare saved checkpoints with runtime skip.
run_diag xsum_det base google/gemma-2-2b-it xsum GEMMA_XSUM 10 96 ""
run_diag xsum_det reverse_saved "$MODELS/gemma_reverse_6" xsum GEMMA_XSUM 10 96 ""
run_diag xsum_det reverse_runtime google/gemma-2-2b-it xsum GEMMA_XSUM 10 96 "20,21,22,23,24,25"
run_diag xsum_det bi_saved "$MODELS/gemma_bi_6" xsum GEMMA_XSUM 10 96 ""
run_diag xsum_det bi_runtime google/gemma-2-2b-it xsum GEMMA_XSUM 10 96 "2,11,20,21,23,24"

# 2. GSM8K deterministic: check whether runtime skip changes the collapse.
run_diag gsm8k_det base google/gemma-2-2b-it gsm8k GEMMA_INSTRUCT 10 256 ""
run_diag gsm8k_det reverse_saved "$MODELS/gemma_reverse_6" gsm8k GEMMA_INSTRUCT 10 256 ""
run_diag gsm8k_det reverse_runtime google/gemma-2-2b-it gsm8k GEMMA_INSTRUCT 10 256 "20,21,22,23,24,25"
run_diag gsm8k_det bi_saved "$MODELS/gemma_bi_6" gsm8k GEMMA_INSTRUCT 10 256 ""
run_diag gsm8k_det bi_runtime google/gemma-2-2b-it gsm8k GEMMA_INSTRUCT 10 256 "2,11,20,21,23,24"

# 3. HumanEval+ baseline recheck with syntax diagnostics.
run_diag humeval_base_diag base google/gemma-2-2b-it humeval GEMMA_HUMEVAL 20 256 ""
