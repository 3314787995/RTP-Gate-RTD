#!/usr/bin/env bash
set -euo pipefail

cd /home/cike/hs
source /home/cike/hs/paper2_layer_pruning/env.sh
source "$ROOT/envs/layer-prune-p100/bin/activate"

TOOLS="$ROOT/experiments/day2_gemma_core_eval/tools"
RESULTS="$ROOT/results/day2_gemma_core_eval"
MODELS="$ROOT/models/day2_gemma_core_eval"

mkdir -p "$RESULTS" "$ROOT/logs/day2_gemma_core_eval"

python - <<'PY'
from transformers import AutoConfig
import os
root = os.environ["ROOT"]
paths = [
    ("gemma_base", "google/gemma-2-2b-it"),
    ("gemma_reverse_6", root + "/models/day2_gemma_core_eval/gemma_reverse_6"),
    ("gemma_bi_6", root + "/models/day2_gemma_core_eval/gemma_bi_6"),
]
for label, path in paths:
    config = AutoConfig.from_pretrained(path, trust_remote_code=True)
    print(label, config.model_type, config.num_hidden_layers)
PY

python "$TOOLS/run_gemma_transformers_eval.py" \
  --model-label gemma_base \
  --model-path google/gemma-2-2b-it \
  --task gsm8k \
  --prompt-key GEMMA_INSTRUCT \
  --samples 2 \
  --split test \
  --output-dir "$RESULTS" \
  --prune-strategy baseline \
  --pruned-layers "" \
  --max-new-tokens 256

python "$TOOLS/run_gemma_transformers_eval.py" \
  --model-label gemma_reverse_6 \
  --model-path "$MODELS/gemma_reverse_6" \
  --task gsm8k \
  --prompt-key GEMMA_INSTRUCT \
  --samples 2 \
  --split test \
  --output-dir "$RESULTS" \
  --prune-strategy reverse \
  --pruned-layers 20,21,22,23,24,25 \
  --max-new-tokens 256

python "$TOOLS/run_gemma_transformers_eval.py" \
  --model-label gemma_bi_6 \
  --model-path "$MODELS/gemma_bi_6" \
  --task gsm8k \
  --prompt-key GEMMA_INSTRUCT \
  --samples 2 \
  --split test \
  --output-dir "$RESULTS" \
  --prune-strategy bi \
  --pruned-layers 2,11,20,21,23,24 \
  --max-new-tokens 256
