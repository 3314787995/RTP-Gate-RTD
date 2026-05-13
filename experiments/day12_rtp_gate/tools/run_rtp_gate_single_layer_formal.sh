#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/cike/hs/paper2_layer_pruning}"
EXP="${RTP_GATE_ROOT:-$ROOT/RTP-Gate}"
GPU_ID="${GPU_ID:-6}"
TOP_K="${TOP_K:-100}"
MAX_SEQ_TOKENS="${MAX_SEQ_TOKENS:-2048}"

mkdir -p "$EXP/traces" "$EXP/rtd_scores" "$EXP/reports" "$EXP/logs"

cd "$ROOT"
if [ -f "$ROOT/env.sh" ]; then
  # Keep using the established project environment, but tolerate its harmless
  # BOM warning on this server image.
  source "$ROOT/env.sh" || true
fi
source "$ROOT/envs/layer-prune-p100/bin/activate"
export ROOT
export CUDA_VISIBLE_DEVICES="$GPU_ID"

echo "[RTP-Gate] formal single-layer run started at $(date -Is)"
echo "[RTP-Gate] ROOT=$ROOT"
echo "[RTP-Gate] EXP=$EXP"
echo "[RTP-Gate] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

if [ -s "$EXP/traces/gsm8k_dense_traces_calibration.jsonl" ] && [ -s "$EXP/traces/gsm8k_dense_traces_holdout.jsonl" ]; then
  echo "[RTP-Gate] full trace cache already exists; skipping collection"
else
  python "$EXP/tools/collect_dense_traces.py" \
    --root "$ROOT" \
    --baseline-json "$ROOT/results/day5_technical_gap_closure/gemma_base_full_eval_gsm8k_1319.json" \
    --output-dir "$EXP/traces" \
    --partition-mode full \
    --top-k "$TOP_K" \
    --max-seq-tokens "$MAX_SEQ_TOKENS"
fi

for layer in $(seq 0 25); do
  echo "[RTP-Gate] scoring single_layer_${layer} at $(date -Is)"
  python "$EXP/tools/score_rtd_candidates.py" \
    --root "$ROOT" \
    --candidate-name "single_layer_${layer}" \
    --runtime-skip-layers "$layer" \
    --trace-jsonl "$EXP/traces/gsm8k_dense_traces_calibration.jsonl" \
    --trace-jsonl "$EXP/traces/gsm8k_dense_traces_holdout.jsonl" \
    --output-dir "$EXP/rtd_scores" \
    --top-k "$TOP_K" \
    --max-seq-tokens "$MAX_SEQ_TOKENS"
  python "$EXP/tools/summarize_rtp_gate.py" \
    --root "$ROOT" \
    --score-dir "$EXP/rtd_scores" \
    --report-dir "$EXP/reports"
done

echo "[RTP-Gate] formal single-layer run completed at $(date -Is)"
