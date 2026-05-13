#!/usr/bin/env bash
set -u

export ROOT="${ROOT:-/home/cike/hs/paper2_layer_pruning}"
cd /home/cike/hs || exit 1
[ -f "$ROOT/env.sh" ] && source "$ROOT/env.sh"

export HF_HOME="$ROOT/cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export XDG_CACHE_HOME="$ROOT/cache"
export PIP_CACHE_DIR="$ROOT/cache/pip"
export TMPDIR="$ROOT/tmp"
export HF_ALLOW_CODE_EVAL=1
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false

source "$ROOT/envs/gemma-recovery-p100/bin/activate"

DAY9="$ROOT/experiments/day9_small_model_repair_completion"
TOOLS="$DAY9/tools"
RESULT="$ROOT/results/day9_small_model_repair_completion"
LOGDIR="$ROOT/logs/day9_small_model_repair_completion"
REPORT="$ROOT/reports/day9_small_model_repair_completion"
MODEL_DAY2="$ROOT/models/day2_gemma_core_eval"
MODEL_DAY8="$ROOT/models/day8_small_model_gap_completion"
mkdir -p "$RESULT"/classification "$RESULT"/generation "$RESULT"/code "$LOGDIR" "$REPORT"

SLEEP_SECONDS="${SLEEP_SECONDS:-300}"
MAX_DAY9_PARALLEL="${MAX_DAY9_PARALLEL:-8}"
export GPU_MEM_LIMIT="${GPU_MEM_LIMIT:-800}"
export GPU_UTIL_LIMIT="${GPU_UTIL_LIMIT:-20}"
DAY9_ENABLE_FULL_CLASSIFICATION="${DAY9_ENABLE_FULL_CLASSIFICATION:-0}"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "$LOGDIR/day9_gpu_watcher.log"
}

json_status() {
  local path="$1"
  [ -f "$path" ] || { echo "missing"; return 0; }
  python - "$path" <<'PY'
import json, sys
try:
    with open(sys.argv[1], encoding="utf-8") as f:
        print(json.load(f).get("status", "unknown"))
except Exception:
    print("bad_json")
PY
}

is_terminal_json() {
  local path="$1"
  local st
  st="$(json_status "$path")"
  [ "$st" = "done" ] || [ "$st" = "failed" ] || [ "$st" = "skipped" ]
}

running_day9_count() {
  pgrep -af "day9_small_model_repair_completion|run_day9_|launch_day9_gpu_watcher" |
    grep -v pgrep |
    grep -v "launch_day9_gpu_watcher" |
    wc -l
}

is_running_key() {
  local key="$1"
  pgrep -af "$key" >/dev/null 2>&1
}

free_gpus() {
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits |
    python -c 'import os, sys
mem_limit = int(os.environ.get("GPU_MEM_LIMIT", "800"))
util_limit = int(os.environ.get("GPU_UTIL_LIMIT", "20"))
for line in sys.stdin:
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 3:
        continue
    try:
        index = int(parts[0]); mem = int(float(parts[1])); util = int(float(parts[2]))
    except ValueError:
        continue
    if mem < mem_limit and util < util_limit:
        print(index)'
}

record_gpu_state() {
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader |
    sed "s/^/[$(date '+%F %T')] gpu /" >> "$LOGDIR/gpu_state.log"
}

classification_routes=(
  "gemma_base|google/gemma-2-2b-it|"
  "gemma_reverse_6|$MODEL_DAY2/gemma_reverse_6|"
  "gemma_bi_6|$MODEL_DAY2/gemma_bi_6|"
  "reverse_alpaca|$MODEL_DAY2/gemma_reverse_6|$ROOT/outputs/day2_7_paper_param_retrain/reverse_alpaca_full_paperlr_seq2048/final_adapter"
  "bi_alpaca|$MODEL_DAY2/gemma_bi_6|$ROOT/outputs/day2_7_paper_param_retrain/bi_alpaca_full_paperlr_seq2048/final_adapter"
  "reverse_sgr_alpaca|$MODEL_DAY2/gemma_reverse_6|$ROOT/outputs/day2_7_paper_param_retrain/reverse_sgr_alpaca_full_paperlr_seq2048/final_adapter"
  "bi_sgr_alpaca|$MODEL_DAY2/gemma_bi_6|$ROOT/outputs/day2_7_paper_param_retrain/bi_sgr_alpaca_full_paperlr_seq2048/final_adapter"
  "reverse_sgr_dolci|$MODEL_DAY2/gemma_reverse_6|$ROOT/outputs/day2_7_paper_param_retrain/reverse_sgr_dolci_full_paperlr_seq2048/final_adapter"
  "bi_sgr_dolci|$MODEL_DAY2/gemma_bi_6|$ROOT/outputs/day2_7_paper_param_retrain/bi_sgr_dolci_full_paperlr_seq2048/final_adapter"
  "reverse_2_sgr_dolci20k|$MODEL_DAY8/gemma_reverse_2|$ROOT/outputs/day8_small_model_gap_completion/reverse_2_sgr_dolci20k/final_adapter"
  "bi_2_sgr_dolci20k|$MODEL_DAY8/gemma_bi_2|$ROOT/outputs/day8_small_model_gap_completion/bi_2_sgr_dolci20k/final_adapter"
)
classification_tasks=(hellaswag piqa mmlu winogrande openbookqa arc_easy arc_challenge)

classification_jobs=()
for samples in 200; do
  for route in "${classification_routes[@]}"; do
    IFS='|' read -r run_id model_path adapter_path <<< "$route"
    for task in "${classification_tasks[@]}"; do
      classification_jobs+=("classification|$run_id|$model_path|$adapter_path|$task|$samples")
    done
  done
done
if [ "$DAY9_ENABLE_FULL_CLASSIFICATION" = "1" ]; then
  for route in "${classification_routes[@]}"; do
    IFS='|' read -r run_id model_path adapter_path <<< "$route"
    for task in "${classification_tasks[@]}"; do
      classification_jobs+=("classification|$run_id|$model_path|$adapter_path|$task|0")
    done
  done
fi

bi_validation_jobs=()
for label in checkpoint_12000 checkpoint_12500 final_adapter; do
  case "$label" in
    checkpoint_12000) adapter="$ROOT/outputs/day2_7_paper_param_retrain/bi_alpaca_full_paperlr_seq2048/checkpoint-12000" ;;
    checkpoint_12500) adapter="$ROOT/outputs/day2_7_paper_param_retrain/bi_alpaca_full_paperlr_seq2048/checkpoint-12500" ;;
    final_adapter) adapter="$ROOT/outputs/day2_7_paper_param_retrain/bi_alpaca_full_paperlr_seq2048/final_adapter" ;;
  esac
  bi_validation_jobs+=("generation|bi_alpaca_${label}_valid|$MODEL_DAY2/gemma_bi_6|$adapter|xsum|100|96|bi_alpaca_early_stop|")
  bi_validation_jobs+=("generation|bi_alpaca_${label}_valid|$MODEL_DAY2/gemma_bi_6|$adapter|gsm8k|100|512|bi_alpaca_early_stop|")
done

generation_jobs=(
  "generation|early_stop_base|google/gemma-2-2b-it||xsum|100|96|bi_alpaca_early_stop|"
  "generation|early_stop_base|google/gemma-2-2b-it||gsm8k|100|512|bi_alpaca_early_stop|"
  "${bi_validation_jobs[@]}"
  "select_bi_xsum|select_bi_xsum|100"
  "generation|selected_bi_alpaca_base|google/gemma-2-2b-it||xsum|500|96|bi_alpaca_selected_test|"
  "generation|selected_bi_alpaca_base|google/gemma-2-2b-it||gsm8k|500|512|bi_alpaca_selected_test|"
  "generation_selected|bi_alpaca_selected_test|$MODEL_DAY2/gemma_bi_6|xsum|500|96|bi_alpaca_selected_test"
  "generation_selected|bi_alpaca_selected_test|$MODEL_DAY2/gemma_bi_6|gsm8k|500|512|bi_alpaca_selected_test"
)

code_jobs=(
  "code|gemma_base|google/gemma-2-2b-it||humeval|prefix_continuation|20|384"
  "code|gemma_base|google/gemma-2-2b-it||humeval|fenced_prefix|20|384"
  "code|gemma_base|google/gemma-2-2b-it||humeval|plain_request|20|384"
  "code|gemma_base|google/gemma-2-2b-it||mbpp|assertion_guided|40|384"
  "code|reverse_alpaca|$MODEL_DAY2/gemma_reverse_6|$ROOT/outputs/day2_7_paper_param_retrain/reverse_alpaca_full_paperlr_seq2048/final_adapter|mbpp|assertion_guided|40|384"
  "code|bi_alpaca|$MODEL_DAY2/gemma_bi_6|$ROOT/outputs/day2_7_paper_param_retrain/bi_alpaca_full_paperlr_seq2048/final_adapter|mbpp|assertion_guided|40|384"
  "code|reverse_sgr_dolci|$MODEL_DAY2/gemma_reverse_6|$ROOT/outputs/day2_7_paper_param_retrain/reverse_sgr_dolci_full_paperlr_seq2048/final_adapter|mbpp|assertion_guided|40|384"
  "code|bi_sgr_dolci|$MODEL_DAY2/gemma_bi_6|$ROOT/outputs/day2_7_paper_param_retrain/bi_sgr_dolci_full_paperlr_seq2048/final_adapter|mbpp|assertion_guided|40|384"
  "code|reverse_2_sgr_dolci20k|$MODEL_DAY8/gemma_reverse_2|$ROOT/outputs/day8_small_model_gap_completion/reverse_2_sgr_dolci20k/final_adapter|mbpp|assertion_guided|40|384"
  "code|bi_2_sgr_dolci20k|$MODEL_DAY8/gemma_bi_2|$ROOT/outputs/day8_small_model_gap_completion/bi_2_sgr_dolci20k/final_adapter|mbpp|assertion_guided|40|384"
)

jobs=("${code_jobs[@]}" "${generation_jobs[@]}" "${classification_jobs[@]}")

selected_adapter_path() {
  python - "$REPORT/bi_alpaca_xsum_selected.json" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as f:
    print(json.load(f)["selected"]["adapter_path"])
PY
}

job_output_json() {
  local spec="$1"
  IFS='|' read -r kind a b c d e f g h <<< "$spec"
  if [ "$kind" = "classification" ]; then
    echo "$RESULT/classification/${a}_${d}_${e}.json"
  elif [ "$kind" = "generation" ]; then
    echo "$RESULT/generation/${a}_${d}_${e}.json"
  elif [ "$kind" = "generation_selected" ]; then
    echo "$RESULT/generation/${a}_${c}_${d}.json"
  elif [ "$kind" = "select_bi_xsum" ]; then
    echo "$REPORT/bi_alpaca_xsum_selected.json"
  elif [ "$kind" = "code" ]; then
    echo "$RESULT/code/${a}_${d}_${e}_${f}.json"
  else
    echo "$RESULT/unknown.json"
  fi
}

job_is_done() {
  is_terminal_json "$(job_output_json "$1")"
}

job_key() {
  local spec="$1"
  IFS='|' read -r kind a b c d e f g h <<< "$spec"
  if [ "$kind" = "classification" ]; then
    echo "run_day9_classification_eval.py.*--run-id ${a}.*--task ${d}.*--samples ${e}"
  elif [ "$kind" = "generation" ]; then
    echo "run_day9_generation_eval.py.*--run-id ${a}.*--task ${d}.*--samples ${e}"
  elif [ "$kind" = "generation_selected" ]; then
    echo "run_day9_generation_eval.py.*--run-id ${a}.*--task ${c}.*--samples ${d}"
  elif [ "$kind" = "select_bi_xsum" ]; then
    echo "select_day9_bi_xsum_checkpoint.py"
  elif [ "$kind" = "code" ]; then
    echo "run_day9_code_eval.py.*--run-id ${a}.*--task ${d}.*--variant ${e}.*--samples ${f}"
  else
    echo "NO_MATCH_FOR_UNKNOWN_DAY9_JOB"
  fi
}

deps_ready() {
  local spec="$1"
  IFS='|' read -r kind a b c d e f g h <<< "$spec"
  if [ "$kind" = "classification" ] && [ -n "$c" ]; then
    [ -f "$c/adapter_model.safetensors" ] || return 1
  fi
  if [ "$kind" = "generation" ] && [ -n "$c" ]; then
    [ -f "$c/adapter_model.safetensors" ] || return 1
  fi
  if [ "$kind" = "code" ] && [ -n "$c" ]; then
    [ -f "$c/adapter_model.safetensors" ] || return 1
  fi
  if [ "$kind" = "select_bi_xsum" ]; then
    for label in checkpoint_12000 checkpoint_12500 final_adapter; do
      is_terminal_json "$RESULT/generation/bi_alpaca_${label}_valid_xsum_100.json" || return 1
      is_terminal_json "$RESULT/generation/bi_alpaca_${label}_valid_gsm8k_100.json" || return 1
    done
  fi
  if [ "$kind" = "generation_selected" ]; then
    is_terminal_json "$REPORT/bi_alpaca_xsum_selected.json" || return 1
  fi
  return 0
}

launch_job() {
  local gpu="$1"
  local spec="$2"
  IFS='|' read -r kind a b c d e f g h <<< "$spec"
  local log_file="$LOGDIR/${kind}_${a}_${d}_${e}_${f}.log"
  if [ "$kind" = "classification" ]; then
    log "launch gpu=$gpu classification run=$a task=$d samples=$e"
    CUDA_VISIBLE_DEVICES="$gpu" nohup python "$TOOLS/run_day9_classification_eval.py" \
      --run-id "$a" --model-path "$b" --adapter-path "${c:-none}" --task "$d" --samples "$e" \
      --output-dir "$RESULT/classification" > "$log_file" 2>&1 &
  elif [ "$kind" = "generation" ]; then
    log "launch gpu=$gpu generation run=$a task=$d samples=$e group=$g"
    CUDA_VISIBLE_DEVICES="$gpu" nohup python "$TOOLS/run_day9_generation_eval.py" \
      --run-id "$a" --model-path "$b" --adapter-path "${c:-none}" --task "$d" --samples "$e" \
      --max-new-tokens "$f" --job-group "$g" --runtime-skip-layers "${h:-}" \
      --output-dir "$RESULT/generation" --repo-root "$ROOT/code/on-the-limits-of-layer-pruning" > "$log_file" 2>&1 &
  elif [ "$kind" = "generation_selected" ]; then
    local selected_adapter
    selected_adapter="$(selected_adapter_path)"
    log "launch gpu=$gpu selected_generation run=$a task=$c samples=$d adapter=$selected_adapter"
    CUDA_VISIBLE_DEVICES="$gpu" nohup python "$TOOLS/run_day9_generation_eval.py" \
      --run-id "$a" --model-path "$b" --adapter-path "$selected_adapter" --task "$c" --samples "$d" \
      --max-new-tokens "$e" --job-group "$f" --output-dir "$RESULT/generation" \
      --repo-root "$ROOT/code/on-the-limits-of-layer-pruning" > "$log_file" 2>&1 &
  elif [ "$kind" = "select_bi_xsum" ]; then
    log "launch gpu=$gpu select_bi_xsum"
    nohup python "$TOOLS/select_day9_bi_xsum_checkpoint.py" \
      --root "$ROOT" --samples "$b" --output-json "$REPORT/bi_alpaca_xsum_selected.json" > "$log_file" 2>&1 &
  elif [ "$kind" = "code" ]; then
    log "launch gpu=$gpu code run=$a task=$d variant=$e samples=$f"
    CUDA_VISIBLE_DEVICES="$gpu" nohup python "$TOOLS/run_day9_code_eval.py" \
      --run-id "$a" --model-path "$b" --adapter-path "${c:-none}" --task "$d" --variant "$e" --samples "$f" \
      --max-new-tokens "$g" --output-dir "$RESULT/code" --repo-root "$ROOT/code/on-the-limits-of-layer-pruning" > "$log_file" 2>&1 &
  fi
}

summarize_day9() {
  python "$TOOLS/summarize_day9_repair_completion.py" --root "$ROOT" >> "$LOGDIR/day9_summarizer.log" 2>&1 || true
}

log "Day9 watcher starting: jobs=${#jobs[@]} max_parallel=$MAX_DAY9_PARALLEL full_classification=$DAY9_ENABLE_FULL_CLASSIFICATION"

while true; do
  record_gpu_state
  summarize_day9
  done_count=0
  running_count=0
  for spec in "${jobs[@]}"; do
    if job_is_done "$spec"; then
      done_count=$((done_count + 1))
    elif is_running_key "$(job_key "$spec")"; then
      running_count=$((running_count + 1))
    fi
  done
  log "status done=$done_count/${#jobs[@]} running=$running_count"
  if [ "$done_count" -eq "${#jobs[@]}" ]; then
    log "all day9 jobs reached terminal status"
    summarize_day9
    exit 0
  fi

  for gpu in $(free_gpus); do
    current_running="$(running_day9_count | tr -d ' ')"
    if [ "$current_running" -ge "$MAX_DAY9_PARALLEL" ]; then
      log "parallel limit reached running=$current_running max=$MAX_DAY9_PARALLEL"
      break
    fi
    launched=0
    for spec in "${jobs[@]}"; do
      job_is_done "$spec" && continue
      is_running_key "$(job_key "$spec")" && continue
      deps_ready "$spec" || continue
      launch_job "$gpu" "$spec"
      launched=1
      sleep 8
      break
    done
    if [ "$launched" -eq 0 ]; then
      log "gpu=$gpu free but no launchable job"
    fi
  done

  sleep "$SLEEP_SECONDS"
done
