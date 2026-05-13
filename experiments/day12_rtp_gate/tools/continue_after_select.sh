#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/hs/paper2_layer_pruning
source "$ROOT/env.sh"
PY="$ROOT/envs/rtp-gate-2080ti/bin/python"
SCRIPT="$ROOT/experiments/day12_rtp_gate/tools/run_lab_root_rtp_gate_full.py"
SELECT_PID_FILE="$ROOT/logs/day12_rtp_gate/current_select_runner.pid"
PREFILL_PID_FILE="$ROOT/logs/day12_rtp_gate/prefill_fixed_raw.pid"
if [[ -f "$SELECT_PID_FILE" ]]; then
  SELECT_PID=$(cat "$SELECT_PID_FILE")
else
  SELECT_PID=$(ps -eo pid,args | awk '/run_lab_root_rtp_gate_full.py/ && /--stage select/ && !/awk/ {print $1; exit}')
fi
echo "[$(date -Is)] continuation watcher started; select_pid=${SELECT_PID:-none}"
if [[ -n "${SELECT_PID:-}" ]]; then
  while kill -0 "$SELECT_PID" 2>/dev/null; do
    echo "[$(date -Is)] waiting for select pid $SELECT_PID"
    sleep 60
  done
fi
echo "[$(date -Is)] select pid finished; checking selection files"
for file in "$ROOT/results/day12_rtp_gate/rtp_gate_selection/rtp_gate_pure_selected.csv" "$ROOT/results/day12_rtp_gate/rtp_gate_selection/rtp_gate_structure_selected.csv"; do
  if [[ ! -s "$file" ]]; then
    echo "[$(date -Is)] missing selection file: $file" >&2
    exit 2
  fi
done
echo "[$(date -Is)] running saved-model consistency"
"$PY" "$SCRIPT" --stage consistency --max-workers 1
if [[ -f "$PREFILL_PID_FILE" ]]; then
  PREFILL_PID=$(cat "$PREFILL_PID_FILE")
  while [[ -n "${PREFILL_PID:-}" ]] && kill -0 "$PREFILL_PID" 2>/dev/null; do
    echo "[$(date -Is)] waiting for prefill fixed raw pid $PREFILL_PID before full raw stage"
    sleep 60
  done
fi
echo "[$(date -Is)] running raw eval"
"$PY" "$SCRIPT" --stage raw --max-workers 6
echo "[$(date -Is)] writing final report"
"$PY" "$SCRIPT" --stage report --max-workers 1
echo "[$(date -Is)] continuation completed"