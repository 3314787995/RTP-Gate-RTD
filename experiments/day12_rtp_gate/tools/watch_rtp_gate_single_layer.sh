#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/cike/hs/paper2_layer_pruning}"
EXP="${RTP_GATE_ROOT:-$ROOT/RTP-Gate}"
POLL_SECONDS="${POLL_SECONDS:-300}"
GPU_MEM_LIMIT="${GPU_MEM_LIMIT:-1200}"
GPU_UTIL_LIMIT="${GPU_UTIL_LIMIT:-5}"
GPU_INCLUDE="${GPU_INCLUDE:-0,1,2,3,4,5,6,7}"
RUN_ONCE="${RUN_ONCE:-1}"
LAUNCHER="$EXP/tools/run_rtp_gate_single_layer_formal.sh"
WATCHER_LOG="$EXP/logs/rtp_gate_single_layer_watcher.log"
WATCHER_PID="$EXP/logs/rtp_gate_single_layer_watcher.pid"
RUN_PID="$EXP/logs/rtp_gate_single_layer_formal.pid"
RUN_LOG="$EXP/logs/rtp_gate_single_layer_formal.log"
STATUS_JSON="$EXP/logs/rtp_gate_single_layer_watcher_status.json"

mkdir -p "$EXP/logs" "$EXP/traces" "$EXP/rtd_scores" "$EXP/reports"

log() {
  echo "[$(date -Is)] $*" | tee -a "$WATCHER_LOG"
}

write_status() {
  local status="$1"
  local gpu="${2:-}"
  local note="${3:-}"
  python - "$STATUS_JSON" "$status" "$gpu" "$note" <<'PY'
import json, sys
from datetime import datetime, timezone

path, status, gpu, note = sys.argv[1:]
payload = {
    "status": status,
    "gpu": gpu,
    "note": note,
    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
}
with open(path, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2, ensure_ascii=False)
    f.write("\n")
PY
}

is_running_pid() {
  local pidfile="$1"
  if [ ! -s "$pidfile" ]; then
    return 1
  fi
  local pid
  pid="$(cat "$pidfile" 2>/dev/null || true)"
  [ -n "$pid" ] && ps -p "$pid" >/dev/null 2>&1
}

gpu_allowed() {
  local gpu="$1"
  IFS=',' read -ra allowed <<< "$GPU_INCLUDE"
  for item in "${allowed[@]}"; do
    if [ "$item" = "$gpu" ]; then
      return 0
    fi
  done
  return 1
}

find_idle_gpu() {
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits |
    while IFS=',' read -r idx mem util; do
      idx="$(echo "$idx" | xargs)"
      mem="$(echo "$mem" | xargs)"
      util="$(echo "$util" | xargs)"
      if gpu_allowed "$idx" && [ "$mem" -le "$GPU_MEM_LIMIT" ] && [ "$util" -le "$GPU_UTIL_LIMIT" ]; then
        echo "$idx"
        return 0
      fi
    done
}

launch_formal() {
  local gpu="$1"
  log "launching RTP-Gate formal single-layer run on physical GPU $gpu"
  nohup bash -lc "export ROOT='$ROOT'; export RTP_GATE_ROOT='$EXP'; export GPU_ID='$gpu'; bash '$LAUNCHER'" > "$RUN_LOG" 2>&1 &
  echo $! > "$RUN_PID"
  write_status "launched" "$gpu" "formal single-layer process started"
  log "formal PID=$(cat "$RUN_PID"), log=$RUN_LOG"
}

main() {
  echo $$ > "$WATCHER_PID"
  log "watcher started"
  log "ROOT=$ROOT"
  log "EXP=$EXP"
  log "thresholds: memory<=${GPU_MEM_LIMIT}MiB utilization<=${GPU_UTIL_LIMIT}% poll=${POLL_SECONDS}s include=$GPU_INCLUDE"
  if [ ! -x "$LAUNCHER" ]; then
    chmod +x "$LAUNCHER"
  fi

  while true; do
    if is_running_pid "$RUN_PID"; then
      log "formal run already active: PID=$(cat "$RUN_PID")"
      write_status "formal_running" "" "formal run already active"
      if [ "$RUN_ONCE" = "1" ]; then
        log "RUN_ONCE=1, watcher exits after confirming active formal run"
        exit 0
      fi
      sleep "$POLL_SECONDS"
      continue
    fi

    local gpu
    gpu="$(find_idle_gpu || true)"
    if [ -n "$gpu" ]; then
      launch_formal "$gpu"
      if [ "$RUN_ONCE" = "1" ]; then
        log "RUN_ONCE=1, watcher exits after launch"
        exit 0
      fi
    else
      log "no idle GPU available; waiting"
      write_status "waiting_for_idle_gpu" "" "no GPU met memory/utilization thresholds"
    fi
    sleep "$POLL_SECONDS"
  done
}

main "$@"

