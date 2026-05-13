# Day12 RTP-Gate

RTP-Gate is an isolated experiment for reasoning-pruning risk diagnosis on
Gemma2-2B-It. It scores candidate layer-removal sets by teacher-forced
reasoning trajectory distortion (RTD), then uses RTD as a gate for multi-layer
pruning candidates.

This directory intentionally does not overwrite earlier reproduction outputs.

## Expected directories

- `results/day12_rtp_gate/traces`
- `results/day12_rtp_gate/rtd_scores`
- `results/day12_rtp_gate/rtp_gate_selection`
- `reports/day12_rtp_gate`

## Minimal flow

```bash
export ROOT=/home/cike/hs/paper2_layer_pruning
source "$ROOT/envs/layer-prune-p100/bin/activate"

python "$ROOT/experiments/day12_rtp_gate/tools/collect_dense_traces.py" \
  --root "$ROOT" \
  --partition-mode smoke

python "$ROOT/experiments/day12_rtp_gate/tools/score_rtd_candidates.py" \
  --root "$ROOT" \
  --candidate-name smoke_drop_layer_24 \
  --runtime-skip-layers 24 \
  --trace-jsonl "$ROOT/results/day12_rtp_gate/traces/gsm8k_dense_traces_smoke.jsonl"

python "$ROOT/experiments/day12_rtp_gate/tools/summarize_rtp_gate.py" \
  --root "$ROOT"
```

Use full partitions only after the smoke trace and one candidate score succeed.

## Full scoring commands

Collect the full trace cache:

```bash
python "$ROOT/experiments/day12_rtp_gate/tools/collect_dense_traces.py" \
  --root "$ROOT" \
  --partition-mode full
```

Score all 26 single-layer candidates:

```bash
for layer in $(seq 0 25); do
  python "$ROOT/experiments/day12_rtp_gate/tools/score_rtd_candidates.py" \
    --root "$ROOT" \
    --candidate-name "single_layer_${layer}" \
    --runtime-skip-layers "$layer" \
    --trace-jsonl "$ROOT/results/day12_rtp_gate/traces/gsm8k_dense_traces_calibration.jsonl" \
    --trace-jsonl "$ROOT/results/day12_rtp_gate/traces/gsm8k_dense_traces_holdout.jsonl"
done
```

Score existing multi-layer baselines:

```bash
declare -A candidates=(
  [reverse_2]="24,25"
  [reverse_3]="23,24,25"
  [reverse_5]="21,22,23,24,25"
  [reverse_6]="20,21,22,23,24,25"
  [bi_2]="23,24"
  [bi_3]="2,23,24"
  [bi_5]="2,20,21,23,24"
  [bi_6]="2,11,20,21,23,24"
  [iterative_proxy_k2]="24,25"
  [iterative_proxy_k3]="1,24,25"
  [iterative_proxy_k5]="1,21,22,24,25"
  [iterative_proxy_k6]="1,21,22,23,24,25"
)

for name in "${!candidates[@]}"; do
  python "$ROOT/experiments/day12_rtp_gate/tools/score_rtd_candidates.py" \
    --root "$ROOT" \
    --candidate-name "$name" \
    --runtime-skip-layers "${candidates[$name]}" \
    --trace-jsonl "$ROOT/results/day12_rtp_gate/traces/gsm8k_dense_traces_calibration.jsonl" \
    --trace-jsonl "$ROOT/results/day12_rtp_gate/traces/gsm8k_dense_traces_holdout.jsonl"
done
```

Run RTP-Gate greedy selection with and without structure-aware penalties:

```bash
python "$ROOT/experiments/day12_rtp_gate/tools/select_rtp_gate_layers.py" \
  --root "$ROOT" \
  --candidate-prefix rtp_gate_pure \
  --trace-jsonl "$ROOT/results/day12_rtp_gate/traces/gsm8k_dense_traces_calibration.jsonl" \
  --trace-jsonl "$ROOT/results/day12_rtp_gate/traces/gsm8k_dense_traces_holdout.jsonl"

python "$ROOT/experiments/day12_rtp_gate/tools/select_rtp_gate_layers.py" \
  --root "$ROOT" \
  --candidate-prefix rtp_gate_structure \
  --structure-aware \
  --trace-jsonl "$ROOT/results/day12_rtp_gate/traces/gsm8k_dense_traces_calibration.jsonl" \
  --trace-jsonl "$ROOT/results/day12_rtp_gate/traces/gsm8k_dense_traces_holdout.jsonl"
```

Summarize after each batch:

```bash
python "$ROOT/experiments/day12_rtp_gate/tools/summarize_rtp_gate.py" --root "$ROOT"
```

## Idle-GPU watcher

On the shared P100 server, prefer launching the formal run through the watcher
instead of starting immediately. It waits until a GPU satisfies both thresholds:

- memory used <= `GPU_MEM_LIMIT` MiB, default `1200`
- utilization <= `GPU_UTIL_LIMIT` percent, default `5`

```bash
export ROOT=/home/cike/hs/paper2_layer_pruning
export RTP_GATE_ROOT="$ROOT/RTP-Gate"
export GPU_MEM_LIMIT=1200
export GPU_UTIL_LIMIT=5
export POLL_SECONDS=300
export RUN_ONCE=1

nohup bash "$RTP_GATE_ROOT/tools/watch_rtp_gate_single_layer.sh" \
  > "$RTP_GATE_ROOT/logs/rtp_gate_single_layer_watcher.nohup.log" 2>&1 &
```

The watcher writes:

- watcher PID: `$RTP_GATE_ROOT/logs/rtp_gate_single_layer_watcher.pid`
- watcher log: `$RTP_GATE_ROOT/logs/rtp_gate_single_layer_watcher.log`
- status JSON: `$RTP_GATE_ROOT/logs/rtp_gate_single_layer_watcher_status.json`
- formal run PID: `$RTP_GATE_ROOT/logs/rtp_gate_single_layer_formal.pid`
- formal run log: `$RTP_GATE_ROOT/logs/rtp_gate_single_layer_formal.log`
