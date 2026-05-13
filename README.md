# RTP-Gate / RTD

This repository contains the RTP-Gate / RTD experiments developed on top of the Paper2 layer-pruning reproduction work. The focus here is our own diagnostic idea: use dense teacher-forced traces to estimate pruning risk before running expensive downstream evaluations.

It is intentionally not a full copy of the Paper2 reproduction repository. Model weights, Hugging Face caches, dense trace JSONL files, virtual environments, and server logs are excluded.

## What Is Included

- `experiments/day12_rtp_gate/tools/`
  - dense trace collection
  - RTD candidate scoring
  - greedy RTP-Gate layer selection
  - lab-root orchestration and continuation scripts
  - report summarization
- `experiments/day9_small_model_repair_completion/tools/`
  - reusable generation/classification evaluators used by the raw-eval stage
- `experiments/day2_gemma_core_eval/tools/make_gemma_pruned_model.py`
  - saved pruned model construction for consistency checks
- `code/on-the-limits-of-layer-pruning/eval/gen_eval/`
  - minimal generation-eval helper code needed by the wrappers
- `reports/day12_rtp_gate/`
  - current lab-root RTD summary snapshot
- `results/day12_rtp_gate/rtd_scores/`
  - per-candidate RTD JSON summaries and item CSV files
- `tests/`
  - lightweight unit tests for lab-root job definitions

## Current Lab-Root Snapshot

Snapshot time: 2026-05-13 06:44 UTC.

Completed:

- Dense trace manifest: `smoke=50`, `calibration=200`, `holdout=100`
- Single-layer RTD sweep: 26 layers
- Known multi-layer baseline sweep: reverse, BI, and iterative proxy candidates
- RTP-Gate candidate scoring in progress at snapshot time

Not final yet:

- RTP-Gate final selection CSV was still being produced on lab-root.
- Raw GSM8K / XSUM / classification evaluation was still running.
- Final downstream task scores should be added in a later commit after the lab-root run finishes.

The strongest current diagnostic signal is that single-layer RTD agrees well with prior GSM8K damage references:

- Spearman against Day11 saved-model GSM8K damage reference: about `0.805`
- AUROC for risky single-layer detection against that reference: about `0.942`
- Spearman against Day8 runtime-skip GSM8K damage reference: about `0.722`
- AUROC against that reference: about `0.861`

These reference comparisons are sanity checks only. The lab-root run is the formal rerun for RTP-Gate/RTD results.

## Quick Start

Create the environment on a CUDA machine:

```bash
python3 -m venv envs/rtp-gate
source envs/rtp-gate/bin/activate
pip install -r experiments/day12_rtp_gate/requirements-lab-root.txt
```

Prepare environment variables:

```bash
export ROOT=/path/to/RTP-Gate-RTD
export HF_HOME="$ROOT/cache/huggingface"
export HF_DATASETS_CACHE="$ROOT/cache/huggingface/datasets"
export MODEL_PATH=/path/to/google/gemma-2-2b-it/local/snapshot
export TOKENIZERS_PARALLELISM=false
```

Run a smoke trace and score:

```bash
python experiments/day12_rtp_gate/tools/run_lab_root_rtp_gate_full.py \
  --root "$ROOT" \
  --model-path "$MODEL_PATH" \
  --stage smoke \
  --max-workers 1
```

Run the full resumable workflow:

```bash
python experiments/day12_rtp_gate/tools/run_lab_root_rtp_gate_full.py \
  --root "$ROOT" \
  --model-path "$MODEL_PATH" \
  --stage all \
  --max-workers 6
```

For lab-root, the production run used:

- project root: `/root/hs/paper2_layer_pruning`
- model: local Gemma2-2B-It HF snapshot
- trace top-k: `100`
- max sequence tokens: `2048`
- GPUs: 6 x RTX 2080 Ti 11GB

## Result Files

- `reports/day12_rtp_gate/rtd_scores.csv`: flattened candidate RTD metrics
- `reports/day12_rtp_gate/top_risky_layers_by_rtd.csv`: current high-risk layer/candidate view
- `reports/day12_rtp_gate/day12_rtp_gate_summary.json`: report generation metadata
- `results/day12_rtp_gate/rtd_scores/*.json`: per-candidate RTD summaries
- `results/day12_rtp_gate/rtd_scores/*_items.csv`: per-trace item metrics
- `results/day12_rtp_gate/traces/gsm8k_dense_trace_manifest.json`: trace counts and configuration only

The dense trace JSONL files are intentionally excluded because they are intermediate artifacts and substantially larger than the summary outputs.

## Interpretation

RTP-Gate/RTD is a pruning-risk diagnostic and gating method. It should be reported as a way to identify risky layer-removal candidates before expensive downstream evaluation, not as a claim that greedy RTP-Gate is a globally optimal pruning algorithm.

