# Project Memory: RTP-Gate/RTD Phase Boundaries

Last updated: 2026-05-14

This note is the canonical project memory for phase naming and output routing.

## Phase Boundary Rule

Only experiments under the phase-2 namespace below are **Phase 2**:

```text
PHASE2_TAG=day13_rtp_phase2_20260514
```

All other RTP-Gate/RTD experiments, including Day12 reruns, Day12 extended k-curves, Day12 stability checks, Day12 k4 addenda, and any outputs already written under `day12_rtp_gate`, are **Phase 1 / exploratory Day12 outputs**, not Phase 2.

## Phase 2 Write Locations

Phase 2 may write only to:

```text
/root/hs/paper2_layer_pruning/results/day13_rtp_phase2_20260514
/root/hs/paper2_layer_pruning/reports/day13_rtp_phase2_20260514
/root/hs/paper2_layer_pruning/logs/day13_rtp_phase2_20260514
```

Phase 2 scripts must not write to:

```text
/root/hs/paper2_layer_pruning/results/day12_rtp_gate
/root/hs/paper2_layer_pruning/reports/day12_rtp_gate
/root/hs/paper2_layer_pruning/logs/day12_rtp_gate
```

Old Day12 files are read-only references. If a Day12 summary is needed for sanity checking, copy only an explicit summary into:

```text
results/day13_rtp_phase2_20260514/sanity/copied_day12_reference_manifest.json
```

Do not mix Day12 raw outputs into Phase 2 main results.

## Current Running Day12/k4 Jobs

Any jobs currently running under paths such as:

```text
results/day12_rtp_gate/raw_eval/generation_k4_20260514
results/day12_rtp_gate/raw_eval/generation
```

are still Phase 1 / exploratory Day12 outputs. When they finish, summarize them only as Day12/Phase-1 addenda unless they are rerun or re-registered under the Phase 2 protocol.

For example, current `rtp_gate_k4`, `reverse_4`, `iterative_proxy_k4`, `bi_4`, and `bi_3` outputs are not Phase 2 merely because they were run later.

## Phase 2 Output Layout

```text
results/day13_rtp_phase2_20260514/
  manifest.json
  traces/
    gsm8k_train_dense_trace_manifest.json
    gsm8k_train_dense_traces_calibration.jsonl
    gsm8k_train_dense_traces_holdout.jsonl
  selection/
    train_proxy_single_layer/
    train_rtd_old/
    train_rtd_r/
    frozen_candidates.json
  rtd_scores/
    old_rtd/
    rtd_r/
  raw_eval/
    gsm8k_test_full/
    svamp_full/
    xsum_500/
    classification/
  sanity/
    copied_day12_reference_manifest.json
```

```text
reports/day13_rtp_phase2_20260514/
  phase2_candidate_registry.csv
  phase2_main_results.csv
  phase2_selection_audit.csv
  phase2_rtd_component_contributions.csv
  phase2_transfer_summary.md
```

```text
logs/day13_rtp_phase2_20260514/
  traces/
  selection/
  rtd_scores/
  raw_eval/
```

## Phase 2 Required Metadata

Every Phase 2 JSON result must include:

```json
{
  "phase2_tag": "day13_rtp_phase2_20260514",
  "selection_split": "...",
  "eval_split": "...",
  "uses_task_labels_for_selection": true,
  "candidate_family": "...",
  "runtime_skip": {}
}
```

`runtime_skip.skip_layers` must match the candidate registry.

## Phase 2 Candidate Naming

All Phase 2 `run_id` values must use the `p2_` prefix.

| run_id | layers | role |
|---|---:|---|
| `p2_dense_base` | none | dense baseline |
| `p2_reverse_3` | `23,24,25` | fixed heuristic |
| `p2_bi_3` | `2,23,24` | BI baseline |
| `p2_test_selected_proxy_k3` | `1,24,25` | old iterative proxy, oracle-ish reference |
| `p2_train_selected_proxy_k3` | newly selected | fair label-aware proxy |
| `p2_train_rtp_gate_old_k3` | newly selected | train-trace old RTD |
| `p2_train_rtp_gate_rtdr_k3` | newly selected | train-trace RTD-R |
| `p2_risky_k5` | existing risky set | collapse/risk reference |

The old Day12 `iterative_proxy_k3` must be renamed to `p2_test_selected_proxy_k3` in Phase 2 references so it is not confused with the fair baseline.

`selection/frozen_candidates.json` is the only Phase 2 candidate registry. Final raw eval must read candidates from this file, not from hand-entered layer lists.

## Phase 2 Flow

1. Generate GSM8K train dense-correct traces into `traces/`.
2. Run train-split selection:
   - `train_proxy_single_layer`: GSM8K train + XSUM train raw retention.
   - `train_rtd_old`: old RTD on GSM8K train dense traces.
   - `train_rtd_r`: RTD-v2 / RTD-R on GSM8K train dense traces.
3. Write `selection/frozen_candidates.json` with candidate layers, source, and label-awareness metadata.
4. Evaluate only frozen candidates:
   - `raw_eval/gsm8k_test_full`
   - `raw_eval/svamp_full`
   - `raw_eval/xsum_500`
   - `raw_eval/classification`

## Phase 2 Checks

- Path check: no writes to Day12 result/report/log directories.
- Registry check: every raw-eval `run_id` exists in `frozen_candidates.json`.
- Metadata check: every result has `phase2_tag`, and `runtime_skip.skip_layers` matches the registry.
- Protocol check: `p2_test_selected_proxy_k3` must have `uses_task_labels_for_selection=true` and `selection_split=gsm8k_test_subset+xsum_test_subset`; it must not enter fair ranking.
- Summary check: `phase2_main_results.csv` includes GSM8K test, SVAMP, XSUM, and classification mean retention.

## k4 Rule

The current Phase 2 main experiment is k=3. Future k4 Phase 2 work must follow the same namespace and naming rules, using `p2_*_k4` names. Existing Day12 k4 jobs are not retroactively Phase 2.
