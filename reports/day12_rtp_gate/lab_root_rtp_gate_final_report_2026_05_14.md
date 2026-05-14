# lab-root RTP-Gate/RTD final snapshot report

Snapshot time: 2026-05-14 08:01 Beijing time  
Remote root: `/root/hs/paper2_layer_pruning`  
Workbook: `reports/day12_rtp_gate/lab_root_rtp_gate_final_results_2026_05_14.xlsx`

## One-line conclusion

This lab-root rerun supports the RTP-Gate/RTD idea as a pruning-risk diagnostic: RTP-Gate k=3 does not recover dense performance, but it avoids the catastrophic GSM8K collapse seen in naive tail pruning and in the deliberately risky k=5 control.

## What has been run

- Dense traces: `smoke=50`, `calibration=200`, `holdout=100`.
- Single-layer RTD: 26/26 layers completed.
- Multi-layer RTD and selection: known baselines plus RTP-Gate pure/structure selections completed.
- Raw evaluation: dense, `reverse_3`, `iterative_proxy_k3`, `rtp_gate_pure_k3`, `rtp_gate_structure_k3`, and `risky_k5`.
- Saved-model consistency: `saved_single_layer_24` and `saved_rtp_gate_structure_k3` completed.
- Classification controls were rerun after fixing the runtime-skip wrapper bug; the fixed outputs are the formal classification results.

## Scope clarification

The current full raw-eval comparison is mainly a **three-layer pruning** comparison. RTP-Gate selected k=2/k=3/k=5 candidates at the RTD stage, but the expensive downstream raw evaluations in this snapshot were run for k=3 RTP-Gate candidates plus the k=5 risky control. Therefore, this is not yet a full k2/k3/k5 raw-eval curve.

## Main raw-eval table

| candidate | layers | GSM8K | GSM8K retention | XSUM | classification avg | calibration RTD |
|---|---:|---:|---:|---:|---:|---:|
| dense_base | - | 0.624 | 1 | 0.202 | 0.624 |  |
| iterative_proxy_k3 | 1,24,25 | 0.396 | 0.635 | 0.165 | 0.534 | 421.005 |
| rtp_gate_pure_k3 | 1,9,24 | 0.332 | 0.532 | 0.189 | 0.56 | 18.787 |
| rtp_gate_structure_k3 | 1,9,24 | 0.332 | 0.532 | 0.189 | 0.56 | 18.787 |
| reverse_3 | 23,24,25 | 0.006 | 0.01 | 0.149 | 0.544 | 2552.351 |
| risky_k5 | 21,22,23,24,25 | 0 | 0 | 0.094 | 0.514 | 12867.493 |

## Interpretation

Dense remains the upper baseline at GSM8K 0.624. The RTP-Gate k=3 candidates score 0.332 on GSM8K, so they lose substantial reasoning ability relative to dense. But the comparison that matters for the gate is the failure mode: `reverse_3` falls to 0.006 and `risky_k5` falls to 0.000. In that sense, RTP-Gate is doing the intended diagnostic job: it avoids layer sets that RTD flags as extremely risky.

`rtp_gate_pure_k3` and `rtp_gate_structure_k3` are identical here because both selected the same layers, `[1,9,24]`. Their raw results therefore match exactly.

The fixed classification controls no longer show the impossible all-candidates-equal pattern. Average classification scores fall from dense 0.624 to about 0.560 for RTP-Gate k=3, 0.544 for reverse_3, 0.534 for iterative_proxy_k3, and 0.514 for risky_k5. These controls suggest broad language capability degrades but does not collapse in the same way as GSM8K reasoning.

## RTD and selection notes

- RTP-Gate pure k2: `[1,24]`
- RTP-Gate pure k3: `[1,9,24]`
- RTP-Gate pure k5: `[1,9,10,19,24]`
- Structure-aware selected the same layer sets for k2/k3/k5 in this run.
- Saved-model consistency produced RTD 4.738 for `saved_single_layer_24` and 19.776 for `saved_rtp_gate_structure_k3`.

## Correct use of the results

RTD should be described as a risk diagnostic or gate, not as a downstream benchmark score. RTP-Gate should be described as a candidate-selection method for reducing expensive raw evaluations, not as a proof of globally optimal pruning.

The old classification directory is retained only for audit purposes. Those outputs are invalid for formal analysis because the wrapper did not actually apply runtime layer skipping before the fix.

## Recommended next experiments

1. Run raw GSM8K/XSUM/classification for RTP-Gate k2 and k5 to produce a real k-curve.
2. Repeat trace construction with another seed or calibration/holdout split to test selection stability.
3. Add an explicit classification evaluator regression test so runtime skip cannot silently fail again.
