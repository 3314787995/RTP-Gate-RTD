# lab-root RTP-Gate/RTD extended report

Snapshot time: 2026-05-14 19:28 Beijing time  
Remote root: `/root/hs/paper2_layer_pruning`  
Workbook: `reports/day12_rtp_gate/lab_root_rtp_gate_extended_results_2026_05_14.xlsx`

## What this second phase adds

- GSM8K 500 k-curve points for RTP-Gate, reverse-tail pruning, and iterative-proxy baselines at k=2 and k=5.
- The existing phase-1 k=3 GSM8K results are reused as the middle point of the curve.
- A seed_5678 trace stability run with true shuffled dense-correct source items.

## GSM8K k-curve

| family | candidate | k | layers | GSM8K | retention vs dense | source |
|---|---|---:|---|---:|---:|---|
| dense | dense_base | 0 | - | 0.624 | 1 | phase1_raw_eval |
| RTP-Gate | rtp_gate_k2 | 2 | 1,24 | 0.54 | 0.865 | phase2_k_curve |
| RTP-Gate | rtp_gate_k3 | 3 | 1,9,24 | 0.332 | 0.532 | phase1_raw_eval |
| RTP-Gate | rtp_gate_k5 | 5 | 1,9,10,19,24 | 0.066 | 0.106 | phase2_k_curve |
| Iterative proxy | iterative_proxy_k2 | 2 | 24,25 | 0.41 | 0.657 | phase2_k_curve |
| Iterative proxy | iterative_proxy_k3 | 3 | 1,24,25 | 0.396 | 0.635 | phase1_raw_eval |
| Iterative proxy | iterative_proxy_k5 | 5 | 1,21,22,24,25 | 0.092 | 0.147 | phase2_k_curve |
| Reverse tail | reverse_2 | 2 | 24,25 | 0.41 | 0.657 | phase2_k_curve |
| Reverse tail | reverse_3 | 3 | 23,24,25 | 0.006 | 0.01 | phase1_raw_eval |
| Reverse tail | reverse_5 | 5 | 21,22,23,24,25 | 0 | 0 | phase2_k_curve |

## k-curve readout

- RTP-Gate is monotonic across the measured strengths: 0.54 at k=2, 0.332 at k=3, and 0.066 at k=5.
- At k=2, RTP-Gate beats the tail/prior proxy layer set by 0.13 GSM8K absolute points.
- At k=5, all methods are weak; RTP-Gate avoids the complete reverse-tail collapse (0) but is slightly below iterative_proxy_k5 (0.092).
- This supports the intended use of RTD as a pruning-risk gate, not a claim that RTP-Gate is globally optimal for every k.

## Stability summary

- Trace seed: 5678
- Trace counts: `{"smoke":50,"calibration":200,"holdout":100}`
- Shuffle enabled: `true`
- Single-layer calibration RTD Spearman: 0.9323
- Base top-5 risky layers: `0,4,16,18,20`
- Seed top-5 risky layers: `0,4,16,18,20`
- Top-5 risky overlap: 5/5

| family | k | base layers | seed layers | Jaccard |
|---|---:|---|---|---:|
| rtp_gate_pure | 2 | 1,24 | 1,24 | 1 |
| rtp_gate_pure | 3 | 1,9,24 | 1,9,24 | 1 |
| rtp_gate_pure | 5 | 1,9,10,19,24 | 1,9,10,19,24 | 1 |
| rtp_gate_structure | 2 | 1,24 | 1,24 | 1 |
| rtp_gate_structure | 3 | 1,9,24 | 1,9,24 | 1 |
| rtp_gate_structure | 5 | 1,9,10,19,24 | 1,9,10,19,24 | 1 |

## Interpretation

This extended phase is meant to strengthen the main claim, not replace the first final report. RTP-Gate/RTD is still framed as a pruning-risk gate: it estimates which layer-removal candidates deserve expensive downstream evaluation. The seed stability check tests whether the RTD ranking and selected layer sets are robust to changing the dense-correct trace sample order.

Here, the RTP-Gate k-curve degrades smoothly as k increases and the seed stability overlap is high. That strengthens the diagnostic story, while still leaving the honest limitation that RTP-Gate is a gate for candidate choice rather than a proof of globally optimal pruning.
