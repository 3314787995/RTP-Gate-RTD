# GSM8K k-curve snapshot

Generated: 2026-05-14 19:28

This table combines phase-1 k=3 raw-eval results with phase-2 k=2/k=5 GSM8K 500 runs.

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

RTD/RTP-Gate remains a pruning-risk diagnostic. The k-curve is a downstream validation view, not the score used to select layers.
