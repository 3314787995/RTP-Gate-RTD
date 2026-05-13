# Day12 RTP-Gate Status

Generated: 2026-05-13 06:44 UTC

## Key Metrics

- RTD single-layer Spearman abs: 
- RTD risky AUROC: 
- scored candidates: 237
- scored single-layer candidates with retention: 0
- scored multi-layer candidates with retention: 0

## Outputs

- `rtd_scores.csv`
- `single_layer_component_metrics.csv`
- `top_risky_layers_by_rtd.csv`
- `multi_layer_comparison.csv`
- `rtd_calibrated_leave_one_out.csv`
- `day12_rtp_gate_summary.json`

## Caveats

- Fixed-weight RTD is the primary result; calibrated scores are exploratory.
- XSUM/classification are controls, not evidence that reasoning is healthy.
- Runtime-skip vs saved-model consistency should be checked before final claims.
