# Lab-Root Snapshot - 2026-05-13

This snapshot records the RTP-Gate / RTD state exported from lab-root on 2026-05-13.

## Completed

- Environment provisioned on lab-root with CUDA-visible 6 x RTX 2080 Ti.
- Local Gemma2-2B-It Hugging Face snapshot loaded successfully.
- Dense traces rebuilt on lab-root:
  - smoke: 50
  - calibration: 200
  - holdout: 100
- Single-layer RTD sweep completed for layers 0 through 25.
- Known multi-layer candidates completed:
  - `reverse_{2,3,5,6}`
  - `bi_{2,3,5,6}`
  - `iterative_proxy_k{2,3,5,6}`

## In Progress At Export Time

- RTP-Gate greedy candidate scoring was still running.
- Fixed raw-eval candidates had started on otherwise idle GPUs.
- Final raw GSM8K / XSUM / classification tables were not yet complete.

## Current Trend

The current RTD diagnostic trend is reasonable and positive:

- Highest-risk single layers include 0, 20, 18, 16, and 4.
- Lower-risk single layers include 1, 9, 24, 8, and 3.
- Early RTP-Gate choices were tracking low-RTD candidates such as `[1]`, `[1,24]`, and `[1,9,24]`.

Sanity comparison to prior non-formal references:

- Spearman against Day11 saved-model GSM8K damage reference: about `0.805`.
- AUROC against that reference: about `0.942`.
- Spearman against Day8 runtime-skip GSM8K damage reference: about `0.722`.
- AUROC against that reference: about `0.861`.

These comparisons are not formal lab-root statistics. They are included to document that the rerun was trending consistently with earlier exploratory evidence.

