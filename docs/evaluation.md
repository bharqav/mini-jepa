# Evaluation

All metrics are computed from real runs by `jev_qwen/evaluate.py` and
`scripts/evaluate.py` + `scripts/baselines.py`. Nothing here is invented.

## Predictive quality
Accuracy (overall + per type), NLL, Brier score, macro-F1 over rows sharing
the modal candidate count (reported with `macro_f1_n`/`macro_f1_C`, since
F1 over mixed candidate counts is ill-defined), `auc_noul` when both classes
present. Variable candidate counts are handled by per-row accumulation, distributions are never padded.

## Calibration
ECE (10 bins) + max calibration gap. Temperature scaling is fit on a seeded
fit-half of the eval rows and reported on the measure-half only
(`eval_calibration.json`); per-checkpoint `reliability.svg` and
`risk_coverage.svg` are generated dependency-free, with accuracy at
coverages 0.2/0.5/0.8/1.0. The compat head ships a learned per-task
temperature instead.

## Robustness
- Candidate-order sensitivity: max/mean total-variation distance over random
  permutations, mapped to candidate IDs (`order_tv_worst/mean`).
- Input validation: invalid requests (empty, duplicate ids, wrong candidate
  counts) are rejected with 422, never answered with fake predictions
  (tested in `tests/test_api.py`).
- Isolation: sibling-mask tests + packed-vs-independent agreement.

## Efficiency
`scripts/benchmark.py` (CUDA-synced, warmed up, repeated; cold start
reported separately). Measured on the RTX 5060 (WSL2), compat+LoRA 12-layer
model: cold 0.81 s; batch-1 median **21.3 ms** (p95 28 ms); batch-16
**369 q/s** (median 43 ms). Peak VRAM **3.3 GB training / 1.3 GB
inference**, measured with `torch.cuda.max_memory_allocated`, not assumed.
Zero tokens generated per decision. On the CPU-only fallback the same
benchmark gives 73 ms at batch 1.

## Uncertainty
Small-n results (100 test questions, 20-question generation subset) are
reported with exact `n`s and no significance claims. Do not over-read
±0.01 accuracy gaps at this scale; the stable signals are NLL/AUC gaps and
the bit-identical invariance results.
