# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

> **Versioning note.** `0.x` releases communicate that the API may still
> change incompatibly between minor versions. Accuracy figures quoted below are
> research measurements on one dataset and one GPU, not product guarantees.

## [Unreleased]

Nothing yet.

## [0.2.0] — 2026-10-19

### Added

- **JEPA-style latent predictor** (`jev_qwen.jepa`) — context encoder, EMA
  target encoder, latent predictor, block masking, VICReg variance/covariance
  terms, and collapse diagnostics (`collapse_ratio`, per-dimension std,
  covariance norm, predictor error, linear CKA). Verified non-collapsed on the
  smoke experiment; explicitly documented as *not* canonical I-JEPA.
- **Order-invariant decision head** (`CompatHead`) — scores each candidate
  independently against the answer vector using shared projections, making
  candidate-order invariance exact by construction rather than approximate.
- **Calibration suite** (`jev_qwen.calibration`) — percentile bootstrap CIs,
  temperature scaling fit/measure split, risk-coverage curves, and
  dependency-free SVG reliability diagrams.
- **Uncertainty utilities** (`jev_qwen.stats`) — bootstrap CIs over evaluation
  rows, cross-seed aggregation, and a Welch-style seed comparison.
- **Stratified evaluation** — accuracy with 95% CIs per domain and per
  candidate count.
- **Serving hardening** — liveness (`/health`) and readiness (`/ready`) split,
  request-size limits, CUDA OOM chunk-halving with retry, structured JSON
  logging, deterministic inference mode.
- **Correctness tests** for batch-vs-single forward agreement and
  packed-vs-independent question equivalence.
- **Teacher labelling** (`jev_qwen.teacher`) — local-LM and external HTTP
  teachers with retry, cost accounting, fingerprint-keyed caching,
  disagreement flagging, and provenance. Exercised end-to-end against a real
  HTTP server in tests.
- **Modular dataset adapters** (`jev_qwen.adapters`) — JSONL and synthetic
  sources validated into the canonical row schema.

### Changed

- **Best configuration:** compat head + LoRA on the last 4 of 12 kept Qwen3
  layers, 3 epochs → accuracy **0.600** on a 2,280-question held-out split,
  versus 0.581 for the reference head at equal epochs.
- **Seed study:** over 3 seeds the compat head reaches 0.5593 ± 0.0008 versus
  0.5247 ± 0.0184 for the reference head (t = 3.26), with complete separation
  between the two systems.
- Length bucketing for evaluation loaders; gradient checkpointing added as an
  opt-in flag; dataset mapping pinned to `num_proc=1`.

### Fixed

- Stale `datasets` on-disk cache silently dropping newly added columns — cache
  fingerprints are now derived from the mapping function's source.
- Row metadata was aligned positionally, which was invalid once length
  bucketing reordered batches; metadata now travels inside the batch.
- Peak-VRAM reporting divided by the wrong constant.
- Multi-worker `DataLoader` collate was not picklable.
- `datasets` / `DataLoader` worker forkserver crashes under WSL2 and
  locked-down Windows; `num_workers: 0` is now the documented safe default.

### Known limitations

- The 3-epoch headline figures are single-seed; only the 1-epoch comparison
  has a multi-seed variance study.
- Accuracy degrades with candidate count (0.665 at C=2 → 0.392 at C=8).
- No authentication, rate limiting, or TLS: the API is intended for local use.
- `load_checkpoint` uses `torch.load`; checkpoints must be trusted.
- The backend is a truncated prefix of Qwen3-0.6B; full-depth training was not
  evaluated.

## [0.1.0] — 2026-10-12

### Added

- Initial decision-model pipeline on `Qwen/Qwen3-0.6B`: packed
  state/question/candidate tokenisation with block-causal masking and
  shared-start candidate position IDs.
- Reference-faithful decision head ported from `avbiswas/bev-train`.
- Typed request/response schema for choice, score and noul questions.
- Training loop with LoRA, gradient accumulation, cosine schedule, warmup,
  checkpointing and resume.
- Evaluation with accuracy, NLL, Brier and per-type breakdowns.
- Local FastAPI service, CLI, and example client.
- Reference-faithful JEPA port, initial correctness tests.

[Unreleased]: https://github.com/bharqav/jev-qwen/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/bharqav/jev-qwen/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/bharqav/jev-qwen/releases/tag/v0.1.0