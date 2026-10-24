# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

> **Versioning note.** `0.x` releases communicate that the API may still
> change incompatibly between minor versions. Accuracy figures quoted below are
> research measurements on one dataset and one GPU, not product guarantees.

## [Unreleased]

### Added

- **Multi-key, scoped authentication.** `JEV_API_KEYS` or
  `JEV_API_KEYS_FILE` accepts a JSON map of key id to token or
  `{token, scopes, rps, burst}`. Keys can be limited to `decide` and/or
  `answer`, carry an independent per-key quota, and multiple keys can be active
  for rotation. Tokens are stored as PBKDF2-HMAC-SHA-256 digests and compared
  with `hmac.compare_digest`; the legacy `JEV_API_TOKEN` path remains.
- **Trusted-proxy client resolution.** `JEV_TRUSTED_PROXIES` lists CIDRs whose
  `X-Forwarded-For` may be used for client attribution and per-client rate
  limiting. The header is ignored from untrusted peers.
- **Request IDs, audit logs and metrics.** Every response carries
  `X-Request-ID`; access logs include method, path, status, latency, resolved
  client and key id. `/metrics` exposes Prometheus counters/histograms for
  requests, latency, in-flight work, rate limits, auth failures, inference
  errors and readiness. Optional OpenTelemetry tracing is available through the
  `otel` extra and `JEV_OTEL_ENABLED`.
- **Bounded concurrency and body limits.** `JEV_MAX_CONCURRENCY` rejects excess
  work with `503` and `Retry-After: 1` instead of queueing without bound;
  `JEV_MAX_BODY_BYTES` rejects declared oversized bodies with `413` before
  JSON parsing.
- **Deployment artefacts.** CPU/CUDA `deploy/Dockerfile`, compose stack with
  internal-only API and nginx TLS/rate-limit edge, hardened systemd unit, env
  examples, and `scripts/generate_api_keys.py` for key creation/rotation.
- **Load/soak harness.** `scripts/load_test.py` reports p50/p95/p99,
  throughput and status counts against a staging endpoint; pure summary logic
  is unit-tested.
- **Model card and third-party notices.** `docs/model_card.md` records intended
  use, metrics, limitations and the unknown dataset license;
  `THIRD_PARTY_NOTICES.md` records backbone/dataset and runtime licenses.
- **Reproducible installs and wider CI.** `requirements/constraints.txt` pins
  transitive dependencies; `requirements/torch-cpu.txt` and
  `requirements/torch-cuda.txt` pin the two torch builds. CI now runs a
  Python 3.11/3.12/3.13 matrix, `mypy` on the production surface, a coverage
  gate, and a container build.

- **Real Qwen JEPA pipeline.** `scripts/train_jepa.py` pretrains the
  JEPA-style latent predictor on packed decision text with the pinned
  Qwen3-0.6B backbone; `src/jev_qwen/jepa_data.py` provides deterministic
  packing/batching. Checkpoints store the context encoder as a raw
  `Qwen3Model` state dict. `scripts/train.py --init-from-jepa` loads it into a
  frozen decision backbone for an apples-to-apples probe. Added causal/frozen
  target configs, a bidirectional probe config, tests, and
  `results/jepa_qwen_results.json`. Measured single-seed result is negative:
  JEPA init 0.4975 vs pretrained baseline 0.5745 accuracy; reported honestly.

### Security

- **Safetensors-only checkpoint loading.** `save_checkpoint` now writes
  `head.safetensors`; `load_checkpoint` reads only that file and refuses a
  legacy `head.pt` instead of falling back to `torch.load`, which can execute
  arbitrary code from an untrusted pickle. Addresses the README "Security"
  and "Roadmap" claims that checkpoints use `torch.load`. Regression tests
  monkeypatch `torch.load` to fail and assert it is never called.
- **One-time checkpoint migration.** `scripts/convert_checkpoints.py`
  converts legacy `head.pt` files with `torch.load(..., weights_only=True)`.
  It keeps the original pickle by default; `--delete-old` is required to
  remove it after a successful conversion, so a bad conversion cannot destroy
  the only copy. No in-repo checkpoints existed at the time of the change, so
  nothing was silently rewritten.
- **Pinned Hugging Face revisions.** The backbone, tokenizer and decision
  dataset now resolve to explicit commit SHAs in `jev_qwen.hub`
  (`Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca`,
  `avbiswas/bev-decision-150K@11ed9b5b879e7f9524e7c89c1037d498c54206f0`).
  Training, inference, the teacher, baselines and the fastparquet fallback all
  pass the revision; a custom model or dataset name without a revision fails
  loudly. Checkpoint `meta.json` records `model_revision` so reloads stay
  pinned.
- **Bearer authentication.** `/v1/decide` and `/v1/answer` require
  `Authorization: Bearer $JEV_API_TOKEN`. The service fails closed at startup
  when the token is unset. Authentication is disabled only in explicit dev
  mode (`JEV_DEV_MODE=1` or `--dev`), which logs a loud warning.
  **Breaking:** starting the app without a token and without dev mode now
  raises instead of serving unauthenticated requests, and `/demo` returns
  `404` when authentication is enabled (it does not send a token).
- **Per-client rate limiting.** A token bucket per peer address
  (`JEV_RATE_LIMIT_RPS`, default 20; `JEV_RATE_LIMIT_BURST`, default 40;
  bounded to `JEV_RATE_LIMIT_MAX_CLIENTS`, default 4096, LRU eviction) returns
  `429` with `Retry-After`. One noisy client cannot exhaust another client's
  budget on the same instance. Documented caveats: N replicas enforce N
  budgets, and behind a proxy all requests share the proxy address unless the
  edge limits per client (the app does not trust `X-Forwarded-For`).
- **Constant-time token comparison.** Bearer tokens are compared with
  `secrets.compare_digest`, not string equality.
- **TLS guidance.** `SECURITY.md` now states plainly that TLS is not
  terminated in-process and that a reverse proxy is required, with a minimal
  nginx example at `deploy/nginx.conf.example`.
- **Log redaction.** Structured logs record `error_type` rather than raw
  exception messages (which can embed request text), and never log request
  bodies, candidate text or bearer tokens. New canary-string tests cover
  successful, unauthorised and failing requests.
- **Dependency audit.** CI runs `pip-audit --skip-editable` on every push and
  pull request; `make audit` runs it locally. The scan found `certifi
  2026.7.22` (PYSEC-2026-4320), which was upgraded to `2026.10.19`; a re-run
  reports no known vulnerabilities. The first CI run also caught the runner's
  preinstalled `setuptools 78.1.0` (PYSEC-2025-49, PYSEC-2026-3447), so the
  build-system floor is now `setuptools>=83.0.0` and CI upgrades build
  tooling before installing. The certifi fix is in the environment, not a
  source pin, so CI will catch a regression.

### Changed

- **The 3-epoch headline is now a 3-seed mean ± std, not a single run.** The
  compat/reference comparison was rerun with seeds 0/1/2 on the pinned
  revisions (seed 0 re-evaluated with the current test loader). Accuracy:
  compat 0.667 ± 0.053, reference 0.629 ± 0.031; per-seed compat 0.6065 /
  0.6935 / 0.7015, reference 0.5950 / 0.6545 / 0.6365. The +0.0385 gap has
  SE 0.0352 (t ≈ 1.10) and is **not significant at 95%**. The previous
  0.600-vs-0.581 single-seed table was an unlucky seed-0 draw for both heads.
  Compat remains higher on accuracy and NLL; the reference head calibrates
  better at 3 epochs (ECE 0.029 vs 0.041). The 1-epoch 3-seed result
  (+0.035, t = 3.26) still stands. Epoch scaling remains a seed-0 ablation,
  so the 5-epoch single run is not directly comparable to the 3-seed mean.
- **Python floor is now 3.11.** `requires-python`, classifiers and the
  ruff/mypy targets moved from 3.10 to 3.11 (3.10 is end-of-life); CI exercises
  3.11, 3.12 and 3.13.
- **Dependency installs are pinned.** CI and the Docker image install from
  `requirements/constraints.txt` after a pinned torch build, so a fresh runner
  does not resolve arbitrary newer transitive versions.
- **Order-invariance numbers are stated as measured.** Head-level TV is
  ≤9e-8 for `CompatHead` and 2.4e-6 for `ChoiceHead` on the 3-epoch
  checkpoints; "exactly 0.0" was replaced with the measured float32 residual.

### Fixed

- **Qwen JEPA dtype, RNG and target-path bugs.** The CPU masking generator is
  used regardless of device; attention-bias dtype follows the embedding/autocast
  dtype; and context/target embed independently so the target encoder cannot
  share the context's embedding computation.
- **Stale dataset config names in the smoke and ablation configs.** The
  pinned dataset revision names the on-disk `data/` config `default`, and
  `datasets` 5.x no longer accepts `data` as a config name. The smoke and
  ablation configs relied on the old default and failed at load time. They now
  pass `subset: default` explicitly; `docs/dataset.md` records the mapping.

### Documentation

- `README.md` serving section, endpoint/auth table, error contract and
  environment variables now describe the actual auth, rate-limit and TLS
  behaviour, and the "Security" and "Roadmap" sections no longer claim
  `torch.load` or unpinned hub fetches.
- `docs/production.md` records the new serving behaviours, checkpoint/hub
  policy and the remaining honest gaps (per-process limiter, proxy TLS).
- `SECURITY.md` threat model updated for safetensors, pinned revisions,
  authentication, rate limiting and reverse-proxy assumptions.
- `SECURITY.md` now covers scoped keys, key rotation, trusted proxies,
  concurrency/body limits, audit logs and metrics; `docs/production.md`
  distinguishes verified behaviours from remaining gaps.
- `docs/model_card.md` and `THIRD_PARTY_NOTICES.md` add model/legal context,
  including the dataset's `unknown` license.
- `deploy/README.md` documents compose, systemd, key rotation, metrics and the
  load harness.


## [0.2.0], 2026-10-19

### Added

- **JEPA-style latent predictor** (`jev_qwen.jepa`), context encoder, EMA
  target encoder, latent predictor, block masking, VICReg variance/covariance
  terms, and collapse diagnostics (`collapse_ratio`, per-dimension std,
  covariance norm, predictor error, linear CKA). Verified non-collapsed on the
  smoke experiment; explicitly documented as *not* canonical I-JEPA.
- **Order-invariant decision head** (`CompatHead`), scores each candidate
  independently against the answer vector using shared projections, making
  candidate-order invariance exact by construction rather than approximate.
- **Calibration suite** (`jev_qwen.calibration`), percentile bootstrap CIs,
  temperature scaling fit/measure split, risk-coverage curves, and
  dependency-free SVG reliability diagrams.
- **Uncertainty utilities** (`jev_qwen.stats`), bootstrap CIs over evaluation
  rows, cross-seed aggregation, and a Welch-style seed comparison.
- **Stratified evaluation**, accuracy with 95% CIs per domain and per
  candidate count.
- **Serving hardening**, liveness (`/health`) and readiness (`/ready`) split,
  request-size limits, CUDA OOM chunk-halving with retry, structured JSON
  logging, deterministic inference mode.
- **Correctness tests** for batch-vs-single forward agreement and
  packed-vs-independent question equivalence.
- **Teacher labelling** (`jev_qwen.teacher`), local-LM and external HTTP
  teachers with retry, cost accounting, fingerprint-keyed caching,
  disagreement flagging, and provenance. Exercised end-to-end against a real
  HTTP server in tests.
- **Modular dataset adapters** (`jev_qwen.adapters`), JSONL and synthetic
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

- Stale `datasets` on-disk cache silently dropping newly added columns, cache
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

## [0.1.0], 2026-10-12

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

[Unreleased]: https://github.com/bharqav/mini-jepa/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/bharqav/mini-jepa/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/bharqav/mini-jepa/releases/tag/v0.1.0