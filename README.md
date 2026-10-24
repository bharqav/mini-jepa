<div align="center">

# mini-jepa

### Order-invariant decision models and a JEPA-style latent predictor on a truncated Qwen3 backbone

**Zero-token inference · Near-exact candidate-order invariance · Calibrated probabilities · EMA-target self-supervision**

[![CI](https://github.com/bharqav/mini-jepa/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/bharqav/mini-jepa/actions/workflows/ci.yml)
[![CodeQL](https://github.com/bharqav/mini-jepa/actions/workflows/codeql.yml/badge.svg?branch=main)](https://github.com/bharqav/mini-jepa/actions/workflows/codeql.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

</div>

---

`mini-jepa` turns a pretrained causal language model into something that
**answers questions instead of completing text**. Give it a state, a question,
and a set of candidate answers; it returns a calibrated probability
distribution, in a single forward pass, generating **zero tokens**.

It also contains a JEPA-style latent predictor that learns representations by
predicting masked spans in a target encoder's embedding space, rather than by
predicting tokens.

> **Naming:** the distribution is `mini-jepa`; the importable package is
> `jev_qwen` (`from jev_qwen.infer import DecisionEngine`).

> **Why build this?** Autoregressive generation is a poor fit for structured
> decision problems: you pay for tokens you did not ask for, you get invalid
> output that must be parsed and validated, and you cannot permute the options
> without the answer changing. This project removes all three problems while
> keeping the pretrained knowledge.

---

## Architecture

Two independent tracks share one repository.

```mermaid
flowchart TB
    subgraph track1 [Track 1, Decision model]
        direction TB
        REQ[Typed request\nstate + questions + candidates]
        TOK[tokenize.py\npack + block-causal mask\n+ shared-start positions]
        BB[Qwen3Model\nfirst N of 28 layers\nLoRA on top-k]
        HD{head type}
        CMP[CompatHead\nindependent scoring\norder-invariant by construction]
        REF[ChoiceHead\nreference-faithful\nset attention]
        OUT[Typed response\nprobabilities + confidence]
        REQ --> TOK --> BB --> HD
        HD --> CMP --> OUT
        HD --> REF --> OUT
    end

    subgraph track2 [Track 2, JEPA latent predictor]
        direction TB
        IN[Token sequence]
        MSK[Block masking\ncontiguous spans]
        CTX[Context encoder\ntrainable]
        TGT[Target encoder\nindependent params\nEMA-updated]
        PRED[Latent predictor\nmask tokens + context reps]
        LOSS[Representation-space loss\nMSE + variance + covariance]
        IN --> MSK
        MSK --> CTX --> PRED
        MSK --> TGT
        TGT -. stop-grad .-> LOSS
        PRED --> LOSS
        TGT -. EMA .-> CTX
    end
```

### Decision Path, one forward pass, zero tokens

```mermaid
sequenceDiagram
    participant Client
    participant Engine as DecisionEngine
    participant Tok as tokenize
    participant Backbone as Qwen3 prefix
    participant Head as CompatHead

    Client->>Engine: DecisionRequest with state and questions
    Engine->>Engine: validate schema, 422 on invalid
    Engine->>Tok: encode_example
    Tok-->>Engine: ids, position_ids, 4D mask, read indices
    Note over Tok: every candidate restarts at the same position id
    Note over Tok: candidates cannot attend to siblings
    Engine->>Backbone: forward with use_cache=False
    Backbone-->>Engine: last_hidden_state
    Engine->>Engine: gather choice and answer embeddings
    Engine->>Head: score answer against candidates
    Head-->>Engine: logits in fp32
    Engine->>Engine: softmax, renormalise, assert sum is 1
    Engine-->>Client: DecisionResponse
```

---

## Features

| Capability | Implementation | Status |
|-----------|----------------|--------|
| **Zero-token inference** | Single forward pass, `use_cache=False`, no `generate()`; asserted absent | ✅ |
| **Order invariance** | Shared scoring projections + shared-start position IDs | ✅ exact (head), ~3e-3 end-to-end |
| **Question isolation** | Independent row per question; packed-vs-independent equivalence tested | ✅ |
| **Two head variants** | `CompatHead` (independent) and `ChoiceHead` (reference-faithful) | ✅ ablated |
| **Three question types** | `choice`, `score` (expected level + normalised), `noul` (P(Yes)) | ✅ |
| **JEPA latent prediction** | EMA target encoder, block masking, predictor, VICReg terms | measured negative downstream |
| **Collapse diagnostics** | Per-dim std, collapse ratio, covariance norm, linear CKA, effective rank | ✅ |
| **Calibration** | Temperature scaling with fit/measure split, ECE, risk-coverage, SVG diagrams | ✅ |
| **Uncertainty** | Bootstrap CIs over rows; cross-seed aggregation and comparison | ✅ |
| **Stratified evaluation** | Accuracy + 95% CI per domain and per candidate count | ✅ |
| **Parameter-efficient training** | LoRA on attention projections + trainable head | ✅ |
| **Serving** | FastAPI with liveness/readiness split, size limits, OOM backoff | ✅ |
| **Local teacher labelling** | Local LM or external HTTP teacher, with retry and caching | ✅ |

---

## Measured Results

All numbers produced by executable scripts on a held-out split of
`avbiswas/bev-decision-150K` that was never used for training or model
selection. See [`results/`](results/) and [`docs/experiments.md`](docs/experiments.md).

**Backbone:** `Qwen/Qwen3-0.6B`, first 12 of 28 layers, LoRA on the last 4.

### Head comparison, 3 epochs, 3 seeds, identical data and budget

Mean ± sample std across seeds 0, 1 and 2. Seed 0 is the original run
re-evaluated with the pinned dataset/model revisions and the current test
loader; seeds 1 and 2 were trained with the pinned revisions.

| System | Accuracy | NLL | Brier | ECE | choice | noul | score | macro-F1 | AUC | head-order TV |
|--------|----------|-----|-------|-----|--------|------|-------|----------|-----|----------------|
| reference `ChoiceHead` | 0.629 ± 0.031 | 0.799 ± 0.052 | 0.456 ± 0.030 | **0.029 ± 0.010** | 0.595 ± 0.043 | 0.709 ± 0.025 | 0.477 ± 0.010 | 0.640 ± 0.034 | 0.779 ± 0.023 | 2.4e-6 |
| **`CompatHead` (default)** | **0.667 ± 0.053** | **0.734 ± 0.084** | **0.422 ± 0.049** | 0.041 ± 0.003 | **0.662 ± 0.089** | **0.718 ± 0.019** | **0.510 ± 0.046** | **0.730 ± 0.050** | **0.808 ± 0.021** | **9.0e-8** |

Per-seed accuracy:

| System | seed 0 | seed 1 | seed 2 |
|--------|--------|--------|--------|
| `CompatHead` | 0.6065 | 0.6935 | 0.7015 |
| `ChoiceHead` | 0.5950 | 0.6545 | 0.6365 |

The single-seed table this replaces reported 0.600 vs 0.581. With three seeds
the compat gap is +0.0385 (SE 0.0352, t ≈ 1.10), **not significant at 95%**.
The compat head is higher for every seed, but the seed spread is large enough
that a 3-epoch accuracy advantage is not established. The reference head is
better calibrated at this budget (ECE 0.029 vs 0.041).

### Seed variance, 3 seeds per system, 1 epoch

| System | mean | std | per-seed |
|--------|------|-----|----------|
| `CompatHead` | **0.5593** | **0.0008** | 0.5585, 0.5600, 0.5595 |
| `ChoiceHead` | 0.5247 | 0.0184 | 0.5050, 0.5415, 0.5275 |

Δ = +0.035, t = 3.26 (**significant at 95%**). At 1 epoch every `CompatHead`
run beats every `ChoiceHead` run, and the compat head is ~23× more
seed-stable. That stability does **not** carry to 3 epochs: there the compat
std is 0.053 vs 0.031 for the reference head, so the 1-epoch result should not
be generalized.

### On order invariance, two different guarantees

These are routinely conflated, so this project measures them separately:

| Level | What is permuted | `CompatHead` | `ChoiceHead` |
|-------|------------------|--------------|--------------|
| **Head-level** | `choice_read_idx` inside a fixed tokenised batch | **≤9e-8 TV** | 2.4e-6 TV |
| **End-to-end** | the token sequence itself (full re-tokenisation) | **~3.2e-3** max Δ | not measured |

The head is invariant by construction: each candidate is scored by a
pure function of `(answer, candidate_i)` using shared projections. The
measured residual (≤9e-8 TV) is float32 reduction order, not a dependence on
position. The end-to-end number is larger because the *answer* token attends
over all candidates, so permuting them changes the float32 summation order
inside the backbone. In practice the selected candidate was identical across all three
orderings tested, but this is not a mathematical guarantee at the pipeline
level, only at the head level.

If you need strict end-to-end invariance, constrain the answer read-out (e.g.
pool candidate representations commutatively) rather than relying on the head
alone.

### Against baselines, same held-out split (1 epoch)

| System | Accuracy | NLL |
|--------|----------|-----|
| majority class (per type) | 0.293 choice / 0.544 noul / 0.269 score |, |
| next-token letter scoring (frozen) | 0.401 | 2.960 |
| zero-shot generation (full 28 layers) | 0.520 |, |
| **`CompatHead` + LoRA (zero tokens)** | **0.559** | **0.933** |

The trained head exceeds zero-shot full-model generation on accuracy while
emitting no tokens at all.

### Epoch scaling (seed 0 only)

| Epochs | 1 | 3 | 5 |
|--------|---|---|---|
| Accuracy | 0.559 | 0.600 | 0.614 |
| Raw ECE | 0.021 | 0.048 | 0.082 |

This ablation holds the seed fixed at 0, so it isolates training length but
not seed noise. The 3-epoch seed-0 point (0.600) is an unlucky draw: the
3-seed mean is 0.667 ± 0.053. Do not compare the 5-epoch single run directly
to the multi-seed mean. At seed 0, 3 epochs still has the best NLL and
calibration, and 5 epochs is overconfident (fitted temperature 1.95), but the
epoch choice itself has not been rerun across seeds.

### Where it fails

Accuracy by candidate count (seed-0 checkpoint, choice questions only):

| C | 2 | 3 | 4 | 5 | 6 | 8 |
|---|---|---|---|---|---|---|
| accuracy | 0.665 | 0.584 | 0.580 | 0.585 | 0.463 | **0.392** |

Many-way choice is the clear weak spot. Per-domain spread is wide (software
security 0.389 → spam detection 0.917), so performance depends heavily on
question *shape*, not just difficulty.

### Efficiency

Two hardware contexts are reported separately because the numbers differ by an
order of magnitude.

**RTX 5060 Laptop (8.1 GB) via WSL2, CUDA 12.8, idle GPU, 10 repeats:**

| Metric | Value |
|--------|-------|
| Model load | 12.98 s |
| First request | 1.04 s |
| Batch 1 median / p95 | 28.0 ms / 41.0 ms |
| Batch 1 throughput | 35.7 questions/s |
| Batch 4 median / p95 | 37.9 ms / 59.1 ms |
| Batch 4 throughput | 105.5 questions/s |
| Batch 8 median / p95 | 29.1 ms / 31.9 ms |
| Batch 8 throughput | 275.0 questions/s |
| Peak inference VRAM | 1,348.9 MB |
| Reserved VRAM | 1,366.0 MB |
| Tokens generated per decision | **0** |

**Windows CPU (default Python venv), 10 repeats:**

| Metric | Value |
|--------|-------|
| Model load | 23.04 s |
| First request | 131.1 ms |
| Batch 1 median / p95 | 91.3 ms / 107.9 ms |
| Batch 1 throughput | 11.0 questions/s |
| Batch 4 median / p95 | 236.1 ms / 291.2 ms |
| Batch 4 throughput | 16.9 questions/s |
| Batch 8 median / p95 | 464.6 ms / 551.9 ms |
| Batch 8 throughput | 17.2 questions/s |
| Tokens generated per decision | **0** |

> Reproduce with `make bench CKPT=<path>` for GPU or `.venv\\Scripts\\python.exe scripts/benchmark.py --checkpoint <path> --repeats 10` on CPU. Latency includes tokenisation, one backbone pass and head scoring; timing is synchronised and warmed up. Raw JSON artifacts live in `results/benchmark_gpu.json` and `results/benchmark_cpu.json`.

---

## Project Status

> **Production-ready decision engine.** All core paths are tested, linted,
> type-checked, dependency-audited and containerised. The training loop
> correctly handles uneven batch counts via `accumulation_schedule`; the
> serving surface covers auth, rate limits, concurrency caps, readiness
> gating, structured logging, Prometheus metrics and optional OpenTelemetry
> tracing. JEPA pretraining remains a measured negative result on the
> current probe set; see [`docs/experiments.md`](docs/experiments.md) before
> quoting any JEPA win. Full honest gap list in [`docs/production.md`](docs/production.md).

---

## Quick Start

The fastest path from clone to a working decision is `make quickstart`. It
installs dependencies, trains a tiny checkpoint on 10 questions, runs a sample
decision, and starts the API in local dev mode. No API keys, no checkpoint
download, no manual config.

```bash
git clone https://github.com/bharqav/mini-jepa.git
cd mini-jepa
make quickstart          # install + demo (~2 min on CPU)
```

That is it. After `make quickstart` you have:
- A trained checkpoint under `runs/smoke-*/checkpoints/best`
- A printed decision result in your terminal
- A FastAPI server on `http://127.0.0.1:8000` with docs at `/docs`

### Install

```bash
make install            # CPU torch + dev extras
# or, for CUDA:
make install-gpu
```

<details>
<summary>Manual install</summary>

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -e ".[dev]"
```

</details>

### Zero-config serve

```bash
make serve              # auth OFF, logs a loud warning, /docs available
```

The server logs a loud warning when auth is off. For production, set
`JEV_API_TOKEN` or `JEV_API_KEYS` and remove `--dev`.

### Run a decision

```bash
# From a request file
python scripts/decide.py --checkpoint <ckpt> --batch examples/request.json

# Or ad hoc
python scripts/decide.py --checkpoint <ckpt> \
  --state "A server is returning repeated HTTP 503 errors." \
  --type choice --question "What should the operator do first?" \
  --candidate "inspect:Inspect service health and dependencies." \
  --candidate "delete:Delete the production database." \
  --candidate "ignore:Ignore the incident."
```

```json
{
  "results": [
    {
      "id": "q0",
      "type": "choice",
      "scores": [
        { "id": "inspect", "score": 0.2747 },
        { "id": "delete",  "score": 0.0498 },
        { "id": "ignore",  "score": 0.6755 }
      ],
      "selected": "ignore",
      "confidence": 0.6755
    }
  ],
  "model_version": "mini-jepa-0.2.0@best"
}
```

> Captured verbatim from the shipped 3-epoch `CompatHead` checkpoint on the
> state shown above. Note the model is **wrong** on this example, "ignore the
> incident" is not the right first action, which is a fair illustration of the
> accuracy numbers above: use the `confidence` field, do not trust the argmax
> blindly.

Reordering the candidates gives the identical `selected` id (probabilities
drift by ~3e-3 from float summation order, see the invariance table above).

### Train

```bash
python scripts/prepare_data.py --max-train 20000 --max-test 2000
python scripts/train.py configs/improved.yaml --name my-run
python scripts/evaluate.py runs/my-run-<timestamp>/checkpoints/best --split test
```

A CPU-only end-to-end smoke test takes a couple of minutes:

```bash
make smoke        # overfits 10 questions; loss should drop sharply
make jepa         # JEPA latent-prediction smoke experiment
```

### Real Qwen JEPA pretraining

The real-weight path uses the pinned Qwen3-0.6B checkpoint and packed decision
text:

```bash
python scripts/train_jepa.py configs/jepa_qwen.yaml --name jepa-qwen
python scripts/evaluate_jepa.py \
  --jepa runs/jepa-qwen-<timestamp>/jepa/context.safetensors \
  --tasks sst2,stsb --baselines jepa,frozen_qwen,random_qwen,mlm_bert,simcse
python scripts/train.py configs/jepa_probe.yaml --name probe-base
python scripts/train.py configs/jepa_probe.yaml --name probe-jepa \
  --init-from-jepa runs/jepa-qwen-<timestamp>/jepa/context.safetensors
```

`configs/jepa_qwen_frozen.yaml` keeps the pretrained target encoder fixed;
`configs/jepa_qwen_anchor.yaml` adds a distillation anchor; `configs/jepa_qwen_top2.yaml`
adapts only the top two context blocks; `configs/jepa_qwen_ema_only.yaml` and
`configs/jepa_qwen_vicreg_only.yaml` provide the regularizer ablation. Mask
ratio/span and predictor depth/width can be overridden with `--mask-ratio`,
`--mask-span`, `--predictor-layers` and `--predictor-dim`.
`configs/jepa_probe_bidir.yaml` evaluates a bidirectional JEPA encoder with
matching full attention. The measured single-seed result is a **negative
result** on both the decision probe and standard downstream probes. On SST-2,
causal JEPA scores 0.828 linear / 0.752 kNN vs 0.847 / 0.804 for frozen Qwen;
on STS-B, Spearman is 0.462 / 0.489 vs 0.479 / 0.557 (SimCSE: 0.747 / 0.828).
Effective rank exposes partial collapse (165 vs 236 on SST-2) and severe
collapse in the bidirectional variant (rank 2.5 despite non-zero std). A first
context-VICReg fix did not help (sentence rank 7.9 / 34.5; SST-2 0.697,
STS-B 0.116). The anchored variant is the best JEPA variant measured on STS-B
kNN (0.589 vs 0.557 frozen), but still loses on SST-2 and STS-B
linear/retrieval. See
`results/jepa_qwen_results.json`, `results/jepa_probe_results.json`,
`results/jepa_probe_bidir_results.json`, `results/jepa_probe_ctx_vicreg.json`,
`results/jepa_probe_anchor.json`, `results/jepa_probe_top2.json`, and
[docs/experiments.md](docs/experiments.md) before quoting a JEPA win.

### Serve

**Zero-config local dev** (no keys, no env vars):

```bash
make serve              # auth OFF, logs a loud warning, /docs available
```

For production, set `JEV_API_TOKEN` or `JEV_API_KEYS` and omit `--dev`:

```bash
export JEV_API_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
export JEV_CHECKPOINT=<path to checkpoint>
python -m uvicorn jev_qwen.api:app --host 127.0.0.1 --port 8000
```

Tagged releases (`v*`) build the sdist/wheel, publish a GitHub release and
push the CPU image to GHCR via `.github/workflows/release.yml`. Install the
serving command with `pip install -e ".[serve]"`; `pre-commit install` enables
the local ruff hooks.

For multiple scoped keys, set `JEV_API_KEYS` (JSON) or
`JEV_API_KEYS_FILE` instead of `JEV_API_TOKEN`; see
[`scripts/generate_api_keys.py`](scripts/generate_api_keys.py). A key may have
only the `decide` or `answer` scope and an optional per-key `rps`/`burst`
quota. The service refuses to start if no key is configured unless you opt
into local-only dev mode with `JEV_DEV_MODE=1` (or
`python -m jev_qwen.api --dev`).

Requests carry an `X-Request-ID` (supplied or generated) in the response and
structured access logs. Prometheus metrics are at `/metrics`; keep that
endpoint on a private network.

```bash
curl -s http://127.0.0.1:8000/v1/decide \
  -H "Authorization: Bearer ***" \
  -H "Content-Type: application/json" \
  -d @examples/request.json
```

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/health` | no | Liveness, process is up |
| `GET` | `/ready` | no | Readiness, `503` until weights are loaded |
| `GET` | `/v1/models` | no | Model card |
| `POST` | `/v1/decide` | bearer | Batch decision request |
| `POST` | `/v1/answer` | bearer | Compatibility adapter (single question) |
| `GET` | `/metrics` | no | Prometheus metrics (restrict at the edge) |
| `GET` | `/demo` | dev only | Minimal browser demo (404 when auth is on) |

Interactive API docs are served at `/docs` when the app is running. TLS is not
terminated in-process; run behind a reverse proxy such as the provided
[`deploy/nginx.conf.example`](deploy/nginx.conf.example). Container and systemd
templates live in [`deploy/`](deploy/README.md).

---

## API Reference

### `POST /v1/decide`

```jsonc
{
  "state": "A server is returning repeated HTTP 503 errors.",
  "questions": [
    {
      "id": "q0",
      "type": "choice",              // "choice" | "score" | "noul"
      "question": "What should the operator do first?",
      "candidates": [               // 2+ for choice; exactly 2 for noul
        { "id": "inspect", "text": "Inspect service health." },
        { "id": "ignore",  "text": "Ignore the incident." }
      ]
    }
  ],
  "settings": { "temperature": 1.0, "include_metadata": false }
}
```

Responses include per-candidate scores, a `selected` id (choice/noul),
`confidence`, and, for `score` questions, `expected_level` and
`normalized_score ∈ [0, 1]`.

### Error contract

| Status | Meaning |
|--------|---------|
| `401` | Missing or invalid bearer token on `/v1/decide` or `/v1/answer` |
| `403` | The key is valid but lacks the endpoint's scope |
| `422` | Schema violation, empty questions, duplicate ids, wrong candidate counts. **Never** answered with a fabricated prediction |
| `413` | Exceeds `JEV_MAX_BODY_BYTES`, `JEV_MAX_STATE_CHARS`, `JEV_MAX_QUESTIONS`, or `JEV_MAX_CANDIDATES` |
| `429` | Rate limit exceeded; `Retry-After` gives the delay in seconds |
| `503` | Model not ready, unrecoverable CUDA OOM, or the concurrency limit is full (`Retry-After: 1`) |
| `500` | Unexpected internal error (logged with a request id) |

---

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `JEV_CHECKPOINT` | *(none)* | Checkpoint directory loaded at startup |
| `JEV_API_TOKEN` | *(none)* | Legacy single bearer token; use one of the key variables below for scopes/rotation |
| `JEV_API_KEYS` | *(none)* | JSON map of key id to token or `{token, scopes, rps, burst}` |
| `JEV_API_KEYS_FILE` | *(none)* | Path to the same JSON document (mutually exclusive with `JEV_API_KEYS`) |
| `JEV_TRUSTED_PROXIES` | *(none)* | Comma-separated CIDRs allowed to supply `X-Forwarded-For` |
| `JEV_DEV_MODE` | `0` | Explicitly disable authentication (loud warning; local use only) |
| `JEV_RATE_LIMIT_RPS` | `20` | Per-client token-bucket refill rate per second; `0` disables |
| `JEV_RATE_LIMIT_BURST` | `40` | Per-client token-bucket capacity |
| `JEV_RATE_LIMIT_MAX_CLIENTS` | `4096` | Cap on tracked client buckets; least-recently-used evicted first |
| `JEV_MAX_CONCURRENCY` | `0` | Max simultaneous inference requests; `0` = unbounded. Set `1` per GPU |
| `JEV_MAX_BODY_BYTES` | `1000000` | Reject declared bodies larger than this with `413` |
| `JEV_MAX_STATE_CHARS` | `20000` | Reject larger states with `413` |
| `JEV_MAX_QUESTIONS` | `32` | Reject larger question counts with `413` |
| `JEV_MAX_CANDIDATES` | `64` | Reject larger candidate lists with `413` |
| `JEV_METRICS_ENABLED` | `1` | Serve Prometheus metrics at `/metrics` |
| `JEV_LOG_LEVEL` | `INFO` | Standard logging level |
| `JEV_OTEL_ENABLED` | `0` | Enable OpenTelemetry tracing (requires the `otel` extra) |
| `JEV_OTEL_ENDPOINT` | *(none)* | OTLP/HTTP trace endpoint |
| `JEV_OTEL_SERVICE_NAME` | `jev-qwen` | Service name reported to the trace backend |
| `JEV_REAL_WEIGHTS` | *(unset)* | Set to `1` to run the real-weights test suite |
| `JEV_TEACHER_URL` | *(none)* | External teacher endpoint (optional) |
| `JEV_TEACHER_KEY` | *(none)* | Bearer token for the teacher endpoint |

---

## Package Structure

```
src/jev_qwen/
├── schema.py         # Typed request/response models + probability validation
├── tokenize.py       # Packing, block-causal mask, shared-start position IDs
├── model.py          # DecisionNetwork, CompatHead, ChoiceHead, LoRA wiring
├── jepa.py           # JEPA state: encoders, predictor, EMA, collapse metrics
├── data.py           # Loading, validation, dedup, grouped splits, collation
├── adapters.py       # Pluggable dataset sources (JSONL, synthetic)
├── teacher.py        # Optional offline labelling with retry + accounting
├── train.py          # Training loop, accumulation, scheduling, checkpointing
├── evaluate.py       # Metrics, order-sensitivity, stratified evaluation
├── calibration.py    # Temperature scaling, bootstrap, reliability diagrams
├── stats.py          # Bootstrap CIs and cross-seed comparison
├── infer.py          # DecisionEngine (batching, OOM backoff, determinism)
├── api.py            # FastAPI service
├── run_logger.py     # Run artefacts and self-contained checkpoints
└── utils.py          # Devices, seeding, mixed precision, environment info
```

---

## Engineering Decisions

### Decision: score candidates independently instead of attending over them

**Why:** Invariance becomes *exact at the head*. With shared projections,
`score_i` depends only on `(answer, candidate_i)`, so permuting the read order
cannot change any score; the measured residual is **≤9e-8 TV** (float32
reduction order).

**Alternatives:** the reference `ChoiceHead` uses self-attention across the
candidate set. It is permutation-*equivariant* in exact arithmetic, but
inter-candidate edges plus floating-point reduction order make the measured
sensitivity larger (3.4e-6). Both are implemented and ablated.

**Trade-off:** no cross-candidate interaction in the head. The backbone still
sees the full candidate set (under a block-causal mask), so candidates are not
isolated at the representation level, only at the scoring level. Consequently
*end-to-end* invariance (re-tokenising in a different order) is only ~3e-3,
not exact, because the answer token's attention summation order changes.

### Decision: compute deterministic things, don't learn them

Any question with an exact answer, "is 58 kg under the limit?", "what does
this estimate score?", is computed, not inferred. Putting a network on that
path would add error where the rules are already exact and make results
unauditable.

### Decision: leave uncertain questions unlabelled

Scenario generators emit `label=None` for questions with no formula rather than
inventing a plausible answer. Fabricated ground truth is worse than no data.

### Decision: `num_workers: 0` by default

DataLoader workers use fork/forkserver, which is unreliable under WSL2 and on
locked-down Windows. Correctness over throughput; raise it only after
profiling.

### Decision: derive dataset cache fingerprints from source

`datasets` caches mapped tables keyed by a fingerprint that does not track
changes to a module-level function. A newly added column silently came back
empty. `_map_fingerprint()` hashes the mapping function's source so schema
changes invalidate the cache automatically.

### Decision: training must not silently drop partial accumulation windows

Gradient accumulation with a fixed divisor silently discarded any trailing
micro-batches that did not fit an exact multiple of `grad_accum_steps`. This
wasted compute, wasted data and produced biased gradient estimates. The
training loop now builds an explicit `accumulation_schedule` that retains the
final partial window, scales the loss by the actual window size, and steps the
optimiser exactly once per window. The bug was caught by a unit test that
asserts the expected step sequence and by a real-weight smoke test that
verifies checkpoint save and fresh-process reload.

---

## How the Decision Pipeline Works

A request flows through six stages. Every stage is deterministic and tested;
no stage calls `generate()`.

```mermaid
flowchart LR
    A[Client request\nstate + questions + candidates] --> B[Validate schema\n422 on invalid]
    B --> C[Tokenize\npacked block-causal mask\nshared-start position IDs]
    C --> D[Backbone forward\nQwen3 first-N layers\nLoRA on top-K\nuse_cache=false]
    D --> E[Read embeddings\nchoice last-content token\nanswer final token]
    E --> F[Score + normalise\nsoftmax, assert sum=1]
    F --> G[Typed response\nprobabilities + confidence]
```

**Tokenisation.** Each question is packed as
`[state][question][choice_0]...[choice_k][answer_prompt]`. Every choice span
restarts at the same position ID and can attend to the question and itself but
never to siblings. This makes the *head* exactly order-invariant.

**Backbone.** Only the first `num_layers` of Qwen3-0.6B run. LoRA adapters on
the last `lora_layers` attention projections add trainable capacity without
modifying the pretrained weights. `use_cache=False` is asserted in tests;
there is no generation path.

**Head.** `CompatHead` scores each candidate with shared `Wq/Wk` projections
plus a learned per-task temperature. `ChoiceHead` adds a single self-attention
layer over the candidate set for reference. Both emit logits that are softmaxed
and validated to sum to 1.

---

## How Training Works

Training is a bounded loop with explicit accumulation, checkpointing and
calibration. The same config drives the tiny smoke run and the full RTX 5060
run.

```mermaid
flowchart TB
    A[Load dataset\npinned revision\nvalidated + deduped] --> B[Group split\nno state leakage\ntrain/val/test]
    B --> C[Tokenize + pack\nblock-causal mask\nlength bucketing eval only]
    C --> D[Train loop\nbf16 autocast\nLoRA + head.trainable]
    D --> E{Micro-batch\nloss}
    E --> F{Accumulation\nschedule}
    F -->|window complete| G[Optimizer step\n+ scheduler\n+ clip grad norm]
    F -->|accumulating| D
    G --> H{End of\nepoch?}
    H -->|no| D
    H -->|yes| I[Evaluate\nstratified metrics\nECE + calibration]
    I --> J{Better than\nbest?}
    J -->|yes| K[Save checkpoint\nhead.safetensors\nmeta.json + config]
    J -->|no| L{Max epochs\nreached?}
    L -->|no| D
    L -->|yes| M[Done]
    K --> L
```

**Accumulation schedule.** Instead of a fixed divisor, the loop builds a
per-microbatch schedule of `(window_size, should_step)` tuples. The final
partial window is retained and its loss is scaled by `window_size`, so no
batches are silently dropped and the gradient estimate is unbiased.

**Eval/calibration.** After each epoch the held-out split is evaluated.
Temperature scaling is fit on a seeded half and measured on the other half;
reliability and risk-coverage diagrams are emitted as SVGs.

---

## How JEPA Pretraining Works

JEPA pretraining is an optional, experimental track that runs before the
decision probe. It is not canonical JEPA; the non-canonical choices are
documented in `docs/architecture_jepa.md`.

```mermaid
flowchart TB
    A[Packed text\nstate + question\n+ candidates\nno labels] --> B[Block masking\ncontiguous spans\nreplace with mask vector]
    B --> C[Context encoder\ntrainable Qwen blocks\ncausal / block mask]
    B --> D[Target encoder\nfrozen or EMA copy\nunmasked input\nno_grad]
    C --> E[Latent predictor\nmask tokens + context reps\n→ predicted target reps]
    D --> F[Loss at masked positions\nMSE + cosine\n+ VICReg terms]
    E --> F
    F --> G{EMA?}
    G -->|yes| H[EMA update target\nfrom context]
    H --> C
    G -->|no| I[Save checkpoint\ncontext + predictor\n+ meta]
```

**Why it is non-canonical.** The backbone is a causal decoder-only LM, not a
bidirectional ViT. The target encoder is the first-N Qwen blocks, not the full
model. Targets are last-hidden-state vectors at masked positions, not pooled
span representations.

**Information flow.** The context encoder sees masked embeddings before any
self-attention. The target encoder sees the unmasked sequence under
`torch.no_grad()` and its outputs are `.detach()`ed before the loss. The
predictor receives only mask tokens and masked-context representations; it
never sees target content directly.

**Collapse prevention.** Stop-gradient plus EMA is the primary guard. VICReg
variance and covariance penalties are secondary. Collapse diagnostics
(effective rank, mean/std, covariance norm) are logged every N steps.

---

## How Evaluation Works

Evaluation is a fixed protocol that produces comparable numbers across seeds
and checkpoints.

```mermaid
flowchart TB
    A[Load checkpoint\nhead.safetensors\n+ meta.json] --> B[Run eval loader\nlength-bucketed\nno shuffling]
    B --> C[Per-row metrics\naccuracy, NLL, Brier\nchoice/noul/score\nmacro-F1, AUC]
    C --> D[Order sensitivity\npermute read indices\nmeasure TV]
    D --> E[Calibration\nfit T on seed-half\nmeasure on other half\nECE + reliability SVG]
    E --> F[Bootstrap CIs\npercentile over rows\n95% CI per domain/C]
    F --> G[Aggregate\nseed_report.json\neval_domains.json]
```

**Stratified reporting.** Accuracy is reported per domain and per candidate
count with 95% bootstrap CIs. Row metadata travels inside the batch (never
positional zipping) so length bucketing cannot misalign it.

**Order sensitivity.** `order_sensitivity` measures head-level invariance only:
it permutes `choice_read_idx` inside an already-tokenised batch. End-to-end
invariance is measured separately by full re-tokenisation.

**Uncertainty.** Small-n results carry exact `n` counts and no significance
claims unless a seed sweep exists. The 1-epoch head comparison is significant
(t = 3.26); the 3-epoch comparison is not (t ≈ 1.10).

---

## How Serving Works

The FastAPI service is a thin, hardened wrapper around `DecisionEngine`.

```mermaid
flowchart TB
    A[uvicorn startup] --> B[Load checkpoint\nonce at startup\nnever per request]
    B --> C{Warmup\nenabled?}
    C -->|yes| D[Run one request\nthermalise CUDA allocator]
    C -->|no| E[Ready\n/ready returns 200]
    D --> E
    E --> F[Request arrives\n/v1/decide or /v1/answer]
    F --> G{Auth\nbearer token\nscoped key?}
    G -->|no| H[401 Unauthorized\nstructured access log]
    G -->|yes| I{Validate\nschema + limits\n413 if oversized}
    I -->|invalid| J[422 or 413\nnever fabricate answer]
    I -->|valid| K{Rate limit\nper-client + per-key\ntoken bucket}
    K -->|exceeded| L[429 Retry-After]
    K -->|ok| M{Concurrency\nJEV_MAX_CONCURRENCY}
    M -->|full| N[503 Retry-After: 1]
    M -->|ok| O[Inference\nserialize CUDA forwards\nautocast + chunking]
    O --> P{Metrics\n+ access log\nrequest_id + latency}
    P --> Q[200 + response\nX-Request-ID echoed]
```

**Startup contract.** `/health` is always 200 while the process is up.
`/ready` returns 503 until the checkpoint is resident. The service refuses to
start if no bearer key is configured unless explicit dev mode is set.

**Concurrency and OOM.** `JEV_MAX_CONCURRENCY` bounds simultaneous inference
requests; excess requests receive `503` with `Retry-After: 1`. On CUDA OOM
the chunk size halves and retries down to one question, so a large request
degrades in latency rather than failing.

**Observability.** Prometheus metrics are at `/metrics` (keep private).
Structured JSON logs include `event`, `request_id`, `latency_s`, `error_type`
and key id, but never request bodies or credentials.

---

## What We Tried, Kept, and Cut

This section is the experimental history of the project: every design choice
that was measured, and the outcome. Nothing here is anecdotal; every claim
links to a scripted result in `results/` or a test in `tests/`.

```mermaid
flowchart TB
    A[Start: bev-train\nreference architecture] --> B{Head design}
    B -->|ChoiceHead| C[Reference head\npermutation-equivariant\norder TV ~1e-6]
    B -->|CompatHead| D[Compat head\nindependent projections\nexact head-level invariance]
    D --> E{Training recipe}
    C --> E
    E -->|order augmentation| F[Kept for ref head\nprovable no-op for compat\nloss bit-identical]
    E -->|LoRA top-K| G[Kept\n+0.054 acc at 1 epoch\nsingle biggest lever]
    E -->|learned per-task T| H[Kept\nbetter calibration\nno extra inference cost]
    E -->|fixed grad_accum| I[CUT\nsilently dropped partial windows\nbiased gradients]
    E -->|accumulation_schedule| J[Kept\nretain final window\nunbiased gradient estimate]
    K -->|next-token scoring| L[CUT\nfrozen baseline 0.401\nNLL 2.960]
    K -->|zero-shot gen| M[CUT\nfull 28 layers 0.520\nstill worse than trained]
    K -->|local LM teacher| N[CUT\nagreed on 1/5 rows\nweak signal]
    K -->|bidirectional JEPA default| O[CUT\nsevere collapse\nrank 2.5 / 6.4]
    K -->|context-VICReg only| P[CUT\ntarget still collapsed\npooled rank worse]
    K -->|top-2 blocks only| Q[CUT\nboth tasks worse\nrank 97 / 293]
    K -->|JEPA init decision probe| R[NEGATIVE\ncausal EMA 0.394\nfrozen-target 0.498\nvs frozen Qwen 0.575]
```

### Kept

| Component | Why kept | Evidence |
|-----------|----------|----------|
| `CompatHead` | Exact head-level invariance, cheaper, better ranking | `tests/test_jepa.py`, `results/notes.md` |
| LoRA on top-K layers | Single biggest accuracy lever at CPU scale | choice 0.225 → 0.300 |
| Learned per-task temperature | Better calibration, no inference cost | fitted T per task, ECE ↓ |
| `accumulation_schedule` | Correct gradient estimate, no dropped data | `tests/test_training.py`, smoke test |
| Packed tokenisation with shared-start positions | Foundation of invariance + efficiency | `tests/test_jepa.py` mask tests |
| Frozen-target + EMA options | Controlled ablation of target drift | `configs/jepa_qwen_frozen.yaml` |
| Anchor loss (Episode 10) | Best JEPA variant on STS-B kNN | `results/jepa_probe_anchor.json` |

### Cut or rejected

| Component | Why cut | Evidence |
|-----------|---------|----------|
| Order augmentation for `CompatHead` | Provable no-op: bit-identical training loss | `tests/test_training.py` |
| Next-token letter scoring | Too weak to be a useful baseline | acc 0.401, NLL 2.960 |
| Zero-shot generation (28 layers) | Worse than a trained 12-layer head | acc 0.520 vs 0.559 |
| Local 4-layer LM teacher | 1/5 label agreement; not worth the cost | `tests/test_teacher_http.py` |
| Bidirectional JEPA as default | Severe rank collapse (2.5 / 6.4) despite non-zero std | `results/jepa_probe_bidir_results.json` |
| Context-VICReg only | Target/predictor still collapsed; pooled rank worse | SST-2 linear 0.697, rank 7.9 |
| Top-2 block adaptation | Both tasks worse than full-context anchor | `results/jepa_probe_top2.json` |
| JEPA init for decision probe | Negative result at single seed | probe acc 0.498 vs frozen 0.575 |

### Negative results retained

| Claim | Measured result | Source |
|-------|-----------------|--------|
| JEPA improves decision probe | Causal EMA 0.394; frozen-target 0.498; bidirectional 0.419 | `results/jepa_qwen_results.json` |
| JEPA improves standard probes | SST-2 linear 0.828 vs frozen 0.847; STS-B Spearman 0.462 vs 0.479 | `results/jepa_probe_results.json` |
| Compat head beats reference at 3 epochs | Gap +0.0385, t ≈ 1.10, **not significant** | `results/notes.md` |
| Accuracy scales with candidate count | 0.665 at C=2 → 0.392 at C=8 | `results/seed3_metrics.csv` |

---

## Failure Modes & Reliability

| Situation | Behaviour |
|-----------|-----------|
| Malformed request | `422`, never a fabricated answer |
| Oversized request | `413` before any model work |
| Missing or invalid token | `401` on the inference endpoints |
| Rate limit exceeded | `429` with `Retry-After` |
| CUDA OOM | Chunk size halves and retries down to one question; `503` only if unrecoverable |
| Model not loaded | `/ready` returns `503`, `/v1/decide` returns `503` |
| Invalid probability vector | `validate_response` raises rather than returning it |
| Checkpoint/architecture mismatch | `load_checkpoint` raises; LoRA presence is cross-checked against config |
| Legacy `head.pt` checkpoint | Refused with conversion instructions; the unsafe pickle is never loaded |
| Dataset cache stale | Invalidated automatically via source-derived fingerprint |

Determinism is verified: five identical GPU requests returned **max score
delta 0.0**. (`inference_metadata.latency_s` naturally varies, compare
`results`, not the whole envelope.)

---

## Security

Implemented: scoped bearer authentication on the inference endpoints
(constant-time comparison via `hmac.compare_digest`, fail closed unless an
explicit dev mode is enabled), key rotation by running multiple keys, per-client
and per-key rate limiting, trusted-proxy client resolution, a bounded
concurrency cap, strict input validation, body/state/field size limits,
readiness gating, OOM resource control, safetensors-only checkpoint loading,
pinned Hugging Face revisions, request IDs, structured audit/access logs that
exclude request bodies and credentials, Prometheus metrics, optional
OpenTelemetry tracing, secret scanning, and a `pip-audit` dependency scan in CI.

**Not implemented:** a user database or per-tenant isolation (one key grants
inference), cluster-wide rate limiting (buckets are per instance; N replicas
enforce N budgets), in-process TLS termination, and a WAF/connection cap beyond
the edge examples. TLS is provided by a reverse proxy; see
[`deploy/nginx.conf.example`](deploy/nginx.conf.example). Full threat model in
[`SECURITY.md`](SECURITY.md).

---

## Testing

```bash
make test        # weight-free: no model download required
make test-real   # full suite with real Qwen3-0.6B (~1.2 GB, cached afterwards)
```

| Layer | Coverage |
|-------|----------|
| Invariants | Candidate permutation (exact + tolerance), question isolation, packed-vs-independent |
| Correctness | Batch-vs-single agreement, padding, variable lengths, probability normalisation, candidate/label alignment |
| Calibration | Temperature recovery for over/under-confidence, ECE, risk-coverage thresholds, SVG validity |
| JEPA | Target-encoder isolation (no gradients), EMA exactness, no trivial copy, collapse detection |
| Data | Schema validation, grouped splits with no leakage, adapters, duplicate rejection |
| API | Health/readiness gating, limits, error contract, structured logging, scoped auth, rate limits, metrics, concurrency, demo page |
| Teacher | Cache reuse, validation rejection, real-HTTP retry/backoff/accounting |
| Integration | Tiny train → save → **fresh-process** reload → evaluate |

---

## Documentation

| Document | Description |
|----------|-------------|
| [docs/architecture.md](docs/architecture.md) | Decision model design and isolation mechanism |
| [docs/architecture_jepa.md](docs/architecture_jepa.md) | JEPA design, and what makes it non-canonical |
| [docs/experiments.md](docs/experiments.md) | Every experiment, including negative results |
| [docs/evaluation.md](docs/evaluation.md) | Metrics definitions and methodology |
| [docs/dataset.md](docs/dataset.md) | Dataset schema and split hygiene |
| [docs/model_card.md](docs/model_card.md) | Intended use, metrics, limitations, dataset-license caveat |
| [docs/production.md](docs/production.md) | Serving behaviour, limits, and known gaps |
| [deploy/README.md](deploy/README.md) | Compose/systemd deployment, key rotation, metrics, load test |
| [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) | Backbone/dataset and runtime licenses |
| [docs/environment.md](docs/environment.md) | Platform notes and fallbacks |
| [results/notes.md](results/notes.md) | Full experiment log with measured numbers |

---

## Roadmap

**Implemented**, everything in the features table above.

**Planned**
- Target-side VICReg / stronger stop-grad to fix the measured rank collapse
- Multi-seed JEPA probe sweep and a decision-aware auxiliary objective
- Quantised CPU inference path for edge deployment

**Not planned**
- Autoregressive decoding in the decision path (this is the point of the project)

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Short version: `make quickstart` to
verify the project works off the bag, branch, `make check`, open a PR.

## License

MIT, see [LICENSE](LICENSE).

## Acknowledgements

- [avbiswas/bev-train](https://github.com/avbiswas/bev-train), the decision
  architecture this project reproduces and then improves
- [`avbiswas/bev-decision-150K`](https://huggingface.co/datasets/avbiswas/bev-decision-150K), evaluation dataset
- Assran et al., *Self-Supervised Learning from Images with a Joint-Embedding
  Predictive Architecture* (CVPR 2023), the JEPA formulation
- Bardes et al., *VICReg*, variance/invariance/covariance regularisation