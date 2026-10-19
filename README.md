<div align="center">

# jev-qwen

### Order-invariant decision models and a JEPA-style latent predictor on a truncated Qwen3 backbone

**Zero-token inference · Exact candidate-order invariance · Calibrated probabilities · EMA-target self-supervision**

[![CI](https://github.com/bharqav/jev-qwen/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/bharqav/jev-qwen/actions/workflows/ci.yml)
[![CodeQL](https://github.com/bharqav/jev-qwen/actions/workflows/codeql.yml/badge.svg?branch=main)](https://github.com/bharqav/jev-qwen/actions/workflows/codeql.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

</div>

---

`jev-qwen` turns a pretrained causal language model into something that
**answers questions instead of completing text**. Give it a state, a question,
and a set of candidate answers; it returns a calibrated probability
distribution — in a single forward pass, generating **zero tokens**.

It also contains a JEPA-style latent predictor that learns representations by
predicting masked spans in a target encoder's embedding space, rather than by
predicting tokens.

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
    subgraph track1 [Track 1 — Decision model]
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

    subgraph track2 [Track 2 — JEPA latent predictor]
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

### Decision Path — one forward pass, zero tokens

```mermaid
sequenceDiagram
    participant Client
    participant Engine as DecisionEngine
    participant Tok as tokenize
    participant Backbone as Qwen3 (N layers)
    participant Head as CompatHead

    Client->>Engine: DecisionRequest(state, questions[])
    Engine->>Engine: validate schema (422 on invalid)
    Engine->>Tok: encode_example(state, question, candidates)
    Tok-->>Engine: input_ids, position_ids, 4-D mask, read indices
    Note over Tok: every candidate restarts at the same position id;<br/>candidates cannot attend to siblings
    Engine->>Backbone: forward(use_cache=False)
    Backbone-->>Engine: last_hidden_state
    Engine->>Engine: gather choice embeddings + answer embedding
    Engine->>Head: score(answer, candidates)
    Head-->>Engine: logits (fp32)
    Engine->>Engine: softmax + renormalise + validate sum == 1
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
| **JEPA latent prediction** | EMA target encoder, block masking, predictor, VICReg terms | ✅ non-collapsed |
| **Collapse diagnostics** | Per-dim std, collapse ratio, covariance norm, linear CKA | ✅ |
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

### Head comparison — 3 epochs, identical data, budget and seed

| System | Accuracy | NLL | ECE | choice | noul | score | AUC | head-order TV |
|--------|----------|-----|-----|--------|------|-------|-----|----------------|
| reference `ChoiceHead` | 0.581 | 0.906 | 0.047 | 0.545 | 0.645 | 0.510 | 0.712 | 3.4e-6 |
| **`CompatHead` (default)** | **0.600** | **0.874** | **0.048** | **0.567** | **0.665** | **0.514** | **0.743** | **0.0** |

### Seed variance — 3 seeds per system, 1 epoch

| System | mean | std | per-seed |
|--------|------|-----|----------|
| `CompatHead` | **0.5593** | **0.0008** | 0.5585, 0.5600, 0.5595 |
| `ChoiceHead` | 0.5247 | 0.0184 | 0.5050, 0.5415, 0.5275 |

Δ = +0.035, t = 3.26 (**significant at 95%**). Every `CompatHead` run beats
every `ChoiceHead` run, and the compat head is ~23× more seed-stable.

### On order invariance — two different guarantees

These are routinely conflated, so this project measures them separately:

| Level | What is permuted | `CompatHead` | `ChoiceHead` |
|-------|------------------|--------------|--------------|
| **Head-level** | `choice_read_idx` inside a fixed tokenised batch | **exactly 0.0** | 3.4e-6 TV |
| **End-to-end** | the token sequence itself (full re-tokenisation) | **~3.2e-3** max Δ | not measured |

The head is exactly invariant by construction: each candidate is scored by a
pure function of `(answer, candidate_i)` using shared projections. The
end-to-end number is larger because the *answer* token attends over all
candidates, so permuting them changes the float32 summation order inside the
backbone. In practice the selected candidate was identical across all three
orderings tested, but this is not a mathematical guarantee at the pipeline
level — only at the head level.

If you need strict end-to-end invariance, constrain the answer read-out (e.g.
pool candidate representations commutatively) rather than relying on the head
alone.

### Against baselines — same held-out split

| System | Accuracy | NLL |
|--------|----------|-----|
| majority class (per type) | 0.293 choice / 0.544 noul / 0.269 score | — |
| next-token letter scoring (frozen) | 0.401 | 2.960 |
| zero-shot generation (full 28 layers) | 0.520 | — |
| **`CompatHead` + LoRA (zero tokens)** | **0.559** | **0.933** |

The trained head exceeds zero-shot full-model generation on accuracy while
emitting no tokens at all.

### Epoch scaling

| Epochs | 1 | 3 | 5 |
|--------|---|---|---|
| Accuracy | 0.559 | 0.600 | 0.614 |
| Raw ECE | 0.021 | 0.048 | 0.082 |

**3 epochs is the default**: best raw NLL and calibration. Five epochs buys
+0.014 accuracy but the model becomes overconfident (fitted temperature 1.95).

### Where it fails

Accuracy by candidate count:

| C | 2 | 3 | 4 | 5 | 6 | 8 |
|---|---|---|---|---|---|---|
| accuracy | 0.665 | 0.584 | 0.580 | 0.585 | 0.463 | **0.392** |

Many-way choice is the clear weak spot. Per-domain spread is wide (software
security 0.389 → spam detection 0.917), so performance depends heavily on
question *shape*, not just difficulty.

### Efficiency

RTX 5060 (8.1 GB) via WSL2, CUDA 12.8 wheels:

| Metric | Value |
|--------|-------|
| Peak training VRAM | 3.3 GB |
| Peak inference VRAM | 1.3 GB |
| Single-question latency (median) | 21 ms |
| Throughput @ batch 16 | 369 q/s |
| Tokens generated per decision | **0** |

> Reproduce with `make bench CKPT=<path>`. Latency includes tokenisation,
> one backbone pass and head scoring; timing is CUDA-synchronised and warmed up.

---

## Project Status

> **Alpha — research code.** Correct and tested, not production-hardened.
> No authentication, no rate limiting, no TLS. Intended for local use.
> The 3-epoch headline figures are single-seed.
> See [`docs/production.md`](docs/production.md) for an honest account of
> what is and is not implemented.

---

## Quick Start

### Install

```bash
git clone https://github.com/bharqav/jev-qwen.git
cd jev-qwen
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
  "model_version": "jev-qwen-0.2.0@best"
}
```

> Captured verbatim from the shipped 3-epoch `CompatHead` checkpoint on the
> state shown above. Note the model is **wrong** on this example — "ignore the
> incident" is not the right first action — which is a fair illustration of the
> accuracy numbers above: use the `confidence` field, do not trust the argmax
> blindly.

Reordering the candidates gives the identical `selected` id (probabilities
drift by ~3e-3 from float summation order — see the invariance table above).

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

### Serve

```bash
export JEV_CHECKPOINT=<path to checkpoint>
python -m uvicorn jev_qwen.api:app --host 127.0.0.1 --port 8000
```

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness — process is up |
| `GET` | `/ready` | Readiness — `503` until weights are loaded |
| `GET` | `/v1/models` | Model card |
| `POST` | `/v1/decide` | Batch decision request |
| `POST` | `/v1/answer` | Compatibility adapter (single question) |
| `GET` | `/demo` | Minimal browser demo |

Interactive API docs are served at `/docs` when the app is running.

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
`confidence`, and — for `score` questions — `expected_level` and
`normalized_score ∈ [0, 1]`.

### Error contract

| Status | Meaning |
|--------|---------|
| `422` | Schema violation — empty questions, duplicate ids, wrong candidate counts. **Never** answered with a fabricated prediction |
| `413` | Exceeds `JEV_MAX_STATE_CHARS`, `JEV_MAX_QUESTIONS`, or `JEV_MAX_CANDIDATES` |
| `503` | Model not ready, or unrecoverable CUDA OOM |
| `500` | Unexpected internal error (logged with a request id) |

---

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `JEV_CHECKPOINT` | *(none)* | Checkpoint directory loaded at startup |
| `JEV_MAX_STATE_CHARS` | `20000` | Reject larger states with `413` |
| `JEV_MAX_QUESTIONS` | `32` | Reject larger question counts with `413` |
| `JEV_MAX_CANDIDATES` | `64` | Reject larger candidate lists with `413` |
| `JEV_LOG_LEVEL` | `INFO` | Standard logging level |
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
cannot change any score — verified at **exactly 0.0** difference.

**Alternatives:** the reference `ChoiceHead` uses self-attention across the
candidate set. It is permutation-*equivariant* in exact arithmetic, but
inter-candidate edges plus floating-point reduction order make the measured
sensitivity larger (3.4e-6). Both are implemented and ablated.

**Trade-off:** no cross-candidate interaction in the head. The backbone still
sees the full candidate set (under a block-causal mask), so candidates are not
isolated at the representation level — only at the scoring level. Consequently
*end-to-end* invariance (re-tokenising in a different order) is only ~3e-3,
not exact, because the answer token's attention summation order changes.

### Decision: compute deterministic things, don't learn them

Any question with an exact answer — "is 58 kg under the limit?", "what does
this estimate score?" — is computed, not inferred. Putting a network on that
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

---

## Failure Modes & Reliability

| Situation | Behaviour |
|-----------|-----------|
| Malformed request | `422`, never a fabricated answer |
| Oversized request | `413` before any model work |
| CUDA OOM | Chunk size halves and retries down to one question; `503` only if unrecoverable |
| Model not loaded | `/ready` returns `503`, `/v1/decide` returns `503` |
| Invalid probability vector | `validate_response` raises rather than returning it |
| Checkpoint/architecture mismatch | `load_checkpoint` raises; LoRA presence is cross-checked against config |
| Dataset cache stale | Invalidated automatically via source-derived fingerprint |

Determinism is verified: five identical GPU requests returned **max score
delta 0.0**. (`inference_metadata.latency_s` naturally varies — compare
`results`, not the whole envelope.)

---

## Security

Implemented: strict input validation, request size limits, readiness gating,
OOM resource control, structured logs that never include credentials, and
secret scanning in CI.

**Not implemented:** authentication, authorisation, rate limiting, TLS
termination, multi-tenancy. Run behind a reverse proxy if exposing this beyond
localhost. `load_checkpoint` uses `torch.load`, so **only load checkpoints you
trust**. Full threat model in [`SECURITY.md`](SECURITY.md).

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
| API | Health/readiness gating, limits, error contract, structured logging, demo page |
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
| [docs/production.md](docs/production.md) | Serving behaviour, limits, and known gaps |
| [docs/environment.md](docs/environment.md) | Platform notes and fallbacks |
| [results/notes.md](results/notes.md) | Full experiment log with measured numbers |

---

## Roadmap

**Implemented** — everything in the features table above.

**Planned**
- `safetensors`-only checkpoint loading to remove the `torch.load` trust issue
- Model revision pinning for reproducible hub fetches
- Downstream probe evaluation for the JEPA representations
- Quantised CPU inference path for edge deployment

**Not planned**
- Autoregressive decoding in the decision path (this is the point of the project)

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Short version: `make install`, branch,
`make check`, open a PR.

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgements

- [avbiswas/bev-train](https://github.com/avbiswas/bev-train) — the decision
  architecture this project reproduces and then improves
- [`avbiswas/bev-decision-150K`](https://huggingface.co/datasets/avbiswas/bev-decision-150K) — evaluation dataset
- Assran et al., *Self-Supervised Learning from Images with a Joint-Embedding
  Predictive Architecture* (CVPR 2023) — the JEPA formulation
- Bardes et al., *VICReg* — variance/invariance/covariance regularisation