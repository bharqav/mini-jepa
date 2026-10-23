# Production notes

What "production ready" means for this project, and what is verified.

## Serving (`jev_qwen.api`, uvicorn)
| behaviour | status |
|---|---|
| model loaded once at startup, never per request | verified |
| `/health` liveness (always 200 while process up) | verified |
| `/ready` readiness (503 until weights resident) | verified |
| bearer auth on `/v1/decide` and `/v1/answer`; fail closed without a token | verified |
| explicit dev mode (`JEV_DEV_MODE=1` / `--dev`) disables auth and logs a warning | verified |
| request limits: state chars, question count, candidate count → 413 | verified (25000-char state → 413, 40 questions → 413) |
| invalid schema → 422, never a fabricated prediction | verified |
| in-process, per-client token-bucket rate limiting → 429 + `Retry-After` | verified |
| CUDA OOM → chunk halving + retry, 503 only if unrecoverable | implemented, CPU-tested path |
| structured JSON logs (`event`, `request_id`, `latency_s`, `error_type`) | verified |
| request bodies, candidate text and tokens absent from logs | tested (canary strings) |
| graceful shutdown releases the model | implemented |
| TLS termination | **not in-process**; reverse proxy required (`deploy/nginx.conf.example`) |

Tunables via env: `JEV_CHECKPOINT`, `JEV_API_TOKEN`, `JEV_DEV_MODE`,
`JEV_RATE_LIMIT_RPS`, `JEV_RATE_LIMIT_BURST`, `JEV_MAX_STATE_CHARS`,
`JEV_MAX_QUESTIONS`, `JEV_MAX_CANDIDATES`, `JEV_LOG_LEVEL`.

The rate limiter keeps one bucket per peer address, bounded to
`JEV_RATE_LIMIT_MAX_CLIENTS` (least-recently-used evicted), so a noisy client
cannot exhaust another client's budget on the same instance. It is still a
single-process design: with N replicas the effective limit is N times the
configured rate, and there is no shared store. Behind a reverse proxy all
requests share the proxy address unless the edge performs its own per-client
limiting; the app does not trust `X-Forwarded-For`.

## Checkpoints and hub fetches
- `save_checkpoint` writes `head.safetensors`; `load_checkpoint` reads only
  that file. A legacy `head.pt` is refused rather than loaded, because
  `torch.load` can execute code from an untrusted pickle.
- `scripts/convert_checkpoints.py` migrates old checkpoints once, using
  `torch.load(..., weights_only=True)`. Pass `--delete-old` to remove the
  pickle after conversion.
- The backbone, tokenizer and dataset resolve to pinned commit SHAs via
  `jev_qwen.hub`. Custom names must supply an explicit revision; no path
  fetches an implicit `main`.

## Determinism
Five identical requests returned **max score delta 0.0** on GPU. Note that
`inference_metadata.latency_s` naturally varies between calls, compare
`results`, not the whole envelope.

## Memory behaviour
- Bounded internal chunking (`settings.max_batch`); on CUDA OOM the chunk
  halves and retries down to a single question, so a large request degrades
  in latency rather than failing.
- Measured: 3.3 GB peak training VRAM, 1.3 GB inference on an 8.1 GB card.
- `max_input_chars` rejects oversized states before any model work instead of
  silently truncating.

## Training / evaluation
- Gradient checkpointing: opt-in via `model.gradient_checkpointing: true`
  (off by default, 3.3 GB already fits the target GPU).
- Length bucketing: `LengthBucketBatchSampler` is used for eval/inference
  loaders to cut padding waste; **training deliberately keeps random
  shuffling** so batch composition stays seed-reproducible.
- `num_workers: 0` is the safe default. DataLoader workers use
  fork/forkserver, which is unreliable under WSL2 and on locked-down
  Windows. Raise it after profiling on a normal host.
- Dataset `.map(num_proc=1)` for the same reason.
- Checkpoints record `peak_vram_mb`; configs are copied beside weights.

## Uncertainty
`jev_qwen.stats` provides percentile bootstrap CIs over evaluation rows and
cross-seed aggregation (`seed_report.py`). Used to show the head comparison
is significant (t=3.26 over 3 seeds) rather than asserting it.

## Stratified evaluation
`scripts/evaluate.py` writes `eval_domains.json`: accuracy with 95% bootstrap
CIs per **domain** and per **candidate count**. Row metadata travels inside
the batch (never positional zipping) so length bucketing cannot misalign it.
Measured degradation: 0.665 at C=2 → 0.463 at C=6 → 0.392 at C=8.

Caveat found the hard way: `datasets` serves mapped tables from an on-disk
cache whose fingerprint does not track changes to a module-level function.
A newly added column silently came back empty. `load_questions` now derives
`new_fingerprint` from the function source + `SCHEMA_VERSION`, so schema
changes invalidate the cache automatically.

## Teacher integration
The external-teacher path is exercised end-to-end against a real local HTTP
server (`tests/test_teacher_http.py`): happy path, transient-failure retry
with backoff, retry exhaustion raising rather than returning junk, and cost
accounting (failed attempts counted separately from successful calls).
Caching makes reruns free. No paid provider required.

## Performance guardrails
`order_sensitivity` is capped at 200 rows: each row costs `1 + n_perm`
unbatched forwards, so a full sweep over thousands of rows is minutes of
overhead for a metric already known to be ~1e-7.

## Known gaps (honest)
- No authorisation or user model: one bearer token grants both inference
  endpoints. No per-tenant isolation.
- Rate limiting is per client within one instance, not cluster-wide; a
  multi-replica deployment needs a shared store or edge limiter. Behind a
  proxy every request shares the proxy address unless the edge limits per
  client.
- TLS is not terminated in-process. A reverse proxy is required; the example
  nginx config has not been load-tested as part of this repo.
- Single-process; no queueing or horizontal scaling.
- The external teacher has been tested against a local HTTP server with real
  sockets and status codes, but never against an actual paid provider.
- 3-epoch headline numbers are single-seed (only the 1-epoch comparison has a
  3-seed variance study).
- Accuracy degrades sharply with candidate count (0.665 at C=2 → 0.392 at
  C=8); many-way choice is the clear weak spot.