# Security Policy

## Scope

`mini-jepa` is a research library. It runs locally, loads a pretrained
backbone, and exposes an optional HTTP API for inference. The security-relevant
surface is:

- the FastAPI inference service (`jev_qwen.api`)
- deserialisation of training checkpoints (`safetensors` only)
- loading models and datasets from the Hugging Face Hub (pinned revisions)
- the optional external-teacher HTTP client (`jev_qwen.teacher.ExternalTeacher`)
- the reverse proxy you place in front of the service (TLS termination)

## Reporting a vulnerability

Please report security issues **privately** via GitHub's
[private vulnerability reporting](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing/privately-reporting-a-security-vulnerability)
on this repository ("Security" → "Report a vulnerability").

Please do not open a public issue for a security bug.

Include, where relevant:

- affected component and version/commit
- reproduction steps
- impact (what an attacker gains)
- whether the issue requires a malicious checkpoint, dataset, or API client

We aim to acknowledge reports within a few days. There is no guaranteed
response SLA, this is an independently maintained research project, not a
commercial product with a security team.

## Security posture (what is and is not implemented)

### Implemented

- **Authentication**, `/v1/decide` and `/v1/answer` require
  `Authorization: Bearer <JEV_API_TOKEN>`. The service **fails closed**: it
  refuses to start when the token is unset. Authentication is disabled only in
  an explicit dev mode (`JEV_DEV_MODE=1` or `python -m jev_qwen.api --dev`),
  which logs a warning at startup. `/health`, `/ready` and `/v1/models`
  remain unauthenticated so probes and model discovery work without a secret.
- **Rate limiting**, a per-client token bucket (`JEV_RATE_LIMIT_RPS`,
  `JEV_RATE_LIMIT_BURST`) returns `429` with `Retry-After`; buckets are
  bounded (`JEV_RATE_LIMIT_MAX_CLIENTS`) and least-recently-used clients are
  evicted. Clients are keyed by the peer address, so one caller cannot spend
  another caller's budget. **Behind multiple replicas** each replica enforces
  its own budget, so the effective cluster rate is `N x` the configured
  value. **Behind a reverse proxy** every request arrives from the proxy
  address unless the edge performs per-client limiting; the app deliberately
  does not trust `X-Forwarded-For`.
- **Input validation**, every API request is validated by Pydantic models
  (`jev_qwen.schema`). Empty questions, duplicate IDs, wrong candidate counts
  and empty texts are rejected with `422`; the service never returns a
  fabricated answer for an invalid request.
- **Request size limits**, state length, question count and candidate count
  are capped via `JEV_MAX_STATE_CHARS`, `JEV_MAX_QUESTIONS`,
  `JEV_MAX_CANDIDATES` (HTTP `413` when exceeded).
- **Readiness gating**, `/ready` returns `503` until model weights are loaded,
  so a reverse proxy will not route traffic to a cold instance.
- **Resource control**, CUDA OOM triggers batch-chunk halving with retry;
  requests over `max_input_chars` are rejected rather than silently truncated.
- **Safe checkpoint loading**, `load_checkpoint` reads `head.safetensors`
  only. A legacy `head.pt` is rejected, never loaded as a fallback.
  `scripts/convert_checkpoints.py` performs a one-time migration with
  `torch.load(..., weights_only=True)`.
- **Pinned hub revisions**, the Qwen3 backbone, tokenizer and decision dataset
  resolve to explicit commit SHAs (`jev_qwen.hub`); no code path fetches an
  implicit `main`. A custom model or dataset name must supply its own
  revision.
- **No secret or body logging**, structured logs contain request IDs,
  latencies, counts and error *types*. Request bodies, candidate text and
  bearer tokens are never logged at INFO or above. Covered by tests.
- **Secret scanning**, `scripts/security_scan.py` runs in CI.
- **Dependency audit**, `pip-audit` runs in CI on every push and pull request.

### Not implemented, do not expose to untrusted networks

- **No authorisation or user model.** One bearer token grants access to both
  inference endpoints; there are no per-user quotas, scopes or audit trails.
- **No cluster-wide rate limiting.** See the single-process caveat above.
- **No in-process TLS termination.** A reverse proxy is required; see below.
- **No multi-tenancy or per-tenant isolation.**
- **No CSRF protection.** The API authenticates with an `Authorization`
  header rather than a cookie, so the classic cookie-CSRF vector does not
  apply, but do not add cookie-based sessions without revisiting this.
- **No protection against denial of service beyond size limits and the
  rate limiter.** There is no WAF, connection cap or queue bound.

### TLS termination (required for network exposure)

The service speaks plain HTTP. Terminate TLS in a reverse proxy and keep the
application bound to loopback. A minimal nginx configuration is provided in
[`deploy/nginx.conf.example`](deploy/nginx.conf.example); it enables TLS 1.2+,
redirects HTTP to HTTPS, caps the request body and forwards the `Authorization`
header unchanged. Replace the certificate paths and `server_name` before use.

Example startup:

```bash
export JEV_API_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
export JEV_CHECKPOINT=/path/to/checkpoint
python -m uvicorn jev_qwen.api:app --host 127.0.0.1 --port 8000
```

The service does not trust or consume `X-Forwarded-*` headers, so no proxy
header configuration can change authentication or rate-limit decisions.

### Trust assumptions

- **Checkpoint files no longer execute code.** Only `head.safetensors` is
  loaded. Safetensors is a tensor container, not a pickle, so a malicious
  checkpoint cannot run arbitrary Python. Model *behaviour* is still whatever
  the weights encode, so load checkpoints whose provenance you accept.
- **Hub repositories are trusted at the pinned revision.** The default backbone
  and dataset resolve to fixed commit SHAs. If you override the name, you must
  supply the revision, and you are trusting that repository.
- **The external teacher endpoint is trusted with your API key.** The teacher
  client sends state/instructions/candidate text to whatever URL you configure
  via `JEV_TEACHER_URL`. It supports `JEV_TEACHER_KEY`. Never point it at an
  endpoint you do not control.
- **The reverse proxy is trusted to terminate TLS correctly** and to restrict
  network access. This project does not verify client certificates.

## Out of scope

- Vulnerabilities in upstream `torch`, `transformers`, `peft`, or `datasets`.
  Report those to the respective maintainers; Dependabot will surface updates.
- Model quality or accuracy issues.
- Denial of service via deliberately unbounded local resource use (this is a
  single-user research tool).