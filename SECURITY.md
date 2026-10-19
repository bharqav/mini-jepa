# Security Policy

## Scope

`jev-qwen` is a research library. It runs locally, loads a pretrained
backbone, and exposes an optional HTTP API for inference. The security-relevant
surface is:

- the FastAPI inference service (`jev_qwen.api`)
- deserialisation of training checkpoints (`torch.load`, `safetensors`)
- loading models and datasets from the Hugging Face Hub
- the optional external-teacher HTTP client (`jev_qwen.teacher.ExternalTeacher`)

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
response SLA — this is an independently maintained research project, not a
commercial product with a security team.

## Security posture (what is and is not implemented)

### Implemented

- **Input validation** — every API request is validated by Pydantic models
  (`jev_qwen.schema`). Empty questions, duplicate IDs, wrong candidate counts
  and empty texts are rejected with `422`; the service never returns a
  fabricated answer for an invalid request.
- **Request size limits** — state length, question count and candidate count
  are capped via `JEV_MAX_STATE_CHARS`, `JEV_MAX_QUESTIONS`,
  `JEV_MAX_CANDIDATES` (HTTP `413` when exceeded).
- **Readiness gating** — `/ready` returns `503` until model weights are loaded,
  so a reverse proxy will not route traffic to a cold instance.
- **Resource control** — CUDA OOM triggers batch-chunk halving with retry;
  requests over `max_input_chars` are rejected rather than silently truncated.
- **No secret logging** — structured logs include request IDs and latencies,
  never headers or credentials. Covered by a test.
- **Secret scanning** — `scripts/security_scan.py` runs in CI.

### Not implemented — do not expose to untrusted networks

- **No authentication or authorisation.** There is no API key, no OAuth, no
  user model. Do not put this service on a public interface as-is.
- **No rate limiting.**
- **No TLS termination.** Run behind a reverse proxy that provides it.
- **No multi-tenancy or per-tenant isolation.**
- **No CSRF protection** (the API is JSON-only and unauthenticated, but this
  would need revisiting if auth is added).

### Trust assumptions

- **Checkpoints are trusted.** `load_checkpoint` uses `torch.load`, which is
  not safe against malicious pickles. Only load checkpoints you produced or
  otherwise trust. A `safetensors`-only loader is on the roadmap.
- **Hub repositories are trusted.** Models and datasets are fetched from the
  Hugging Face Hub. Pin revisions if supply-chain integrity matters to you
  (note: `load_backbone` does not currently pin a revision).
- **The external teacher endpoint is trusted with your API key.** The teacher
  client sends state/instructions/candidate text to whatever URL you configure
  via `JEV_TEACHER_URL`. It supports `JEV_TEACHER_KEY`. Never point it at an
  endpoint you do not control.

## Out of scope

- Vulnerabilities in upstream `torch`, `transformers`, `peft`, or `datasets`.
  Report those to the respective maintainers; Dependabot will surface updates.
- Model quality or accuracy issues.
- Denial of service via deliberately unbounded local resource use (this is a
  single-user research tool).