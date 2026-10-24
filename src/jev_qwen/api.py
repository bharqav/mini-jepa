"""Local FastAPI service. Model loads ONCE at startup, never per request.

Production behaviours:
- readiness/liveness split (/health vs /ready) so a load balancer does not
  send traffic before weights are resident
- hard request-size limits enforced before any model work
- multiple scoped bearer tokens (fail closed unless explicit dev mode)
- per-client and per-key token-bucket rate limiting
- bounded concurrency so one burst cannot queue unbounded work
- request IDs on every response, structured access logs and Prometheus metrics
- typed errors (422 validation, 413 too large, 429 rate limited, 503 not
  ready, 500 unexpected)
- graceful shutdown releases the model
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Iterable, Mapping
from contextlib import asynccontextmanager
from typing import Any

import torch
from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from .auth import (
    SCOPES,
    ApiKey,
    ApiKeyRegistry,
    AuthContext,
    TrustedProxies,
    extract_bearer,
    load_keys_from_env,
    parse_api_keys,
    single_key,
)
from .infer import DecisionEngine
from .metrics import PROMETHEUS_AVAILABLE, Metrics
from .schema import DecisionRequest
from .tracing import configure_tracing

logging.basicConfig(
    stream=sys.stdout, level=os.environ.get("JEV_LOG_LEVEL", "INFO"), format="%(message)s"
)
log = logging.getLogger("jev_qwen.api")
ENGINE: DecisionEngine | None = None
READY = False

MAX_STATE_CHARS = int(os.environ.get("JEV_MAX_STATE_CHARS", "20000"))
MAX_QUESTIONS = int(os.environ.get("JEV_MAX_QUESTIONS", "32"))
MAX_CANDIDATES = int(os.environ.get("JEV_MAX_CANDIDATES", "64"))
MAX_BODY_BYTES = int(os.environ.get("JEV_MAX_BODY_BYTES", "1000000"))

DEFAULT_RATE_LIMIT_RPS = 20.0
DEFAULT_RATE_LIMIT_BURST = 40.0
DEFAULT_RATE_LIMIT_MAX_CLIENTS = 4096
DEFAULT_MAX_CONCURRENCY = 0
DEFAULT_METRICS_ENABLED = True

_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._:-]{1,64}$")


def _log(event: str, **fields: Any) -> None:
    rec = {"ts": round(time.time(), 3), "event": event, **fields}
    log.info(json.dumps(rec, default=str))


def _error_type(exc: BaseException) -> str:
    """Loggable error identity.

    Exception messages can embed user-supplied text (a tokenizer or validation
    error is not guaranteed to avoid it), so only the class name is logged.
    """
    return type(exc).__name__


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _request_id(request: Request) -> str:
    supplied = request.headers.get("x-request-id", "").strip()
    if _REQUEST_ID_RE.match(supplied):
        return supplied
    return uuid.uuid4().hex[:16]


class TokenBucket:
    """Token bucket for one client or key.

    Rate limiting happens per client address so one noisy caller cannot spend
    another caller's budget. Per-key quotas add a second, independent budget
    for a scoped credential. Behind a reverse proxy the client address is only
    taken from ``X-Forwarded-For`` when the peer is a configured trusted
    proxy; see docs/production.md.
    """

    def __init__(self, rate: float, burst: float):
        if rate < 0 or burst < 1:
            raise ValueError("rate must be >= 0 and burst >= 1")
        self.rate = rate
        self.capacity = float(burst)
        self.tokens = float(burst)
        self.updated = time.monotonic()
        self.lock = threading.Lock()

    def take(self) -> tuple[bool, float]:
        """Return (allowed, seconds_until_one_token)."""
        if self.rate <= 0:
            return True, 0.0
        with self.lock:
            now = time.monotonic()
            self.tokens = min(self.capacity, self.tokens + (now - self.updated) * self.rate)
            self.updated = now
            if self.tokens >= 1.0:
                self.tokens -= 1.0
                return True, 0.0
            return False, (1.0 - self.tokens) / self.rate


class BucketRegistry:
    """Bounded buckets, evicting the least recently used client."""

    def __init__(self, rate: float, burst: float, max_clients: int):
        if max_clients < 1:
            raise ValueError("max_clients must be >= 1")
        self.rate = rate
        self.burst = burst
        self.max_clients = max_clients
        self._buckets: OrderedDict[str, TokenBucket] = OrderedDict()
        self._lock = threading.Lock()

    def take(
        self, client: str, rate: float | None = None, burst: float | None = None
    ) -> tuple[bool, float]:
        effective_rate = self.rate if rate is None else rate
        effective_burst = self.burst if burst is None else burst
        if effective_rate <= 0:
            return True, 0.0
        with self._lock:
            bucket = self._buckets.get(client)
            if bucket is None:
                bucket = TokenBucket(effective_rate, effective_burst)
                self._buckets[client] = bucket
                if len(self._buckets) > self.max_clients:
                    self._buckets.popitem(last=False)
            else:
                self._buckets.move_to_end(client)
            return bucket.take()


class ModelCard(BaseModel):
    id: str
    head: str
    layers: int | None = None


def _check_size(req: DecisionRequest) -> None:
    if len(req.state) > MAX_STATE_CHARS:
        raise HTTPException(413, f"state too long: {len(req.state)} > {MAX_STATE_CHARS} chars")
    if len(req.questions) > MAX_QUESTIONS:
        raise HTTPException(413, f"too many questions: {len(req.questions)} > {MAX_QUESTIONS}")
    for q in req.questions:
        if len(q.candidates) > MAX_CANDIDATES:
            raise HTTPException(
                413, f"question {q.id}: {len(q.candidates)} candidates > {MAX_CANDIDATES}"
            )


def _require_scope(auth: AuthContext, scope: str) -> None:
    if auth.dev:
        return
    if scope not in auth.scopes:
        raise HTTPException(403, f"key {auth.key_id!r} lacks the {scope!r} scope")


def _build_registry(
    api_token: str | None,
    api_keys: Mapping[str, Any] | Iterable[ApiKey] | None,
) -> ApiKeyRegistry:
    if api_keys is not None:
        if isinstance(api_keys, Mapping):
            return ApiKeyRegistry(parse_api_keys(json.dumps(api_keys), "api_keys"))
        return ApiKeyRegistry(api_keys)
    if api_token is not None:
        token = api_token.strip()
        return ApiKeyRegistry([single_key(token)] if token else [])
    return ApiKeyRegistry(load_keys_from_env())


def create_app(
    checkpoint: str | None = None,
    *,
    dev_mode: bool | None = None,
    api_token: str | None = None,
    api_keys: Mapping[str, Any] | Iterable[ApiKey] | None = None,
    trusted_proxies: str | None = None,
    rate_limit_rps: float | None = None,
    rate_limit_burst: float | None = None,
    rate_limit_max_clients: int | None = None,
    max_concurrency: int | None = None,
    max_body_bytes: int | None = None,
    metrics_enabled: bool | None = None,
) -> FastAPI:
    dev = _env_flag("JEV_DEV_MODE") if dev_mode is None else dev_mode
    registry = _build_registry(api_token, api_keys)
    rps = (
        float(os.environ.get("JEV_RATE_LIMIT_RPS", DEFAULT_RATE_LIMIT_RPS))
        if rate_limit_rps is None
        else float(rate_limit_rps)
    )
    burst = (
        float(os.environ.get("JEV_RATE_LIMIT_BURST", DEFAULT_RATE_LIMIT_BURST))
        if rate_limit_burst is None
        else float(rate_limit_burst)
    )
    max_clients = (
        int(os.environ.get("JEV_RATE_LIMIT_MAX_CLIENTS", DEFAULT_RATE_LIMIT_MAX_CLIENTS))
        if rate_limit_max_clients is None
        else int(rate_limit_max_clients)
    )
    concurrency = (
        int(os.environ.get("JEV_MAX_CONCURRENCY", DEFAULT_MAX_CONCURRENCY))
        if max_concurrency is None
        else int(max_concurrency)
    )
    body_limit = (
        int(os.environ.get("JEV_MAX_BODY_BYTES", MAX_BODY_BYTES))
        if max_body_bytes is None
        else int(max_body_bytes)
    )
    metrics_on = (
        _env_flag("JEV_METRICS_ENABLED", DEFAULT_METRICS_ENABLED)
        if metrics_enabled is None
        else metrics_enabled
    )
    proxies = (
        TrustedProxies.from_env()
        if trusted_proxies is None
        else TrustedProxies(trusted_proxies.split(","))
    )
    client_buckets = BucketRegistry(rps, burst, max_clients)
    key_buckets = BucketRegistry(rps, burst, max(1, len(registry)))
    semaphore = threading.BoundedSemaphore(concurrency) if concurrency > 0 else None
    metrics = Metrics(metrics_on)

    def require_token(
        request: Request, authorization: str | None = Header(default=None)
    ) -> AuthContext:
        if dev:
            return AuthContext("dev", SCOPES, dev=True)
        rid = getattr(request.state, "request_id", None)
        supplied = extract_bearer(authorization)
        if supplied is None:
            _log("auth_failed", request_id=rid, reason="missing_bearer")
            metrics.auth_failed("missing_bearer")
            raise HTTPException(401, "missing bearer token", headers={"WWW-Authenticate": "Bearer"})
        key = registry.authenticate(supplied)
        if key is None:
            _log("auth_failed", request_id=rid, reason="invalid_token")
            metrics.auth_failed("invalid_token")
            raise HTTPException(401, "invalid bearer token", headers={"WWW-Authenticate": "Bearer"})
        request.scope["key_id"] = key.key_id
        return AuthContext(key.key_id, key.scopes, key.rps, key.burst)

    def rate_limit(request: Request, auth: AuthContext = Depends(require_token)) -> None:
        client = proxies.client_ip(
            request.client.host if request.client else None,
            request.headers.get("x-forwarded-for"),
        )
        allowed, retry_after = client_buckets.take(client)
        if auth.rps is not None or auth.burst is not None:
            key_allowed, key_retry = key_buckets.take(auth.key_id, auth.rps, auth.burst)
            allowed = allowed and key_allowed
            retry_after = max(retry_after, key_retry)
        if not allowed:
            rid = getattr(request.state, "request_id", None)
            _log(
                "rate_limited",
                request_id=rid,
                client=client,
                key_id=auth.key_id,
                retry_after_s=round(retry_after, 3),
            )
            metrics.limited()
            raise HTTPException(
                429,
                "rate limit exceeded",
                headers={"Retry-After": f"{max(retry_after, 0.001):.3f}"},
            )

    def limit_concurrency():
        if semaphore is None:
            yield
            return
        if not semaphore.acquire(blocking=False):
            raise HTTPException(503, "server busy", headers={"Retry-After": "1"})
        try:
            yield
        finally:
            semaphore.release()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        global ENGINE, READY
        if dev:
            log.warning(
                json.dumps(
                    {
                        "ts": round(time.time(), 3),
                        "event": "auth_disabled",
                        "detail": "dev mode: bearer authentication is OFF; never expose this "
                        "process to an untrusted network",
                    }
                )
            )
        elif len(registry) == 0:
            raise RuntimeError(
                "no API keys configured; refusing to start with authentication disabled. "
                "Set JEV_API_KEYS or JEV_API_TOKEN, or opt into local-only dev mode with "
                "JEV_DEV_MODE=1 (or `python -m jev_qwen.api --dev`)."
            )
        t0 = time.perf_counter()
        ENGINE = DecisionEngine(checkpoint=checkpoint or os.environ.get("JEV_CHECKPOINT"))
        if _env_flag("JEV_WARMUP", False) and hasattr(ENGINE, "warmup"):
            ENGINE.warmup()
            _log("engine_warmed", model=ENGINE.model_version)
        READY = True
        metrics.set_ready(True)
        _log(
            "engine_ready",
            model=ENGINE.model_version,
            device=str(ENGINE.device),
            load_s=round(time.perf_counter() - t0, 2),
        )
        try:
            yield
        finally:
            READY = False
            metrics.set_ready(False)
            _log("engine_shutdown")
            ENGINE = None

    app = FastAPI(title="jev-qwen decision API", version="0.2.0", lifespan=lifespan)

    @app.middleware("http")
    async def observe(request: Request, call_next):
        rid = _request_id(request)
        request.state.request_id = rid
        method = request.method
        path = request.url.path
        client = proxies.client_ip(
            request.client.host if request.client else None,
            request.headers.get("x-forwarded-for"),
        )
        declared = request.headers.get("content-length")
        if declared is None and method in {"POST", "PUT", "PATCH"}:
            metrics.request_started()
            metrics.request_finished(method, path, 411, 0.0)
            _log(
                "request",
                request_id=rid,
                method=method,
                path=path,
                status=411,
                client=client,
                reason="missing_content_length",
            )
            return JSONResponse(
                {"detail": "Content-Length is required for request bodies"},
                status_code=411,
                headers={
                    "X-Request-ID": rid,
                    "X-Content-Type-Options": "nosniff",
                    "Cache-Control": "no-store",
                    "Referrer-Policy": "no-referrer",
                },
            )
        if declared is not None:
            try:
                too_large = int(declared) > body_limit
            except ValueError:
                too_large = False
            if too_large:
                metrics.request_started()
                metrics.request_finished(method, path, 413, 0.0)
                _log(
                    "request",
                    request_id=rid,
                    method=method,
                    path=path,
                    status=413,
                    client=client,
                    reason="body_too_large",
                )
                return JSONResponse(
                    {"detail": f"request body exceeds {body_limit} bytes"},
                    status_code=413,
                    headers={
                        "X-Request-ID": rid,
                        "X-Content-Type-Options": "nosniff",
                        "Cache-Control": "no-store",
                        "Referrer-Policy": "no-referrer",
                    },
                )
        metrics.request_started()
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            metrics.request_finished(method, path, 500, time.perf_counter() - start)
            _log(
                "request_error",
                request_id=rid,
                method=method,
                path=path,
                status=500,
                client=client,
            )
            raise
        elapsed = time.perf_counter() - start
        response.headers["X-Request-ID"] = rid
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Cache-Control"] = "no-store"
        response.headers["Referrer-Policy"] = "no-referrer"
        metrics.request_finished(method, path, response.status_code, elapsed)
        _log(
            "request",
            request_id=rid,
            method=method,
            path=path,
            status=response.status_code,
            latency_s=round(elapsed, 4),
            client=client,
            key_id=request.scope.get("key_id"),
        )
        return response

    @app.get("/health")
    def health() -> dict[str, Any]:
        """Liveness: process is up. Does not require the model."""
        return {"status": "ok", "model": ENGINE.model_version if ENGINE else None}

    @app.get("/ready")
    def ready() -> JSONResponse:
        """Readiness: model is loaded and can serve traffic."""
        if not READY or ENGINE is None:
            return JSONResponse({"status": "loading"}, status_code=503)
        return JSONResponse(
            {"status": "ready", "model": ENGINE.model_version, "device": str(ENGINE.device)}
        )

    @app.get("/v1/models")
    def models() -> dict[str, Any]:
        if ENGINE is None:
            raise HTTPException(503, "model not ready")
        card = ModelCard(
            id=ENGINE.model_version,
            head=str(ENGINE.meta.get("head_config", {}).get("type", "?")),
            layers=ENGINE.meta.get("num_layers"),
        )
        return {"data": [card.model_dump()]}

    @app.get("/metrics")
    def prometheus_metrics():
        if not metrics_on:
            raise HTTPException(404, "metrics disabled")
        rendered = metrics.render()
        if rendered is None:
            raise HTTPException(
                503,
                "metrics unavailable: install the serve extra (prometheus-client)"
                if not PROMETHEUS_AVAILABLE
                else "metrics unavailable",
            )
        body, content_type = rendered
        return Response(content=body, media_type=content_type)

    @app.post("/v1/decide")
    def decide(
        req: DecisionRequest,
        request: Request,
        auth: AuthContext = Depends(require_token),
        _: None = Depends(rate_limit),
        __: None = Depends(limit_concurrency),
    ):
        _require_scope(auth, "decide")
        if not READY or ENGINE is None:
            raise HTTPException(503, "model not ready")
        rid = getattr(request.state, "request_id", None)
        _check_size(req)
        t0 = time.perf_counter()
        try:
            resp = ENGINE.decide(req)
        except ValueError as e:
            metrics.inference_error(_error_type(e))
            _log("decide_invalid", request_id=rid, error_type=_error_type(e))
            raise HTTPException(422, str(e)) from e
        except torch.cuda.OutOfMemoryError as e:
            metrics.inference_error(_error_type(e))
            _log("decide_oom", request_id=rid, error_type=_error_type(e))
            raise HTTPException(
                503, "out of memory: retry with fewer questions or lower max_state_tokens"
            ) from e
        except Exception as e:  # noqa: BLE001
            metrics.inference_error(_error_type(e))
            _log("decide_error", request_id=rid, error_type=_error_type(e))
            raise HTTPException(500, "internal error") from e
        _log(
            "decide",
            request_id=rid,
            key_id=auth.key_id,
            questions=len(req.questions),
            latency_s=round(time.perf_counter() - t0, 4),
        )
        return JSONResponse(resp.model_dump())

    @app.post("/v1/answer")
    def answer(
        body: dict[str, Any],
        request: Request,
        auth: AuthContext = Depends(require_token),
        _: None = Depends(rate_limit),
        __: None = Depends(limit_concurrency),
    ):
        """Compatibility adapter for the reference ecosystem's answer()
        protocol: {state, question:{type, instructions, criteria, label?}} ->
        {choice, probabilities} / {score, probabilities} / {noul}."""
        _require_scope(auth, "answer")
        if not READY or ENGINE is None:
            raise HTTPException(503, "model not ready")
        rid = getattr(request.state, "request_id", None)
        try:
            from .data import question_to_choices

            q = body["question"]
            choices, _ = question_to_choices(q)
            if len(choices) > MAX_CANDIDATES:
                raise HTTPException(413, "too many candidates")
            req = DecisionRequest.model_validate(
                {
                    "state": body["state"],
                    "questions": [
                        {
                            "id": "q0",
                            "type": q["type"],
                            "question": q["instructions"],
                            "candidates": [
                                {"id": str(i), "text": c} for i, c in enumerate(choices)
                            ],
                        }
                    ],
                }
            )
            _check_size(req)
            resp = ENGINE.decide(req)
            r = resp.results[0]
            probs = [s.score for s in r.scores]
            if q["type"] == "choice":
                keys = list(q["criteria"])
                return {
                    "choice": keys[probs.index(max(probs))],
                    "probabilities": dict(zip(keys, probs)),
                }
            if q["type"] == "score":
                return {"score": r.expected_level, "probabilities": probs}
            return {"noul": probs[1]}
        except HTTPException:
            raise
        except (KeyError, ValueError) as e:
            metrics.inference_error(_error_type(e))
            _log("answer_invalid", request_id=rid, key_id=auth.key_id, error_type=_error_type(e))
            raise HTTPException(422, f"invalid answer payload: {e}") from e

    @app.get("/demo", response_class=HTMLResponse)
    def demo():
        """Minimal local demo page (dev mode only; it does not send a token)."""
        if not dev:
            raise HTTPException(404, "demo is only available in dev mode")
        return """<!doctype html><html><head><meta charset="utf-8">
<title>jev-qwen demo</title></head><body style="font-family:sans-serif;max-width:720px;margin:2em auto">
<h2>jev-qwen decision demo</h2>
<label>State<br><textarea id="state" rows="3" cols="70">A server is returning repeated HTTP 503 errors.</textarea></label><br><br>
<label>Question<br><input id="q" size="70" value="What should the operator do first?"></label><br><br>
<label>Candidates (one per line, id:text)<br><textarea id="c" rows="4" cols="70">inspect:Inspect service health and dependencies.
delete:Delete the production database.
ignore:Ignore the incident.</textarea></label><br><br>
<button onclick="go()">Decide (zero text generation)</button>
<pre id="out"></pre>
<script>async function go(){
 const cands=document.getElementById('c').value.split('\\n').filter(l=>l.trim())
   .map(l=>{const i=l.indexOf(':');return{id:l.slice(0,i).trim(),text:l.slice(i+1).trim()}});
 const body={state:document.getElementById('state').value,
   questions:[{id:'q0',type:'choice',question:document.getElementById('q').value,candidates:cands}]};
 const r=await fetch('/v1/decide',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
 document.getElementById('out').textContent=JSON.stringify(await r.json(),null,2);
}</script></body></html>"""

    configure_tracing(app)
    return app


app = create_app()


def main() -> None:
    import argparse

    import uvicorn

    ap = argparse.ArgumentParser(description="Serve the decision API")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument(
        "--dev",
        action="store_true",
        help="disable bearer authentication (loud warning; local use only)",
    )
    args = ap.parse_args()
    uvicorn.run(create_app(dev_mode=args.dev), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
