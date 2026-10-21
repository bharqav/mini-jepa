"""Local FastAPI service. Model loads ONCE at startup, never per request.

Production behaviours:
- readiness/liveness split (/health vs /ready) so a load balancer does not
  send traffic before weights are resident
- hard request-size limits enforced before any model work
- bearer-token authentication on /v1/decide and /v1/answer (fail closed unless
  an explicit dev mode is enabled)
- in-process token-bucket rate limiting (single instance only; see docs)
- typed errors (422 validation, 413 too large, 429 rate limited, 503 not
  ready, 500 unexpected)
- structured JSON logs with latency; request bodies and credentials are never
  logged
- graceful shutdown releases the model
"""

from __future__ import annotations

import hmac
import json
import logging
import os
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

import torch
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from .infer import DecisionEngine
from .schema import DecisionRequest

logging.basicConfig(
    stream=sys.stdout, level=os.environ.get("JEV_LOG_LEVEL", "INFO"), format="%(message)s"
)
log = logging.getLogger("jev_qwen.api")
ENGINE: DecisionEngine | None = None
READY = False

MAX_STATE_CHARS = int(os.environ.get("JEV_MAX_STATE_CHARS", "20000"))
MAX_QUESTIONS = int(os.environ.get("JEV_MAX_QUESTIONS", "32"))
MAX_CANDIDATES = int(os.environ.get("JEV_MAX_CANDIDATES", "64"))

DEFAULT_RATE_LIMIT_RPS = 20.0
DEFAULT_RATE_LIMIT_BURST = 40.0


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


class TokenBucket:
    """In-process token bucket. One bucket per process, not per client.

    This is correct for a single instance. Behind multiple replicas each
    replica enforces its own limit, so the effective rate is N times the
    configured value; a shared store is required for a cluster-wide limit.
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


def create_app(
    checkpoint: str | None = None,
    *,
    dev_mode: bool | None = None,
    api_token: str | None = None,
    rate_limit_rps: float | None = None,
    rate_limit_burst: float | None = None,
) -> FastAPI:
    dev = _env_flag("JEV_DEV_MODE") if dev_mode is None else dev_mode
    token = (os.environ.get("JEV_API_TOKEN", "") if api_token is None else api_token).strip()
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
    bucket = TokenBucket(rps, burst)

    def require_token(authorization: str | None = Header(default=None)) -> None:
        if dev:
            return
        scheme, _, supplied = (authorization or "").partition(" ")
        supplied = supplied.strip()
        if scheme.lower() != "bearer" or not supplied:
            _log("auth_failed", reason="missing_bearer")
            raise HTTPException(401, "missing bearer token", headers={"WWW-Authenticate": "Bearer"})
        if not hmac.compare_digest(supplied, token):
            _log("auth_failed", reason="invalid_token")
            raise HTTPException(401, "invalid bearer token", headers={"WWW-Authenticate": "Bearer"})

    def rate_limit() -> None:
        allowed, retry_after = bucket.take()
        if not allowed:
            _log("rate_limited", retry_after_s=round(retry_after, 3))
            raise HTTPException(
                429,
                "rate limit exceeded",
                headers={"Retry-After": f"{max(retry_after, 0.001):.3f}"},
            )

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
        elif not token:
            raise RuntimeError(
                "JEV_API_TOKEN is not set; refusing to start with authentication disabled. "
                "Set JEV_API_TOKEN to a secret, or opt into local-only dev mode with "
                "JEV_DEV_MODE=1 (or `python -m jev_qwen.api --dev`)."
            )
        t0 = time.perf_counter()
        ENGINE = DecisionEngine(checkpoint=checkpoint or os.environ.get("JEV_CHECKPOINT"))
        READY = True
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
            _log("engine_shutdown")
            ENGINE = None

    app = FastAPI(title="jev-qwen decision API", version="0.2.0", lifespan=lifespan)

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
        card = ModelCard(
            id=ENGINE.model_version,
            head=str(ENGINE.meta.get("head_config", {}).get("type", "?")),
            layers=ENGINE.meta.get("num_layers"),
        )
        return {"data": [card.model_dump()]}

    @app.post("/v1/decide")
    def decide(
        req: DecisionRequest,
        request: Request,
        _: None = Depends(require_token),
        __: None = Depends(rate_limit),
    ):
        if not READY or ENGINE is None:
            raise HTTPException(503, "model not ready")
        rid = request.headers.get("x-request-id", uuid.uuid4().hex[:12])
        _check_size(req)
        t0 = time.perf_counter()
        try:
            resp = ENGINE.decide(req)
        except ValueError as e:
            _log("decide_invalid", request_id=rid, error_type=_error_type(e))
            raise HTTPException(422, str(e)) from e
        except torch.cuda.OutOfMemoryError as e:
            _log("decide_oom", request_id=rid, error_type=_error_type(e))
            raise HTTPException(
                503, "out of memory: retry with fewer questions or lower max_state_tokens"
            ) from e
        except Exception as e:  # noqa: BLE001
            _log("decide_error", request_id=rid, error_type=_error_type(e))
            raise HTTPException(500, "internal error") from e
        _log(
            "decide",
            request_id=rid,
            questions=len(req.questions),
            latency_s=round(time.perf_counter() - t0, 4),
        )
        return JSONResponse(resp.model_dump())

    @app.post("/v1/answer")
    def answer(
        body: dict[str, Any],
        _: None = Depends(require_token),
        __: None = Depends(rate_limit),
    ):
        """Compatibility adapter for the reference ecosystem's answer()
        protocol: {state, question:{type, instructions, criteria, label?}} ->
        {choice, probabilities} / {score, probabilities} / {noul}."""
        if not READY or ENGINE is None:
            raise HTTPException(503, "model not ready")
        try:
            from .data import question_to_choices

            q = body["question"]
            choices, _ = question_to_choices(q)
            if len(choices) > MAX_CANDIDATES:
                raise HTTPException(413, "too many candidates")
            req = DecisionRequest(
                state=body["state"],
                questions=[
                    {
                        "id": "q0",
                        "type": q["type"],
                        "question": q["instructions"],
                        "candidates": [{"id": str(i), "text": c} for i, c in enumerate(choices)],
                    }
                ],
            )
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
            _log("answer_invalid", error_type=_error_type(e))
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
