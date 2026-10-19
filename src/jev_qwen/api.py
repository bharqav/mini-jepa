"""Local FastAPI service. Model loads ONCE at startup, never per request.

Production behaviours:
- readiness/liveness split (/health vs /ready) so a load balancer does not
  send traffic before weights are resident
- hard request-size limits enforced before any model work
- typed errors (422 validation, 413 too large, 503 not ready, 500 unexpected)
- structured JSON logs with latency; no secrets ever logged
- graceful shutdown releases the model
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

import torch
from fastapi import FastAPI, HTTPException, Request
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


def _log(event: str, **fields: Any) -> None:
    rec = {"ts": round(time.time(), 3), "event": event, **fields}
    log.info(json.dumps(rec, default=str))


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


def create_app(checkpoint: str | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        global ENGINE, READY
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
    def decide(req: DecisionRequest, request: Request):
        if not READY or ENGINE is None:
            raise HTTPException(503, "model not ready")
        rid = request.headers.get("x-request-id", uuid.uuid4().hex[:12])
        _check_size(req)
        t0 = time.perf_counter()
        try:
            resp = ENGINE.decide(req)
        except ValueError as e:
            _log("decide_invalid", request_id=rid, error=str(e)[:200])
            raise HTTPException(422, str(e)) from e
        except torch.cuda.OutOfMemoryError as e:
            _log("decide_oom", request_id=rid, error=str(e)[:200])
            raise HTTPException(
                503, "out of memory: retry with fewer questions or lower max_state_tokens"
            ) from e
        except Exception as e:  # noqa: BLE001
            _log("decide_error", request_id=rid, error=str(e)[:200])
            raise HTTPException(500, "internal error") from e
        _log(
            "decide",
            request_id=rid,
            questions=len(req.questions),
            latency_s=round(time.perf_counter() - t0, 4),
        )
        return JSONResponse(resp.model_dump())

    @app.post("/v1/answer")
    def answer(body: dict[str, Any]):
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
            raise HTTPException(422, f"invalid answer payload: {e}") from e

    @app.get("/demo", response_class=HTMLResponse)
    def demo():
        """Minimal local demo page (no build step, no dependencies)."""
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
