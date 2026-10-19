"""API contract + hardening tests (stub engine; no weights needed)."""

import json

import pytest
from fastapi.testclient import TestClient

import jev_qwen.api as api
from jev_qwen.schema import CandidateScore, DecisionResponse, QuestionResponse


class StubEngine:
    model_version = "stub-0"
    meta = {}
    device = "cpu"

    def decide(self, req):
        return DecisionResponse(
            results=[
                QuestionResponse(
                    id=q.id,
                    type=q.type,
                    scores=[
                        CandidateScore(id=c.id, score=1.0 / len(q.candidates)) for c in q.candidates
                    ],
                    selected=q.candidates[0].id if q.type != "score" else None,
                    confidence=1.0 / len(q.candidates),
                )
                for q in req.questions
            ],
            model_version="stub-0",
        )


def client(ready=True):
    api.ENGINE = StubEngine()
    api.READY = ready
    return TestClient(api.create_app())


GOOD = {
    "state": "server 503s",
    "questions": [
        {
            "id": "q0",
            "type": "choice",
            "question": "What first?",
            "candidates": [{"id": "a", "text": "Inspect"}, {"id": "b", "text": "Delete"}],
        }
    ],
}


def test_health_is_liveness_and_always_ok():
    c = client(ready=False)
    assert c.get("/health").json()["status"] == "ok"


def test_ready_gates_traffic():
    assert client(ready=False).get("/ready").status_code == 503
    assert client(ready=True).get("/ready").json()["status"] == "ready"


def test_decide_rejected_when_not_ready():
    c = client(ready=False)
    assert c.post("/v1/decide", json=GOOD).status_code == 503


def test_models_and_valid_decide():
    c = client()
    assert c.get("/v1/models").json()["data"][0]["id"] == "stub-0"
    r = c.post("/v1/decide", json=GOOD)
    assert r.status_code == 200
    body = r.json()
    assert abs(sum(s["score"] for s in body["results"][0]["scores"]) - 1.0) < 1e-6


@pytest.mark.parametrize(
    "bad,why",
    [
        ({"state": "x", "questions": []}, "no questions"),
        (
            {
                "state": "x",
                "questions": [
                    {
                        "id": "q",
                        "type": "choice",
                        "question": "?",
                        "candidates": [{"id": "a", "text": "only"}],
                    }
                ],
            },
            "1 candidate",
        ),
        (
            {
                "state": "x",
                "questions": [
                    {
                        "id": "q",
                        "type": "choice",
                        "question": "?",
                        "candidates": [{"id": "a", "text": "x"}, {"id": "a", "text": "y"}],
                    }
                ],
            },
            "dup ids",
        ),
        (
            {
                "state": "x",
                "questions": [
                    {
                        "id": "q",
                        "type": "noul",
                        "question": "?",
                        "candidates": [{"id": "a", "text": "x"}],
                    }
                ],
            },
            "noul needs 2",
        ),
    ],
)
def test_invalid_requests_are_422_never_faked(bad, why):
    assert client().post("/v1/decide", json=bad).status_code == 422, why


def test_oversized_requests_are_413():
    c = client()
    big = {"state": "s" * (api.MAX_STATE_CHARS + 1), "questions": GOOD["questions"]}
    assert c.post("/v1/decide", json=big).status_code == 413
    many = {
        "state": "s",
        "questions": [
            {
                "id": f"q{i}",
                "type": "choice",
                "question": "?",
                "candidates": [{"id": "a", "text": "x"}, {"id": "b", "text": "y"}],
            }
            for i in range(api.MAX_QUESTIONS + 1)
        ],
    }
    assert c.post("/v1/decide", json=many).status_code == 413


def test_answer_adapter_contract():
    c = client()
    r = c.post(
        "/v1/answer",
        json={
            "state": "s",
            "question": {"type": "noul", "instructions": "Safe?", "criteria": {}, "label": 1},
        },
    )
    assert r.status_code == 200 and "noul" in r.json()
    bad = c.post("/v1/answer", json={"state": "s", "question": {"type": "wat"}})
    assert bad.status_code == 422


def test_demo_page_served():
    r = client().get("/demo")
    assert r.status_code == 200 and "jev-qwen decision demo" in r.text


def test_structured_logs_are_json(caplog):
    import logging

    caplog.set_level(logging.INFO)
    c = client()
    c.post("/v1/decide", json=GOOD)
    recs = [r for r in caplog.records if r.message.startswith("{")]
    assert recs, "expected structured JSON logs"
    payload = json.loads(recs[-1].message)
    assert payload["event"] == "decide"
    assert "latency_s" in payload and "questions" in payload
    # secrets must never appear in logs
    assert "Authorization" not in recs[-1].message
