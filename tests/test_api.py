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

    def __init__(self, checkpoint=None):
        self.checkpoint = checkpoint

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
    return TestClient(api.create_app(dev_mode=True, rate_limit_rps=0))


def secure_client(token="s3cret-token"):
    api.ENGINE = StubEngine()
    api.READY = True
    return TestClient(api.create_app(api_token=token, rate_limit_rps=0))


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


def test_demo_page_served_in_dev_mode():
    r = client().get("/demo")
    assert r.status_code == 200 and "jev-qwen decision demo" in r.text


def test_demo_is_hidden_when_authentication_is_enabled():
    assert secure_client().get("/demo").status_code == 404


def test_structured_logs_are_json(caplog):
    import logging

    caplog.set_level(logging.INFO)
    c = client()
    c.post("/v1/decide", json=GOOD)
    recs = [r for r in caplog.records if r.message.startswith("{")]
    assert recs, "expected structured JSON logs"
    payloads = [json.loads(r.message) for r in recs]
    decide = next(p for p in payloads if p["event"] == "decide")
    assert "latency_s" in decide and "questions" in decide
    access = next(p for p in payloads if p["event"] == "request")
    assert access["status"] == 200 and access["path"] == "/v1/decide"
    # secrets must never appear in logs
    assert all("Authorization" not in r.message for r in recs)


def test_missing_or_wrong_bearer_token_is_401():
    c = secure_client()
    assert c.post("/v1/decide", json=GOOD).status_code == 401
    assert (
        c.post("/v1/decide", json=GOOD, headers={"Authorization": "Bearer wrong"}).status_code
        == 401
    )
    assert (
        c.post("/v1/answer", json={"state": "s", "question": {"type": "noul"}}).status_code == 401
    )


def test_valid_bearer_token_is_accepted():
    c = secure_client()
    r = c.post("/v1/decide", json=GOOD, headers={"Authorization": "Bearer s3cret-token"})
    assert r.status_code == 200
    # public operational endpoints stay reachable for probes and model cards
    assert c.get("/health").status_code == 200
    assert c.get("/ready").status_code == 200
    assert c.get("/v1/models").status_code == 200


def test_service_refuses_to_start_without_token_or_dev_mode(monkeypatch):
    monkeypatch.setattr(api, "DecisionEngine", StubEngine)
    app = api.create_app(api_token="", dev_mode=False, rate_limit_rps=0)
    with pytest.raises(RuntimeError, match="JEV_API_TOKEN"), TestClient(app):
        pass


def test_dev_mode_logs_loud_warning(monkeypatch, caplog):
    import logging

    caplog.set_level(logging.INFO)
    monkeypatch.setattr(api, "DecisionEngine", StubEngine)
    app = api.create_app(dev_mode=True, rate_limit_rps=0)
    with TestClient(app) as c:
        assert c.post("/v1/decide", json=GOOD).status_code == 200
    messages = [r.message for r in caplog.records]
    assert any("auth_disabled" in m and "never expose" in m for m in messages)


def test_rate_limit_returns_429_with_retry_after():
    api.ENGINE = StubEngine()
    api.READY = True
    c = TestClient(api.create_app(dev_mode=True, rate_limit_rps=1, rate_limit_burst=1))
    assert c.post("/v1/decide", json=GOOD).status_code == 200
    r = c.post("/v1/decide", json=GOOD)
    assert r.status_code == 429
    assert float(r.headers["Retry-After"]) > 0


def test_rate_limit_is_per_client_not_global():
    api.ENGINE = StubEngine()
    api.READY = True
    app = api.create_app(dev_mode=True, rate_limit_rps=1, rate_limit_burst=1)
    a = TestClient(app, client=("198.51.100.10", 1000))
    b = TestClient(app, client=("198.51.100.11", 1000))
    assert a.post("/v1/decide", json=GOOD).status_code == 200
    assert a.post("/v1/decide", json=GOOD).status_code == 429
    assert b.post("/v1/decide", json=GOOD).status_code == 200


def test_rate_limit_bucket_registry_is_bounded():
    from jev_qwen.api import BucketRegistry

    reg = BucketRegistry(rate=1, burst=1, max_clients=2)
    for client in ("a", "b", "c"):
        reg.take(client)
    assert len(reg._buckets) == 2
    assert "a" not in reg._buckets


def test_logs_never_contain_request_state_or_tokens(caplog):
    import logging

    caplog.set_level(logging.INFO)
    canary = "CANARY-STATE-9f3c"
    c = secure_client()
    c.post("/v1/decide", json={**GOOD, "state": canary}, headers={"Authorization": "Bearer wrong"})
    c.post(
        "/v1/decide",
        json={**GOOD, "state": canary},
        headers={"Authorization": "Bearer s3cret-token"},
    )
    for rec in caplog.records:
        assert canary not in rec.message
        assert "s3cret-token" not in rec.message


def test_exception_messages_are_not_logged(caplog):
    import logging

    caplog.set_level(logging.INFO)

    class LeakyEngine(StubEngine):
        def decide(self, req):
            raise ValueError(f"rejected state {req.state}")

    api.ENGINE = LeakyEngine()
    api.READY = True
    canary = "CANARY-EXC-4a1b"
    c = TestClient(api.create_app(dev_mode=True, rate_limit_rps=0))
    assert c.post("/v1/decide", json={**GOOD, "state": canary}).status_code == 422
    for rec in caplog.records:
        assert canary not in rec.message


def scoped_client(keys, **kwargs):
    api.ENGINE = StubEngine()
    api.READY = True
    return TestClient(api.create_app(api_keys=keys, rate_limit_rps=0, **kwargs))


def test_scoped_keys_are_enforced_per_endpoint():
    c = scoped_client({"decide-only": {"token": "decide-token", "scopes": ["decide"]}})
    assert (
        c.post(
            "/v1/decide", json=GOOD, headers={"Authorization": "Bearer decide-token"}
        ).status_code
        == 200
    )
    r = c.post(
        "/v1/answer",
        json={"state": "s", "question": {"type": "noul"}},
        headers={"Authorization": "Bearer decide-token"},
    )
    assert r.status_code == 403


def test_multiple_keys_and_rotation():
    c = scoped_client({"old": "old-token", "new": "new-token"})
    assert (
        c.post("/v1/decide", json=GOOD, headers={"Authorization": "Bearer old-token"}).status_code
        == 200
    )
    assert (
        c.post("/v1/decide", json=GOOD, headers={"Authorization": "Bearer new-token"}).status_code
        == 200
    )
    assert (
        c.post("/v1/decide", json=GOOD, headers={"Authorization": "Bearer gone-token"}).status_code
        == 401
    )


def test_per_key_quota_is_independent_of_global_limit():
    api.ENGINE = StubEngine()
    api.READY = True
    c = TestClient(
        api.create_app(
            api_keys={"limited": {"token": "limited-token", "rps": 1, "burst": 1}},
            rate_limit_rps=0,
        )
    )
    h = {"Authorization": "Bearer limited-token"}
    assert c.post("/v1/decide", json=GOOD, headers=h).status_code == 200
    assert c.post("/v1/decide", json=GOOD, headers=h).status_code == 429


def test_forwarded_for_is_used_only_from_a_trusted_proxy():
    api.ENGINE = StubEngine()
    api.READY = True
    app = api.create_app(
        dev_mode=True,
        rate_limit_rps=1,
        rate_limit_burst=1,
        trusted_proxies="10.0.0.1",
    )
    c = TestClient(app, client=("10.0.0.1", 1000))
    assert (
        c.post("/v1/decide", json=GOOD, headers={"X-Forwarded-For": "198.51.100.7"}).status_code
        == 200
    )
    assert (
        c.post("/v1/decide", json=GOOD, headers={"X-Forwarded-For": "198.51.100.7"}).status_code
        == 429
    )
    assert (
        c.post("/v1/decide", json=GOOD, headers={"X-Forwarded-For": "198.51.100.8"}).status_code
        == 200
    )


def test_request_id_is_echoed_or_generated():
    c = client()
    r = c.get("/health", headers={"X-Request-ID": "trace-abc_123"})
    assert r.headers["X-Request-ID"] == "trace-abc_123"
    r2 = c.get("/health", headers={"X-Request-ID": "bad id with spaces"})
    assert r2.headers["X-Request-ID"] != "bad id with spaces"
    assert len(r2.headers["X-Request-ID"]) == 16


def test_body_size_limit_is_enforced_before_parsing():
    api.ENGINE = StubEngine()
    api.READY = True
    c = TestClient(api.create_app(dev_mode=True, rate_limit_rps=0, max_body_bytes=64))
    r = c.post("/v1/decide", json=GOOD)
    assert r.status_code == 413
    assert "X-Request-ID" in r.headers


def test_metrics_endpoint_reports_requests():
    pytest.importorskip("prometheus_client")
    c = client()
    c.post("/v1/decide", json=GOOD)
    r = c.get("/metrics")
    assert r.status_code == 200
    assert "jev_requests_total" in r.text
    assert 'path="/v1/decide"' in r.text


def test_metrics_can_be_disabled():
    api.ENGINE = StubEngine()
    api.READY = True
    c = TestClient(api.create_app(dev_mode=True, rate_limit_rps=0, metrics_enabled=False))
    assert c.get("/metrics").status_code == 404


def test_concurrency_limit_rejects_excess_work():
    import threading

    class BlockingEngine(StubEngine):
        def __init__(self):
            super().__init__()
            self.entered = threading.Event()
            self.release = threading.Event()

        def decide(self, req):
            self.entered.set()
            assert self.release.wait(5)
            return super().decide(req)

    engine = BlockingEngine()
    api.ENGINE = engine
    api.READY = True
    app = api.create_app(dev_mode=True, rate_limit_rps=0, max_concurrency=1)
    first = TestClient(app)
    second = TestClient(app)
    result = {}

    def call():
        result["response"] = first.post("/v1/decide", json=GOOD)

    thread = threading.Thread(target=call)
    thread.start()
    try:
        assert engine.entered.wait(5)
        r = second.post("/v1/decide", json=GOOD)
        assert r.status_code == 503
        assert r.headers["Retry-After"] == "1"
    finally:
        engine.release.set()
        thread.join(5)
    assert result["response"].status_code == 200
