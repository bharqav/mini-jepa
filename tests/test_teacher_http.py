"""External-teacher integration test against a real local HTTP endpoint.

Exercises paths a stub object cannot: real sockets, real HTTP status codes,
retry/backoff, and cost accounting on failure. No paid provider, no network.
"""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

ATTEMPTS = {}
LOCK = threading.Lock()


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        choices = body.get("choices", [])
        if not choices:
            self.send_response(422)
            self.end_headers()
            return
        if self.path.startswith("/fail"):
            self.send_response(500)
            self.end_headers()
            return
        if self.path.startswith("/flaky/"):
            n = int(self.path.split("/")[-1])
            key = body["state"][:40] + body["instructions"][:40]
            with LOCK:
                seen = ATTEMPTS.get(key, 0)
                ATTEMPTS[key] = seen + 1
            if seen < n:
                self.send_response(500)
                self.end_headers()
                return
        target = len(body.get("state", "")) + len(body.get("instructions", ""))
        best = min(range(len(choices)), key=lambda i: abs(len(choices[i]) - target))
        out = json.dumps({"label": best, "confidence": 0.77, "model": "stub-teacher-1"}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)


@pytest.fixture(scope="module")
def endpoint():
    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    time.sleep(0.2)
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def recs():
    return [
        {
            "state": "A server is returning repeated HTTP 503 errors.",
            "instructions": "What should the operator do first?",
            "choices": [
                "Inspect service health and dependencies.",
                "Delete the production database.",
                "Ignore the incident.",
            ],
            "label": None,
            "example_id": "r0",
        },
        {
            "state": "The change adds logging only.",
            "instructions": "Is this change safe to deploy?",
            "choices": ["No", "Yes"],
            "label": 1,
            "example_id": "r1",
        },
    ]


def test_external_teacher_end_to_end(endpoint, tmp_path):
    from jev_qwen.teacher import ExternalTeacher, TeacherCache, label_records

    cache = TeacherCache(str(tmp_path / "c.jsonl"))
    t = ExternalTeacher(url=f"{endpoint}/label")
    work = recs()
    rep = label_records(work, t, cache)
    assert rep["labeled"] == 2 and rep["cache_hits"] == 0
    assert t.calls == 2 and t.failures == 0 and t.chars_sent > 0
    for r in work:
        assert 0.0 <= r["teacher_confidence"] <= 1.0
        assert r["teacher"] == "external:stub-teacher-1"
        assert 0 <= r["teacher_label"] < len(r["choices"])
    t2 = ExternalTeacher(url=f"{endpoint}/label")
    rep2 = label_records(recs(), t2, TeacherCache(str(tmp_path / "c.jsonl")))
    assert rep2["cache_hits"] == 2 and t2.calls == 0, "cache must make reruns free"


def test_external_teacher_retries_transient_failures(endpoint, tmp_path):
    from jev_qwen.teacher import ExternalTeacher, TeacherCache, label_records

    t = ExternalTeacher(url=f"{endpoint}/flaky/2")
    rep = label_records(recs(), t, TeacherCache(str(tmp_path / "f.jsonl")))
    assert rep["labeled"] == 2
    assert t.failures > 0, "retry path never exercised"


def test_external_teacher_raises_and_accounts_failures(endpoint, tmp_path):
    from jev_qwen.teacher import ExternalTeacher, TeacherCache, label_records

    t = ExternalTeacher(url=f"{endpoint}/fail")
    with pytest.raises(RuntimeError):
        label_records(recs()[:1], t, TeacherCache(str(tmp_path / "x.jsonl")))
    assert t.failures == 3, f"expected 3 recorded failures, got {t.failures}"
    assert t.calls == 0, "failed calls must not count as successful calls"


def test_external_teacher_requires_url(monkeypatch):
    from jev_qwen.teacher import ExternalTeacher

    monkeypatch.delenv("JEV_TEACHER_URL", raising=False)
    with pytest.raises(ValueError):
        ExternalTeacher()
