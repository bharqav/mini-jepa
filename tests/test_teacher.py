"""Adapter + teacher tests. Local-LM teacher test needs weights (gated)."""

import os

import pytest

REAL = os.environ.get("JEV_REAL_WEIGHTS") == "1"
needs_weights = pytest.mark.skipif(not REAL, reason="needs Qwen3-0.6B download")


def test_jsonl_adapter_schema():
    from jev_qwen.adapters import get

    recs = get("jsonl")("tests/fixtures/sample.jsonl")
    assert len(recs) == 3
    assert recs[0]["task_type"] == 0 and recs[0]["label"] == 0  # id -> index
    assert recs[1]["task_type"] == 2 and recs[1]["label"] == 2
    assert recs[2]["choices"] == ["No", "Yes"]
    from jev_qwen.data import validate_records

    rep = validate_records(recs)
    assert rep["malformed"] == 0 and rep["n"] == 3


def test_jsonl_adapter_rejects_bad_label():
    import tempfile

    from jev_qwen.adapters import get

    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        f.write(
            '{"state": "s", "type": "choice", "instructions": "q", '
            '"choices": ["a", "b"], "label": 7}\n'
        )
        path = f.name
    with pytest.raises(ValueError):
        get("jsonl")(path)


def test_unknown_adapter():
    from jev_qwen.adapters import get

    with pytest.raises(KeyError):
        get("does-not-exist")


def test_teacher_cache_and_validation(tmp_path):
    from jev_qwen.teacher import TeacherCache, label_records

    class StubTeacher:
        def __init__(self):
            self.calls = 0

        def label(self, state, instructions, choices):
            self.calls += 1
            return {"label": 0, "confidence": 0.9, "teacher": "stub"}

    recs = [
        {"state": "s", "instructions": "q", "choices": ["a", "b"], "label": 1, "example_id": "e0"}
    ]
    cache = TeacherCache(str(tmp_path / "c.jsonl"))
    rep = label_records(recs, StubTeacher(), cache)
    assert rep == {
        "n": 1,
        "cache_hits": 0,
        "labeled": 1,
        "agree": 0,
        "disagree": 1,
        "unlabeled_before": 0,
    }
    assert recs[0]["label"] == 1  # original kept, disagreement flagged
    assert recs[0]["label_disagreement"] is True
    # second pass hits cache: teacher not called again
    stub = StubTeacher()
    rep2 = label_records(recs, stub, cache)
    assert rep2["cache_hits"] == 1 and stub.calls == 0


def test_teacher_rejects_bad_confidence(tmp_path):
    from jev_qwen.teacher import TeacherCache, label_records

    class BadTeacher:
        def label(self, state, instructions, choices):
            return {"label": 0, "confidence": 42.0, "teacher": "bad"}

    with pytest.raises(ValueError):
        label_records(
            [
                {
                    "state": "s",
                    "instructions": "q",
                    "choices": ["a", "b"],
                    "label": None,
                    "example_id": "e",
                }
            ],
            BadTeacher(),
            TeacherCache(str(tmp_path / "c.jsonl")),
        )


@needs_weights
def test_local_lm_teacher_labels(tmp_path):
    from jev_qwen.teacher import LocalLMTeacher, TeacherCache, label_records

    t = LocalLMTeacher(num_layers=2)
    recs = [
        {
            "state": "server 503s",
            "instructions": "What first?",
            "choices": ["Inspect health", "Delete database"],
            "label": None,
            "example_id": "e0",
        }
    ]
    rep = label_records(recs, t, TeacherCache(str(tmp_path / "c.jsonl")))
    assert rep["labeled"] == 1
    assert recs[0]["teacher_label"] in (0, 1)
    assert 0.0 <= recs[0]["teacher_confidence"] <= 1.0
