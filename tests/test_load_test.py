"""Pure helpers in scripts/load_test.py (no server needed)."""

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "load_test.py"
_spec = importlib.util.spec_from_file_location("load_test_script", _SCRIPT)
assert _spec and _spec.loader
load_test = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = load_test
_spec.loader.exec_module(load_test)


def test_percentile_handles_empty_and_bounds():
    assert load_test.percentile([], 95) == 0.0
    values = [1.0, 2.0, 3.0, 4.0]
    assert load_test.percentile(values, 50) == 2.0
    assert load_test.percentile(values, 99) == 4.0


def test_result_summary_counts_statuses_and_errors():
    result = load_test.Result()
    result.add(200, 0.01)
    result.add(429, 0.02)
    result.add(500, 0.03)
    result.add(None, 0.04, "ConnectError")
    summary = result.summary(1.0)
    assert summary["requests"] == 4
    assert summary["statuses"] == {"200": 1, "429": 1, "500": 1}
    assert summary["errors"] == {"ConnectError": 1}
    assert summary["error_rate"] == pytest.approx(0.5)


def test_payload_has_choice_shape():
    body = load_test.payload("state", "question", [{"id": "a", "text": "A"}])
    question = body["questions"][0]
    assert question["type"] == "choice"
    assert question["candidates"] == [{"id": "a", "text": "A"}]
