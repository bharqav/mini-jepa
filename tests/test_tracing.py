"""Tracing must be opt-in and must never break startup."""

from jev_qwen.tracing import configure_tracing


def test_tracing_is_off_by_default(monkeypatch):
    monkeypatch.delenv("JEV_OTEL_ENABLED", raising=False)
    assert configure_tracing(object()) is False


def test_tracing_flag_accepts_truthy_values(monkeypatch):
    monkeypatch.setenv("JEV_OTEL_ENABLED", "0")
    assert configure_tracing(object()) is False
