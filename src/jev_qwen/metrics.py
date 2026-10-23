"""Prometheus metrics for the inference service.

Metrics are optional at import time so the training-only install does not need
``prometheus-client``. The ``serve`` extra includes it; if it is missing the
``/metrics`` endpoint returns 503 instead of breaking the API.
"""

from __future__ import annotations

from typing import Any

try:  # pragma: no cover - exercised by the dependency-present path in CI
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
        generate_latest,
    )

    PROMETHEUS_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"
    PROMETHEUS_AVAILABLE = False

KNOWN_PATHS = frozenset(
    {"/health", "/ready", "/v1/models", "/v1/decide", "/v1/answer", "/demo", "/metrics"}
)


def metric_path(path: str) -> str:
    """Bound label cardinality for unknown URLs."""
    return path if path in KNOWN_PATHS else "other"


class Metrics:
    def __init__(self, enabled: bool = True):
        self.enabled = bool(enabled) and PROMETHEUS_AVAILABLE
        self.registry: Any = None
        if not self.enabled:
            return
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "jev_requests_total",
            "HTTP requests by method, route and status.",
            ["method", "path", "status"],
            registry=self.registry,
        )
        self.duration = Histogram(
            "jev_request_duration_seconds",
            "HTTP request latency in seconds.",
            ["method", "path"],
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60),
            registry=self.registry,
        )
        self.in_flight = Gauge(
            "jev_in_flight_requests",
            "Requests currently being served.",
            registry=self.registry,
        )
        self.rate_limited = Counter(
            "jev_rate_limited_total",
            "Requests rejected by the rate limiter.",
            registry=self.registry,
        )
        self.auth_failures = Counter(
            "jev_auth_failures_total",
            "Authentication failures by reason.",
            ["reason"],
            registry=self.registry,
        )
        self.inference_errors = Counter(
            "jev_inference_errors_total",
            "Inference failures by exception type.",
            ["error_type"],
            registry=self.registry,
        )
        self.ready = Gauge(
            "jev_model_ready",
            "1 when the model is loaded and ready, 0 otherwise.",
            registry=self.registry,
        )

    def request_started(self) -> None:
        if self.enabled:
            self.in_flight.inc()

    def request_finished(self, method: str, path: str, status: int, seconds: float) -> None:
        if not self.enabled:
            return
        path = metric_path(path)
        self.in_flight.dec()
        self.requests.labels(method=method, path=path, status=str(status)).inc()
        self.duration.labels(method=method, path=path).observe(max(seconds, 0.0))

    def auth_failed(self, reason: str) -> None:
        if self.enabled:
            self.auth_failures.labels(reason=reason).inc()

    def limited(self) -> None:
        if self.enabled:
            self.rate_limited.inc()

    def inference_error(self, error_type: str) -> None:
        if self.enabled:
            self.inference_errors.labels(error_type=error_type).inc()

    def set_ready(self, value: bool) -> None:
        if self.enabled:
            self.ready.set(1 if value else 0)

    def render(self) -> tuple[bytes, str] | None:
        if not self.enabled:
            return None
        return generate_latest(self.registry), CONTENT_TYPE_LATEST
