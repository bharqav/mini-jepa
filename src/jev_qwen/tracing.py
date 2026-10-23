"""Optional OpenTelemetry tracing.

Tracing is off by default and its packages live in the ``otel`` extra. When
``JEV_OTEL_ENABLED=1`` the app is instrumented and spans are exported to
``JEV_OTEL_ENDPOINT`` (OTLP/HTTP) if set. A missing package logs a warning and
leaves the service running; tracing must never be a startup dependency.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

log = logging.getLogger("jev_qwen.tracing")


def _enabled() -> bool:
    return os.environ.get("JEV_OTEL_ENABLED", "").strip().lower() in {"1", "true", "yes", "on"}


def configure_tracing(app: Any) -> bool:
    if not _enabled():
        return False
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        log.warning(
            json.dumps(
                {
                    "ts": round(time.time(), 3),
                    "event": "tracing_unavailable",
                    "detail": "install the otel extra to enable JEV_OTEL_ENABLED",
                }
            )
        )
        return False
    service_name = os.environ.get("JEV_OTEL_SERVICE_NAME", "jev-qwen")
    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    endpoint = os.environ.get("JEV_OTEL_ENDPOINT", "").strip()
    if endpoint:
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
    trace.set_tracer_provider(provider)
    FastAPIInstrumentor.instrument_app(app)
    return True
