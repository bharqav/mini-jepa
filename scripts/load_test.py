"""Small load/soak harness for the decision API.

This is deliberately dependency-light (``httpx`` only) so it can run against a
staging deployment without installing a load-testing framework::

    JEV_API_TOKEN=... python scripts/load_test.py \
        --url http://127.0.0.1:8000 --concurrency 8 --duration 60

The script prints latency percentiles and status counts, and exits non-zero on
connection errors or 5xx responses. 429 responses are reported separately
because they are the expected backpressure signal, not a failure.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
from collections import Counter
from dataclasses import dataclass, field

import httpx

DEFAULT_STATE = "A server is returning repeated HTTP 503 errors to customers."
DEFAULT_QUESTION = "What should the operator do first?"
DEFAULT_CANDIDATES = [
    {"id": "inspect", "text": "Inspect service health and dependencies."},
    {"id": "delete", "text": "Delete the production database."},
    {"id": "ignore", "text": "Ignore the incident."},
    {"id": "restart", "text": "Restart every service immediately."},
]


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(p / 100.0 * len(ordered)) - 1))
    return ordered[index]


@dataclass
class Result:
    latencies: list[float] = field(default_factory=list)
    statuses: Counter[int] = field(default_factory=Counter)
    errors: Counter[str] = field(default_factory=Counter)

    def add(self, status: int | None, latency: float, error: str | None = None) -> None:
        self.latencies.append(latency)
        if status is not None:
            self.statuses[status] += 1
        if error:
            self.errors[error] += 1

    def summary(self, elapsed: float) -> dict[str, object]:
        total = len(self.latencies)
        return {
            "requests": total,
            "throughput_rps": round(total / elapsed, 2) if elapsed > 0 else 0.0,
            "latency_ms": {
                "p50": round(percentile(self.latencies, 50) * 1000, 2),
                "p95": round(percentile(self.latencies, 95) * 1000, 2),
                "p99": round(percentile(self.latencies, 99) * 1000, 2),
                "max": round(max(self.latencies, default=0.0) * 1000, 2),
            },
            "statuses": {str(k): v for k, v in sorted(self.statuses.items())},
            "errors": dict(self.errors),
            "error_rate": round(
                (sum(v for k, v in self.statuses.items() if k >= 500) + sum(self.errors.values()))
                / total,
                4,
            )
            if total
            else 0.0,
        }


def payload(state: str, question: str, candidates: list[dict[str, str]]) -> dict[str, object]:
    return {
        "state": state,
        "questions": [
            {
                "id": "q0",
                "type": "choice",
                "question": question,
                "candidates": candidates,
            }
        ],
    }


async def worker(
    client: httpx.AsyncClient,
    *,
    url: str,
    body: dict[str, object],
    result: Result,
    deadline: float,
    interval: float,
) -> None:
    next_at = time.perf_counter()
    while time.perf_counter() < deadline:
        if interval > 0:
            now = time.perf_counter()
            if now < next_at:
                await asyncio.sleep(next_at - now)
            next_at = max(next_at + interval, time.perf_counter())
        started = time.perf_counter()
        try:
            response = await client.post(url, json=body)
            result.add(response.status_code, time.perf_counter() - started)
        except httpx.HTTPError as e:
            result.add(None, time.perf_counter() - started, type(e).__name__)


async def run(args: argparse.Namespace) -> dict[str, object]:
    body = payload(args.state, args.question, DEFAULT_CANDIDATES)
    headers = {"Authorization": f"Bearer {args.token}"} if args.token else {}
    result = Result()
    timeout = httpx.Timeout(args.timeout)
    limits = httpx.Limits(
        max_connections=args.concurrency, max_keepalive_connections=args.concurrency
    )
    interval = 1.0 / args.rps if args.rps > 0 else 0.0
    async with httpx.AsyncClient(headers=headers, timeout=timeout, limits=limits) as client:
        started = time.perf_counter()
        deadline = started + args.duration
        await asyncio.gather(
            *[
                worker(
                    client,
                    url=args.url,
                    body=body,
                    result=result,
                    deadline=deadline,
                    interval=interval,
                )
                for _ in range(args.concurrency)
            ]
        )
        elapsed = time.perf_counter() - started
    return result.summary(elapsed)


def main() -> int:
    ap = argparse.ArgumentParser(description="Load/soak the decision API")
    ap.add_argument("--url", default="http://127.0.0.1:8000/v1/decide")
    ap.add_argument("--token", default="")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--rps", type=float, default=0.0, help="per-worker rate cap; 0 = unthrottled")
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--state", default=DEFAULT_STATE)
    ap.add_argument("--question", default=DEFAULT_QUESTION)
    args = ap.parse_args()
    if args.concurrency < 1:
        ap.error("--concurrency must be >= 1")
    if args.duration <= 0:
        ap.error("--duration must be > 0")
    summary = asyncio.run(run(args))
    print(json.dumps(summary, indent=2))
    statuses = summary["statuses"]
    errors = summary["errors"]
    assert isinstance(statuses, dict) and isinstance(errors, dict)
    failed = any(int(code) >= 500 for code in statuses) or bool(errors)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
