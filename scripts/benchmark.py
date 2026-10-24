"""Benchmark decision-engine load time, inference latency and throughput.

Usage::

    python scripts/benchmark.py --checkpoint CKPT --batch-sizes 1 4 8 --repeats 20

The model-load and first-request timings are reported separately. The p95 uses
nearest-rank quantiles, so small sample counts remain well-defined.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def percentile(values: Sequence[float], p: float) -> float:
    """Return the nearest-rank percentile; reject undefined requests."""
    if not values:
        raise ValueError("percentile requires at least one sample")
    if not 0 <= p <= 100:
        raise ValueError("percentile must be between 0 and 100")
    ordered = sorted(values)
    index = max(0, math.ceil(p / 100 * len(ordered)) - 1)
    return ordered[index]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark the decision engine")
    parser.add_argument(
        "--checkpoint", required=True, help="trained decision checkpoint to measure"
    )
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 4, 8])
    parser.add_argument("--repeats", type=int, default=20)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.repeats < 1:
        parser.error("--repeats must be >= 1")
    if any(size < 1 for size in args.batch_sizes):
        parser.error("all --batch-sizes must be >= 1")
    return args


def _synchronize(device: Any) -> None:
    import torch

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark(args: argparse.Namespace) -> dict[str, Any]:
    # Resolve the package from a source checkout when the project is not installed.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from jev_qwen.infer import DecisionEngine
    from jev_qwen.schema import DecisionRequest

    started = time.perf_counter()
    engine = DecisionEngine(checkpoint=args.checkpoint)
    _synchronize(engine.device)
    model_load_s = time.perf_counter() - started

    base_question = {
        "id": "q0",
        "type": "choice",
        "question": "What first?",
        "candidates": [
            {"id": "a", "text": "Inspect health"},
            {"id": "b", "text": "Delete database"},
            {"id": "c", "text": "Ignore it"},
        ],
    }
    state = "A server is returning repeated HTTP 503 errors."
    first_request = DecisionRequest(state=state, questions=[base_question])
    started = time.perf_counter()
    engine.decide(first_request)
    _synchronize(engine.device)
    first_request_s = time.perf_counter() - started

    report: dict[str, Any] = {
        "device": str(engine.device),
        "model": engine.model_version,
        "model_load_s": round(model_load_s, 4),
        "first_request_latency_s": round(first_request_s, 4),
        "repeats": args.repeats,
    }
    for batch_size in args.batch_sizes:
        request = DecisionRequest(
            state=state,
            questions=[dict(base_question, id=f"q{i}") for i in range(batch_size)],
        )
        engine.decide(request)  # warmup outside the measured sample
        _synchronize(engine.device)
        latencies = []
        for _ in range(args.repeats):
            started = time.perf_counter()
            engine.decide(request)
            _synchronize(engine.device)
            latencies.append(time.perf_counter() - started)
        median_s = statistics.median(latencies)
        report[f"batch_{batch_size}"] = {
            "median_s": round(median_s, 6),
            "p95_s": round(percentile(latencies, 95), 6),
            "throughput_questions_per_s": round(batch_size / median_s, 2),
        }

    import torch

    if engine.device.type == "cuda":
        report["peak_vram_mb"] = round(torch.cuda.max_memory_allocated(engine.device) / 2**20, 1)
        report["reserved_vram_mb"] = round(torch.cuda.max_memory_reserved(engine.device) / 2**20, 1)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    print(json.dumps(benchmark(args), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
