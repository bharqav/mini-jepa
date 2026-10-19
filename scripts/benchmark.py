"""Benchmark: warm/cold latency (median/p95, CUDA-synced), throughput, peak VRAM.

Usage: python scripts/benchmark.py --checkpoint CKPT [--batch-sizes 1 4 8]
"""

import argparse
import json
import statistics
import sys
import time

import torch

sys.path.insert(0, "src")
from jev_qwen.infer import DecisionEngine
from jev_qwen.schema import DecisionRequest

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", default=None)
ap.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 4, 8])
ap.add_argument("--repeats", type=int, default=20)
args = ap.parse_args()

engine = DecisionEngine(checkpoint=args.checkpoint)
base_q = {
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
report = {"device": str(engine.device), "model": engine.model_version, "samples": args.repeats}
# cold start (fresh sync, single run)
t0 = time.perf_counter()
engine.decide(DecisionRequest(state=state, questions=[base_q]))
if torch.cuda.is_available():
    torch.cuda.synchronize()
report["cold_latency_s"] = time.perf_counter() - t0
for bs in args.batch_sizes:
    req = DecisionRequest(state=state, questions=[dict(base_q, id=f"q{i}") for i in range(bs)])
    engine.decide(req)  # warmup
    lats = []
    for _ in range(args.repeats):
        t0 = time.perf_counter()
        engine.decide(req)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        lats.append(time.perf_counter() - t0)
    lats.sort()
    report[f"batch_{bs}"] = {
        "median_s": statistics.median(lats),
        "p95_s": lats[int(0.95 * len(lats)) - 1],
        "throughput_qps": bs / statistics.median(lats),
    }
if torch.cuda.is_available():
    report["peak_vram_mb"] = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 1)
    report["reserved_vram_mb"] = round(torch.cuda.max_memory_reserved() / (1024 * 1024), 1)
print(json.dumps(report, indent=2))
