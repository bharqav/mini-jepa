"""Uncertainty estimation: bootstrap confidence intervals over evaluation
rows, and cross-seed aggregation.

Motivation: single-seed deltas are not trustworthy. Two sources of noise are
distinguished here:
  - sampling noise (bootstrap over eval rows) -> CI for a metric
  - training noise (variance across seeds) -> mean/std/half-width across runs
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from typing import Any


def bootstrap_ci(
    values: Sequence[float],
    metric_fn: Callable[[Sequence[float]], float],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, float]:
    """Percentile bootstrap CI of metric_fn over resampled rows."""
    n = len(values)
    if n == 0:
        return {"point": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    point = metric_fn(values)
    rng = random.Random(seed)
    stats = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        stats.append(metric_fn(sample))
    stats.sort()
    lo = stats[int((alpha / 2) * n_boot)]
    hi = stats[min(int((1 - alpha / 2) * n_boot), n_boot - 1)]
    return {"point": point, "lo": lo, "hi": hi, "n": n}


def accuracy_ci(
    correct: Sequence[bool], n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
) -> dict[str, float]:
    return bootstrap_ci(
        [1.0 if c else 0.0 for c in correct], lambda v: sum(v) / len(v), n_boot, alpha, seed
    )


def mean_ci(
    values: Sequence[float], n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
) -> dict[str, float]:
    return bootstrap_ci(list(values), lambda v: sum(v) / len(v), n_boot, alpha, seed)


def paired_delta_ci(
    correct_a: Sequence[bool],
    correct_b: Sequence[bool],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, float]:
    """CI for accuracy(A) - accuracy(B) resampling ROWS (paired). This is the
    right test for 'system A beats system B on the same questions'."""
    assert len(correct_a) == len(correct_b)
    d = [1.0 if a else 0.0 for a in correct_a], [1.0 if b else 0.0 for b in correct_b]
    n = len(d[0])
    if n == 0:
        return {"point": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}

    def delta(vals):
        a = [x[0] for x in vals]
        b = [x[1] for x in vals]
        return sum(a) / len(a) - sum(b) / len(b)

    pairs = list(zip(d[0], d[1]))
    rng = random.Random(seed)
    stats = []
    for _ in range(n_boot):
        sample = [pairs[rng.randrange(n)] for _ in range(n)]
        stats.append(delta(sample))
    stats.sort()
    return {
        "point": delta(pairs),
        "lo": stats[int((alpha / 2) * n_boot)],
        "hi": stats[min(int((1 - alpha / 2) * n_boot), n_boot - 1)],
        "n": n,
    }


def seed_summary(values: Sequence[float]) -> dict[str, float]:
    """Mean / std / half-width / n across independent training seeds."""
    vals = list(values)
    n = len(vals)
    if n == 0:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    mean = sum(vals) / n
    if n == 1:
        return {"mean": mean, "std": 0.0, "half_width": 0.0, "min": mean, "max": mean, "n": 1}
    var = sum((v - mean) ** 2 for v in vals) / (n - 1)
    sd = math.sqrt(var)
    return {
        "mean": mean,
        "std": sd,
        "half_width": 1.96 * sd / math.sqrt(n) if n > 1 else 0.0,
        "min": min(vals),
        "max": max(vals),
        "n": n,
    }


def compare_seeds(a: Sequence[float], b: Sequence[float]) -> dict[str, Any]:
    """Welch-style comparison of two systems across seeds."""
    sa, sb = seed_summary(a), seed_summary(b)
    diff = sa["mean"] - sb["mean"]
    se = math.sqrt(sa["std"] ** 2 / max(sa["n"], 1) + sb["std"] ** 2 / max(sb["n"], 1))
    return {
        "a": sa,
        "b": sb,
        "delta": diff,
        "delta_se": se,
        "t_like": diff / se if se > 0 else float("inf"),
        "significant_95": bool(se > 0 and abs(diff / se) > 2.776),
    }
