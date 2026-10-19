"""Calibration: temperature scaling (fit on held-out rows only), reliability
bins, risk-coverage curves, dependency-free SVG diagrams.

Convention: caller splits eval rows into fit/measure subsets (seeded);
temperature is fit on fit-rows and all reported post-scaling metrics are on
measure-rows. Never fit and measure on the same rows.
"""

from __future__ import annotations

import math
from typing import Any


def softmax_rows(logits_list: list[list[float]], temperature: float = 1.0) -> list[list[float]]:
    t = max(temperature, 1e-3)
    out = []
    for row in logits_list:
        m = max(row)
        exps = [math.exp((v - m) / t) for v in row]
        s = sum(exps)
        out.append([e / s for e in exps])
    return out


def nll_of(logits_list: list[list[float]], labels: list[int], temperature: float = 1.0) -> float:
    tot = 0.0
    for row, lab in zip(softmax_rows(logits_list, temperature), labels):
        tot += -math.log(max(row[lab], 1e-12))
    return tot / max(len(labels), 1)


def fit_temperature(logits_list: list[list[float]], labels: list[int]) -> float:
    """1-D grid + refine search for scalar T minimizing NLL (no scipy needed)."""
    lo, hi = 0.05, 20.0
    best_t, best_nll = 1.0, nll_of(logits_list, labels, 1.0)
    for _ in range(3):
        cand = [lo + (hi - lo) * i / 40 for i in range(41)]
        for t in cand:
            v = nll_of(logits_list, labels, t)
            if v < best_nll:
                best_nll, best_t = v, t
        span = (hi - lo) / 40
        lo, hi = max(0.05, best_t - span * 2), min(20.0, best_t + span * 2)
    return best_t


def conf_hit_pairs(
    logits_list: list[list[float]], labels: list[int], temperature: float = 1.0
) -> list[tuple[float, bool]]:
    pairs = []
    for row, lab in zip(softmax_rows(logits_list, temperature), labels):
        pred = max(range(len(row)), key=row.__getitem__)
        pairs.append((row[pred], pred == lab))
    return pairs


def ece_of(pairs: list[tuple[float, bool]], n_bins: int = 10) -> dict[str, float]:
    ece, worst, bins = 0.0, 0.0, []
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        sel = [p for p in pairs if (p[0] > lo and p[0] <= hi) or (b == 0 and p[0] <= hi)]
        if not sel:
            bins.append({"bin": [lo, hi], "n": 0, "acc": None, "conf": None})
            continue
        acc = sum(1 for _, h in sel if h) / len(sel)
        conf = sum(c for c, _ in sel) / len(sel)
        ece += len(sel) / len(pairs) * abs(acc - conf)
        worst = max(worst, abs(acc - conf))
        bins.append({"bin": [lo, hi], "n": len(sel), "acc": acc, "conf": conf})
    return {"ece": ece, "max_gap": worst, "bins": bins}


def risk_coverage(pairs: list[tuple[float, bool]]) -> dict[str, Any]:
    """Sort by confidence desc; accuracy over top-k fraction. Reports curve +
    accuracy at fixed coverage thresholds."""
    ordered = sorted(pairs, key=lambda p: -p[0])
    n = len(ordered)
    curve = []
    for k in range(1, n + 1):
        acc = sum(1 for _, h in ordered[:k] if h) / k
        curve.append({"coverage": k / n, "accuracy": acc})
    out = {"curve": curve}
    for cov in (0.2, 0.5, 0.8, 1.0):
        k = max(1, int(round(n * cov)))
        out[f"acc_at_{cov}"] = sum(1 for _, h in ordered[:k] if h) / k
    return out


def temperature_report(fit_logits, fit_labels, measure_logits, measure_labels) -> dict[str, Any]:
    t = fit_temperature(fit_logits, fit_labels)
    pre = ece_of(conf_hit_pairs(measure_logits, measure_labels, 1.0))
    post = ece_of(conf_hit_pairs(measure_logits, measure_labels, t))
    return {
        "temperature": t,
        "fit_n": len(fit_labels),
        "measure_n": len(measure_labels),
        "pre_nll": nll_of(measure_logits, measure_labels, 1.0),
        "post_nll": nll_of(measure_logits, measure_labels, t),
        "pre_ece": pre["ece"],
        "post_ece": post["ece"],
        "post_bins": post["bins"],
    }


# ----------------------------------------------------------------------------
# Dependency-free SVG diagrams (no matplotlib needed)
# ----------------------------------------------------------------------------


def _svg_open(w: int, h: int, title: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">',
        f"<title>{title}</title>",
        '<rect x="0" y="0" width="%d" height="%d" fill="white"/>' % (w, h),
    ]


def reliability_svg(bins: list[dict[str, Any]], title: str = "Reliability diagram") -> str:
    W, H, pad = 420, 360, 48
    s = _svg_open(W, H, title)
    s.append(f'<text x="{W // 2}" y="24" text-anchor="middle" font-size="14">{title}</text>')
    X = lambda v: pad + v * (W - 2 * pad)
    Y = lambda v: H - pad - v * (H - 2 * pad)
    s.append(
        f'<line x1="{X(0)}" y1="{Y(0)}" x2="{X(1)}" y2="{Y(1)}" stroke="#999" stroke-dasharray="4"/>'
    )
    for b in bins:
        if not b["n"]:
            continue
        lo, hi = b["bin"]
        s.append(
            f'<rect x="{X(lo) + 1}" y="{Y(b["acc"])}" width="{X(hi) - X(lo) - 2}" '
            f'height="{Y(0) - Y(b["acc"])}" fill="#4472c4"/>'
        )
        s.append(f'<circle cx="{(X(lo) + X(hi)) / 2}" cy="{Y(b["conf"])}" r="3" fill="#c00000"/>')
    s.append(
        f'<text x="{W // 2}" y="{H - 8}" text-anchor="middle" font-size="11">confidence (bars=accuracy, red dots=mean conf)</text>'
    )
    s.append("</svg>")
    return "\n".join(s)


def risk_coverage_svg(curve: list[dict[str, float]], title: str = "Risk-coverage") -> str:
    W, H, pad = 420, 360, 48
    s = _svg_open(W, H, title)
    s.append(f'<text x="{W // 2}" y="24" text-anchor="middle" font-size="14">{title}</text>')
    X = lambda v: pad + v * (W - 2 * pad)
    Y = lambda v: H - pad - v * (H - 2 * pad)
    pts = " ".join(f"{X(p['coverage']):.1f},{Y(p['accuracy']):.1f}" for p in curve)
    s.append(f'<polyline points="{pts}" fill="none" stroke="#4472c4" stroke-width="2"/>')
    s.append(
        f'<text x="{W // 2}" y="{H - 8}" text-anchor="middle" font-size="11">coverage vs accuracy (high-confidence first)</text>'
    )
    s.append("</svg>")
    return "\n".join(s)
