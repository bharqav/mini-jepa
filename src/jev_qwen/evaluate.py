"""Evaluation: predictive quality, calibration, robustness, efficiency.

Every number is computed from real runs — no invented metrics. Metrics are
returned as plain dicts (JSON-serializable) and can be dumped to CSV.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch
import torch.nn.functional as F

from .data import TASK_TYPES
from .utils import autocast, to_device

TYPE_NAMES = {v: k for k, v in TASK_TYPES.items()}


def evaluate_by_domain(network, loader, device, questions=None) -> dict[str, Any]:
    """Stratified metrics by dataset domain and by candidate count. Uses the
    per-row metadata carried in the batch (never positional zipping)."""
    rows = collect_rows(network, loader, device)
    if not rows:
        return {"error": "no rows"}
    from .stats import accuracy_ci

    def stratify(keyfn):
        buckets: dict[str, list[bool]] = {}
        for r in rows:
            pred = max(range(len(r["logits"])), key=r["logits"].__getitem__)
            buckets.setdefault(str(keyfn(r)), []).append(pred == r["label"])
        return {
            d: {**accuracy_ci(h), "accuracy": sum(h) / len(h), "n": len(h)}
            for d, h in sorted(buckets.items(), key=lambda kv: -len(kv[1]))
        }

    return {
        "by_domain": stratify(lambda r: r.get("domain", "unknown")),
        "by_candidate_count": stratify(lambda r: r.get("choice_count", 0)),
    }


@torch.no_grad()
def collect_rows(network, loader, device):
    """Per-row (logits, label) pairs for calibration/stratified analysis.
    Metadata (domain, candidate count) rides along WITH the row, so results
    stay correct even when the loader reorders rows (length bucketing)."""
    network.eval()
    rows = []
    for batch in loader:
        batch = to_device(batch, device)
        with autocast(device):
            logits = network(**batch).float()
        doms = batch.get("domains") or ["unknown"] * len(logits)
        counts = batch.get("choice_counts")
        for i, (lg, lab) in enumerate(zip(logits.tolist(), batch["labels"].tolist())):
            rows.append(
                {
                    "logits": lg,
                    "label": lab,
                    "domain": doms[i] if i < len(doms) else "unknown",
                    "choice_count": int(counts[i]) if counts is not None else len(lg),
                }
            )
    network.train()
    return rows


@torch.no_grad()
def evaluate(network, loader, device) -> dict[str, float]:
    """Variable candidate counts are handled by accumulating per-row
    statistics (never by padding distributions)."""
    network.eval()
    tot_loss, correct, count = 0.0, defaultdict(int), defaultdict(int)
    confs, hits, briers = [], [], []
    per_row_pred, per_row_label, per_row_c = [], [], []  # for modal-C macro-F1
    noul_scores, noul_labels = [], []
    n = 0
    for batch in loader:
        batch = to_device(batch, device)
        with autocast(device):
            logits = network(**batch)
        tot_loss += F.cross_entropy(logits, batch["labels"], reduction="sum").item()
        probs = torch.softmax(logits.float(), -1)
        conf, pred = probs.max(-1)
        hit = pred == batch["labels"]
        C = probs.shape[1]
        onehot = torch.zeros_like(probs)
        onehot.scatter_(1, batch["labels"].unsqueeze(1), 1.0)
        brier = ((probs - onehot) ** 2).sum(1)
        for t, h, cf, br, p, l in zip(
            batch["task_type"].tolist(),
            hit.tolist(),
            conf.tolist(),
            brier.tolist(),
            pred.tolist(),
            batch["labels"].tolist(),
        ):
            count[TYPE_NAMES[t]] += 1
            correct[TYPE_NAMES[t]] += h
            confs.append(cf)
            hits.append(h)
            briers.append(br)
            per_row_pred.append(p)
            per_row_label.append(l)
            per_row_c.append(C)
            if t == TASK_TYPES["noul"]:
                noul_scores.append(cf if p == 1 else 1.0 - cf)
                noul_labels.append(l)
        n += len(hit)
    network.train()
    if n == 0:
        raise ValueError("evaluate: empty loader (check split sizes)")
    m: dict[str, float] = {
        "loss": tot_loss / n,
        "accuracy": sum(correct.values()) / n,
        "nll": tot_loss / n,
        "brier": sum(briers) / n,
        **_ece_from_pairs(confs, hits),
    }
    for name in count:
        m[f"accuracy_{name}"] = correct[name] / count[name]
    # macro-F1 over rows sharing the modal candidate count (noted explicitly)
    from collections import Counter

    modal_c = Counter(per_row_c).most_common(1)[0][0]
    sel = [i for i, c in enumerate(per_row_c) if c == modal_c]
    m["macro_f1"] = _macro_f1_pure_lists(
        [per_row_pred[i] for i in sel], [per_row_label[i] for i in sel]
    )
    m["macro_f1_n"] = len(sel)
    m["macro_f1_C"] = modal_c
    if noul_labels and len(set(noul_labels)) == 2:
        m["auc_noul"] = float(binary_auc(noul_scores, noul_labels))
    return m


def _ece_from_pairs(confs: list[float], hits: list[bool], n_bins: int = 10) -> dict[str, float]:
    ece, worst = 0.0, 0.0
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        sel = [i for i, c in enumerate(confs) if (c > lo and c <= hi) or (b == 0 and c <= hi)]
        if not sel:
            continue
        gap = abs(sum(hits[i] for i in sel) / len(sel) - sum(confs[i] for i in sel) / len(sel))
        ece += len(sel) / len(confs) * gap
        worst = max(worst, gap)
    return {"ece": ece, "max_calibration_gap": worst}


def _macro_f1_pure_lists(pred: list[int], labels: list[int]) -> float:
    classes = sorted(set(pred) | set(labels))
    f1s = []
    for c in classes:
        tp = sum(1 for p, l in zip(pred, labels) if p == c and l == c)
        fp = sum(1 for p, l in zip(pred, labels) if p == c and l != c)
        fn = sum(1 for p, l in zip(pred, labels) if p != c and l == c)
        denom = 2 * tp + fp + fn
        f1s.append(2 * tp / denom if denom else 0.0)
    return sum(f1s) / max(len(f1s), 1)


def binary_auc(scores: list[float], labels: list[int]) -> float:
    s = torch.tensor(scores)
    lab = torch.tensor(labels, dtype=torch.bool)
    pos, neg = s[lab], s[~lab].sort().values
    below = torch.searchsorted(neg, pos, right=False)
    tied = torch.searchsorted(neg, pos, right=True)
    return float((below + tied).sum() / 2 / (len(pos) * len(neg)))


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Temperature scaling on VALIDATION data only (never test)."""
    t = torch.ones((), requires_grad=True)
    opt = torch.optim.LBFGS([t], lr=0.1, max_iter=50)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / t.clamp(0.05, 20.0), labels)
        loss.backward()
        return loss

    opt.step(closure)
    return float(t.detach().clamp(0.05, 20.0))


def order_sensitivity(
    network, loader, device, n_perm: int = 5, seed: int = 0, max_rows: int = 200
) -> dict[str, float]:
    """Total-variation distance under candidate permutation.

    IMPORTANT — this measures HEAD-LEVEL invariance only. It permutes
    `choice_read_idx` inside an already-tokenised batch, so the token sequence
    (and therefore the backbone computation) is unchanged; only the head
    re-reads candidates in a different order. For `CompatHead` this is exactly
    0.0, because its shared projections make each score a pure function of
    `(answer, candidate_i)`.

    Full end-to-end invariance — re-tokenising with a different candidate
    order, which *does* change the token sequence — is NOT measured here and is
    weaker: the answer token attends over all candidates, so float32 summation
    order changes and probabilities drift by ~3e-3. The argmax is unaffected in
    practice. See docs/architecture.md.
    """
    import random

    rng = random.Random(seed)
    network.eval()
    worst, mean, cnt = 0.0, 0.0, 0
    with torch.no_grad():
        for batch in loader:
            B = batch["input_ids"].shape[0]
            for b in range(B):
                if cnt >= max_rows * (n_perm + 1):
                    break
                C = int(batch["choice_mask"][b].sum())
                if C < 2:
                    continue
                dev = device
                base_in = {
                    "input_ids": batch["input_ids"][b : b + 1],
                    "position_ids": batch["position_ids"][b : b + 1],
                    "attention_mask": batch["attention_mask"][b : b + 1],
                    "choice_read_idx": batch["choice_read_idx"][b : b + 1, :C],
                    "choice_mask": batch["choice_mask"][b : b + 1, :C],
                    "answer_end_idx": batch["answer_end_idx"][b : b + 1],
                    "task_type": batch["task_type"][b : b + 1],
                }
                base = torch.softmax(network(**to_device(base_in, dev))[0].float(), -1)
                ids = list(range(C))
                for _ in range(n_perm):
                    order = ids[:]
                    rng.shuffle(order)
                    inv = [0] * C
                    for new, old in enumerate(order):
                        inv[old] = new
                    pin = dict(base_in)
                    pin["choice_read_idx"] = base_in["choice_read_idx"][:, order]
                    p = torch.softmax(network(**to_device(pin, dev))[0].float(), -1)
                    tv = float(0.5 * (base - p[inv]).abs().sum())
                    worst = max(worst, tv)
                    mean += tv
                    cnt += 1
    network.train()
    return {"order_tv_worst": worst, "order_tv_mean": mean / max(cnt, 1)}
