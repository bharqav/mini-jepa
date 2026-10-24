"""Dataset: schema, validation, dedup, splits, synthetic, adapters."""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict

import torch

TASK_TYPES = {"choice": 0, "noul": 1, "score": 2}
TASK_NAMES = {v: k for k, v in TASK_TYPES.items()}
DATASET_NAME = "avbiswas/bev-decision-150K"
DATASET_REVISION = "11ed9b5b879e7f9524e7c89c1037d498c54206f0"
SCHEMA_VERSION = "v3-domain"  # bump when the output schema changes


def question_to_choices(question: dict):
    """Reference semantics (dataset.py): choice -> criteria keys; noul -> [No, Yes]
    with label=int(label) = P(Yes) index; score -> enumerated levels, label=index."""
    t = question["type"]
    if t == "choice":
        keys = list(question["criteria"])
        choices = [
            f"{k}: {question['criteria'][k]}" if question["criteria"][k] else k for k in keys
        ]
        label = keys.index(question["label"]) if "label" in question else None
    elif t == "score":
        choices = [f"{i}: {lv}" for i, lv in enumerate(question["criteria"])]
        label = question.get("label")
    elif t == "noul":
        choices = ["No", "Yes"]
        label = int(question["label"]) if "label" in question else None
    else:
        raise ValueError(f"unknown question type: {t!r} (expected choice/score/noul)")
    return choices, label


def explode_questions(rows: dict) -> dict:
    out = {
        "state": [],
        "task_type": [],
        "instructions": [],
        "choices": [],
        "label": [],
        "example_id": [],
        "provenance": [],
        "domain": [],
    }
    for i, (state, qj, domain) in enumerate(
        zip(rows["state"], rows["questions_json"], rows.get("domain", [""] * len(rows["state"])))
    ):
        for qid, q in json.loads(qj).items():
            choices, label = question_to_choices(q)
            out["state"].append(state)
            out["task_type"].append(TASK_TYPES[q["type"]])
            out["instructions"].append(q["instructions"])
            out["choices"].append(choices)
            out["label"].append(label)
            out["example_id"].append(f"r{i}:{qid}")
            out["provenance"].append(DATASET_NAME)
            out["domain"].append(domain)
    return out


def fingerprint(state: str, instructions: str, choices: list) -> str:
    h = hashlib.sha256()
    h.update(json.dumps([state.strip(), instructions.strip(), choices], sort_keys=True).encode())
    return h.hexdigest()[:16]


def validate_records(records: list[dict]) -> dict:
    """Returns quality report; drops nothing (caller decides)."""
    report = {
        "n": len(records),
        "malformed": 0,
        "impossible_label": 0,
        "exact_dupes": 0,
        "conflicts": 0,
        "by_type": defaultdict(int),
    }
    seen: dict[str, int] = {}
    for r in records:
        try:
            if not r["state"] or not r["instructions"] or not r["choices"]:
                raise ValueError("empty field")
            lab = r["label"]
            if lab is not None and not (0 <= lab < len(r["choices"])):
                report["impossible_label"] += 1
                raise ValueError("label out of range")
            report["by_type"][TASK_NAMES.get(r["task_type"], "?")] += 1
            fp = fingerprint(r["state"], r["instructions"], r["choices"])
            if fp in seen:
                report["exact_dupes"] += 1
                if seen[fp] != lab:
                    report["conflicts"] += 1
            else:
                seen[fp] = lab
        except Exception:
            report["malformed"] += 1
    report["by_type"] = dict(report["by_type"])
    return report


def dedup_records(records: list[dict]) -> list[dict]:
    seen = {}
    out = []
    for r in records:
        fp = fingerprint(r["state"], r["instructions"], r["choices"])
        if fp not in seen:
            seen[fp] = r["label"]
            out.append(r)
    return out


def group_split(
    records: list[dict], seed: int = 0, ratios=(0.8, 0.1, 0.1)
) -> dict[str, list[dict]]:
    """Group by state fingerprint BEFORE splitting to avoid leakage."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        groups[fingerprint(r["state"], "", [])].append(r)
    keys = sorted(groups)
    rng = random.Random(seed)
    rng.shuffle(keys)
    n = len(keys)
    n_tr = int(n * ratios[0])
    n_va = int(n * ratios[1])
    out = {"train": [], "validation": [], "test": []}
    for k in keys[:n_tr]:
        out["train"].extend(groups[k])
    for k in keys[n_tr : n_tr + n_va]:
        out["validation"].extend(groups[k])
    for k in keys[n_tr + n_va :]:
        out["test"].extend(groups[k])
    return out


def _map_fingerprint():
    """Cache token derived from the function source.

    `datasets` caches mapped datasets on disk keyed by a fingerprint that does
    NOT reliably track changes to a module-level function. When the output
    schema changed (adding `domain`), a stale cached table kept being served
    and the new column silently came back empty. Deriving the token from the
    source hash makes cache invalidation automatic.
    """
    import hashlib
    import inspect

    src = inspect.getsource(explode_questions) + SCHEMA_VERSION
    return "jevqwen-" + hashlib.sha1(src.encode()).hexdigest()[:12]


def load_questions(
    split: str = "train",
    max_questions: int | None = None,
    seed: int = 0,
    subset: str = "data",
    revision: str | None = None,
):
    """Load reference questions. Uses HF `datasets` when available (pyarrow);
    otherwise falls back to huggingface_hub + fastparquet (pure-python-friendly
    path for locked-down machines). Same schema either way.

    The dataset revision is a pinned commit SHA; implicit ``main`` is never
    fetched.
    """
    from .hub import resolve_dataset_revision

    rev = resolve_dataset_revision(DATASET_NAME, revision)
    try:
        from datasets import load_dataset

        ds = load_dataset(DATASET_NAME, subset, split=split, revision=rev).shuffle(seed=seed)
        if max_questions is not None:
            ds = ds.select(range(min(max_questions, len(ds))))
        # num_proc=1 keeps this deterministic and avoids forkserver crashes on
        # Python 3.14 + WSL (and on locked-down Windows hosts).
        ds = ds.map(
            explode_questions,
            batched=True,
            remove_columns=ds.column_names,
            num_proc=1,
            desc="exploding questions",
            new_fingerprint=_map_fingerprint(),
        )
        if max_questions is not None:
            ds = ds.select(range(min(max_questions, len(ds))))
        return ds
    except ImportError:
        return _load_questions_fastparquet(split, max_questions, seed, subset, rev)


def _load_questions_fastparquet(
    split: str, max_questions: int | None, seed: int, subset: str, revision: str
):
    import random as _random

    import fastparquet
    from huggingface_hub import snapshot_download

    folder = {"default": "data"}.get(subset, subset)  # datasets-config -> parquet folder
    snap = snapshot_download(
        DATASET_NAME,
        repo_type="dataset",
        revision=revision,
        allow_patterns=[f"{folder}/{split}.parquet"],
    )
    pf = fastparquet.ParquetFile(f"{snap}/{folder}/{split}.parquet")
    cols = ["state", "questions_json"]
    names = pf.columns
    if "domain" in names:
        cols.append("domain")
    rows = pf.to_pandas()[cols].to_dict("records")
    rng = _random.Random(seed)
    rng.shuffle(rows)
    if max_questions is not None:
        rows = rows[:max_questions]
    recs = []
    for i, r in enumerate(rows):
        for qid, q in json.loads(r["questions_json"]).items():
            choices, label = question_to_choices(q)
            recs.append(
                {
                    "state": r["state"],
                    "task_type": TASK_TYPES[q["type"]],
                    "instructions": q["instructions"],
                    "choices": choices,
                    "label": label,
                    "example_id": f"{subset}/{split}:{i}:{qid}",
                    "provenance": DATASET_NAME,
                    "domain": r.get("domain", subset),
                }
            )
    if max_questions is not None:
        recs = recs[:max_questions]
    return recs


SYNTH_TOPICS = [
    (
        "A server is returning repeated HTTP 503 errors.",
        "What should the operator do first?",
        [
            "Inspect service health and dependencies",
            "Delete the production database",
            "Ignore the incident",
            "Restart every unrelated service",
        ],
        0,
    ),
    (
        "The proposed patch fixes the reproduced crash and passes the relevant regression tests.",
        "How likely is this patch to resolve the reported defect?",
        ["very unlikely", "unlikely", "likely", "very likely"],
        2,
    ),
    (
        "The change adds logging only and touches no control flow.",
        "Is this change safe to deploy without further review?",
        ["No", "Yes"],
        1,
    ),
]


def synthetic_records(n: int = 60, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    out = []
    for i in range(n):
        s, q, ch, lab = rng.choice(SYNTH_TOPICS)
        t = "noul" if ch == ["No", "Yes"] else ("score" if q.startswith("How likely") else "choice")
        out.append(
            {
                "state": s,
                "task_type": TASK_TYPES[t],
                "instructions": q,
                "choices": list(ch),
                "label": lab,
                "example_id": f"synth:{i}",
                "provenance": "synthetic-v1",
            }
        )
    return out


class DecisionDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        questions,
        max_state_tokens: int,
        max_choice_tokens: int,
        tokenizer,
        permute: bool = False,
        seed: int = 0,
    ):
        self.q = questions
        self.ms, self.mc = max_state_tokens, max_choice_tokens
        self.tok = tokenizer
        self.permute = permute
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.q)

    def __getitem__(self, idx):
        from .tokenize import encode_example

        row = self.q[idx]
        task = row["task_type"]
        name = TASK_NAMES[task] if isinstance(task, int) else task
        ex = encode_example(
            row["state"], row["instructions"], row["choices"], name, self.ms, self.mc, self.tok
        )
        choices, label = list(row["choices"]), row["label"]
        if self.permute and len(choices) > 1:
            order = list(range(len(choices)))
            self.rng.shuffle(order)
            ex["choice_read_idx"] = [ex["choice_read_idx"][i] for i in order]
            choices = [choices[i] for i in order]
            if label is not None:
                label = order.index(label)
        ex["task_type"] = TASK_TYPES[name]
        ex["label"] = label
        ex["domain"] = str(row.get("domain", "") or "unknown")
        ex["choice_count"] = len(choices)
        return ex


def collate_with_pad(batch, pad_id: int = 0):
    """Module-level (picklable) collate wrapper for multi-worker DataLoaders."""
    return collate_fn(batch, pad_id=pad_id)


class LengthBucketBatchSampler(torch.utils.data.Sampler):
    """Groups indices into batches of similar encoded length to cut padding.

    Deterministic given `seed` (a fixed permutation is reshuffled per epoch by
    the caller). Falls back to sequential behaviour when lengths are uniform.
    Only used for evaluation/inference-style loading; training keeps plain
    random shuffling so batch composition stays seed-reproducible.
    """

    def __init__(
        self,
        lengths,
        batch_size: int,
        shuffle: bool = False,
        seed: int = 0,
        drop_last: bool = False,
    ):
        self.lengths = list(lengths)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

    def __iter__(self):
        import random as _random

        rng = _random.Random(self.seed + self.epoch)
        idx = list(range(len(self.lengths)))
        if self.shuffle:
            rng.shuffle(idx)
        # sort within large chunks so batches are length-homogeneous but the
        # global order is still varied between epochs
        chunk = self.batch_size * 32
        buckets = []
        for i in range(0, len(idx), chunk):
            part = sorted(idx[i : i + chunk], key=lambda j: self.lengths[j])
            for b in range(0, len(part), self.batch_size):
                batch = part[b : b + self.batch_size]
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                buckets.append(batch)
        if self.shuffle:
            rng.shuffle(buckets)
        self.epoch += 1
        return iter(buckets)

    def __len__(self):
        n = len(self.lengths)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size


def collate_fn(batch, pad_id: int = 0):
    import torch as _t

    B = len(batch)
    L = max(len(ex["input_ids"]) for ex in batch)
    C = max(len(ex["choice_read_idx"]) for ex in batch)
    input_ids = _t.full((B, L), pad_id, dtype=_t.long)
    position_ids = _t.zeros((B, L), dtype=_t.long)
    attention_mask = _t.eye(L).repeat(B, 1, 1)
    choice_read_idx = _t.zeros((B, C), dtype=_t.long)
    choice_mask = _t.zeros((B, C), dtype=_t.bool)
    answer_end_idx = _t.zeros(B, dtype=_t.long)
    for b, ex in enumerate(batch):
        n, c = len(ex["input_ids"]), len(ex["choice_read_idx"])
        input_ids[b, :n] = _t.tensor(ex["input_ids"])
        position_ids[b, :n] = _t.tensor(ex["position_ids"])
        attention_mask[b, :n, :n] = ex["attention_mask"]
        choice_read_idx[b, :c] = _t.tensor(ex["choice_read_idx"])
        choice_mask[b, :c] = True
        answer_end_idx[b] = ex["answer_end_idx"]
    attention_mask = _t.where(attention_mask == 0, -float("inf"), 0.0).unsqueeze(1)
    out = {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "attention_mask": attention_mask,
        "token_mask": input_ids != pad_id,
        "choice_read_idx": choice_read_idx,
        "choice_mask": choice_mask,
        "answer_end_idx": answer_end_idx,
        "task_type": _t.tensor([ex["task_type"] for ex in batch], dtype=_t.long),
    }
    if all(ex["label"] is not None for ex in batch):
        out["labels"] = _t.tensor([ex["label"] for ex in batch], dtype=_t.long)
    # per-row metadata travels WITH the row so bucketing/reordering cannot
    # misalign it (positional zipping is wrong once batches are reordered).
    out["domains"] = [ex.get("domain", "unknown") for ex in batch]
    out["choice_counts"] = _t.tensor(
        [ex.get("choice_count", len(ex["choice_read_idx"])) for ex in batch], dtype=_t.long
    )
    return out
