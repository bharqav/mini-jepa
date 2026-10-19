"""Modular dataset adapters: every adapter yields canonical row dicts
(state, task_type, instructions, choices, label, example_id, provenance).

Adapters let new public datasets plug in without touching the core pipeline.
Only adapters whose output validates are accepted (see data.validate_records).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from .data import TASK_TYPES

Adapter = Callable[[str], list[dict]]
_REGISTRY: dict[str, Adapter] = {}


def register(name: str):
    def deco(fn: Adapter):
        _REGISTRY[name] = fn
        return fn

    return deco


def get(name: str) -> Adapter:
    if name not in _REGISTRY:
        raise KeyError(f"unknown adapter {name!r}; registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def _canon(
    state: str,
    task: str,
    instructions: str,
    choices: list[str],
    label,
    example_id: str,
    provenance: str,
) -> dict:
    if task not in TASK_TYPES:
        raise ValueError(f"bad task type {task!r} in {example_id}")
    if label is not None and not (0 <= label < len(choices)):
        raise ValueError(f"label {label} out of range in {example_id}")
    return {
        "state": state,
        "task_type": TASK_TYPES[task],
        "instructions": instructions,
        "choices": choices,
        "label": label,
        "example_id": example_id,
        "provenance": provenance,
    }


@register("jsonl")
def jsonl_adapter(path: str) -> list[dict]:
    """JSONL where each line is {state, type, instructions, choices, label?}.

    `choices` may be a list of texts (ids auto-assigned) or a list of
    {id, text}. `label` is a choice id, index, or null.
    """
    recs = []
    for i, line in enumerate(Path(path).read_text().splitlines()):
        if not line.strip():
            continue
        r = json.loads(line)
        ch = r["choices"]
        if ch and isinstance(ch[0], dict):
            ids, texts = [c["id"] for c in ch], [c["text"] for c in ch]
        else:
            ids, texts = [f"c{j}" for j in range(len(ch))], list(ch)
        lab = r.get("label")
        if isinstance(lab, str):
            lab = ids.index(lab)
        recs.append(
            _canon(
                r["state"],
                r["type"],
                r["instructions"],
                texts,
                lab,
                f"jsonl:{Path(path).name}:{i}",
                f"jsonl:{Path(path).name}",
            )
        )
    return recs


@register("synthetic")
def synthetic_adapter(spec: str) -> list[dict]:
    """spec = '<n>:<seed>' (e.g. '200:0')."""
    from .data import synthetic_records

    n, _, seed = spec.partition(":")
    return synthetic_records(int(n or 60), int(seed or 0))
