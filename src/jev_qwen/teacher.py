"""Offline teacher labeling (optional; never required for the baseline).

Two teacher kinds:
  - local-lm: frozen Qwen causal LM, letter-scoring (A/B/C...) — real
    confidence from softmax, runs on CPU, no API key, no cost.
  - external: user-configured HTTP endpoint (env JEV_TEACHER_URL + optional
    JEV_TEACHER_KEY). Retry with backoff, per-call accounting (chars sent,
    calls, failures), provenance + model version recorded per label.

Every teacher label records provenance; teacher outputs are validated
(label in range, finite confidence) and disagreements with existing labels
are flagged, never silently overwritten. Results are cached (fingerprint
keyed JSONL) so reruns are free.
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
from pathlib import Path
from typing import Any

import torch

LETTERS = "ABCDEFGHIJKLMNOP"


def fingerprint_of(state: str, instructions: str, choices: list[str]) -> str:
    from .data import fingerprint

    return fingerprint(state, instructions, choices)


class TeacherCache:
    def __init__(self, path: str):
        self.path = Path(path)
        self.rows: dict[str, dict] = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    self.rows[r["fingerprint"]] = r

    def get(self, fp: str) -> dict | None:
        return self.rows.get(fp)

    def put(self, fp: str, row: dict) -> None:
        self.rows[fp] = {"fingerprint": fp, **row}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as f:
            f.write(json.dumps({"fingerprint": fp, **row}) + "\n")


class LocalLMTeacher:
    """Letter-scoring teacher on a frozen causal LM. Confidence = softmax
    probability of the chosen letter (genuine, not invented)."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        num_layers: int | None = 8,
        device=None,
        revision: str | None = None,
    ):
        from transformers import Qwen3ForCausalLM

        from .hub import resolve_model_revision
        from .tokenize import get_tokenizer
        from .utils import get_device

        self.device = device or get_device()
        rev = resolve_model_revision(model_name, revision)
        self.tok = get_tokenizer(model_name, revision=rev)
        self.lm = Qwen3ForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float32,
            revision=rev,
            **({} if num_layers is None else {"num_hidden_layers": num_layers}),
        )
        self.lm = self.lm.to(self.device).eval()
        self.version = f"local-lm:{model_name}@{rev[:8]}:L{num_layers}"

    @torch.no_grad()
    def label(self, state: str, instructions: str, choices: list[str]) -> dict[str, Any]:
        opts = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(choices))
        prompt = f"State: {state}\nQuestion: {instructions}\n{opts}\nAnswer:"
        ids = torch.tensor([self.tok.encode(prompt)]).to(self.device)
        logits = self.lm(ids).logits[0, -1]
        letter_ids = []
        for letter in LETTERS[: len(choices)]:
            cands = self.tok.encode(" " + letter) or self.tok.encode(letter)
            letter_ids.append(cands[-1])
        probs = torch.softmax(torch.stack([logits[i] for i in letter_ids]).float(), -1)
        best = int(probs.argmax())
        return {
            "label": best,
            "confidence": float(probs[best]),
            "probs": [float(p) for p in probs],
            "teacher": self.version,
        }


class ExternalTeacher:
    """POST {state, instructions, choices} -> {label, confidence?, model?}.
    Accounting counts every call; failures retry 3x then are recorded, never hidden."""

    def __init__(self, url: str | None = None, key: str | None = None):
        self.url = url or os.environ.get("JEV_TEACHER_URL", "")
        self.key = key or os.environ.get("JEV_TEACHER_KEY", "")
        if not self.url:
            raise ValueError("no teacher URL (pass url= or set JEV_TEACHER_URL)")
        self.calls = 0
        self.failures = 0
        self.chars_sent = 0

    def label(self, state: str, instructions: str, choices: list[str]) -> dict[str, Any]:
        payload = json.dumps(
            {"state": state, "instructions": instructions, "choices": choices}
        ).encode()
        self.chars_sent += len(payload)
        last: Exception | None = None
        for attempt in range(3):
            try:
                req = urllib.request.Request(
                    self.url,
                    data=payload,
                    headers={
                        "Content-Type": "application/json",
                        **({"Authorization": f"Bearer {self.key}"} if self.key else {}),
                    },
                )
                with urllib.request.urlopen(req, timeout=60) as r:
                    body = json.load(r)
                self.calls += 1
                lab = body["label"]
                if isinstance(lab, str):
                    lab = choices.index(lab)
                assert 0 <= lab < len(choices), f"teacher label {lab} out of range"
                conf = float(body.get("confidence", float("nan")))
                return {
                    "label": lab,
                    "confidence": conf,
                    "teacher": f"external:{body.get('model', self.url)}",
                    "cost": {"calls": self.calls, "chars_sent": self.chars_sent},
                }
            except Exception as e:  # noqa: BLE001 - retried, then recorded
                last = e
                self.failures += 1
                time.sleep(2**attempt)
        raise RuntimeError(f"teacher failed 3x: {last}")


def label_records(
    records: list[dict], teacher, cache: TeacherCache, keep_disagreements: bool = True
) -> dict[str, Any]:
    """Label records in place (adds teacher_label/confidence/teacher). Cached
    rows cost nothing. Returns a report with agreement stats."""
    report = {
        "n": len(records),
        "cache_hits": 0,
        "labeled": 0,
        "agree": 0,
        "disagree": 0,
        "unlabeled_before": 0,
    }
    for r in records:
        fp = fingerprint_of(r["state"], r["instructions"], r["choices"])
        hit = cache.get(fp)
        if hit is None:
            hit = teacher.label(r["state"], r["instructions"], r["choices"])
            if not (0 <= hit["label"] < len(r["choices"])):
                raise ValueError(f"teacher label out of range for {r.get('example_id')}")
            if not (hit["confidence"] != hit["confidence"] or 0.0 <= hit["confidence"] <= 1.0):
                raise ValueError("teacher confidence not in [0,1]")
            cache.put(fp, hit)
            report["labeled"] += 1
        else:
            report["cache_hits"] += 1
        r["teacher_label"] = hit["label"]
        r["teacher_confidence"] = hit.get("confidence")
        r["teacher"] = hit.get("teacher", "?")
        if r.get("label") is None:
            report["unlabeled_before"] += 1
        elif r["label"] == hit["label"]:
            report["agree"] += 1
        else:
            report["disagree"] += 1
            if keep_disagreements:
                r["label_disagreement"] = True  # flagged, original kept
    return report
