"""Inference engine: batched, zero-token-generation, typed in/out.

The core path never calls generate()/sample() — one backbone forward pass
per (state, question-set) pack, then head scoring. Multi-question packing is
done as independent rows (never cross-attending) unless the caller passes
verified packed encodings.
"""

from __future__ import annotations

import time
from pathlib import Path

import torch

from .data import TASK_TYPES, collate_fn
from .model import build_network
from .run_logger import load_checkpoint
from .schema import (
    CandidateScore,
    DecisionRequest,
    DecisionResponse,
    QuestionResponse,
    validate_response,
)
from .tokenize import encode_example, get_tokenizer
from .utils import autocast, get_device, to_device


class DecisionEngine:
    def __init__(
        self,
        checkpoint: str | None = None,
        model_name: str = "Qwen/Qwen3-0.6B",
        num_layers: int | None = None,
        head_config: dict | None = None,
        device: torch.device | None = None,
        precision: str = "bf16",
        max_state_tokens: int = 512,
        max_choice_tokens: int = 64,
        max_input_chars: int = 20000,
        deterministic: bool = True,
        model_version: str = "jev-qwen-0.2.0",
        model_revision: str | None = None,
    ):
        self.device = device or get_device()
        self.precision = precision
        if deterministic:
            torch.use_deterministic_algorithms(False)  # custom 4D masks have no
            torch.backends.cudnn.deterministic = True  # deterministic kernel
            torch.backends.cudnn.benchmark = False
        self.max_state_tokens = max_state_tokens
        self.max_choice_tokens = max_choice_tokens
        self.max_input_chars = max_input_chars
        self.oom_retries = 0
        self.deterministic = deterministic
        self.model_version = model_version
        if checkpoint:
            self.network, self.meta = load_checkpoint(checkpoint, self.device)
            self.max_state_tokens = self.meta.get("max_state_tokens", max_state_tokens)
            self.max_choice_tokens = self.meta.get("max_choice_tokens", max_choice_tokens)
            self.model_version = f"{model_version}@{Path(checkpoint).name}"
            # Use the tokenizer from the same pinned revision as the weights,
            # not whatever the default happens to be today.
            self.tokenizer = get_tokenizer(
                self.meta.get("model_name", model_name),
                revision=self.meta.get("model_revision"),
            )
        else:
            self.tokenizer = get_tokenizer(model_name, revision=model_revision)
            self.network = build_network(
                model_name,
                num_layers,
                8,
                16,
                0.0,
                0,
                head_config or {"type": "compat", "new_dim": 128},
                revision=model_revision,
            )
            self.network = self.network.to(self.device).eval()
            self.meta = {}
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        assert not hasattr(self.network, "generate"), (
            "decision backbone must not expose a generation path"
        )

    @torch.no_grad()
    def decide(self, req: DecisionRequest) -> DecisionResponse:
        t0 = time.perf_counter()
        if self.max_input_chars and len(req.state) > self.max_input_chars:
            raise ValueError(
                f"state is {len(req.state)} chars, limit is {self.max_input_chars}; "
                "truncate upstream or raise engine.max_input_chars"
            )
        rows = []
        for q in req.questions:
            texts = [c.text for c in q.candidates]
            ex = encode_example(
                req.state,
                q.question,
                texts,
                q.type,
                req.settings.max_state_tokens or self.max_state_tokens,
                req.settings.max_candidate_tokens or self.max_choice_tokens,
                self.tokenizer,
            )
            ex["task_type"] = TASK_TYPES[q.type]
            ex["label"] = None
            rows.append((q, ex))
        logits = self._forward(rows, req.settings.max_batch)
        temp = max(float(req.settings.temperature), 1e-3)
        probs = torch.softmax(logits / temp, -1).tolist()
        out: list[QuestionResponse] = []
        for (q, _), p in zip(rows, probs):
            p = p[: len(q.candidates)]
            s = sum(p)
            p = [v / s for v in p]  # exact renormalization after truncation
            scores = [CandidateScore(id=c.id, score=v) for c, v in zip(q.candidates, p)]
            conf = max(p)
            if q.type == "choice" or q.type == "noul":
                sel = q.candidates[p.index(conf)].id
                out.append(
                    QuestionResponse(
                        id=q.id, type=q.type, scores=scores, selected=sel, confidence=conf
                    )
                )
            else:
                exp_level = sum(i * v for i, v in enumerate(p))
                norm = exp_level / max(len(p) - 1, 1)
                out.append(
                    QuestionResponse(
                        id=q.id,
                        type=q.type,
                        scores=scores,
                        selected=None,
                        confidence=conf,
                        expected_level=exp_level,
                        normalized_score=norm,
                    )
                )
        resp = DecisionResponse(
            results=out,
            model_version=self.model_version,
            inference_metadata={"latency_s": time.perf_counter() - t0, "device": str(self.device)}
            if req.settings.include_metadata
            else None,
        )
        validate_response(resp)  # never return silently invalid distributions
        return resp

    def _forward(self, rows, max_batch: int | None = None) -> torch.Tensor:
        """Run the backbone in bounded chunks. On CUDA OOM the chunk size is
        halved and retried (up to single-question), so one oversized request
        degrades in latency instead of failing the whole call."""
        exs = [ex for _, ex in rows]
        if max_batch:
            chunk = max(1, min(len(exs), int(max_batch)))
        else:
            chunk = len(exs)
        parts: list[torch.Tensor] = []
        i = 0
        while i < len(exs):
            batch_ex = exs[i : i + chunk]
            try:
                batch = to_device(
                    collate_fn(batch_ex, pad_id=self.tokenizer.pad_token_id or 0), self.device
                )
                with autocast(self.device, self.precision):
                    out = self.network(**batch).float()
                parts.append(out.cpu())
                i += len(batch_ex)
            except torch.cuda.OutOfMemoryError:
                if self.device.type != "cuda":
                    raise
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                if chunk == 1:
                    raise
                chunk = max(1, chunk // 2)
                self.oom_retries += 1
        return torch.cat(parts, dim=0)
