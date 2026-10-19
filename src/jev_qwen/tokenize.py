"""Reference-faithful tokenization with packed state/question/candidate isolation.

Mechanism (from avbiswas/bev-train tokenization.py):
- Sequence layout: [question+state tokens][choice_0 tokens][choice_1 tokens]...[answer prompt tokens].
- Position IDs: question tokens 0..Q-1; EVERY choice restarts at position Q
  (shared start => candidate-order invariance: no candidate carries a
  position offset that depends on sibling order/length); answer tokens start
  at max(choice position)+1 so every choice token is in their causal past.
- Attention mask (block-causal): question block is causal; each choice block
  is internally causal, sees the question block, sees NO sibling choice
  block; answer block sees everything before it and is causal within itself.
"""

from __future__ import annotations

import os

import torch

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

TASK_PROMPTS = {
    "choice": "Choose the best option.",
    "score": "Rate it on the given scale.",
    "noul": "Answer yes or no.",
}

_tokenizer = None


def get_tokenizer(model_name: str = "Qwen/Qwen3-0.6B"):
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer

        _tokenizer = AutoTokenizer.from_pretrained(model_name)
    return _tokenizer


def tokenize_prompt(question: str, choices: list[str], tokenizer) -> tuple[list[int], dict]:
    choice_begin = tokenizer.encode("<option>")
    choice_end = tokenizer.encode("</option>")
    answer_prompt = "The answer is:"
    q_toks = tokenizer.encode(question)
    a_toks = tokenizer.encode(answer_prompt)
    c_toks = [tokenizer.encode(c) for c in choices]
    out: list[int] = []
    out.extend(q_toks)
    span = {"question_end_idx": len(q_toks) - 1, "choices_end_idx": [], "answer_end_idx": 0}
    for ct in c_toks:
        out.extend(choice_begin)
        out.extend(ct)
        out.extend(choice_end)
        span["choices_end_idx"].append(len(out) - 1)
    out.extend(a_toks)
    span["answer_end_idx"] = len(out) - 1
    return out, span


def generate_position_ids(span: dict) -> list[int]:
    pos: list[int] = []
    start = 0
    for _ in range(span["question_end_idx"] + 1):
        pos.append(start)
        start += 1
    next_begin = start
    for end in span["choices_end_idx"]:
        for idx in range(end - next_begin + 1):
            pos.append(start + idx)
        next_begin = end + 1
    ans_start = max(pos) + 1
    for idx in range(span["answer_end_idx"] + 1 - len(pos)):
        pos.append(ans_start + idx)
    return pos


def get_attention_mask(span: dict) -> torch.Tensor:
    L = span["answer_end_idx"] + 1
    m = torch.zeros((L, L))
    qlen = span["question_end_idx"] + 1
    m[:qlen, :qlen] = torch.tril(torch.ones(qlen, qlen))
    spans = []
    s = qlen
    for e in span["choices_end_idx"]:
        spans.append((s, e))
        s = e + 1
    for s, e in spans:
        m[s : e + 1, s : e + 1] = torch.tril(torch.ones(e - s + 1, e - s + 1))
    m[qlen:, :qlen] = 1
    ab = spans[-1][1] + 1
    m[ab:, :ab] = 1
    m[ab:, ab:] = torch.tril(torch.ones(L - ab, L - ab))
    return m


def truncate(text: str, max_tokens: int, tokenizer) -> str:
    toks = tokenizer.encode(text)
    if len(toks) <= max_tokens:
        return text
    return tokenizer.decode(toks[:max_tokens])


def build_question(
    state: str, instructions: str, task_type: str, max_state_tokens: int, tokenizer
) -> str:
    return f"{TASK_PROMPTS[task_type]}\nQuestion: {instructions}\n\n{truncate(state, max_state_tokens, tokenizer)}\n"


def encode_example(
    state, instructions, choices, task_type, max_state_tokens, max_choice_tokens, tokenizer
) -> dict:
    if not choices:
        raise ValueError("encode_example requires >= 1 choice text")
    question = build_question(state, instructions, task_type, max_state_tokens, tokenizer)
    choices = [truncate(c, max_choice_tokens, tokenizer) for c in choices]
    ids, span = tokenize_prompt(question, choices, tokenizer)
    pos = generate_position_ids(span)
    mask = get_attention_mask(span)
    end_len = len(tokenizer.encode("</option>"))
    read_idx = [e - end_len for e in span["choices_end_idx"]]
    return {
        "input_ids": ids,
        "position_ids": pos,
        "attention_mask": mask,
        "choice_read_idx": read_idx,
        "answer_end_idx": span["answer_end_idx"],
    }
