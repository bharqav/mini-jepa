"""Data helpers for JEPA pretraining on packed decision text."""

from __future__ import annotations

import random
from collections.abc import Iterable, Sequence

import torch


def record_text(record: dict) -> str:
    """Flatten a decision record into one self-supervised text stream.

    The label is deliberately excluded: JEPA pretraining is unsupervised.
    """
    choices = "\n".join(record.get("choices") or [])
    return f"{record['state']}\n{record['instructions']}\n{choices}".strip()


def pack_sequences(
    sequences: Iterable[Sequence[int]],
    max_len: int,
    eos_id: int,
    max_sequences: int | None = None,
) -> list[list[int]]:
    """Pack token streams into fixed-length rows separated by EOS.

    Rows are always exactly ``max_len`` tokens, so the JEPA masking path never
    sees padding tokens as if they were content. A final partial row is padded
    with EOS; callers can treat that as a no-op separator.
    """
    if max_len < 2:
        raise ValueError("max_len must be >= 2")
    packed: list[list[int]] = []
    current: list[int] = []
    for raw in sequences:
        ids = [int(t) for t in raw][:max_len]
        if not ids:
            continue
        if len(ids) == max_len:
            if current:
                packed.append(current + [eos_id] * (max_len - len(current)))
                current = []
            packed.append(ids)
        else:
            if current and len(current) + 1 + len(ids) > max_len:
                packed.append(current + [eos_id] * (max_len - len(current)))
                current = []
            if current:
                current.append(eos_id)
            current.extend(ids)
        if max_sequences is not None and len(packed) >= max_sequences:
            return packed[:max_sequences]
    if current and (max_sequences is None or len(packed) < max_sequences):
        packed.append(current + [eos_id] * (max_len - len(current)))
    return packed[:max_sequences] if max_sequences is not None else packed


def build_packed_dataset(
    records: Iterable[dict],
    tokenizer,
    max_len: int,
    max_sequences: int | None = None,
    seed: int = 0,
) -> list[list[int]]:
    """Tokenize and pack records deterministically for one pretraining run."""
    texts: list[list[int]] = []
    for record in records:
        text = record_text(record)
        if text:
            texts.append(tokenizer.encode(text, add_special_tokens=False))
    random.Random(seed).shuffle(texts)
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("tokenizer must define eos_token_id for packing")
    return pack_sequences(texts, max_len, int(eos_id), max_sequences=max_sequences)


def make_batches(
    sequences: list[list[int]], batch_size: int, seed: int = 0, shuffle: bool = True
) -> list[torch.Tensor]:
    """Materialise deterministic batches; the caller moves them to the device."""
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    order = list(range(len(sequences)))
    if shuffle:
        random.Random(seed).shuffle(order)
    batches = []
    for start in range(0, len(order), batch_size):
        rows = [sequences[i] for i in order[start : start + batch_size]]
        batches.append(torch.tensor(rows, dtype=torch.long))
    return batches
