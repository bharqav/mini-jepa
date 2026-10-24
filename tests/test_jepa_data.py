"""Packing helpers for real-Qwen JEPA pretraining."""

import torch

from jev_qwen.jepa_data import build_packed_dataset, make_batches, pack_sequences, record_text


class WordTokenizer:
    eos_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [1 + (hash(word) % 50) for word in text.split()]


def test_record_text_includes_state_instructions_and_choices():
    text = record_text({"state": "s", "instructions": "i", "choices": ["a", "b"]})
    assert text == "s\ni\na\nb"


def test_pack_sequences_uses_exact_rows_and_eos():
    packed = pack_sequences([[1, 2, 3], [4, 5], [6]], max_len=8, eos_id=0)
    assert all(len(row) == 8 for row in packed)
    assert packed[0][:3] == [1, 2, 3]
    assert packed[0][3] == 0


def test_pack_sequences_caps_output():
    packed = pack_sequences([[1] * 8] * 10, max_len=8, eos_id=0, max_sequences=3)
    assert len(packed) == 3


def test_build_packed_dataset_is_deterministic():
    records = [
        {"state": "alpha beta", "instructions": "pick", "choices": ["one", "two"]},
        {"state": "gamma delta", "instructions": "pick", "choices": ["three"]},
    ]
    tokenizer = WordTokenizer()
    a = build_packed_dataset(records, tokenizer, max_len=8, seed=3)
    b = build_packed_dataset(records, tokenizer, max_len=8, seed=3)
    assert a == b
    assert all(len(row) == 8 for row in a)


def test_make_batches_shapes():
    sequences = [[i] * 4 for i in range(5)]
    batches = make_batches(sequences, batch_size=2, seed=0, shuffle=False)
    assert [tuple(b.shape) for b in batches] == [(2, 4), (2, 4), (1, 4)]
    assert torch.equal(batches[0][0], torch.full((4,), 0))
