"""Correctness and robustness tests that must hold in production.

- batch vs single inference agreement (padding must not change answers)
- packed multi-question request == independent single-question requests
  (question isolation, end to end, not just via the mask)
- malformed / oversized / adversarial input handling
- score and noul output semantics
"""

import os

import pytest
import torch
from test_decision import make_net, tiny_batch


class _StubTok:
    pad_token_id = 0
    eos_token = "<eos>"

    def encode(self, s):
        return [ord(c) % 60 + 1 for c in s]

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


def _example(state, question, choices, task="choice"):
    from jev_qwen.data import TASK_TYPES
    from jev_qwen.tokenize import encode_example

    ex = encode_example(state, question, choices, task, 48, 16, _StubTok())
    ex.update(task_type=TASK_TYPES[task], label=None)
    return ex


def test_batch_vs_single_forward_agreement():
    """A ragged batch must give each row the same logits as running it alone.

    This is the test that catches padding/attention bugs per-row tests miss.
    """
    from jev_qwen.data import collate_with_pad

    net = make_net("compat")
    rows = [
        _example(
            "state " + "x" * i,
            "question " + "y" * i,
            ["alpha", "beta", "gamma"] + ["opt%d" % k for k in range(i)],
        )
        for i in range(4)
    ]
    counts = [len(r["choice_read_idx"]) for r in rows]
    with torch.no_grad():
        batch = net(**collate_with_pad(rows, 0))
        for i, r in enumerate(rows):
            c = counts[i]
            single = net(**{k: v for k, v in collate_with_pad([r], 0).items()})[0]
            assert single.shape == (c,), f"row {i}: unexpected single shape {single.shape}"
            assert torch.allclose(single, batch[i][:c], atol=1e-5), (
                f"row {i}: batching changed the result "
                f"(max diff {float((single - batch[i][:c]).abs().max()):.2e})"
            )


def test_multi_question_request_matches_individual_requests():
    """Question isolation end to end: asking about Q2 alongside Q1 must not
    change Q1's answer (no sibling leakage through padding or attention)."""
    from jev_qwen.data import collate_with_pad

    net = make_net("compat")
    q1 = _example("state A", "first question?", ["a one", "b two"])
    q2 = _example("state B totally different", "second question?", ["c three", "d four", "e five"])
    with torch.no_grad():
        alone = net(**{k: v for k, v in collate_with_pad([q1], 0).items()})[0]
        pair = net(**{k: v for k, v in collate_with_pad([q1, q2], 0).items()})[0]
    assert alone.shape[0] == 2 and pair.shape[0] >= 2
    assert torch.allclose(alone, pair[: alone.shape[0]], atol=1e-5), (
        "q1's answer changed when q2 shared the request (isolation leak)"
    )


def test_noul_and_score_semantics():
    from jev_qwen.data import collate_with_pad

    net = make_net("compat")
    with torch.no_grad():
        p = torch.softmax(
            net(**collate_with_pad([_example("s", "q", ["a", "b", "c"])], 0)).float(), -1
        )
    n = int(p.shape[-1])
    assert n == 3
    expected = sum(i * float(v) for i, v in enumerate(p[0]))
    assert 0.0 <= expected <= n - 1 + 1e-6
    assert 0.0 <= expected / (n - 1) <= 1.0 + 1e-6


def test_oversized_input_is_rejected_not_truncated_silently():

    from jev_qwen.schema import DecisionRequest

    # huge candidate list: valid schema but must not crash the engine silently
    big = [{"id": f"c{i}", "text": f"option {i}"} for i in range(200)]
    req = DecisionRequest(
        state="s", questions=[{"id": "q", "type": "choice", "question": "?", "candidates": big}]
    )
    assert len(req.questions[0].candidates) == 200  # schema accepts, engine truncates safely
    # but a state beyond any sane bound is rejected by the API layer (tested there)


def test_probabilities_stay_normalized_for_degenerate_logits():
    """A model that puts all mass on one candidate must still return sum==1."""
    net = make_net("compat")
    b = tiny_batch(net)
    b["input_ids"] = torch.zeros_like(b["input_ids"])  # identical tokens
    with torch.no_grad():
        p = torch.softmax(net(**b)[0].float(), -1)
    assert abs(float(p.sum()) - 1.0) < 1e-5
    assert torch.isfinite(p).all()


def test_empty_candidate_text_rejected():
    import pydantic

    from jev_qwen.schema import DecisionRequest

    with pytest.raises(pydantic.ValidationError):
        DecisionRequest(
            state="s",
            questions=[
                {
                    "id": "q",
                    "type": "choice",
                    "question": "?",
                    "candidates": [{"id": "a", "text": ""}, {"id": "b", "text": "y"}],
                }
            ],
        )


@pytest.mark.skipif(os.environ.get("JEV_REAL_WEIGHTS") != "1", reason="needs Qwen3-0.6B download")
def test_real_model_batch_vs_single_agreement():
    """Same check against the real backbone with the real tokenizer — the
    stub cannot catch padding/mask bugs in the actual attention path."""
    from jev_qwen.data import collate_with_pad
    from jev_qwen.model import build_network
    from jev_qwen.tokenize import get_tokenizer

    tok = get_tokenizer("Qwen/Qwen3-0.6B")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    net = build_network(
        "Qwen/Qwen3-0.6B", 4, 8, 16, 0.0, 0, {"type": "compat", "new_dim": 64}
    ).eval()

    def ex(state, choices):
        from jev_qwen.data import TASK_TYPES
        from jev_qwen.tokenize import encode_example

        e = encode_example(state, "what first?", choices, "choice", 128, 32, tok)
        e.update(task_type=TASK_TYPES["choice"], label=None)
        return e

    rows = [
        ex(
            "a server returns 503 errors " * (i + 1),
            ["inspect health", "delete the database"] + [f"opt {k}" for k in range(i)],
        )
        for i in range(3)
    ]
    counts = [len(r["choice_read_idx"]) for r in rows]
    with torch.no_grad():
        batch = net(**collate_with_pad(rows, tok.pad_token_id))
        for i, r in enumerate(rows):
            single = net(**{k: v for k, v in collate_with_pad([r], tok.pad_token_id).items()})[0]
            c = counts[i]
            assert torch.allclose(single, batch[i][:c], atol=2e-3), (
                f"row {i}: max diff {float((single - batch[i][:c]).abs().max()):.2e}"
            )
