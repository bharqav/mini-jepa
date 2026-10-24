"""Decision-path tests. Weight-free tests run everywhere; tests needing the
real Qwen checkpoint skip unless JEV_REAL_WEIGHTS=1 (local + nightly, not CI)."""

import os
import random

import pytest
import torch

REAL = os.environ.get("JEV_REAL_WEIGHTS") == "1"
needs_weights = pytest.mark.skipif(not REAL, reason="needs Qwen3-0.6B download")


class FakeBackbone(torch.nn.Module):
    """Deterministic stub: hidden states depend only on input_ids content."""

    def __init__(self, dim=16):
        super().__init__()
        self.config = type("C", (), {"hidden_size": dim, "num_hidden_layers": 2})()
        self.emb = torch.nn.Embedding(64, dim)
        torch.manual_seed(0)
        torch.nn.init.normal_(self.emb.weight)

    def forward(self, input_ids, position_ids=None, attention_mask=None, **kw):
        h = self.emb(input_ids % 64)
        if position_ids is not None:
            h = h + 0.01 * position_ids.unsqueeze(-1).float()
        out = type("O", (), {})()
        out.last_hidden_state = h
        return out


def make_net(head_type="compat", dim=16):
    from jev_qwen.model import DecisionNetwork, build_head

    torch.manual_seed(1)
    return DecisionNetwork(
        FakeBackbone(dim), build_head({"type": head_type, "new_dim": 8, "num_layers": 1}, dim)
    ).eval()


def tiny_batch(net, order=(0, 1, 2)):
    B, L, C = 1, 12, 3
    torch.manual_seed(0)
    return {
        "input_ids": torch.randint(0, 64, (B, L)),
        "position_ids": torch.zeros(B, L, dtype=torch.long),
        "attention_mask": torch.zeros(B, 1, L, L),
        "choice_read_idx": torch.tensor([[2, 5, 8]])[:, list(order)],
        "choice_mask": torch.ones(B, C, dtype=torch.bool),
        "answer_end_idx": torch.tensor([11]),
        "task_type": torch.tensor([0]),
    }


def test_bidirectional_mask_masks_padding_only():
    from jev_qwen.model import DecisionNetwork, build_head

    class Recorder(FakeBackbone):
        def forward(self, input_ids, position_ids=None, attention_mask=None, **kw):
            self.seen_mask = attention_mask
            return super().forward(input_ids, position_ids, attention_mask, **kw)

    backbone = Recorder(16)
    net = DecisionNetwork(
        backbone,
        build_head({"type": "compat", "new_dim": 8}, 16),
        bidirectional=True,
    ).eval()
    batch = tiny_batch(net)
    batch["token_mask"] = torch.ones_like(batch["input_ids"], dtype=torch.bool)
    batch["token_mask"][:, -1] = False
    net(**batch)
    mask = backbone.seen_mask
    assert mask.shape == (1, 1, 12, 12)
    assert torch.equal(mask[0, 0, :, 0], torch.zeros(12))
    assert torch.equal(mask[0, 0, :, -1], torch.full((12,), torch.finfo(mask.dtype).min))


def test_schema_rejects_invalid():
    import pydantic

    from jev_qwen.schema import DecisionRequest

    with pytest.raises(pydantic.ValidationError):
        DecisionRequest(state="s", questions=[])  # empty questions
    with pytest.raises(pydantic.ValidationError):
        DecisionRequest(
            state="s",
            questions=[
                {
                    "id": "q",
                    "type": "choice",
                    "question": "?",
                    "candidates": [{"id": "a", "text": "only one"}],
                }
            ],
        )  # choice needs >= 2
    with pytest.raises(pydantic.ValidationError):
        DecisionRequest(
            state="s",
            questions=[
                {
                    "id": "q",
                    "type": "noul",
                    "question": "?",
                    "candidates": [{"id": "a", "text": "x"}],
                }
            ],
        )  # noul needs exactly 2
    # duplicate candidate ids rejected
    with pytest.raises(pydantic.ValidationError):
        DecisionRequest(
            state="s",
            questions=[
                {
                    "id": "q",
                    "type": "choice",
                    "question": "?",
                    "candidates": [{"id": "a", "text": "x"}, {"id": "a", "text": "y"}],
                }
            ],
        )


def test_tokenize_positions_and_mask_shape():

    from jev_qwen.tokenize import generate_position_ids, get_attention_mask, tokenize_prompt

    tok = lambda s: [ord(c) % 64 for c in s]  # stub tokenizer
    tok = type("T", (), {"encode": staticmethod(tok)})()
    ids, span = tokenize_prompt("state?", ["opt A", "opt B"], tok)
    pos = generate_position_ids(span)
    assert len(pos) == len(ids) == span["answer_end_idx"] + 1
    m = get_attention_mask(span)
    assert m.shape == (len(ids), len(ids))
    # every choice restarts at the same position id (order-invariance mechanism)
    qlen = span["question_end_idx"] + 1
    first_choice_pos = pos[qlen]
    s = qlen
    for e in span["choices_end_idx"]:
        assert pos[s] == first_choice_pos, "choices must share starting position id"
        s = e + 1
    # sibling isolation: choice blocks must not attend to each other
    spans, s = [], qlen
    for e in span["choices_end_idx"]:
        spans.append((s, e))
        s = e + 1
    (a0, a1), (b0, b1) = spans[0], spans[1]
    assert m[a0 : a1 + 1, b0 : b1 + 1].sum() == 0
    assert m[b0 : b1 + 1, a0 : a1 + 1].sum() == 0
    # choices see the question; answer sees everything before it
    assert m[a0 : a1 + 1, :qlen].sum() == (a1 - a0 + 1) * qlen
    ab = spans[-1][1] + 1
    assert m[ab:, :ab].sum() == (len(ids) - ab) * ab


def test_compat_head_exact_permutation_invariance():
    net = make_net("compat")
    with torch.no_grad():
        base = torch.softmax(net(**tiny_batch(net))[0], -1)
        for _ in range(5):
            order = [0, 1, 2]
            random.Random(_).shuffle(order)
            inv = [0] * 3
            for new, old in enumerate(order):
                inv[old] = new
            b = tiny_batch(net, tuple(order))
            p = torch.softmax(net(**b)[0], -1)
            assert torch.allclose(base, p[inv], atol=1e-5), (
                "CompatHead must be exactly order-invariant (independent scoring)"
            )


def test_reference_head_near_invariance():
    net = make_net("reference")
    with torch.no_grad():
        base = torch.softmax(net(**tiny_batch(net))[0], -1)
        worst = 0.0
        for seed in range(5):
            order = [0, 1, 2]
            random.Random(seed).shuffle(order)
            inv = [0] * 3
            for new, old in enumerate(order):
                inv[old] = new
            p = torch.softmax(net(**tiny_batch(net, tuple(order)))[0], -1)
            worst = max(worst, float(0.5 * (base - p[inv]).abs().sum()))
    assert worst < 1e-4, f"reference head order sensitivity {worst} exceeds tolerance"


def test_probabilities_valid_and_aligned():
    net = make_net("compat")
    logits = net(**tiny_batch(net))
    assert torch.isfinite(logits).all()
    probs = torch.softmax(logits, -1)
    assert ((probs >= 0) & (probs <= 1)).all()
    assert torch.allclose(probs.sum(-1), torch.ones(1), atol=1e-5)


def test_no_generation_path():
    net = make_net("compat")
    assert not hasattr(net, "generate")
    assert not hasattr(net.backbone, "generate")


def test_variable_lengths_and_padding():
    from jev_qwen.data import collate_fn

    rows = [
        {
            "input_ids": [1, 2, 3, 4, 5],
            "position_ids": [0, 1, 2, 2, 3],
            "attention_mask": torch.ones(5, 5),
            "choice_read_idx": [2],
            "answer_end_idx": 4,
            "task_type": 0,
            "label": 0,
        },
        {
            "input_ids": [1, 2, 3, 4, 5, 6, 7, 8],
            "position_ids": [0, 1, 2, 2, 2, 3, 4, 5],
            "attention_mask": torch.ones(8, 8),
            "choice_read_idx": [2, 4],
            "answer_end_idx": 7,
            "task_type": 1,
            "label": 1,
        },
    ]
    b = collate_fn(rows, pad_id=0)
    assert b["input_ids"].shape == (2, 8)
    assert b["choice_mask"].tolist() == [[True, False], [True, True]]
    # padded query rows attend only to themselves (no NaN softmax rows)
    assert b["attention_mask"][0, 0, 5, 5] == 0.0
    assert b["attention_mask"][0, 0, 5, 0] == float("-inf")


@needs_weights
def test_real_tokenizer_and_backbone_load():
    from jev_qwen.model import load_backbone
    from jev_qwen.tokenize import get_tokenizer

    tok = get_tokenizer("Qwen/Qwen3-0.6B")
    assert tok.encode("hello")
    bb = load_backbone("Qwen/Qwen3-0.6B", num_layers=2)
    assert bb.config.num_hidden_layers == 2


@needs_weights
def test_real_forward_finite():
    from jev_qwen.data import collate_fn
    from jev_qwen.model import build_network
    from jev_qwen.tokenize import encode_example, get_tokenizer

    tok = get_tokenizer("Qwen/Qwen3-0.6B")
    net = build_network(
        "Qwen/Qwen3-0.6B", 2, 8, 16, 0.0, 0, {"type": "compat", "new_dim": 32}
    ).eval()
    ex = encode_example(
        "A server returns 503.",
        "What first?",
        ["Inspect health", "Delete database"],
        "choice",
        64,
        16,
        tok,
    )
    ex.update(task_type=0, label=None)
    b = collate_fn([ex], pad_id=tok.pad_token_id or 0)
    with torch.no_grad():
        logits = net(**{k: v for k, v in b.items() if k != "labels"})
    assert torch.isfinite(logits).all() and logits.shape == (1, 2)


def test_deterministic_candidate_ordering_in_infer():
    from jev_qwen.infer import DecisionEngine

    class FakeTokenizer:
        pad_token_id = 0
        eos_token_id = 0

        def encode(self, text, **kwargs):
            return [ord(c) % 64 for c in text]

    engine = DecisionEngine.__new__(DecisionEngine)
    engine.tokenizer = FakeTokenizer()
    engine.network = make_net("compat", 16)
    engine.max_state_tokens = 64
    engine.max_choice_tokens = 16
    engine.max_input_chars = 20000
    engine.model_version = "test"
    engine._inference_lock = None
    engine.device = torch.device("cpu")
    engine.precision = "cpu"

    req = type(
        "R",
        (),
        {
            "state": "s",
            "settings": type(
                "S",
                (),
                {
                    "max_state_tokens": 64,
                    "max_candidate_tokens": 16,
                    "temperature": 1.0,
                    "max_batch": None,
                    "include_metadata": False,
                },
            )(),
            "questions": [
                type(
                    "Q",
                    (),
                    {
                        "id": "q",
                        "type": "choice",
                        "question": "?",
                        "candidates": [
                            type("C", (), {"id": "b", "text": "B"})(),
                            type("C", (), {"id": "a", "text": "A"})(),
                        ],
                    },
                )()
            ],
        },
    )()
    resp = engine._decide(req)
    assert [s.id for s in resp.results[0].scores] == ["b", "a"]
    # Same logical request with different input order must produce identical scores
    req2 = type(
        "R",
        (),
        {
            "state": "s",
            "settings": req.settings,
            "questions": [
                type(
                    "Q",
                    (),
                    {
                        "id": "q",
                        "type": "choice",
                        "question": "?",
                        "candidates": [
                            type("C", (), {"id": "a", "text": "A"})(),
                            type("C", (), {"id": "b", "text": "B"})(),
                        ],
                    },
                )()
            ],
        },
    )()
    resp2 = engine._decide(req2)
    assert [s.id for s in resp2.results[0].scores] == ["a", "b"]
    by_id = {s.id: s for s in resp.results[0].scores}
    for s2 in resp2.results[0].scores:
        s1 = by_id[s2.id]
        assert abs(s1.score - s2.score) < 1e-6
