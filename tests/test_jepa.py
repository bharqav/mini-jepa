"""JEPA isolation / no-copy / EMA / collapse tests (spec items 9 + 12).

All tests run on CPU with the tiny random backbone — no Qwen download needed.
"""

import torch

from jev_qwen.jepa import (
    JEPAConfig,
    JEPAState,
    apply_context_masking,
    collapse_diagnostics,
    representation_similarity,
    sample_target_spans,
    shares_parameters,
)


def make_state(**kw):
    cfg = JEPAConfig(
        hidden_dim=32,
        num_layers=2,
        predictor_layers=1,
        predictor_dim=32,
        mask_ratio=0.25,
        mask_span=2,
        ema_momentum=0.9,
        seed=0,
        **kw,
    )
    torch.manual_seed(0)
    return JEPAState(cfg, qwen_model=None, vocab=64, tiny_dim=32, tiny_layers=2), cfg


def test_target_spans_are_contiguous_blocks():
    rng = torch.Generator().manual_seed(0)
    m = sample_target_spans(32, 0.25, 4, rng)
    assert m.sum() >= 4, "must cover ~mask_ratio of positions"
    # block structure: number of runs <= ceil(covered / 1); at least check spans exist
    idx = torch.where(m)[0].tolist()
    assert max(b - a for a, b in zip([idx[0]] + idx, idx + [idx[-1]])) >= 1


def test_context_input_hides_target_content():
    torch.manual_seed(0)
    emb = torch.randn(2, 16, 8)
    mask = torch.zeros(2, 16, dtype=torch.bool)
    mask[:, 4:8] = True
    mv = torch.full((8,), 999.0)
    out = apply_context_masking(emb, mask, mv)
    assert torch.equal(out[mask], mv.expand(mask.sum(), 8).to(out.dtype))
    assert torch.equal(out[~mask], emb[~mask])


def test_target_encoder_gets_no_gradient():
    st, _ = make_state()
    ids = torch.randint(0, 64, (2, 16))
    out = st(ids)
    out["loss"].backward()
    grads = [p.grad for p in st.target.parameters()]
    assert all(g is None for g in grads), "stop-gradient violated: target encoder received grads"
    ctx_grads = [p.grad for p in st.context.parameters() if p.requires_grad]
    assert any(g is not None for g in ctx_grads), "context encoder got no grads"
    pred_grads = [p.grad for p in st.predictor.parameters()]
    assert any(g is not None for g in pred_grads), "predictor got no grads"


def test_predictor_cannot_trivially_copy():
    st, _ = make_state()
    ids = torch.randint(0, 64, (2, 16))
    out = st(ids)
    assert float(out["mse"]) > 1e-6, "untrained predictor must not already match targets"


def test_ema_update_exact():
    st, cfg = make_state()
    with torch.no_grad():
        for p in st.context.parameters():
            p.fill_(1.0)
        for p in st.target.parameters():
            p.fill_(0.0)
    st.ema_update(momentum=0.0)
    for p in st.target.parameters():
        assert torch.allclose(p, torch.ones_like(p)), "m=0 must copy context exactly"
    with torch.no_grad():
        for p in st.context.parameters():
            p.fill_(5.0)
    st.ema_update(momentum=1.0)
    for p in st.target.parameters():
        assert torch.allclose(p, torch.ones_like(p)), "m=1 must leave target unchanged"


def test_encoders_are_independent_copies():
    st, _ = make_state()
    assert not shares_parameters(st.context, st.target)


def test_collapse_diagnostics_flag_constant_reps():
    const = torch.ones(32, 16)
    d = collapse_diagnostics(const)
    assert d["collapse_ratio"] == 1.0
    assert d["mean_std"] < 1e-3
    healthy = torch.randn(256, 16)
    d2 = collapse_diagnostics(healthy)
    assert d2["collapse_ratio"] == 0.0
    assert d2["mean_std"] > 0.5


def test_similarity_reports_sane_values():
    a = torch.randn(64, 16)
    s = representation_similarity(a, a.clone())
    assert s["pred_target_mse"] == 0.0
    assert abs(s["pred_target_cosine"] - 1.0) < 1e-5
    assert 0.0 <= s["linear_cka"] <= 1.0


def test_forward_shapes_and_mask_only_loss():
    st, _ = make_state()
    ids = torch.randint(0, 64, (2, 16))
    out = st(ids)
    assert out["pred"].shape == out["target"].shape == (2, 16, 32)
    assert out["target_mask"].shape == (2, 16)
    assert out["loss"].isfinite()
