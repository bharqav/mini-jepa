"""Pure metric helpers in jev_qwen.evaluate (no model weights needed)."""

import pytest
import torch

from jev_qwen.evaluate import (
    _ece_from_pairs,
    _macro_f1_pure_lists,
    binary_auc,
    fit_temperature,
)


def test_ece_is_zero_for_perfectly_calibrated_bins():
    out = _ece_from_pairs([0.5] * 10, [True, False] * 5)
    assert out["ece"] == pytest.approx(0.0)
    assert out["max_calibration_gap"] == pytest.approx(0.0)


def test_ece_penalises_overconfidence():
    out = _ece_from_pairs([0.9] * 10, [False] * 10)
    assert out["ece"] == pytest.approx(0.9)
    assert out["max_calibration_gap"] == pytest.approx(0.9)


def test_macro_f1_perfect_and_partial():
    assert _macro_f1_pure_lists([0, 1, 1], [0, 1, 1]) == pytest.approx(1.0)
    partial = _macro_f1_pure_lists([0, 1, 1], [0, 0, 1])
    assert 0 < partial < 1


def test_binary_auc_perfect_ranking_and_ties():
    assert binary_auc([0.1, 0.4, 0.8, 0.9], [0, 0, 1, 1]) == pytest.approx(1.0)
    assert binary_auc([0.5, 0.5, 0.5, 0.5], [0, 1, 0, 1]) == pytest.approx(0.5)


def test_fit_temperature_returns_a_positive_scalar():
    logits = torch.tensor([[3.0, 0.0], [0.0, 3.0], [2.5, 0.0], [0.0, 2.5]])
    labels = torch.tensor([0, 1, 0, 1])
    t = fit_temperature(logits, labels)
    assert t > 0
