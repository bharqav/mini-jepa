"""Uncertainty helpers: bootstrap CIs and cross-seed aggregation."""

import math

import pytest

from jev_qwen.stats import (
    accuracy_ci,
    bootstrap_ci,
    compare_seeds,
    mean_ci,
    paired_delta_ci,
    seed_summary,
)


def test_bootstrap_ci_empty_is_nan():
    out = bootstrap_ci([], lambda v: sum(v) / len(v))
    assert out["n"] == 0
    assert math.isnan(out["point"])


def test_bootstrap_ci_constant_values_are_tight():
    out = mean_ci([1.0] * 20, n_boot=200)
    assert out["point"] == pytest.approx(1.0)
    assert out["lo"] == pytest.approx(1.0)
    assert out["hi"] == pytest.approx(1.0)
    assert out["n"] == 20


def test_accuracy_ci_matches_point_estimate():
    correct = [True] * 7 + [False] * 3
    out = accuracy_ci(correct, n_boot=300)
    assert out["point"] == pytest.approx(0.7)
    assert out["lo"] <= out["point"] <= out["hi"]


def test_paired_delta_ci_detects_a_consistent_gap():
    a = [True] * 8 + [False] * 2
    b = [True] * 4 + [False] * 6
    out = paired_delta_ci(a, b, n_boot=300)
    assert out["point"] == pytest.approx(0.4)
    assert out["lo"] > 0


def test_paired_delta_ci_rejects_mismatched_lengths():
    with pytest.raises(AssertionError):
        paired_delta_ci([True], [True, False])


def test_seed_summary_handles_empty_single_and_multiple():
    assert math.isnan(seed_summary([])["mean"])
    one = seed_summary([0.5])
    assert one == {"mean": 0.5, "std": 0.0, "half_width": 0.0, "min": 0.5, "max": 0.5, "n": 1}
    many = seed_summary([0.4, 0.6, 0.5])
    assert many["mean"] == pytest.approx(0.5)
    assert many["std"] == pytest.approx(0.1)
    assert many["n"] == 3


def test_compare_seeds_reports_effect_and_significance():
    out = compare_seeds([0.60, 0.62, 0.61], [0.50, 0.51, 0.49])
    assert out["delta"] == pytest.approx(0.11)
    assert out["significant_95"] is True
    weak = compare_seeds([0.60, 0.62, 0.61], [0.59, 0.60, 0.58])
    assert weak["significant_95"] is False
