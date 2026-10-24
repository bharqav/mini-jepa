"""Unit tests for benchmark measurement helpers."""

import pytest

from scripts.benchmark import parse_args, percentile


def test_percentile_uses_nearest_rank_for_small_samples():
    assert percentile([1.0, 2.0, 3.0], 95) == 3.0
    assert percentile([1.0, 2.0, 3.0, 4.0], 50) == 2.0


def test_percentile_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="at least one"):
        percentile([], 95)
    with pytest.raises(ValueError, match="between 0 and 100"):
        percentile([1.0], 101)


def test_benchmark_requires_a_checkpoint():
    with pytest.raises(SystemExit):
        parse_args([])


def test_benchmark_rejects_nonpositive_repeats_and_batches():
    with pytest.raises(SystemExit):
        parse_args(["--checkpoint", "model", "--repeats", "0"])
    with pytest.raises(SystemExit):
        parse_args(["--checkpoint", "model", "--batch-sizes", "1", "0"])
