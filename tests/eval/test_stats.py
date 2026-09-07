import pytest

from doc_intel.eval.stats import bootstrap_mean, paired_difference


def test_bootstrap_interval_contains_the_mean_and_narrows_with_n() -> None:
    small = bootstrap_mean([1.0, 0.0, 1.0, 1.0, 0.0])
    large = bootstrap_mean([1.0, 0.0] * 50)
    assert small.low <= small.mean <= small.high
    assert small.mean == 0.6 and large.mean == 0.5
    assert (large.high - large.low) < (small.high - small.low)
    assert bootstrap_mean([]).n == 0


def test_paired_difference_detects_a_consistent_improvement() -> None:
    a = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    b = [1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0]
    diff = paired_difference(a, b)
    assert diff.mean == 0.3 and diff.low > 0
    noise = paired_difference(a, a[1:] + a[:1])
    assert noise.low <= 0 <= noise.high
    with pytest.raises(ValueError):
        paired_difference([1.0], [1.0, 0.0])


def test_intervals_are_reproducible() -> None:
    assert bootstrap_mean([0.2, 0.9, 0.4], seed=7) == bootstrap_mean([0.2, 0.9, 0.4], seed=7)
    assert str(bootstrap_mean([1.0, 0.0])).startswith("50.0% [")
