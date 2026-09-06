"""Uncertainty for eval numbers: bootstrap confidence intervals and paired A/B differences.

A single accuracy number over 24 documents or 100 questions is a point estimate with wide
error bars. Resampling the per-item outcomes with replacement gives an interval; comparing
two configurations on the *same* items (paired) removes the item difficulty from the
difference, so a small real improvement is visible and a lucky one is not mistaken for it.
"""

import random
import statistics
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Interval:
    mean: float
    low: float
    high: float
    n: int

    def __str__(self) -> str:
        return f"{self.mean:.1%} [{self.low:.1%}, {self.high:.1%}] n={self.n}"


def bootstrap_mean(
    values: Sequence[float], samples: int = 2000, confidence: float = 0.95, seed: int = 0
) -> Interval:
    if not values:
        return Interval(0.0, 0.0, 0.0, 0)
    rng = random.Random(seed)
    n = len(values)
    means = sorted(statistics.fmean(rng.choices(values, k=n)) for _ in range(samples))
    tail = (1 - confidence) / 2
    return Interval(
        statistics.fmean(values),
        means[int(tail * samples)],
        means[int((1 - tail) * samples) - 1],
        n,
    )


def paired_difference(
    a: Sequence[float],
    b: Sequence[float],
    samples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Interval:
    """Bootstrap interval of mean(b - a) over paired items. Excludes 0 -> a real difference."""
    if len(a) != len(b):
        raise ValueError(f"paired series must have equal length, got {len(a)} and {len(b)}")
    diffs = [y - x for x, y in zip(a, b, strict=True)]
    return bootstrap_mean(diffs, samples, confidence, seed)
