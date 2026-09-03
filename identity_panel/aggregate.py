"""Aggregation strategies for turning a round's judge scores into one group score.

The choice of aggregator materially affects the H1 result (independent panels), so
it is a first-class, swappable knob rather than a hardcoded mean.
"""

from __future__ import annotations

from collections import Counter
from statistics import mean as _mean, median as _median
from typing import Callable, Sequence

Aggregator = Callable[[Sequence[int]], float]


def mean(scores: Sequence[int]) -> float:
    return float(_mean(scores)) if scores else float("nan")


def median(scores: Sequence[int]) -> float:
    return float(_median(scores)) if scores else float("nan")


def majority(scores: Sequence[int]) -> float:
    """Modal score; ties broken toward the lower score."""
    if not scores:
        return float("nan")
    counts = Counter(scores)
    top = max(counts.values())
    return float(min(s for s, n in counts.items() if n == top))


AGGREGATORS: dict[str, Aggregator] = {
    "mean": mean,
    "median": median,
    "majority": majority,
}


def aggregate(name: str, scores: Sequence[int]) -> float:
    try:
        fn = AGGREGATORS[name]
    except KeyError:
        raise ValueError(f"unknown aggregator {name!r}; options: {sorted(AGGREGATORS)}") from None
    return fn(scores)
