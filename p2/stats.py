"""The statistics p2 score reports: paired intervals, the minimum detectable difference, Wilson intervals.

- paired_bootstrap: resample the queries (10,000 times, seed 0) and take the middle 95% of the
  resampled mean differences; when the interval includes 0, the two systems are not distinguishable
  on this gold set.
- unpaired_bootstrap: the same for two separate groups of queries (your hand queries and the ones a
  model drafted, say): each resample draws the queries of each group from that group only, and the
  interval is the middle 95% of the resampled differences of the two group means.
- mdd: the smallest true difference a paired test on n queries detects 80% of the time at the 5%
  level, from the standard deviation of the per-query differences:
  (z(0.975) + z(0.80)) * sd / sqrt(n), about 2.8 * sd / sqrt(n).
- queries_needed: the same formula solved for n, about (2.8 * sd / difference) ** 2.
- wilson: a 95% interval for a share (k successes out of n), which stays sensible near 0 and 1.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from statistics import NormalDist

import numpy as np

RESAMPLES = 10_000
SEED = 0


def z(p: float) -> float:
    """The standard normal quantile."""
    return NormalDist().inv_cdf(p)


def sd(values: Sequence[float]) -> float:
    """The sample standard deviation (n - 1 in the denominator); 0 for fewer than two values."""
    n = len(values)
    if n < 2:
        return 0.0
    m = sum(values) / n
    return math.sqrt(sum((v - m) ** 2 for v in values) / (n - 1))


def paired_bootstrap(a: Sequence[float], b: Sequence[float], resamples: int = RESAMPLES, seed: int = SEED, level: float = 0.95) -> dict:
    """The mean of a - b over paired per-query values and a percentile bootstrap interval for it."""
    if len(a) != len(b):
        raise ValueError("paired values need the same queries on both sides")
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    n = len(d)
    if n == 0:
        return {"n": 0, "mean_diff": 0.0, "ci95": [0.0, 0.0], "sd_diff": 0.0}
    rng = np.random.default_rng(seed)
    means = np.empty(resamples, dtype=np.float64)
    step = max(1, 2_000_000 // n)  # bounded memory for large query sets
    for start in range(0, resamples, step):
        stop = min(start + step, resamples)
        idx = rng.integers(0, n, size=(stop - start, n))
        means[start:stop] = d[idx].mean(axis=1)
    tail = (1 - level) / 2 * 100
    lo, hi = np.percentile(means, [tail, 100 - tail])
    return {"n": n, "mean_diff": float(d.mean()), "ci95": [float(lo), float(hi)], "sd_diff": sd(d.tolist())}


def _resampled_means(rng: np.random.Generator, values: np.ndarray, resamples: int) -> np.ndarray:
    """The means of `resamples` resamples (with replacement) of `values`, in bounded memory."""
    n = len(values)
    means = np.empty(resamples, dtype=np.float64)
    step = max(1, 2_000_000 // n)
    for start in range(0, resamples, step):
        stop = min(start + step, resamples)
        means[start:stop] = values[rng.integers(0, n, size=(stop - start, n))].mean(axis=1)
    return means


def unpaired_bootstrap(a: Sequence[float], b: Sequence[float], resamples: int = RESAMPLES, seed: int = SEED, level: float = 0.95) -> dict:
    """The difference of two group means, mean(a) - mean(b), and a percentile bootstrap interval for it.

    The groups are separate queries (not the same queries scored twice), so each resample draws len(a)
    values from a and len(b) values from b, each group on its own; the interval is the middle `level`
    of the resampled differences. Both groups need at least one value."""
    va = np.asarray(a, dtype=np.float64)
    vb = np.asarray(b, dtype=np.float64)
    if len(va) == 0 or len(vb) == 0:
        raise ValueError("each group needs at least one value")
    rng = np.random.default_rng(seed)
    diffs = _resampled_means(rng, va, resamples) - _resampled_means(rng, vb, resamples)
    tail = (1 - level) / 2 * 100
    lo, hi = np.percentile(diffs, [tail, 100 - tail])
    return {"n_a": len(va), "n_b": len(vb), "mean_diff": float(va.mean() - vb.mean()), "ci95": [float(lo), float(hi)]}


def mdd(sd_diff: float, n: int, alpha: float = 0.05, power: float = 0.80) -> float:
    """Minimum detectable difference of a paired test on n queries."""
    if n <= 0:
        return float("inf")
    return (z(1 - alpha / 2) + z(power)) * sd_diff / math.sqrt(n)


def queries_needed(sd_diff: float, difference: float, alpha: float = 0.05, power: float = 0.80) -> int:
    """How many queries a paired test needs to detect `difference` with the given power."""
    if difference <= 0:
        raise ValueError("the difference to detect must be above 0")
    return math.ceil(((z(1 - alpha / 2) + z(power)) * sd_diff / difference) ** 2)


def wilson(k: int, n: int, level: float = 0.95) -> list[float]:
    """Wilson score interval [low, high] for k successes in n trials; [0, 1] when n is 0."""
    if n == 0:
        return [0.0, 1.0]
    q = z(1 - (1 - level) / 2)
    p = k / n
    centre = (p + q * q / (2 * n)) / (1 + q * q / n)
    half = q * math.sqrt(p * (1 - p) / n + q * q / (4 * n * n)) / (1 + q * q / n)
    # At k = 0 and k = n the bound is exactly 0 or 1; the formula lands one rounding step short on
    # some builds of Python, so those ends are set exactly.
    low = 0.0 if k == 0 else max(0.0, centre - half)
    high = 1.0 if k == n else min(1.0, centre + half)
    return [low, high]


IDENTICAL = "identical on every query"


def reading(ci95: Sequence[float], a: str, b: str) -> str:
    """How to read a paired interval of a - b."""
    lo, hi = ci95
    if lo > 0:
        return f"{a} higher"
    if hi < 0:
        return f"{b} higher"
    return "not distinguishable"
