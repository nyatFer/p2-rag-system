"""Paired bootstrap, minimum detectable difference, and Wilson intervals."""

import pytest

from p2 import stats


def test_wilson_known_values():
    lo, hi = stats.wilson(8, 10)
    assert (round(lo, 4), round(hi, 4)) == (0.4902, 0.9433)
    lo, hi = stats.wilson(0, 4)
    assert lo == 0.0 and round(hi, 4) == 0.4899
    assert stats.wilson(4, 4)[1] == 1.0
    assert stats.wilson(0, 0) == [0.0, 1.0]


def test_mdd_and_queries_needed():
    # About 2.8 * sd / sqrt(n): sd 0.3 and 30 queries detect about 0.15.
    assert stats.mdd(0.3, 30) == pytest.approx(2.8016 * 0.3 / 30**0.5, rel=1e-4)
    assert round(stats.mdd(0.3, 30), 3) == 0.153
    assert stats.queries_needed(0.3, 0.10) == 71
    assert stats.queries_needed(0.2, 0.10) == 32


def test_paired_bootstrap_is_deterministic_and_sensible():
    a = [1.0, 0.5, 0.0, 1.0, 0.333, 0.25, 1.0, 0.0, 0.5, 1.0]
    b = [0.5, 0.5, 0.0, 1.0, 1.0, 0.25, 0.2, 0.0, 0.5, 0.0]
    first = stats.paired_bootstrap(a, b)
    assert first == stats.paired_bootstrap(a, b)
    lo, hi = first["ci95"]
    assert lo <= first["mean_diff"] <= hi
    assert first["mean_diff"] == pytest.approx(sum(x - y for x, y in zip(a, b)) / len(a))
    same = stats.paired_bootstrap(a, a)
    assert same["mean_diff"] == 0.0 and same["ci95"] == [0.0, 0.0]
    shifted = stats.paired_bootstrap([x + 0.1 for x in b], b)
    assert shifted["ci95"] == pytest.approx([0.1, 0.1])
    assert stats.reading(shifted["ci95"], "a", "b") == "a higher"
    assert stats.reading([-0.2, -0.1], "a", "b") == "b higher"
    assert stats.reading([-0.1, 0.2], "a", "b") == "not distinguishable"


def test_sd():
    assert stats.sd([1.0]) == 0.0
    assert stats.sd([0.0, 2.0]) == pytest.approx(2**0.5)


def test_unpaired_bootstrap_resamples_each_group_on_its_own():
    hand = [1.0, 0.5, 0.0, 1.0, 0.333, 0.25]
    claude = [1.0, 1.0, 0.5, 1.0, 1.0, 0.5, 1.0, 0.0]
    first = stats.unpaired_bootstrap(hand, claude)
    assert first == stats.unpaired_bootstrap(hand, claude)  # seed 0, so the same every time
    assert (first["n_a"], first["n_b"]) == (6, 8)
    assert first["mean_diff"] == pytest.approx(sum(hand) / 6 - sum(claude) / 8)
    lo, hi = first["ci95"]
    assert lo < first["mean_diff"] < hi
    # Two groups with no spread inside them: every resample gives the same difference.
    flat = stats.unpaired_bootstrap([0.75] * 5, [0.25] * 9)
    assert flat["mean_diff"] == pytest.approx(0.5) and flat["ci95"] == pytest.approx([0.5, 0.5])
    # A clear gap reads as one group higher; groups of different sizes are fine.
    gap = stats.unpaired_bootstrap([0.9, 1.0, 0.8, 0.95, 0.85], [0.1, 0.0, 0.2, 0.15, 0.05, 0.1, 0.0])
    assert stats.reading(gap["ci95"], "hand", "claude") == "hand higher"
    with pytest.raises(ValueError):
        stats.unpaired_bootstrap([], [1.0])
