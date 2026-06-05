"""Tests for adjusted_mutual_info and macro_f1 (#312).

TDD: these tests are written to fail until the functions are implemented in
metrics.py.  sklearn is imported ONLY inside individual test functions — it is
not a src/ dependency.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from nichelens_st.metrics import adjusted_mutual_info, macro_f1


def _assert_nan(x: float) -> None:
    assert math.isnan(x), f"expected NaN, got {x!r}"


# ---------------------------------------------------------------------------
# adjusted_mutual_info
# ---------------------------------------------------------------------------


def test_ami_perfect_and_permuted():
    """AMI == 1.0 for identical partition regardless of label permutation."""
    pred = np.array([0, 0, 1, 1, 2, 2])
    true = np.array([2, 2, 0, 0, 1, 1])  # same structure, permuted labels
    assert adjusted_mutual_info(pred, true) == pytest.approx(1.0)


def test_ami_n_less_than_2_is_nan():
    """n < 2 → NaN (mirrors adjusted_rand contract, metrics.py:11)."""
    _assert_nan(
        adjusted_mutual_info(np.array([], dtype=np.int64), np.array([], dtype=np.int64))
    )
    _assert_nan(adjusted_mutual_info(np.array([0]), np.array([0])))


def test_ami_both_single_cluster_is_nan():
    """Both partitions degenerate to one cluster → denom 0 → NaN."""
    _assert_nan(adjusted_mutual_info(np.array([0, 0, 0]), np.array([0, 0, 0])))


def test_ami_independent_is_near_zero():
    """AMI ≈ 0 (small absolute value) for independent random labelings."""
    rng = np.random.default_rng(0)
    pred = rng.integers(0, 4, size=300)
    true = rng.integers(0, 4, size=300)
    ami = adjusted_mutual_info(pred, true)
    assert abs(ami) < 0.1, f"expected AMI near 0 for independent labelings, got {ami}"


def test_ami_mismatched_shapes_raises():
    """Mismatched shapes raise ValueError."""
    with pytest.raises(ValueError, match="shapes differ"):
        adjusted_mutual_info(np.array([0, 1]), np.array([0]))


def test_ami_known_value_regression():
    """AMI exact value for a fixed 3x3 case, anchored to sklearn.

    The expected value was cross-checked against
    ``sklearn.metrics.adjusted_mutual_info_score(true, pred, average_method='max')``
    (== 0.31967265056964705). Pinned here as a sklearn-free regression guard so
    CI — where sklearn is not installed — still validates the hypergeometric
    E[MI] adjustment, not just the perfect/independent limits.
    """
    pred = np.array([0, 0, 0, 1, 1, 1, 2, 2])
    true = np.array([0, 0, 1, 1, 1, 2, 2, 2])
    assert adjusted_mutual_info(pred, true) == pytest.approx(
        0.31967265056964705, abs=1e-9
    )


def test_ami_sklearn_cross_check():
    """AMI matches sklearn (average_method='max') within 1e-9 on random labelings.

    Skipped where sklearn is absent (e.g. CI) — sklearn is not a src/ or test
    dependency; :func:`test_ami_known_value_regression` covers the exact value
    without it. This test adds breadth (random labelings) where sklearn exists.
    """
    pytest.importorskip("sklearn")
    from sklearn.metrics import adjusted_mutual_info_score  # cross-check only

    rng = np.random.default_rng(42)
    for trial in range(3):
        pred = rng.integers(0, 5, size=80)
        true = rng.integers(0, 4, size=80)
        expected = float(adjusted_mutual_info_score(true, pred, average_method="max"))
        actual = adjusted_mutual_info(pred, true)
        assert actual == pytest.approx(expected, abs=1e-9), (
            f"trial={trial}: ours={actual:.12f}, sklearn={expected:.12f}"
        )


# ---------------------------------------------------------------------------
# macro_f1
# ---------------------------------------------------------------------------


def test_macro_f1_perfect_and_permuted():
    """macro_f1 == 1.0 for identical partitions up to label permutation."""
    pred = np.array([1, 1, 0, 0, 2, 2])
    true = np.array([0, 0, 1, 1, 2, 2])
    assert macro_f1(pred, true) == pytest.approx(1.0)


def test_macro_f1_known_partial_overlap():
    """Hand-computed case: pred has 2 clusters, true has 2 classes, imperfect.

    contingency (pred-row × true-col):
        [[2, 1],
         [0, 1]]
    Hungarian assigns pred-0 → true-0, pred-1 → true-1 (maximise overlap).
    true-0: TP=2, FP=1, FN=0  → F1 = 4/5
    true-1: TP=1, FP=0, FN=1  → F1 = 2/3
    macro_f1 = (4/5 + 2/3) / 2 = 11/15
    """
    pred = np.array([0, 0, 0, 1])
    true = np.array([0, 0, 1, 1])
    assert macro_f1(pred, true) == pytest.approx(11.0 / 15.0, abs=1e-12)


def test_macro_f1_unmatched_true_class_contributes_zero():
    """When more true classes than pred clusters, unmatched ones contribute F1=0."""
    # 1 pred cluster, 2 true classes → one true class necessarily unmatched
    pred = np.array([0, 0, 0, 0])
    true = np.array([0, 0, 1, 1])
    # contingency: [[2, 2]]
    # Hungarian: pred-0 → true-0 (or true-1, both give 2; scipy picks lower index)
    # best match: pred-0 → true-0 (TP=2, FP=2, FN=0, F1=4/6=2/3)
    # true-1: unmatched → F1=0
    # macro = (2/3 + 0) / 2 = 1/3
    result = macro_f1(pred, true)
    assert result == pytest.approx(1.0 / 3.0, abs=1e-12)


def test_macro_f1_empty_is_nan():
    """Empty input → NaN (issue #83 NaN policy)."""
    _assert_nan(macro_f1(np.array([], dtype=np.int64), np.array([], dtype=np.int64)))


def test_macro_f1_mismatched_shapes_raises():
    """Mismatched shapes raise ValueError."""
    with pytest.raises(ValueError, match="shapes differ"):
        macro_f1(np.array([0, 1, 2]), np.array([0, 1]))
