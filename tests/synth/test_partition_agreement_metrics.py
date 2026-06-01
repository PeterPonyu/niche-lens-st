"""Issue #87: information-theoretic partition-agreement metrics.

ARI is a single pair-counting index that conflates over- and under-
segmentation under a cluster-count mismatch (model k=10 vs truth k=8).
These tests lock NMI / homogeneity / completeness / V-measure and their
decomposition behaviour so the failure mode is diagnosable.
"""

from __future__ import annotations

import numpy as np

from nichelens_st.metrics import (
    completeness,
    homogeneity,
    normalized_mutual_info,
    v_measure,
)


PERFECT = (
    np.array([0, 0, 1, 1, 2, 2], dtype=np.int64),  # true
    np.array([2, 2, 0, 0, 1, 1], dtype=np.int64),  # pred (relabelled)
)


def test_perfect_agreement_is_one():
    true, pred = PERFECT
    assert normalized_mutual_info(pred, true) == 1.0
    assert homogeneity(pred, true) == 1.0
    assert completeness(pred, true) == 1.0
    assert v_measure(pred, true) == 1.0


def test_permutation_invariance():
    true, pred = PERFECT
    pred2 = np.where(pred == 0, 9, np.where(pred == 1, 7, 5))
    for fn in (normalized_mutual_info, homogeneity, completeness, v_measure):
        assert fn(pred, true) == fn(pred2, true)


def test_single_cluster_gives_zero_nmi():
    """All points in one cluster carries no information about the truth."""
    true = np.array([0, 1, 2, 3], dtype=np.int64)
    pred = np.zeros(4, dtype=np.int64)
    assert normalized_mutual_info(pred, true) == 0.0
    assert v_measure(pred, true) == 0.0


def test_over_segmentation_homogeneous_not_complete():
    """k_pred > k_true, every predicted cluster pure -> homogeneity 1 > completeness."""
    true = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
    pred = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int64)  # each point its own
    h = homogeneity(pred, true)
    c = completeness(pred, true)
    assert h == 1.0
    assert c < 1.0
    assert h > c


def test_under_segmentation_complete_not_homogeneous():
    """k_pred < k_true, each true class fully inside one cluster -> completeness 1 > homogeneity."""
    true = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int64)
    pred = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)  # merge into 2
    h = homogeneity(pred, true)
    c = completeness(pred, true)
    assert c == 1.0
    assert h < 1.0
    assert c > h


def test_v_measure_is_harmonic_mean_and_equals_arithmetic_nmi():
    rng = np.random.default_rng(0)
    true = rng.integers(0, 4, size=200).astype(np.int64)
    pred = rng.integers(0, 6, size=200).astype(np.int64)
    h = homogeneity(pred, true)
    c = completeness(pred, true)
    expected_v = 2 * h * c / (h + c)
    assert abs(v_measure(pred, true) - expected_v) < 1e-12
    # V-measure is exactly NMI with the arithmetic-mean normalisation.
    assert abs(v_measure(pred, true) - normalized_mutual_info(pred, true)) < 1e-12


def test_random_partitions_are_near_zero():
    rng = np.random.default_rng(1)
    true = rng.integers(0, 8, size=2000).astype(np.int64)
    pred = rng.integers(0, 10, size=2000).astype(np.int64)
    assert normalized_mutual_info(pred, true) < 0.05


def test_shape_mismatch_raises():
    import pytest

    with pytest.raises(ValueError):
        normalized_mutual_info(np.zeros(3, np.int64), np.zeros(4, np.int64))
