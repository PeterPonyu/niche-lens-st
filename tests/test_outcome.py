"""Niche-abundance to per-sample outcome association (#320 / #339).

These pin the contract of :mod:`nichelens_st.outcome`, which turns a per-cell
prototype assignment into a per-(sample x prototype) relative-abundance matrix
and tests whether each prototype's abundance associates with a *per-sample*
label -- the "is this niche enriched in responders / high-grade / short-survival
samples?" question. The unit of analysis is the **sample**, never the cell (no
per-cell pseudo-replication), and the test is compositional-aware (centered
log-ratio) with Benjamini-Hochberg FDR across prototypes.

Everything here is pure numpy/scipy and deterministic given a seed.
"""

from __future__ import annotations

import numpy as np
import pytest

from nichelens_st.outcome import (
    benjamini_hochberg,
    clr_transform,
    niche_outcome_association,
    sample_prototype_abundance,
)


# --------------------------------------------------------------------------
# sample_prototype_abundance
# --------------------------------------------------------------------------


def test_sample_prototype_abundance_hand_example():
    sample_id = np.array([0, 0, 0, 1, 1])
    prototype_id = np.array([0, 1, 1, 0, 0])
    abundance, samples = sample_prototype_abundance(
        sample_id, prototype_id, n_prototypes=2
    )
    assert samples.tolist() == [0, 1]
    np.testing.assert_allclose(abundance, [[1 / 3, 2 / 3], [1.0, 0.0]])
    # rows are relative abundances -> sum to 1
    np.testing.assert_allclose(abundance.sum(axis=1), [1.0, 1.0])


def test_sample_prototype_abundance_includes_empty_prototype_columns():
    sample_id = np.array([0, 0, 1])
    prototype_id = np.array([0, 0, 0])
    abundance, _ = sample_prototype_abundance(
        sample_id, prototype_id, n_prototypes=3
    )
    assert abundance.shape == (2, 3)
    # prototypes 1 and 2 never occur -> all-zero columns
    np.testing.assert_allclose(abundance[:, 1:], 0.0)


def test_sample_prototype_abundance_infers_n_prototypes_from_max():
    sample_id = np.array([0, 1])
    prototype_id = np.array([0, 2])
    abundance, _ = sample_prototype_abundance(sample_id, prototype_id)
    assert abundance.shape == (2, 3)  # max code 2 -> 3 columns


# --------------------------------------------------------------------------
# clr_transform
# --------------------------------------------------------------------------


def test_clr_transform_uniform_row_is_zero():
    comp = np.full((2, 4), 0.25)
    out = clr_transform(comp)
    np.testing.assert_allclose(out, 0.0, atol=1e-9)


def test_clr_transform_rows_sum_to_zero():
    rng = np.random.default_rng(0)
    comp = rng.dirichlet(np.ones(5), size=7)
    out = clr_transform(comp)
    np.testing.assert_allclose(out.sum(axis=1), 0.0, atol=1e-9)


def test_clr_transform_handles_zeros_without_inf():
    comp = np.array([[0.0, 0.5, 0.5], [1.0, 0.0, 0.0]])
    out = clr_transform(comp)
    assert np.isfinite(out).all()


# --------------------------------------------------------------------------
# benjamini_hochberg
# --------------------------------------------------------------------------


def test_benjamini_hochberg_known_values():
    q = benjamini_hochberg([0.001, 0.5])
    np.testing.assert_allclose(q, [0.002, 0.5])


def test_benjamini_hochberg_is_monotone_and_bounded():
    q = benjamini_hochberg([0.01, 0.02, 0.03, 0.04, 0.05])
    # equal-spaced p_i = i*0.01 with n=5 all collapse to 0.05 after BH
    np.testing.assert_allclose(q, 0.05)
    assert ((q >= 0.0) & (q <= 1.0)).all()


# --------------------------------------------------------------------------
# niche_outcome_association -- planted signal recovery + null behaviour
# --------------------------------------------------------------------------


def _planted_cohort(seed: int = 0):
    """12 samples, 5 prototypes; prototype 0 is enriched in label-positive
    samples and prototype 1 is depleted (mirror), while prototypes 2-4 carry the
    same per-sample value in both groups (no group signal)."""
    rng = np.random.default_rng(seed)
    n_per_group = 6
    rest = rng.uniform(0.10, 0.20, size=(n_per_group, 3))  # protos 2,3,4 shared
    pos = np.column_stack(
        [np.full(n_per_group, 0.45), np.full(n_per_group, 0.05), rest]
    )
    neg = np.column_stack(
        [np.full(n_per_group, 0.05), np.full(n_per_group, 0.45), rest]
    )
    comp = np.vstack([pos, neg])
    comp = comp / comp.sum(axis=1, keepdims=True)  # renormalise to compositions
    labels = np.array([1] * n_per_group + [0] * n_per_group)
    return comp, labels


def test_association_recovers_planted_enrichment():
    comp, labels = _planted_cohort()
    res = niche_outcome_association(comp, labels, label_kind="binary")
    proto = {r["prototype"]: r for r in res["prototypes"]}
    # enriched prototype 0: significant, positive effect (higher in label 1)
    assert proto[0]["q_value"] < 0.05
    assert proto[0]["effect_size"] > 0.0
    # depleted prototype 1: significant, negative effect
    assert proto[1]["q_value"] < 0.05
    assert proto[1]["effect_size"] < 0.0
    # shared prototypes 2-4: no group signal -> not significant
    for p in (2, 3, 4):
        assert proto[p]["q_value"] > 0.05


def test_planted_signal_vanishes_under_label_permutation():
    """The enriched/depleted prototypes (0 and 1) must lose significance when the
    labels no longer track them: each group is given 3 of the high-proto0 samples
    and 3 of the low-proto0 samples, so proto0/proto1 no longer separate."""
    comp, _ = _planted_cohort()
    labels = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
    res = niche_outcome_association(comp, labels, label_kind="binary")
    proto = {r["prototype"]: r for r in res["prototypes"]}
    assert proto[0]["q_value"] > 0.05
    assert proto[1]["q_value"] > 0.05


def test_association_continuous_outcome_spearman():
    rng = np.random.default_rng(1)
    n = 14
    proto0 = np.linspace(0.05, 0.6, n)  # monotone in the outcome
    rest = rng.uniform(0.1, 0.2, size=(n, 3))
    comp = np.column_stack([proto0, 0.7 - proto0, rest])
    comp = comp / comp.sum(axis=1, keepdims=True)
    outcome = np.arange(n, dtype=float)  # monotone increasing
    res = niche_outcome_association(comp, outcome, label_kind="continuous")
    proto = {r["prototype"]: r for r in res["prototypes"]}
    assert proto[0]["q_value"] < 0.05
    assert proto[0]["effect_size"] > 0.0  # positive Spearman rho


def test_association_binary_requires_exactly_two_groups():
    comp = np.full((4, 3), 1 / 3)
    with pytest.raises(ValueError):
        niche_outcome_association(comp, np.array([0, 0, 0, 0]), label_kind="binary")


def test_association_result_is_json_serialisable():
    import json

    comp, labels = _planted_cohort()
    res = niche_outcome_association(comp, labels, label_kind="binary")
    json.dumps(res)  # must not raise (all values are plain python/float)
