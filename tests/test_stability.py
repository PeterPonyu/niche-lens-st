"""Tests for the #296 niche-stability primitives (seed reproducibility + cohort).

Torch-free: these are pure numpy/scipy reductions over already-fitted
``prototype_id`` labellings. The heavy multi-seed *refit* lives in the driver
(scripts/compute_niche_stability.py); here we pin the math.
"""

from __future__ import annotations

import numpy as np
import pytest

from nichelens_st import stability


# --------------------------------------------------------------------------
# Seed-stability ARI matrix
# --------------------------------------------------------------------------
def test_pairwise_ari_matrix_identity_for_same_labelling():
    lab = np.array([0, 0, 1, 1, 2, 2])
    M = stability.pairwise_ari_matrix([lab, lab.copy(), lab.copy()])
    assert M.shape == (3, 3)
    assert np.allclose(np.diag(M), 1.0)
    # identical labellings -> ARI 1 off-diagonal too
    assert np.allclose(M, 1.0)
    # symmetric
    assert np.allclose(M, M.T)


def test_pairwise_ari_matrix_label_permutation_is_invariant():
    a = np.array([0, 0, 1, 1, 2, 2])
    b = np.array([2, 2, 0, 0, 1, 1])  # relabel of the same partition
    M = stability.pairwise_ari_matrix([a, b])
    assert M[0, 1] == pytest.approx(1.0)


def test_seed_stability_summary_offdiagonal_stats():
    M = np.array([[1.0, 0.8, 0.6], [0.8, 1.0, 0.4], [0.6, 0.4, 1.0]])
    s = stability.seed_stability_summary(M)
    assert s["n_seeds"] == 3
    # off-diagonal upper-triangle = [0.8, 0.6, 0.4]
    assert s["mean_offdiag_ari"] == pytest.approx(0.6)
    assert s["min_offdiag_ari"] == pytest.approx(0.4)
    assert s["sd_offdiag_ari"] == pytest.approx(np.std([0.8, 0.6, 0.4]))


def test_seed_stability_summary_single_seed_is_undefined():
    s = stability.seed_stability_summary(np.array([[1.0]]))
    assert s["n_seeds"] == 1
    assert s["mean_offdiag_ari"] is None


# --------------------------------------------------------------------------
# tag_conserved parity with model._separation_head (no drift)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("cov", [1.0, 0.8, 0.5])
def test_tag_conserved_matches_separation_head(cov):
    try:  # model._separation_head pulls the torch encoder module
        from nichelens_st.model import _separation_head
    except (ImportError, AttributeError) as exc:
        pytest.skip(f"torch encoder unavailable: {exc}")

    rng = np.random.default_rng(0)
    proto = rng.integers(0, 4, size=60)
    section = rng.integers(0, 5, size=60)
    n_protos = 4
    assert stability.tag_conserved(proto, section, n_protos, cov) == _separation_head(
        proto, section, n_protos, cov
    )


def test_tag_conserved_single_section_is_unknown():
    proto = np.array([0, 1, 0, 1])
    section = np.zeros(4, dtype=int)
    assert stability.tag_conserved(proto, section, 2, 1.0) == ["unknown", "unknown"]


# --------------------------------------------------------------------------
# Coverage sweep
# --------------------------------------------------------------------------
def test_coverage_sweep_monotone_conserved_fraction():
    # A prototype present in 4/5 sections is conserved at cov<=0.8 but not at 1.0.
    proto = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1])
    section = np.array([0, 1, 2, 3, 0, 1, 2, 3, 4])  # p0 in {0,1,2,3}, p1 in all 5
    rows = stability.coverage_sweep(proto, section, n_protos=2, thresholds=[1.0, 0.8])
    by_t = {r["min_section_coverage"]: r for r in rows}
    # at 1.0 only p1 conserved (1/2); at 0.8 both (2/2)
    assert by_t[1.0]["conserved_fraction"] == pytest.approx(0.5)
    assert by_t[0.8]["conserved_fraction"] == pytest.approx(1.0)
    # section_overlap_rate present and in [0,1]
    assert 0.0 <= by_t[1.0]["section_overlap_rate"] <= 1.0


# --------------------------------------------------------------------------
# Prototype matching (Sankey content)
# --------------------------------------------------------------------------
def test_prototype_matching_hungarian_overlap():
    # section A protos {0,1}; section B is A with labels swapped -> match 0<->1, 1<->0
    a = np.array([0, 0, 1, 1])
    b = np.array([1, 1, 0, 0])
    matches = stability.prototype_matching(a, b)
    pairs = {(m["proto_a"], m["proto_b"]): m["overlap"] for m in matches}
    assert pairs[(0, 1)] == 2
    assert pairs[(1, 0)] == 2
    # total matched overlap == n_cells for a clean relabel
    assert sum(m["overlap"] for m in matches) == 4
