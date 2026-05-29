"""Regression tests for #81 + #82: synthetic harness scores model output.

#81: harness compared ground truth to itself; ``fit_niche_model`` was
never invoked. #82: ``NicheModelResult`` had no ``marker_table`` so
marker recall could only be run truth-vs-truth. These tests pin the
fixed contract without requiring torch (encoder is gated).
"""

from __future__ import annotations

import math

import numpy as np

from nichelens_st.metrics import (
    adjusted_rand,
    marker_recall_at_k,
    score_against_truth,
    section_overlap_rate,
)
from nichelens_st.model import NicheModelResult, _compute_marker_table
from nichelens_st.synth import generate_instance


def _assert_nan(x: float) -> None:
    assert math.isnan(x), f"expected NaN, got {x!r}"


def test_empty_input_not_perfect():
    """Issue #83: empty/degenerate inputs must NOT score 1.0 across metrics."""
    # adjusted_rand: empty and single-cell inputs are undefined.
    _assert_nan(adjusted_rand(np.array([], dtype=np.int64), np.array([], dtype=np.int64)))
    _assert_nan(adjusted_rand(np.array([0], dtype=np.int64), np.array([0], dtype=np.int64)))

    # marker_recall_at_k: empty catalog is undefined.
    _assert_nan(marker_recall_at_k([], [], k=5))
    # marker_recall_at_k: empty per-prototype truth must not be a free point.
    _assert_nan(marker_recall_at_k([[1, 2]], [[]], k=5))

    # section_overlap_rate: empty catalog is undefined.
    z3 = np.zeros(3, dtype=np.int64)
    _assert_nan(section_overlap_rate(z3, z3, []))


def test_compute_marker_table_ranks_by_mean_expression():
    """``_compute_marker_table`` derives markers from X means (issue #82)."""
    X = np.array(
        [[1.0, 5.0, 2.0, 0.5], [1.2, 5.5, 2.1, 0.4], [0.1, 0.2, 9.0, 3.0]],
        dtype=np.float32,
    )
    proto = np.array([0, 0, 1], dtype=np.int64)
    table = _compute_marker_table(X, proto, n_protos=2, top_k=3)
    assert table[0][0] == 1  # gene 1 dominates proto 0
    assert table[1][0] == 2  # gene 2 dominates proto 1


def test_niche_model_result_has_marker_table_field():
    """marker_table is part of the MVP output contract (#82)."""
    r = NicheModelResult(
        H=np.zeros((1, 1), dtype=np.float32),
        prototype_id=np.array([0], dtype=np.int64),
        proto_kind=["conserved"],
        marker_table=[[0]],
    )
    assert r.marker_table == [[0]]


def test_shuffled_prototypes_metric_drops():
    """Regression for #81: random-shuffled model output must NOT score
    1.0 like the old truth-vs-truth harness did."""
    inst = generate_instance(
        n_sections=2, n_cells_per_section=50, n_genes=20, seed=0
    )
    rng = np.random.default_rng(42)
    shuffled = rng.permutation(inst.prototype_id)
    score = score_against_truth(
        pred_prototype_id=shuffled,
        pred_marker_table=[list(m) for m in inst.marker_genes],
        pred_proto_kind=inst.proto_kind,
        true_prototype_id=inst.prototype_id,
        true_marker_genes=inst.marker_genes,
        section_id=inst.section_id,
        edges=inst.edges,
        k=5,
    )
    assert score["ARI"] < 0.5, (
        f"shuffled-prototype scoring still returned ARI={score['ARI']}; "
        "harness is not evaluating predicted assignments"
    )
    # Ceiling sanity: identity input recovers ARI=1 on the same harness.
    identity = score_against_truth(
        pred_prototype_id=inst.prototype_id,
        pred_marker_table=[list(m) for m in inst.marker_genes],
        pred_proto_kind=inst.proto_kind,
        true_prototype_id=inst.prototype_id,
        true_marker_genes=inst.marker_genes,
        section_id=inst.section_id,
        edges=inst.edges,
        k=5,
    )
    assert identity["ARI"] == 1.0
    assert identity["marker_recall_at_k"] == 1.0
