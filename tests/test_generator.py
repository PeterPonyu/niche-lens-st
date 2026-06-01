"""Regression tests for the synthetic generator.

Marker depth must match the evaluated recall@k: the synthetic generator emits
``n_markers`` ground-truth markers per prototype, and ``marker_recall_at_k``
must not silently degrade to recall@n_markers when called with ``k > n_markers``
(issue #80).

Separately, when ``J_specific > n_sections`` the indices
``K_conserved + n_sections .. K_conserved + J_specific - 1`` were never assigned
to any cell but still appeared in ``proto_kind`` and ``marker_genes``, silently
inflating ``section_overlap_rate``. The fix clips ``J_specific = min(J_specific,
n_sections)`` with a warning so the catalog and the assignments stay consistent
(issue #72).
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from nichelens_st.metrics import marker_recall_at_k
from nichelens_st.synth import generate_instance

_KW = dict(
    n_sections=2,
    n_cells_per_section=20,
    n_genes=30,
    K_conserved=3,
    J_specific=1,
    k_nn=3,
    seed=0,
)


def test_marker_recall_k_gt_5_not_silent():
    # Default generator emits 5 markers per prototype.
    inst = generate_instance(**_KW)
    assert all(len(m) == 5 for m in inst.marker_genes)
    pred = [list(m) for m in inst.marker_genes]

    # k > available markers must raise instead of silently scoring recall@5.
    with pytest.raises(ValueError, match="silently degrade"):
        marker_recall_at_k(pred, inst.marker_genes, k=10)

    # With matching marker depth, recall@k is well-defined (perfect self-recall).
    inst10 = generate_instance(**_KW, n_markers=10)
    assert all(len(m) == 10 for m in inst10.marker_genes)
    pred10 = [list(m) for m in inst10.marker_genes]
    assert marker_recall_at_k(pred10, inst10.marker_genes, k=10) == 1.0


def test_n_markers_controls_argsort_depth():
    inst = generate_instance(**_KW, n_markers=7)
    assert inst.proto_means is not None
    for p, markers in enumerate(inst.marker_genes):
        expected = list(map(int, np.argsort(inst.proto_means[p])[-7:][::-1]))
        assert markers == expected


def test_n_markers_validated():
    with pytest.raises(ValueError, match="n_markers"):
        generate_instance(**_KW, n_markers=0)


def test_J_specific_clipped():
    """With J_specific > n_sections, the realised catalog has at most n_sections specific protos."""
    n_sections = 4
    K_conserved = 3
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        inst = generate_instance(
            n_sections=n_sections,
            n_cells_per_section=20,
            n_genes=8,
            K_conserved=K_conserved,
            J_specific=10,
            noise_sigma=0.1,
            k_nn=4,
            seed=0,
        )

    specific_count = sum(1 for k in inst.proto_kind if k == "sample_specific")
    assert specific_count <= n_sections, (
        f"expected at most {n_sections} sample_specific prototypes; got {specific_count}"
    )

    # No phantom prototypes: every catalog entry is assigned to ≥1 cell.
    assigned_ids = set(int(p) for p in inst.prototype_id)
    catalog_ids = set(range(len(inst.proto_kind)))
    assert catalog_ids.issubset(assigned_ids), (
        f"phantom prototypes in catalog: {catalog_ids - assigned_ids}"
    )

    # marker_genes must mirror the realised catalog.
    assert len(inst.marker_genes) == len(inst.proto_kind)

    # The clip should warn the caller about the silent reduction.
    assert any("J_specific" in str(w.message) for w in caught), (
        "expected a warning that J_specific was clipped"
    )
