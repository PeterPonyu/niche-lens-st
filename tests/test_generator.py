"""Generator marker depth must match the evaluated recall@k.

The synthetic generator emits ``n_markers`` ground-truth markers per prototype.
``marker_recall_at_k`` must not silently degrade to recall@n_markers when called
with ``k > n_markers``: it should raise, and a generator instance with matching
marker depth should make recall@k well-defined.
"""

from __future__ import annotations

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
