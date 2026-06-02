"""Sound synthetic-benchmark metrics (issue #53).

(A) ``neighbor_label_agreement`` replaces Moran's I on nominal prototype codes:
a categorical, label-numbering-invariant spatial-coherence measure.
(B) ``proto_kind_accuracy`` compares predicted conserved/sample_specific tags
against the *ground-truth* proto_kind (Hungarian-matched) rather than the
circular ``section_overlap_rate`` self-check.
"""

import math

import numpy as np

from nichelens_st.metrics import (
    neighbor_label_agreement,
    proto_kind_accuracy,
)


# --- (A) neighbor_label_agreement -------------------------------------------

def test_agreement_all_same_label_edges():
    labels = np.array([0, 0, 1, 1])
    edges = np.array([[0, 2], [1, 3]])  # 0-1 (both 0), 2-3 (both 1)
    assert neighbor_label_agreement(labels, edges) == 1.0


def test_agreement_all_crossing_edges():
    labels = np.array([0, 0, 1, 1])
    edges = np.array([[0, 1], [2, 3]])  # 0-2 and 1-3 cross label boundary
    assert neighbor_label_agreement(labels, edges) == 0.0


def test_agreement_is_label_numbering_invariant():
    labels = np.array([0, 0, 1, 1])
    edges = np.array([[0, 1, 2], [1, 2, 3]])
    a = neighbor_label_agreement(labels, edges)
    b = neighbor_label_agreement(np.array([5, 5, 9, 9]), edges)
    assert a == b


def test_agreement_empty_edges_is_nan():
    labels = np.array([0, 1])
    assert math.isnan(neighbor_label_agreement(labels, np.empty((2, 0), dtype=np.int64)))


# --- (B) proto_kind_accuracy (truth-matched, non-circular) ------------------

def test_proto_kind_accuracy_perfect_under_permutation():
    true_id = np.array([0, 0, 1, 1, 2, 2])
    true_kind = ["conserved", "sample_specific", "conserved"]
    # Predicted clusters are a relabeling: truth0->2, truth1->0, truth2->1.
    pred_id = np.array([2, 2, 0, 0, 1, 1])
    pred_kind = ["sample_specific", "conserved", "conserved"]  # indexed by pred id
    assert proto_kind_accuracy(pred_kind, true_kind, pred_id, true_id) == 1.0


def test_proto_kind_accuracy_penalizes_wrong_tag():
    true_id = np.array([0, 0, 1, 1, 2, 2])
    true_kind = ["conserved", "sample_specific", "conserved"]
    pred_id = np.array([2, 2, 0, 0, 1, 1])
    # Flip the tag for the cluster matched to truth-prototype 0 -> 2/3 correct.
    pred_kind = ["sample_specific", "conserved", "sample_specific"]
    acc = proto_kind_accuracy(pred_kind, true_kind, pred_id, true_id)
    assert acc == 2 / 3


def test_proto_kind_accuracy_empty_truth_is_nan():
    assert math.isnan(
        proto_kind_accuracy([], [], np.array([0, 1]), np.array([0, 1]))
    )
