"""Synthetic benchmark metrics for NicheLens-ST."""

from __future__ import annotations

from math import comb

import numpy as np
from scipy.optimize import linear_sum_assignment


def adjusted_rand(pred_id: np.ndarray, true_id: np.ndarray) -> float:
    """Adjusted Rand Index, invariant to label permutations.

    Returns ``NaN`` when the score is undefined (empty input, single cell, or
    both partitions degenerate to one cluster) — issue #83. Previously these
    cases returned ``1.0`` ("perfect"), silently masking collapsed outputs.
    """
    pred = np.asarray(pred_id)
    true = np.asarray(true_id)
    if pred.shape != true.shape:
        raise ValueError(f"pred_id and true_id shapes differ: {pred.shape} != {true.shape}")
    n = pred.size
    if n < 2:
        return float("nan")
    pred_labels, pred_inv = np.unique(pred, return_inverse=True)
    true_labels, true_inv = np.unique(true, return_inverse=True)
    contingency = np.zeros((pred_labels.size, true_labels.size), dtype=np.int64)
    np.add.at(contingency, (pred_inv, true_inv), 1)
    sum_comb = sum(comb(int(v), 2) for v in contingency.ravel())
    row_comb = sum(comb(int(v), 2) for v in contingency.sum(axis=1))
    col_comb = sum(comb(int(v), 2) for v in contingency.sum(axis=0))
    total = comb(n, 2)
    expected = row_comb * col_comb / total if total else 0.0
    max_index = 0.5 * (row_comb + col_comb)
    denom = max_index - expected
    if denom == 0:
        # Both partitions collapse to one cluster — ARI is undefined.
        return float("nan")
    return float((sum_comb - expected) / denom)


def _contingency(pred_id: np.ndarray, true_id: np.ndarray) -> tuple[np.ndarray, int]:
    """Joint count matrix between predicted clusters (rows) and true classes (cols)."""
    pred = np.asarray(pred_id)
    true = np.asarray(true_id)
    if pred.shape != true.shape:
        raise ValueError(f"pred_id and true_id shapes differ: {pred.shape} != {true.shape}")
    n = pred.size
    _, pred_inv = np.unique(pred, return_inverse=True)
    _, true_inv = np.unique(true, return_inverse=True)
    table = np.zeros(
        (int(pred_inv.max(initial=-1)) + 1, int(true_inv.max(initial=-1)) + 1),
        dtype=np.int64,
    )
    if n:
        np.add.at(table, (pred_inv, true_inv), 1)
    return table, n


def _entropy(counts: np.ndarray, n: int) -> float:
    """Shannon entropy (nats) of a count vector; 0.0 for an empty/degenerate set."""
    counts = counts[counts > 0]
    if counts.size <= 1 or n == 0:
        return 0.0
    p = counts / n
    return float(-np.sum(p * np.log(p)))


def _conditional_entropy(table: np.ndarray, n: int, *, given_rows: bool) -> float:
    """Conditional entropy (nats); ``given_rows`` conditions on the row marginal.

    ``given_rows=True`` -> ``H(true | pred)`` (column uncertainty given a cluster).
    Computed directly (not as ``H - I``) so it is exactly 0.0 when each
    conditioning group is pure, avoiding the float-rounding noise of subtraction.
    """
    if n == 0:
        return 0.0
    rows, cols = np.nonzero(table > 0)
    nij = table[rows, cols].astype(np.float64)
    marginal = (table.sum(axis=1)[rows] if given_rows else table.sum(axis=0)[cols]).astype(
        np.float64
    )
    return float(-np.sum((nij / n) * (np.log(nij) - np.log(marginal))))


def homogeneity(pred_id: np.ndarray, true_id: np.ndarray) -> float:
    """Each cluster contains members of a single class (1.0 = fully homogeneous).

    ``homogeneity = 1 - H(true | pred) / H(true)``; defined as 1.0 when the truth
    has no entropy (a single class). Label-permutation invariant like ARI.
    """
    table, n = _contingency(pred_id, true_id)
    h_true = _entropy(table.sum(axis=0), n)
    if h_true == 0.0:
        return 1.0
    cond = _conditional_entropy(table, n, given_rows=True)
    return float(min(1.0, max(0.0, 1.0 - cond / h_true)))


def completeness(pred_id: np.ndarray, true_id: np.ndarray) -> float:
    """All members of a class are assigned to the same cluster (1.0 = complete).

    ``completeness = 1 - H(pred | true) / H(pred)``; defined as 1.0 when the
    prediction has no entropy (a single cluster). Counterpart of homogeneity.
    """
    table, n = _contingency(pred_id, true_id)
    h_pred = _entropy(table.sum(axis=1), n)
    if h_pred == 0.0:
        return 1.0
    cond = _conditional_entropy(table, n, given_rows=False)
    return float(min(1.0, max(0.0, 1.0 - cond / h_pred)))


def v_measure(pred_id: np.ndarray, true_id: np.ndarray) -> float:
    """Harmonic mean of homogeneity and completeness (equals arithmetic-mean NMI)."""
    h = homogeneity(pred_id, true_id)
    c = completeness(pred_id, true_id)
    if h + c == 0.0:
        return 0.0
    return float(2 * h * c / (h + c))


def normalized_mutual_info(pred_id: np.ndarray, true_id: np.ndarray) -> float:
    """NMI with arithmetic-mean normalisation: ``I / ((H(pred)+H(true))/2)``.

    1.0 for identical partitions (up to label permutation), ~0 for independent
    ones. Equals the V-measure; reported alongside ARI so a cluster-count
    mismatch can be diagnosed as over- vs under-segmentation.
    """
    table, n = _contingency(pred_id, true_id)
    h_pred = _entropy(table.sum(axis=1), n)
    h_true = _entropy(table.sum(axis=0), n)
    denom = 0.5 * (h_pred + h_true)
    if denom == 0.0:
        return 0.0
    # I(true; pred) = H(true) - H(true | pred); direct conditional form is exact.
    mi = h_true - _conditional_entropy(table, n, given_rows=True)
    return float(min(1.0, max(0.0, mi / denom)))


def morans_i(values: np.ndarray, edges: np.ndarray) -> float:
    """Moran's I over graph edges for scalar labels/values."""
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("values must be one-dimensional")
    if edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError(f"edges must be (2, n_edges); got {edges.shape}")
    if x.size == 0 or edges.shape[1] == 0:
        return 0.0
    centered = x - x.mean()
    denom = float(np.dot(centered, centered))
    if denom == 0.0:
        return 0.0
    src = edges[0]
    dst = edges[1]
    if src.size and (src.min() < 0 or dst.min() < 0 or src.max() >= x.size or dst.max() >= x.size):
        raise ValueError("edges contain indices outside values")
    w = edges.shape[1]
    return float((x.size / w) * np.sum(centered[src] * centered[dst]) / denom)


def neighbor_label_agreement(labels: np.ndarray, edges: np.ndarray) -> float:
    """Fraction of graph edges whose endpoints share a label (issue #53).

    A sound, label-numbering-invariant measure of the spatial coherence of a
    *categorical* field (e.g. ``prototype_id``). Moran's I instead treats the
    nominal integer codes as a magnitude, so its value depends on the arbitrary
    cluster numbering (which ``_kmeans`` assigns "in order of first appearance");
    edge label-agreement does not. Returns ``NaN`` for an empty edge set.
    """
    lab = np.asarray(labels)
    if edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError(f"edges must be (2, n_edges); got {edges.shape}")
    if edges.shape[1] == 0:
        return float("nan")
    src = edges[0]
    dst = edges[1]
    if src.size and (
        src.min() < 0 or dst.min() < 0 or src.max() >= lab.size or dst.max() >= lab.size
    ):
        raise ValueError("edges contain indices outside labels")
    return float(np.mean(lab[src] == lab[dst]))


def section_overlap_rate(
    prototype_id: np.ndarray, section_id: np.ndarray, proto_kind: list[str]
) -> float:
    """Fraction of prototype tags whose observed section coverage matches kind.

    Conserved prototypes must appear in every section; sample-specific prototypes
    must appear in a strict subset. Unassigned catalog entries count as correct
    only when tagged sample-specific.
    """
    proto = np.asarray(prototype_id)
    section = np.asarray(section_id)
    if proto.shape != section.shape:
        raise ValueError(f"prototype_id and section_id shapes differ: {proto.shape} != {section.shape}")
    # Empty catalog → undefined, not a free "perfect" score (issue #83).
    if len(proto_kind) == 0:
        return float("nan")
    sections = set(section.tolist())
    # "unknown" tags are emitted when the conserved/sample_specific
    # distinction is undefined (single-section input, issue #85); they
    # contribute neither to the numerator nor the denominator.
    scorable = [(p, k) for p, k in enumerate(proto_kind) if k != "unknown"]
    if not scorable:
        return float("nan")
    correct = 0
    for p, kind in scorable:
        seen = set(section[proto == p].tolist())
        if kind == "conserved":
            correct += seen == sections
        elif kind == "sample_specific":
            correct += len(seen) < len(sections)
        else:
            raise ValueError(f"invalid proto_kind at {p}: {kind!r}")
    return float(correct / len(scorable))


def marker_recall_at_k(
    pred_markers: list[list[int]], true_markers: list[list[int]], k: int = 5
) -> float:
    """Mean marker recall@k over prototypes.

    Raises ``ValueError`` if ``k`` exceeds the number of available ground-truth
    markers for any (non-empty) prototype. Otherwise recall@k would silently
    degrade to recall@len(true) -- a result labeled recall@k but measured at a
    smaller depth. Generate the ground truth with at least ``k`` markers per
    prototype (``generate_instance(n_markers=...)``).
    """
    if k < 1:
        raise ValueError(f"k must be >= 1; got {k}")
    if len(pred_markers) != len(true_markers):
        raise ValueError("pred_markers and true_markers must have the same length")
    # Empty catalog → undefined, not a free "perfect" score (issue #83).
    if not true_markers:
        return float("nan")
    recalls = []
    for pred, true in zip(pred_markers, true_markers, strict=True):
        if true and len(true) < k:
            raise ValueError(
                f"k={k} exceeds available true markers ({len(true)}) for a "
                "prototype; recall@k would silently degrade to "
                f"recall@{len(true)}. Provide at least k markers per prototype "
                "(see generate_instance(n_markers=...))."
            )
        true_top = list(true[:k])
        if not true_top:
            # Per-prototype truth is empty: recall is undefined for that row.
            # Skip it so a catalog with no true markers anywhere yields NaN
            # rather than the previous silent 1.0 (issue #83).
            continue
        pred_top = set(pred[:k])
        recalls.append(len(pred_top.intersection(true_top)) / len(true_top))
    if not recalls:
        return float("nan")
    return float(np.mean(recalls))


def proto_kind_accuracy(
    pred_proto_kind: list[str],
    true_proto_kind: list[str],
    pred_prototype_id: np.ndarray,
    true_prototype_id: np.ndarray,
) -> float:
    """Accuracy of predicted conserved/sample_specific tags vs truth (issue #53).

    Hungarian-matches predicted prototypes to truth prototypes by cell overlap
    (the same alignment :func:`score_against_truth` uses for markers), then
    compares each matched pair's *predicted* ``proto_kind`` against the
    *ground-truth* tag. Unlike :func:`section_overlap_rate` -- which re-derives
    section coverage from the very ``prototype_id`` that produced ``proto_kind``
    and is therefore ~1.0 by construction (a self-consistency check, not an
    accuracy) -- this evaluates against the independent truth ``proto_kind``.

    Returns ``NaN`` when there is nothing to score (no truth tags, no predicted
    cells, or no overlap-matched pairs).
    """
    pred_id = np.asarray(pred_prototype_id, dtype=np.int64)
    true_id = np.asarray(true_prototype_id, dtype=np.int64)
    if not true_proto_kind or pred_id.size == 0 or true_id.size == 0:
        return float("nan")
    n_pred = int(pred_id.max()) + 1
    n_true = int(true_id.max()) + 1
    contingency = np.zeros((n_pred, n_true), dtype=np.int64)
    np.add.at(contingency, (pred_id, true_id), 1)
    row_ind, col_ind = linear_sum_assignment(-contingency)
    correct = 0
    total = 0
    for p, t in zip(row_ind, col_ind):
        if (
            p >= len(pred_proto_kind)
            or t >= len(true_proto_kind)
            or contingency[p, t] == 0
        ):
            continue
        total += 1
        correct += int(pred_proto_kind[p] == true_proto_kind[t])
    if total == 0:
        return float("nan")
    return float(correct / total)


def score_against_truth(
    pred_prototype_id: np.ndarray,
    pred_marker_table: list[list[int]],
    pred_proto_kind: list[str],
    true_prototype_id: np.ndarray,
    true_marker_genes: list[list[int]],
    section_id: np.ndarray,
    edges: np.ndarray,
    k: int = 5,
    true_proto_kind: list[str] | None = None,
) -> dict[str, float]:
    """Score model output against synthetic ground truth (issues #81/#82).

    ARI is label-permutation-invariant; marker recall is computed after
    Hungarian-aligning predicted prototypes to truth via the contingency
    overlap so per-prototype marker lists are compared on matched
    indices. Unmatched truth prototypes contribute empty pred markers
    (recall 0). Required so the harness scores the *fitted model's
    output* — not truth-vs-truth as in #81.
    """
    pred_id = np.asarray(pred_prototype_id, dtype=np.int64)
    true_id = np.asarray(true_prototype_id, dtype=np.int64)
    n_true = len(true_marker_genes)
    truth_to_pred: dict[int, int] = {}
    if pred_id.size:
        n_pred = int(pred_id.max()) + 1
        n_true_ids = int(true_id.max()) + 1
        contingency = np.zeros((n_pred, n_true_ids), dtype=np.int64)
        np.add.at(contingency, (pred_id, true_id), 1)
        # Hungarian minimises cost; negate counts to maximise overlap.
        row_ind, col_ind = linear_sum_assignment(-contingency)
        truth_to_pred = {int(t): int(p) for p, t in zip(row_ind, col_ind)}
    aligned_pred_markers: list[list[int]] = []
    for t in range(n_true):
        p = truth_to_pred.get(t)
        if p is None or p >= len(pred_marker_table):
            aligned_pred_markers.append([])
        else:
            aligned_pred_markers.append(pred_marker_table[p])
    return {
        "ARI": adjusted_rand(pred_id, true_id),
        "NMI": normalized_mutual_info(pred_id, true_id),
        "homogeneity": homogeneity(pred_id, true_id),
        "completeness": completeness(pred_id, true_id),
        "v_measure": v_measure(pred_id, true_id),
        "label_agreement": neighbor_label_agreement(pred_id, edges),
        "proto_kind_accuracy": (
            proto_kind_accuracy(pred_proto_kind, true_proto_kind, pred_id, true_id)
            if true_proto_kind is not None
            else float("nan")
        ),
        "marker_recall_at_k": marker_recall_at_k(
            aligned_pred_markers, true_marker_genes, k=k
        ),
    }
