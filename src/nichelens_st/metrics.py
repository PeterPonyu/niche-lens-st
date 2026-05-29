"""Synthetic benchmark metrics for NicheLens-ST."""

from __future__ import annotations

from math import comb

import numpy as np
from scipy.optimize import linear_sum_assignment


def adjusted_rand(pred_id: np.ndarray, true_id: np.ndarray) -> float:
    """Adjusted Rand Index, invariant to label permutations."""
    pred = np.asarray(pred_id)
    true = np.asarray(true_id)
    if pred.shape != true.shape:
        raise ValueError(f"pred_id and true_id shapes differ: {pred.shape} != {true.shape}")
    n = pred.size
    if n < 2:
        return 1.0
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
        return 1.0
    return float((sum_comb - expected) / denom)


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
    if len(proto_kind) == 0:
        return 1.0
    sections = set(section.tolist())
    correct = 0
    for p, kind in enumerate(proto_kind):
        seen = set(section[proto == p].tolist())
        if kind == "conserved":
            correct += seen == sections
        elif kind == "sample_specific":
            correct += len(seen) < len(sections)
        else:
            raise ValueError(f"invalid proto_kind at {p}: {kind!r}")
    return float(correct / len(proto_kind))


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
    if not true_markers:
        return 1.0
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
            recalls.append(1.0)
            continue
        pred_top = set(pred[:k])
        recalls.append(len(pred_top.intersection(true_top)) / len(true_top))
    return float(np.mean(recalls))


def score_against_truth(
    pred_prototype_id: np.ndarray,
    pred_marker_table: list[list[int]],
    pred_proto_kind: list[str],
    true_prototype_id: np.ndarray,
    true_marker_genes: list[list[int]],
    section_id: np.ndarray,
    edges: np.ndarray,
    k: int = 5,
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
        "MoranI": morans_i(pred_id, edges),
        "section_overlap_rate": section_overlap_rate(
            pred_id, section_id, pred_proto_kind
        ),
        "marker_recall_at_k": marker_recall_at_k(
            aligned_pred_markers, true_marker_genes, k=k
        ),
    }
