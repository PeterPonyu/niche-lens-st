"""Synthetic benchmark metrics for NicheLens-ST."""

from __future__ import annotations

from math import comb

import numpy as np


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
    """Mean marker recall@k over prototypes."""
    if k < 1:
        raise ValueError(f"k must be >= 1; got {k}")
    if len(pred_markers) != len(true_markers):
        raise ValueError("pred_markers and true_markers must have the same length")
    if not true_markers:
        return 1.0
    recalls = []
    for pred, true in zip(pred_markers, true_markers, strict=True):
        true_top = list(true[:k])
        if not true_top:
            recalls.append(1.0)
            continue
        pred_top = set(pred[:k])
        recalls.append(len(pred_top.intersection(true_top)) / len(true_top))
    return float(np.mean(recalls))
