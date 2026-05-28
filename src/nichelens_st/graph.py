"""Neighborhood graph and cell-centered subgraph utilities."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable

import numpy as np


class GraphError(ValueError):
    """Raised when graph construction or extraction inputs are invalid."""


def build_graph(
    coords: np.ndarray,
    section_id: np.ndarray,
    k: int = 8,
    method: str = "knn",
) -> np.ndarray:
    """Build a per-section neighborhood graph as int64 COO edges.

    ``method="knn"`` uses ``scipy.spatial.cKDTree`` and clamps each section to
    at most ``m - 1`` outgoing neighbors, so singleton sections produce no edges.
    ``method="delaunay"`` is reserved for a future implementation.
    """
    if method != "knn":
        if method == "delaunay":
            raise NotImplementedError("method='delaunay' is not implemented yet")
        raise GraphError(f"unsupported graph method: {method!r}")
    if k < 0:
        raise GraphError(f"k must be non-negative; got {k}")
    if coords.ndim != 2 or coords.shape[1] not in (2, 3):
        raise GraphError(f"coords must be (n_cells, 2) or (n_cells, 3); got {coords.shape}")
    if section_id.ndim != 1 or section_id.shape[0] != coords.shape[0]:
        raise GraphError(f"section_id must be (n_cells,); got {section_id.shape}")
    if not np.isfinite(coords).all():
        raise GraphError("coords contains NaN or Inf")
    if coords.shape[0] == 0 or k == 0:
        return np.zeros((2, 0), dtype=np.int64)

    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:  # pragma: no cover - dependency declared in pyproject
        raise GraphError("build_graph(method='knn') requires scipy") from exc

    src_chunks: list[np.ndarray] = []
    dst_chunks: list[np.ndarray] = []
    for section in np.unique(section_id):
        idx = np.where(section_id == section)[0]
        m = idx.size
        k_eff = min(k, m - 1)
        if k_eff <= 0:
            continue
        pts = coords[idx]
        # Query k_eff+1 neighbors so we can drop the self-row even when
        # coincident/tied coordinates push self off column 0 (issue #60).
        _dist, nn_local = cKDTree(pts).query(pts, k=k_eff + 1)
        nn_local = np.asarray(nn_local)
        if nn_local.ndim == 1:
            nn_local = nn_local[:, None]
        # Per-row mask: drop the first occurrence of the row's own index,
        # then take the first k_eff survivors. Positional slicing
        # (nn_local[:, 1:]) is only correct when column 0 is guaranteed
        # to be self; cKDTree breaks that with coincident coords.
        row_idx = np.arange(m)
        self_mask = nn_local == row_idx[:, None]
        # argmax returns the first True per row; if no True, drops first col.
        first_self = np.where(self_mask.any(axis=1), self_mask.argmax(axis=1), 0)
        keep = np.ones_like(nn_local, dtype=bool)
        keep[row_idx, first_self] = False
        nn_local = nn_local[keep].reshape(m, k_eff + 1 - 1)[:, :k_eff]
        rows = np.repeat(row_idx, k_eff)
        cols = nn_local.reshape(-1)
        src_chunks.append(idx[rows])
        dst_chunks.append(idx[cols])

    if not src_chunks:
        return np.zeros((2, 0), dtype=np.int64)
    return np.stack([np.concatenate(src_chunks), np.concatenate(dst_chunks)]).astype(np.int64)


@dataclass(frozen=True)
class Subgraph:
    """A cell-centered induced subgraph using original node identifiers."""

    node_ids: np.ndarray
    edges: np.ndarray


def _adjacency(edges: np.ndarray) -> dict[int, set[int]]:
    adj: dict[int, set[int]] = {}
    if edges.size == 0:
        return adj
    for src, dst in edges.T.tolist():
        adj.setdefault(src, set()).add(dst)
        adj.setdefault(dst, set()).add(src)
    return adj


def extract_subgraph(edges: np.ndarray, center: int, k_hop: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Return original node ids and induced original-id edges around ``center``.

    Traversal treats COO edges as undirected neighborhood links. Returned edges
    keep original node ids to remain compatible with the package's COO contract.
    """
    return extract_subgraphs(edges, [center], k_hop=k_hop)[0]


def extract_subgraphs(
    edges: np.ndarray, centers: Iterable[int], k_hop: int = 1
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Batch cell-centered subgraph extraction with one shared adjacency build."""
    if k_hop < 0:
        raise GraphError(f"k_hop must be non-negative; got {k_hop}")
    if edges.ndim != 2 or edges.shape[0] != 2:
        raise GraphError(f"edges must be (2, n_edges); got {edges.shape}")
    if edges.dtype != np.int64:
        raise GraphError(f"edges must be int64; got {edges.dtype}")
    adj = _adjacency(edges)
    out: list[tuple[np.ndarray, np.ndarray]] = []
    for center in centers:
        seen = {int(center)}
        q: deque[tuple[int, int]] = deque([(int(center), 0)])
        while q:
            node, depth = q.popleft()
            if depth == k_hop:
                continue
            for nbr in adj.get(node, ()):  # isolated centers are allowed
                if nbr not in seen:
                    seen.add(nbr)
                    q.append((nbr, depth + 1))
        node_ids = np.array(sorted(seen), dtype=np.int64)
        if edges.size:
            mask = np.isin(edges[0], node_ids) & np.isin(edges[1], node_ids)
            sub_edges = edges[:, mask].astype(np.int64, copy=False)
        else:
            sub_edges = np.zeros((2, 0), dtype=np.int64)
        out.append((node_ids, sub_edges))
    return out
