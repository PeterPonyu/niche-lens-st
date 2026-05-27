"""Deterministic synthetic niche-recovery generator.

Schema mirrors ``docs/SYNTHETIC_BENCHMARK.md``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SynthInstance:
    X: np.ndarray            # (n_cells, n_genes) float32
    coords: np.ndarray       # (n_cells, 2) float32
    section_id: np.ndarray   # (n_cells,) int64
    edges: np.ndarray        # (2, n_cells * k_nn) int64
    prototype_id: np.ndarray # (n_cells,) int64
    proto_kind: list[str]    # len == K_conserved + J_specific
    marker_genes: list[list[int]]  # len == K_conserved + J_specific


def generate_instance(
    n_sections: int = 4,
    n_cells_per_section: int = 2000,
    n_genes: int = 500,
    K_conserved: int = 6,
    J_specific: int = 2,
    noise_sigma: float = 0.5,
    k_nn: int = 8,
    seed: int = 0,
) -> SynthInstance:
    """Build one synthetic niche-recovery instance.

    Section ``s`` always contains all conserved prototypes plus exactly one
    sample-specific prototype (index ``K_conserved + (s % J_specific)``) when
    ``J_specific > 0``.
    """
    rng = np.random.default_rng(seed)
    n_cells = n_sections * n_cells_per_section
    n_protos = K_conserved + J_specific

    proto_means = rng.exponential(scale=1.0, size=(n_protos, n_genes)).astype(np.float32)

    coords = rng.uniform(0.0, 1.0, size=(n_cells, 2)).astype(np.float32)
    section_id = np.repeat(np.arange(n_sections), n_cells_per_section).astype(np.int64)

    prototype_id = np.empty(n_cells, dtype=np.int64)
    for s in range(n_sections):
        rows = np.arange(s * n_cells_per_section, (s + 1) * n_cells_per_section)
        if J_specific > 0:
            allowed = list(range(K_conserved)) + [K_conserved + (s % J_specific)]
        else:
            allowed = list(range(K_conserved))
        prototype_id[rows] = rng.choice(allowed, size=rows.size).astype(np.int64)

    proto_kind = ["conserved"] * K_conserved + ["sample_specific"] * J_specific

    X = proto_means[prototype_id] + rng.normal(
        0.0, noise_sigma, size=(n_cells, n_genes)
    ).astype(np.float32)

    edges = _build_knn_edges_per_section(coords, section_id, k=k_nn)

    marker_genes = [
        list(map(int, np.argsort(proto_means[p])[-5:][::-1])) for p in range(n_protos)
    ]

    return SynthInstance(
        X=X,
        coords=coords,
        section_id=section_id,
        edges=edges,
        prototype_id=prototype_id,
        proto_kind=proto_kind,
        marker_genes=marker_genes,
    )


def _build_knn_edges_per_section(
    coords: np.ndarray, section_id: np.ndarray, k: int
) -> np.ndarray:
    src_chunks: list[np.ndarray] = []
    dst_chunks: list[np.ndarray] = []
    for s in np.unique(section_id):
        idx = np.where(section_id == s)[0]
        pts = coords[idx]
        d2 = np.sum((pts[:, None, :] - pts[None, :, :]) ** 2, axis=-1)
        np.fill_diagonal(d2, np.inf)
        nn = np.argpartition(d2, k, axis=1)[:, :k]
        rows = np.repeat(np.arange(pts.shape[0]), k)
        cols = nn.flatten()
        src_chunks.append(idx[rows])
        dst_chunks.append(idx[cols])
    return np.stack([np.concatenate(src_chunks), np.concatenate(dst_chunks)]).astype(np.int64)
