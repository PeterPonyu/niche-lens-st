"""Deterministic synthetic niche-recovery generator.

Schema mirrors ``docs/SYNTHETIC_BENCHMARK.md``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from nichelens_st.graph import build_graph


@dataclass
class SynthInstance:
    X: np.ndarray            # (n_cells, n_genes) float32
    coords: np.ndarray       # (n_cells, 2) float32
    section_id: np.ndarray   # (n_cells,) int64
    edges: np.ndarray        # (2, n_cells * k_nn) int64
    prototype_id: np.ndarray # (n_cells,) int64
    proto_kind: list[str]    # len == K_conserved + J_specific
    marker_genes: list[list[int]]  # len == K_conserved + J_specific
    proto_means: np.ndarray | None = None  # (n_protos, n_genes) ground truth means


def generate_instance(
    n_sections: int = 4,
    n_cells_per_section: int = 2000,
    n_genes: int = 500,
    K_conserved: int = 6,
    J_specific: int = 2,
    noise_sigma: float = 0.5,
    k_nn: int = 8,
    n_markers: int = 5,
    seed: int = 0,
) -> SynthInstance:
    """Build one synthetic niche-recovery instance.

    Section ``s`` always contains all conserved prototypes plus exactly one
    sample-specific prototype (index ``K_conserved + (s % J_specific)``) when
    ``J_specific > 0``.

    ``n_markers`` sets the number of ground-truth marker genes emitted per
    prototype. It must be at least the largest ``k`` at which
    :func:`nichelens_st.metrics.marker_recall_at_k` will be evaluated, otherwise
    recall@k would silently degrade to recall@n_markers.
    """
    if n_sections < 1:
        raise ValueError(f"n_sections must be >= 1; got {n_sections}")
    if n_cells_per_section < 1:
        raise ValueError(f"n_cells_per_section must be >= 1; got {n_cells_per_section}")
    if n_genes < 1:
        raise ValueError(f"n_genes must be >= 1; got {n_genes}")
    if K_conserved < 0 or J_specific < 0:
        raise ValueError("K_conserved and J_specific must be non-negative")
    if K_conserved + J_specific < 1:
        raise ValueError("at least one prototype is required")
    if k_nn < 0:
        raise ValueError(f"k_nn must be non-negative; got {k_nn}")
    if n_markers < 1:
        raise ValueError(f"n_markers must be >= 1; got {n_markers}")

    # Section ``s`` assigns the sample-specific prototype
    # ``K_conserved + (s % J_specific)``, so the indices
    # ``K_conserved + n_sections .. K_conserved + J_specific - 1`` would never
    # be assigned to any cell and would appear as phantom catalog entries
    # (issue #72). Clip ``J_specific`` to the number of sections so the
    # catalog matches the realised assignments.
    if J_specific > n_sections:
        warnings.warn(
            f"J_specific={J_specific} > n_sections={n_sections}; clipping to "
            f"{n_sections} so every sample_specific prototype is realised "
            "(issue #72)",
            stacklevel=2,
        )
        J_specific = n_sections

    rng = np.random.default_rng(seed)
    n_cells = n_sections * n_cells_per_section
    n_protos = K_conserved + J_specific

    proto_means = rng.exponential(scale=1.0, size=(n_protos, n_genes)).astype(np.float32)

    coords = rng.uniform(0.0, 1.0, size=(n_cells, 2)).astype(np.float32)
    section_id = np.repeat(np.arange(n_sections), n_cells_per_section).astype(np.int64)

    # Spatially-structured niche assignment (issue #59). Within each section
    # every allowed prototype owns a Voronoi zone seeded by a random centre, and
    # each cell takes the prototype of its nearest centre. This makes niches
    # spatially contiguous, so the kNN graph carries real niche signal and
    # ``prototype_id`` has positive spatial autocorrelation -- unlike the former
    # position-independent ``rng.choice`` assignment, whose Moran's I was ~ 0.
    prototype_id = np.empty(n_cells, dtype=np.int64)
    for s in range(n_sections):
        rows = np.arange(s * n_cells_per_section, (s + 1) * n_cells_per_section)
        if J_specific > 0:
            allowed = list(range(K_conserved)) + [K_conserved + (s % J_specific)]
        else:
            allowed = list(range(K_conserved))
        allowed_arr = np.array(allowed, dtype=np.int64)
        m = allowed_arr.size
        sec_coords = coords[rows]
        centres = rng.uniform(0.0, 1.0, size=(m, 2)).astype(np.float32)
        # Nearest-centre (Voronoi) assignment -> contiguous niche zones.
        d2 = ((sec_coords[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)
        nearest = d2.argmin(axis=1)
        # Guarantee every allowed prototype is realised in this section (the
        # conserved/sample_specific contract needs each conserved prototype in
        # every section): if a centre claimed no cell, hand it the cell sitting
        # closest to that centre.
        if rows.size >= m:
            for j in range(m):
                if not np.any(nearest == j):
                    nearest[int(d2[:, j].argmin())] = j
        prototype_id[rows] = allowed_arr[nearest]

    proto_kind = ["conserved"] * K_conserved + ["sample_specific"] * J_specific

    X = proto_means[prototype_id] + rng.normal(
        0.0, noise_sigma, size=(n_cells, n_genes)
    ).astype(np.float32)

    edges = build_graph(coords, section_id, k=k_nn, method="knn")

    marker_genes = [
        list(map(int, np.argsort(proto_means[p])[-n_markers:][::-1]))
        for p in range(n_protos)
    ]

    return SynthInstance(
        X=X,
        coords=coords,
        section_id=section_id,
        edges=edges,
        prototype_id=prototype_id,
        proto_kind=proto_kind,
        marker_genes=marker_genes,
        proto_means=proto_means,
    )

