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
    seed: int = 0,
) -> SynthInstance:
    """Build one synthetic niche-recovery instance.

    Section ``s`` always contains all conserved prototypes plus exactly one
    sample-specific prototype (index ``K_conserved + (s % J_specific)``) when
    ``J_specific > 0``.
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

    prototype_id = np.empty(n_cells, dtype=np.int64)
    for s in range(n_sections):
        rows = np.arange(s * n_cells_per_section, (s + 1) * n_cells_per_section)
        if J_specific > 0:
            allowed = list(range(K_conserved)) + [K_conserved + (s % J_specific)]
        else:
            allowed = list(range(K_conserved))
        assigned = rng.choice(allowed, size=rows.size).astype(np.int64)
        if rows.size >= len(allowed):
            assigned[: len(allowed)] = np.array(allowed, dtype=np.int64)
            rng.shuffle(assigned)
        prototype_id[rows] = assigned

    proto_kind = ["conserved"] * K_conserved + ["sample_specific"] * J_specific

    X = proto_means[prototype_id] + rng.normal(
        0.0, noise_sigma, size=(n_cells, n_genes)
    ).astype(np.float32)

    edges = build_graph(coords, section_id, k=k_nn, method="knn")

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
        proto_means=proto_means,
    )

