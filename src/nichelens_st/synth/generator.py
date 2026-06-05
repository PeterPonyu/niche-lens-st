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
    # Optional ligand-receptor / cell-cell-communication ground truth (issue
    # #347/#319/#353), populated only when ``generate_instance(n_ligrec_pairs>0)``.
    # Each entry is a ``(ligand_gene, receptor_gene, source_proto, target_proto)``
    # tuple, mirroring communication.py's (ligand, receptor, source, target) key.
    # Positives: ligand up in source-proto cells, receptor up in target-proto
    # cells, with source/target spatially adjacent over the kNN graph. Decoys:
    # the same elevation pattern between *non*-adjacent prototypes, so a
    # proximity-aware detector should rank positives above them.
    ligrec_truth: list[tuple[int, int, int, int]] | None = None
    ligrec_decoys: list[tuple[int, int, int, int]] | None = None


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
    n_ligrec_pairs: int = 0,
    n_ligrec_decoys: int | None = None,
    ligrec_strength: float = 8.0,
) -> SynthInstance:
    """Build one synthetic niche-recovery instance.

    Section ``s`` always contains all conserved prototypes plus exactly one
    sample-specific prototype (index ``K_conserved + (s % J_specific)``) when
    ``J_specific > 0``.

    ``n_markers`` sets the number of ground-truth marker genes emitted per
    prototype. It must be at least the largest ``k`` at which
    :func:`nichelens_st.metrics.marker_recall_at_k` will be evaluated, otherwise
    recall@k would silently degrade to recall@n_markers.

    Ligand-receptor / cell-cell-communication (CCC) ground truth is **off by
    default** (``n_ligrec_pairs == 0``). When ``n_ligrec_pairs > 0`` the function
    plants that many known-positive ligand->receptor interactions plus
    ``n_ligrec_decoys`` negatives (defaulting to ``n_ligrec_pairs``), exposing
    them on :attr:`SynthInstance.ligrec_truth` / :attr:`SynthInstance.ligrec_decoys`
    (schema documented in ``docs/SYNTHETIC_BENCHMARK.md``). The planting only
    perturbs ``X`` at the chosen ``(prototype, gene)`` entries and draws from the
    RNG *after* the base instance is built, so with ``n_ligrec_pairs == 0`` every
    output is byte-identical to before for a fixed seed.
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
    if n_ligrec_pairs < 0:
        raise ValueError(f"n_ligrec_pairs must be non-negative; got {n_ligrec_pairs}")
    n_decoys = n_ligrec_pairs if n_ligrec_decoys is None else n_ligrec_decoys
    if n_decoys < 0:
        raise ValueError(f"n_ligrec_decoys must be non-negative; got {n_decoys}")

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

    ligrec_truth: list[tuple[int, int, int, int]] | None = None
    ligrec_decoys: list[tuple[int, int, int, int]] | None = None
    if n_ligrec_pairs > 0:
        # Plant CCC ground truth AFTER the base instance so the disabled path
        # (and every base array) stays byte-identical for a fixed seed -- the
        # only effect is in-place boosts to ``X`` at the chosen entries.
        ligrec_truth, ligrec_decoys = _plant_ligrec(
            rng=rng,
            X=X,
            prototype_id=prototype_id,
            edges=edges,
            K_conserved=K_conserved,
            n_genes=n_genes,
            marker_genes=marker_genes,
            n_pairs=n_ligrec_pairs,
            n_decoys=n_decoys,
            strength=ligrec_strength,
        )

    return SynthInstance(
        X=X,
        coords=coords,
        section_id=section_id,
        edges=edges,
        prototype_id=prototype_id,
        proto_kind=proto_kind,
        marker_genes=marker_genes,
        proto_means=proto_means,
        ligrec_truth=ligrec_truth,
        ligrec_decoys=ligrec_decoys,
    )


def _plant_ligrec(
    *,
    rng: np.random.Generator,
    X: np.ndarray,
    prototype_id: np.ndarray,
    edges: np.ndarray,
    K_conserved: int,
    n_genes: int,
    marker_genes: list[list[int]],
    n_pairs: int,
    n_decoys: int,
    strength: float,
) -> tuple[list[tuple[int, int, int, int]], list[tuple[int, int, int, int]]]:
    """Plant ligand-receptor ground truth in-place on ``X`` (issue #347/#319/#353).

    Positives use spatially *adjacent* conserved prototype pairs (>=1 directed
    ``source -> target`` kNN edge) and elevate the ligand gene in source-prototype
    cells and the receptor gene in target-prototype cells by ``strength``, so a
    proximity + co-expression detector recovers them.

    Decoys are negatives that such a detector should rank *below* the positives.
    Two flavours are used: (a) *elevated-but-not-colocated* -- same elevation
    pattern but between a **non**-adjacent prototype pair (no source->target edges,
    so spatial co-expression is ~0); (b) when too few non-adjacent pairs exist
    (contiguous Voronoi zones make most conserved pairs adjacent), *random
    unboosted* candidates -- distinct genes with no planted signal, scoring at
    baseline. All ligand/receptor genes are disjoint from the marker panels and
    from each other, so the planted signal never collides with the niche-recovery
    ground truth.

    Returns ``(ligrec_truth, ligrec_decoys)`` as lists of
    ``(ligand_gene, receptor_gene, source_proto, target_proto)`` int tuples.
    Restricting source/target to conserved prototypes keeps them present in every
    section, so adjacency is stable across sections.
    """
    if K_conserved < 2:
        raise ValueError(
            "CCC ground truth (n_ligrec_pairs>0) needs K_conserved >= 2 so "
            f"distinct source/target prototypes exist; got K_conserved={K_conserved}"
        )

    src, dst = edges[0], edges[1]
    p_src = prototype_id[src]
    p_dst = prototype_id[dst]
    conserved_edge = (p_src < K_conserved) & (p_dst < K_conserved)
    cross = np.zeros((K_conserved, K_conserved), dtype=np.int64)
    np.add.at(cross, (p_src[conserved_edge], p_dst[conserved_edge]), 1)

    adjacent: list[tuple[int, int]] = []
    non_adjacent: list[tuple[int, int]] = []
    for a in range(K_conserved):
        for b in range(K_conserved):
            if a == b:
                continue
            if cross[a, b] > 0:
                adjacent.append((a, b))
            else:
                non_adjacent.append((a, b))
    # Most-connected adjacent pairs first so positives have the strongest spatial
    # co-location signal; deterministic tie-break by (source, target).
    adjacent.sort(key=lambda ab: (-int(cross[ab[0], ab[1]]), ab[0], ab[1]))

    if len(adjacent) < n_pairs:
        raise ValueError(
            f"only {len(adjacent)} spatially adjacent conserved prototype pairs "
            f"are available but n_ligrec_pairs={n_pairs}; raise k_nn, "
            "n_cells_per_section, or K_conserved"
        )

    marker_used = {g for panel in marker_genes for g in panel}
    available = np.array(
        [g for g in range(n_genes) if g not in marker_used], dtype=np.int64
    )
    need = 2 * (n_pairs + n_decoys)
    if available.size < need:
        raise ValueError(
            f"CCC planting needs {need} non-marker genes (2 per interaction) but "
            f"only {available.size} are free; raise n_genes or lower "
            "n_ligrec_pairs/n_ligrec_decoys"
        )
    genes = available[rng.permutation(available.size)][:need]

    # Decoy (source, target) pairs: prefer non-adjacent (hard, elevated) pairs;
    # fall back to any distinct conserved pair (random, unboosted) when the dense
    # graph leaves too few non-adjacent options.
    hard_pairs = [non_adjacent[i] for i in rng.permutation(len(non_adjacent))]
    all_pairs = [(a, b) for a in range(K_conserved) for b in range(K_conserved) if a != b]
    random_pairs = [all_pairs[i] for i in rng.permutation(len(all_pairs))]

    gene_cursor = 0

    def _next_genes() -> tuple[int, int]:
        nonlocal gene_cursor
        ligand = int(genes[gene_cursor])
        receptor = int(genes[gene_cursor + 1])
        gene_cursor += 2
        return ligand, receptor

    def _boost(source: int, target: int, ligand: int, receptor: int) -> None:
        X[prototype_id == source, ligand] += strength
        X[prototype_id == target, receptor] += strength

    ligrec_truth: list[tuple[int, int, int, int]] = []
    for source, target in adjacent[:n_pairs]:
        ligand, receptor = _next_genes()
        _boost(source, target, ligand, receptor)
        ligrec_truth.append((ligand, receptor, int(source), int(target)))

    ligrec_decoys: list[tuple[int, int, int, int]] = []
    rand_cursor = 0
    for i in range(n_decoys):
        ligand, receptor = _next_genes()
        if i < len(hard_pairs):
            # Elevated-but-not-colocated: planted signal, no spatial proximity.
            source, target = hard_pairs[i]
            _boost(source, target, ligand, receptor)
        else:
            # Random unboosted candidate: distinct genes, no planted signal.
            source, target = random_pairs[rand_cursor % len(random_pairs)]
            rand_cursor += 1
        ligrec_decoys.append((ligand, receptor, int(source), int(target)))

    return ligrec_truth, ligrec_decoys

