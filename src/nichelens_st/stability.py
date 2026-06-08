"""Niche-stability primitives (#296): seed reproducibility + cohort consistency.

Reproducibility benchmarks require a niche method to give stable labels across
random seeds and prototypes that recur across samples — instability undermines
any biological claim built on the labels. These are torch-free reductions over
already-fitted ``prototype_id`` labellings; the heavy multi-seed *refit* lives
in ``scripts/compute_niche_stability.py``.

- :func:`pairwise_ari_matrix` / :func:`seed_stability_summary` — seed-to-seed
  agreement (pairwise ARI of ``prototype_id`` across N seeds).
- :func:`tag_conserved` / :func:`coverage_sweep` — conserved-vs-sample_specific
  fraction and section-overlap across a ``min_section_coverage`` sweep (#105).
- :func:`prototype_matching` — Hungarian-matched prototype correspondence
  between two sections (the content of the #296 prototype-matching Sankey).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

from .metrics import adjusted_rand, section_overlap_rate


# ---------------------------------------------------------------------------
# Seed stability
# ---------------------------------------------------------------------------
def pairwise_ari_matrix(labelings: "list[np.ndarray]") -> np.ndarray:
    """Symmetric pairwise adjusted-Rand matrix across seed labellings.

    Each entry ``M[i, j]`` is the ARI between ``labelings[i]`` and
    ``labelings[j]`` (label-permutation invariant). Diagonal is 1.0.
    """
    n = len(labelings)
    M = np.eye(n, dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            ari = adjusted_rand(labelings[i], labelings[j])
            M[i, j] = M[j, i] = ari
    return M


def seed_stability_summary(matrix: np.ndarray) -> dict:
    """Off-diagonal stability stats from a pairwise-ARI matrix.

    With < 2 seeds the off-diagonal is empty → stats are ``None`` (undefined),
    never a free "perfect" score.
    """
    M = np.asarray(matrix, dtype=float)
    n = int(M.shape[0])
    if n < 2:
        return {
            "n_seeds": n,
            "mean_offdiag_ari": None,
            "sd_offdiag_ari": None,
            "min_offdiag_ari": None,
        }
    iu = np.triu_indices(n, k=1)
    vals = M[iu]
    return {
        "n_seeds": n,
        "mean_offdiag_ari": float(np.mean(vals)),
        "sd_offdiag_ari": float(np.std(vals)),  # population sd (ddof=0)
        "min_offdiag_ari": float(np.min(vals)),
    }


# ---------------------------------------------------------------------------
# Cohort reproducibility (conserved vs sample_specific)
# ---------------------------------------------------------------------------
def tag_conserved(
    prototype_id: np.ndarray,
    section_id: np.ndarray,
    n_protos: int,
    min_section_coverage: float = 1.0,
) -> "list[str]":
    """Tag each prototype conserved/sample_specific by cross-section presence.

    Torch-free mirror of :func:`nichelens_st.model._separation_head` (kept in
    the stability lane so this module never imports torch). A prototype is
    ``conserved`` iff its cells cover ``>= ceil(min_section_coverage *
    n_sections)`` sections; ``< 2`` sections → all ``"unknown"`` (the
    distinction is degenerate, #85). Parity with ``_separation_head`` is pinned
    by ``tests/test_stability.py``.
    """
    if not 0.0 < min_section_coverage <= 1.0:
        raise ValueError(
            f"min_section_coverage must be in (0, 1]; got {min_section_coverage}"
        )
    proto = np.asarray(prototype_id)
    section = np.asarray(section_id)
    all_sections = set(section.tolist())
    n_sections = len(all_sections)
    if n_sections < 2:
        return ["unknown"] * n_protos
    required = max(1, int(np.ceil(min_section_coverage * n_sections)))
    kinds: list[str] = []
    for p in range(n_protos):
        seen = set(section[proto == p].tolist())
        kinds.append("conserved" if seen and len(seen) >= required else "sample_specific")
    return kinds


def _conserved_fraction(kinds: "list[str]") -> Optional[float]:
    scorable = [k for k in kinds if k != "unknown"]
    if not scorable:
        return None
    return sum(k == "conserved" for k in scorable) / len(scorable)


def coverage_sweep(
    prototype_id: np.ndarray,
    section_id: np.ndarray,
    n_protos: int,
    thresholds: "list[float]",
) -> "list[dict]":
    """Sweep ``min_section_coverage`` → conserved_fraction + section_overlap_rate.

    Shows robustness of the conserved/sample_specific call to unequal section
    depth (#105). Returns one row per threshold.
    """
    rows: list[dict] = []
    for t in thresholds:
        kinds = tag_conserved(prototype_id, section_id, n_protos, t)
        n_conserved = sum(k == "conserved" for k in kinds)
        rows.append(
            {
                "min_section_coverage": float(t),
                "conserved_fraction": _conserved_fraction(kinds),
                "section_overlap_rate": section_overlap_rate(
                    prototype_id, section_id, kinds
                ),
                "n_conserved": int(n_conserved),
                "n_prototypes": int(n_protos),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Prototype matching (Sankey content)
# ---------------------------------------------------------------------------
def prototype_matching(
    prototype_id_a: np.ndarray, prototype_id_b: np.ndarray
) -> "list[dict]":
    """Hungarian-matched prototype correspondence between two labellings.

    Builds the prototype×prototype overlap (contingency) table and maximises
    total overlap via ``linear_sum_assignment`` (the same matching used in
    ``metrics.py``). Returns matched ``{proto_a, proto_b, overlap}`` rows — the
    edges of the #296 prototype-matching Sankey.
    """
    a = np.asarray(prototype_id_a)
    b = np.asarray(prototype_id_b)
    if a.shape != b.shape:
        raise ValueError(f"labelling shapes differ: {a.shape} != {b.shape}")
    a_ids = np.unique(a)
    b_ids = np.unique(b)
    table = np.zeros((a_ids.size, b_ids.size), dtype=np.int64)
    a_pos = {int(v): i for i, v in enumerate(a_ids)}
    b_pos = {int(v): j for j, v in enumerate(b_ids)}
    for av, bv in zip(a.tolist(), b.tolist()):
        table[a_pos[int(av)], b_pos[int(bv)]] += 1
    row_ind, col_ind = linear_sum_assignment(-table)
    matches: list[dict] = []
    for i, j in zip(row_ind, col_ind):
        matches.append(
            {
                "proto_a": int(a_ids[i]),
                "proto_b": int(b_ids[j]),
                "overlap": int(table[i, j]),
            }
        )
    return matches
