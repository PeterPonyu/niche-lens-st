"""CCC (ligand-receptor) ground-truth planting + known-answer recovery.

The generator's CCC ground truth is OFF by default (``n_ligrec_pairs=0``); these
tests exercise the opt-in path and prove the disabled path is byte-identical.
"""

import numpy as np
import pytest

from nichelens_st.synth import generate_instance
from nichelens_st.synth.ccc import score_ccc_topk

# Params chosen so each section has enough cells for contiguous Voronoi zones
# (spatially adjacent prototype pairs) AND enough non-adjacent pairs for decoys.
CCC_KW = dict(
    n_sections=3,
    n_cells_per_section=400,
    n_genes=60,
    K_conserved=6,
    J_specific=0,
    noise_sigma=0.1,
    k_nn=8,
    n_markers=5,
    n_ligrec_pairs=3,
)


def _spatial_ccc_scores(inst, candidates):
    """Trivial reference detector: mean ligand*receptor co-expression over the
    directed source->target edges of the kNN graph. Non-adjacent prototype pairs
    have no such edges and score 0, so spatially co-located planted positives
    must outrank elevated-but-not-colocated decoys."""
    src, dst = inst.edges
    psrc = inst.prototype_id[src]
    pdst = inst.prototype_id[dst]
    scores = []
    for lig, rec, s, t in candidates:
        mask = (psrc == s) & (pdst == t)
        if not mask.any():
            scores.append(0.0)
            continue
        u = src[mask]
        v = dst[mask]
        scores.append(float(np.mean(inst.X[u, lig] * inst.X[v, rec])))
    return np.asarray(scores)


def test_disabled_by_default_truth_is_none():
    inst = generate_instance(n_sections=2, n_cells_per_section=20, n_genes=10,
                             K_conserved=2, J_specific=1, k_nn=3, seed=0)
    assert inst.ligrec_truth is None
    assert inst.ligrec_decoys is None


def test_enabling_ccc_only_perturbs_X():
    """Backward-compat: enabling CCC must leave coords/section_id/edges/
    prototype_id/proto_means/marker_genes byte-identical and change X ONLY at the
    planted (prototype, gene) entries -- the existing RNG stream is untouched."""
    off = generate_instance(**{**CCC_KW, "n_ligrec_pairs": 0}, seed=0)
    on = generate_instance(**{**CCC_KW, "n_ligrec_pairs": 3}, seed=0)

    np.testing.assert_array_equal(off.coords, on.coords)
    np.testing.assert_array_equal(off.section_id, on.section_id)
    np.testing.assert_array_equal(off.edges, on.edges)
    np.testing.assert_array_equal(off.prototype_id, on.prototype_id)
    np.testing.assert_array_equal(off.proto_means, on.proto_means)
    assert off.marker_genes == on.marker_genes

    # X differs only on the boosted (ligand-in-source / receptor-in-target) cells.
    diff_cols = np.where(np.any(off.X != on.X, axis=0))[0]
    planted_genes = set()
    for lig, rec, _s, _t in on.ligrec_truth + on.ligrec_decoys:
        planted_genes.update((lig, rec))
    assert set(int(c) for c in diff_cols).issubset(planted_genes)


def test_determinism_same_seed_identical_truth_and_expression():
    a = generate_instance(**CCC_KW, seed=0)
    b = generate_instance(**CCC_KW, seed=0)
    assert a.ligrec_truth == b.ligrec_truth
    assert a.ligrec_decoys == b.ligrec_decoys
    np.testing.assert_array_equal(a.X, b.X)


def test_determinism_different_seed_diverges():
    a = generate_instance(**CCC_KW, seed=0)
    b = generate_instance(**CCC_KW, seed=1)
    assert (a.ligrec_truth != b.ligrec_truth) or (not np.array_equal(a.X, b.X))


def test_truth_schema():
    inst = generate_instance(**CCC_KW, seed=0)
    assert isinstance(inst.ligrec_truth, list)
    assert len(inst.ligrec_truth) == 3
    assert len(inst.ligrec_decoys) == 3
    used_markers = set()
    for m in inst.marker_genes:
        used_markers.update(m)
    planted_genes = []
    for tup in inst.ligrec_truth + inst.ligrec_decoys:
        assert len(tup) == 4
        lig, rec, s, t = tup
        assert 0 <= s < inst.proto_means.shape[0]
        assert 0 <= t < inst.proto_means.shape[0]
        planted_genes.extend((lig, rec))
    # Genes are disjoint from marker genes and unique across candidates.
    assert not (set(planted_genes) & used_markers)
    assert len(planted_genes) == len(set(planted_genes))


def test_known_answer_positives_outrank_decoys():
    inst = generate_instance(**CCC_KW, seed=0)
    candidates = list(inst.ligrec_truth) + list(inst.ligrec_decoys)
    scores = _spatial_ccc_scores(inst, candidates)
    order = np.argsort(-scores, kind="stable")
    ranked = [candidates[i] for i in order]

    k_true = len(inst.ligrec_truth)
    out = score_ccc_topk(pred_ranked=ranked, truth=inst.ligrec_truth, k=k_true)
    assert out["precision_at_k"] == 1.0
    assert out["recall_at_k"] == 1.0

    # Every planted positive scores strictly above every decoy in the low-noise limit.
    pos_scores = scores[: len(inst.ligrec_truth)]
    decoy_scores = scores[len(inst.ligrec_truth) :]
    assert pos_scores.min() > decoy_scores.max()


def test_decoy_count_override():
    inst = generate_instance(**{**CCC_KW, "n_ligrec_decoys": 5}, seed=0)
    assert len(inst.ligrec_truth) == 3
    assert len(inst.ligrec_decoys) == 5


def test_too_many_pairs_for_genes_raises():
    with pytest.raises(ValueError):
        generate_instance(n_sections=2, n_cells_per_section=200, n_genes=14,
                          K_conserved=6, J_specific=0, k_nn=8, n_markers=2,
                          n_ligrec_pairs=8, seed=0)
