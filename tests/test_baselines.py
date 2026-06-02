"""Tests for the license-clean spatial baselines (issue #152).

These pin the neighborhood-augmented baseline's contract:

* the reduction property -- ``alpha == 0`` (or ``k == 0``) collapses the
  augmented embedding to the non-augmented (self-only) features;
* determinism -- a fixed seed reproduces the embedding and the prototype
  assignment exactly;
* the headline result -- on data with planted, spatially contiguous niches and
  noisy per-cell expression, clustering the neighborhood-augmented embedding
  recovers the niches markedly better (ARI / silhouette) than clustering the
  non-spatial PCA baseline.

They run without torch (the baselines are pure numpy/scipy), so the contract is
verified even when the optional ``[model]`` extra is absent.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

from nichelens_st import baselines as B
from nichelens_st.metrics import adjusted_rand
from nichelens_st.synth.generator import generate_instance

# Import the runner module by path (it lives under scripts/, not on sys.path).
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO_ROOT / "scripts" / "run_baselines_niche.py"
sys.path.insert(0, str(_REPO_ROOT / "src"))
_spec = importlib.util.spec_from_file_location("run_baselines_niche", _SCRIPT)
runner = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(runner)


def _noisy_niche_instance(noise_sigma: float = 3.0, seed: int = 0):
    """Synthetic instance with contiguous niches and noisy per-cell expression.

    ``noise_sigma`` is set high enough that per-cell expression alone is
    ambiguous, so spatial neighborhood averaging has signal to exploit.
    """
    return generate_instance(
        n_sections=2,
        n_cells_per_section=600,
        n_genes=60,
        K_conserved=4,
        J_specific=2,
        noise_sigma=noise_sigma,
        k_nn=8,
        seed=seed,
    )


# --------------------------------------------------------------------------
# Reduction property: alpha == 0 / k == 0 -> non-augmented self features
# --------------------------------------------------------------------------
def test_alpha_zero_reduces_to_self_features():
    inst = _noisy_niche_instance()
    X = inst.X.astype(np.float64)
    emb = B.neighborhood_augmented_embedding(
        X, inst.coords, k=8, alpha=0.0, section_id=inst.section_id
    )
    np.testing.assert_array_equal(emb, X)


def test_k_zero_reduces_to_self_features_for_any_alpha():
    inst = _noisy_niche_instance()
    X = inst.X.astype(np.float64)
    emb = B.neighborhood_augmented_embedding(
        X, inst.coords, k=0, alpha=0.7, section_id=inst.section_id
    )
    np.testing.assert_array_equal(emb, X)


def test_neighborhood_average_singleton_section_falls_back_to_self():
    """A cell with no in-section neighbor uses its own feature row (finite)."""
    X = np.arange(12, dtype=np.float64).reshape(3, 4)
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [5.0, 5.0]], dtype=np.float64)
    section_id = np.array([0, 0, 1], dtype=np.int64)  # cell 2 alone in section 1
    nbr = B.neighborhood_averaged_features(X, coords, k=4, section_id=section_id)
    assert np.isfinite(nbr).all()
    np.testing.assert_array_equal(nbr[2], X[2])  # singleton -> self


# --------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------
def test_embedding_and_assignment_are_deterministic():
    inst = _noisy_niche_instance()
    X = inst.X.astype(np.float64)
    a = B.neighborhood_augmented_embedding(
        X, inst.coords, k=8, alpha=0.5, section_id=inst.section_id
    )
    b = B.neighborhood_augmented_embedding(
        X, inst.coords, k=8, alpha=0.5, section_id=inst.section_id
    )
    np.testing.assert_array_equal(a, b)
    la = B.assign_prototypes(a, n_clusters=6, seed=0)
    lb = B.assign_prototypes(b, n_clusters=6, seed=0)
    np.testing.assert_array_equal(la, lb)


# --------------------------------------------------------------------------
# Headline result: neighborhood augmentation beats plain PCA on niche recovery
# --------------------------------------------------------------------------
def test_neighborhood_beats_pca_on_niche_recovery():
    inst = _noisy_niche_instance(noise_sigma=3.0, seed=0)
    X = inst.X.astype(np.float64)
    gt = inst.prototype_id
    n_protos = int(gt.max()) + 1

    emb_pca = B.pca_embedding(X, n_components=32, seed=0)
    lab_pca = B.assign_prototypes(emb_pca, n_clusters=n_protos, seed=0)
    ari_pca = adjusted_rand(lab_pca, gt)

    emb_nb = B.neighborhood_augmented_embedding(
        X, inst.coords, k=8, alpha=0.5, section_id=inst.section_id
    )
    lab_nb = B.assign_prototypes(emb_nb, n_clusters=n_protos, seed=0)
    ari_nb = adjusted_rand(lab_nb, gt)

    # The neighborhood-augmented baseline recovers niches well AND clearly beats
    # the non-spatial PCA baseline on this noisy, spatially-structured data.
    assert ari_nb > 0.8
    assert ari_nb > ari_pca + 0.1


def test_zero_augmentation_matches_pca_free_self_clustering():
    """alpha=0 clustering equals clustering the raw self features (no spatial gain)."""
    inst = _noisy_niche_instance()
    X = inst.X.astype(np.float64)
    emb0 = B.neighborhood_augmented_embedding(
        X, inst.coords, k=8, alpha=0.0, section_id=inst.section_id
    )
    lab0 = B.assign_prototypes(emb0, n_clusters=6, seed=0)
    lab_self = B.assign_prototypes(X, n_clusters=6, seed=0)
    np.testing.assert_array_equal(lab0, lab_self)


# --------------------------------------------------------------------------
# Runner core: emits the uniform-contract intrinsic metric keys
# --------------------------------------------------------------------------
def test_runner_core_emits_model_contract_metric_keys():
    inst = _noisy_niche_instance()
    X = inst.X.astype(np.float64)
    proto, emb, metrics, _notes = runner.run_baseline(
        X,
        inst.coords,
        inst.section_id,
        baseline="neighborhood",
        k=8,
        alpha=0.5,
        n_prototypes=8,
        n_components=32,
        seed=0,
    )
    # Same keys the model's intrinsic-metrics path emits, so the rows line up.
    for key in (
        "n_prototypes",
        "prototype_size_min",
        "prototype_size_median",
        "prototype_size_max",
        "prototype_size_entropy",
        "niche_morans_i",
        "embedding_silhouette",
        "embedding_silhouette_n_subsample",
    ):
        assert key in metrics
    assert metrics["n_prototypes"] >= 2
    assert proto.shape == (X.shape[0],)
    assert emb.shape[0] == X.shape[0]


def test_runner_core_pca_and_neighborhood_are_deterministic():
    inst = _noisy_niche_instance()
    X = inst.X.astype(np.float64)
    for baseline in ("neighborhood", "pca"):
        p1, _e1, m1, _ = runner.run_baseline(
            X, inst.coords, inst.section_id, baseline=baseline,
            k=8, alpha=0.5, n_prototypes=8, n_components=32, seed=0,
        )
        p2, _e2, m2, _ = runner.run_baseline(
            X, inst.coords, inst.section_id, baseline=baseline,
            k=8, alpha=0.5, n_prototypes=8, n_components=32, seed=0,
        )
        np.testing.assert_array_equal(p1, p2)
        assert m1 == m2
