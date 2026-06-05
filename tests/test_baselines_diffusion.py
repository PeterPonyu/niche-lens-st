"""Tests for spatial_diffusion_embedding (issue #328).

Pinned contracts:

* Reduction properties: n_steps=0, alpha=0, or k=0 -> returns X exactly.
* n_steps=1 equals neighborhood_augmented_embedding exactly (single-step case).
* Determinism: same inputs -> identical outputs.
* Monotone smoothing: more diffusion steps reduce within-niche variance on
  spatially contiguous niche instances.
* Invalid arguments raise ValueError.
* Runner dispatch routes 'diffusion' to spatial_diffusion_embedding.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from nichelens_st import baselines as B
from nichelens_st.synth.generator import generate_instance

# Import the runner module by path (lives under scripts/, not on sys.path).
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO_ROOT / "scripts" / "run_baselines_niche.py"
sys.path.insert(0, str(_REPO_ROOT / "src"))
_spec = importlib.util.spec_from_file_location("run_baselines_niche", _SCRIPT)
runner = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(runner)


def _small_instance(seed: int = 0):
    """Small synthetic instance with spatially contiguous niches."""
    return generate_instance(
        n_sections=2,
        n_cells_per_section=200,
        n_genes=20,
        K_conserved=4,
        J_specific=2,
        noise_sigma=2.0,
        k_nn=6,
        seed=seed,
    )


# --------------------------------------------------------------------------
# Reduction properties: n_steps=0, alpha=0, k=0 -> X exactly
# --------------------------------------------------------------------------


def test_n_steps_zero_returns_X_exactly():
    inst = _small_instance()
    X = inst.X.astype(np.float64)
    emb = B.spatial_diffusion_embedding(
        X, inst.coords, k=8, n_steps=0, alpha=0.5, section_id=inst.section_id
    )
    np.testing.assert_array_equal(emb, X)


def test_alpha_zero_returns_X_exactly():
    inst = _small_instance()
    X = inst.X.astype(np.float64)
    emb = B.spatial_diffusion_embedding(
        X, inst.coords, k=8, n_steps=5, alpha=0.0, section_id=inst.section_id
    )
    np.testing.assert_array_equal(emb, X)


def test_k_zero_returns_X_exactly():
    inst = _small_instance()
    X = inst.X.astype(np.float64)
    emb = B.spatial_diffusion_embedding(
        X, inst.coords, k=0, n_steps=3, alpha=0.5, section_id=inst.section_id
    )
    np.testing.assert_array_equal(emb, X)


# --------------------------------------------------------------------------
# n_steps=1 is identical to neighborhood_augmented_embedding
# --------------------------------------------------------------------------


def test_n_steps_one_equals_neighborhood_augmented_embedding():
    """n_steps=1 must be identical to neighborhood_augmented_embedding (same code path)."""
    inst = _small_instance()
    X = inst.X.astype(np.float64)
    emb_diff = B.spatial_diffusion_embedding(
        X, inst.coords, k=8, n_steps=1, alpha=0.5, section_id=inst.section_id
    )
    emb_nbr = B.neighborhood_augmented_embedding(
        X, inst.coords, k=8, alpha=0.5, section_id=inst.section_id
    )
    np.testing.assert_array_equal(emb_diff, emb_nbr)


def test_n_steps_one_equals_neighborhood_augmented_various_params():
    """n_steps=1 identity holds for several (k, alpha) combinations."""
    inst = _small_instance(seed=7)
    X = inst.X.astype(np.float64)
    for k, alpha in [(4, 0.3), (6, 0.7), (8, 1.0)]:
        emb_d = B.spatial_diffusion_embedding(
            X, inst.coords, k=k, n_steps=1, alpha=alpha, section_id=inst.section_id
        )
        emb_n = B.neighborhood_augmented_embedding(
            X, inst.coords, k=k, alpha=alpha, section_id=inst.section_id
        )
        np.testing.assert_array_equal(
            emb_d, emb_n, err_msg=f"mismatch at k={k}, alpha={alpha}"
        )


# --------------------------------------------------------------------------
# Determinism: same inputs -> identical outputs
# --------------------------------------------------------------------------


def test_diffusion_embedding_is_deterministic():
    inst = _small_instance()
    X = inst.X.astype(np.float64)
    a = B.spatial_diffusion_embedding(
        X, inst.coords, k=6, n_steps=3, alpha=0.5, section_id=inst.section_id
    )
    b = B.spatial_diffusion_embedding(
        X, inst.coords, k=6, n_steps=3, alpha=0.5, section_id=inst.section_id
    )
    np.testing.assert_array_equal(a, b)


# --------------------------------------------------------------------------
# Monotone smoothing: more steps -> within-niche variance does not increase
# --------------------------------------------------------------------------


def _within_niche_variance(emb: np.ndarray, niche_labels: np.ndarray) -> float:
    """Mean per-gene variance averaged over niches (cells in the same niche)."""
    variances = []
    for lbl in np.unique(niche_labels):
        mask = niche_labels == lbl
        if mask.sum() > 1:
            variances.append(float(emb[mask].var(axis=0).mean()))
    return float(np.mean(variances)) if variances else 0.0


def test_more_steps_reduces_within_niche_variance():
    """8 diffusion steps yield no more within-niche variance than 1 step.

    On spatially contiguous niches the diffusion operator averages mostly
    same-niche neighbours, so repeated application reduces within-niche
    feature variance monotonically.
    """
    inst = _small_instance(seed=42)
    X = inst.X.astype(np.float64)
    gt = inst.prototype_id  # ground-truth spatially contiguous niche labels

    var1 = _within_niche_variance(
        B.spatial_diffusion_embedding(
            X, inst.coords, k=6, n_steps=1, alpha=0.5, section_id=inst.section_id
        ),
        gt,
    )
    var8 = _within_niche_variance(
        B.spatial_diffusion_embedding(
            X, inst.coords, k=6, n_steps=8, alpha=0.5, section_id=inst.section_id
        ),
        gt,
    )
    assert var8 <= var1, (
        f"Within-niche variance increased: n_steps=1 -> {var1:.6f}, "
        f"n_steps=8 -> {var8:.6f}"
    )


# --------------------------------------------------------------------------
# Invalid argument validation
# --------------------------------------------------------------------------


def test_negative_n_steps_raises_value_error():
    X = np.ones((5, 4), dtype=np.float64)
    coords = np.zeros((5, 2), dtype=np.float64)
    with pytest.raises(ValueError, match="n_steps"):
        B.spatial_diffusion_embedding(X, coords, k=2, n_steps=-1, alpha=0.5)


def test_alpha_above_one_raises_value_error():
    X = np.ones((5, 4), dtype=np.float64)
    coords = np.zeros((5, 2), dtype=np.float64)
    with pytest.raises(ValueError, match="alpha"):
        B.spatial_diffusion_embedding(X, coords, k=2, n_steps=2, alpha=1.5)


def test_alpha_below_zero_raises_value_error():
    X = np.ones((5, 4), dtype=np.float64)
    coords = np.zeros((5, 2), dtype=np.float64)
    with pytest.raises(ValueError, match="alpha"):
        B.spatial_diffusion_embedding(X, coords, k=2, n_steps=2, alpha=-0.1)


def test_negative_k_raises_value_error():
    X = np.ones((5, 4), dtype=np.float64)
    coords = np.zeros((5, 2), dtype=np.float64)
    with pytest.raises(ValueError, match="k"):
        B.spatial_diffusion_embedding(X, coords, k=-1, n_steps=2, alpha=0.5)


# --------------------------------------------------------------------------
# Runner dispatch: 'diffusion' routes to spatial_diffusion_embedding
# --------------------------------------------------------------------------


def test_runner_dispatch_diffusion_returns_correct_embedding():
    """run_baseline(..., baseline='diffusion') matches spatial_diffusion_embedding directly."""
    inst = _small_instance()
    X = inst.X.astype(np.float64)
    proto, emb, _metrics, _notes = runner.run_baseline(
        X,
        inst.coords,
        inst.section_id,
        baseline="diffusion",
        k=6,
        alpha=0.5,
        n_prototypes=6,
        n_components=32,
        seed=0,
        n_steps=3,
    )
    assert emb.shape == X.shape
    assert proto.shape == (X.shape[0],)
    assert emb.dtype == np.float64
    # Must match direct call with identical parameters.
    emb_direct = B.spatial_diffusion_embedding(
        X, inst.coords, k=6, n_steps=3, alpha=0.5, section_id=inst.section_id
    )
    np.testing.assert_array_equal(emb, emb_direct)


def test_runner_dispatch_diffusion_emits_contract_metric_keys():
    """Diffusion runner row has the same metric keys as neighborhood/pca rows."""
    inst = _small_instance()
    X = inst.X.astype(np.float64)
    _, _, metrics, _ = runner.run_baseline(
        X,
        inst.coords,
        inst.section_id,
        baseline="diffusion",
        k=6,
        alpha=0.5,
        n_prototypes=6,
        n_components=32,
        seed=0,
        n_steps=3,
    )
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
        assert key in metrics, f"missing contract metric key: {key!r}"


def test_runner_dispatch_diffusion_is_deterministic():
    """Repeated runner calls with baseline='diffusion' yield identical prototypes."""
    inst = _small_instance()
    X = inst.X.astype(np.float64)
    p1, _, m1, _ = runner.run_baseline(
        X, inst.coords, inst.section_id, baseline="diffusion",
        k=6, alpha=0.5, n_prototypes=6, n_components=32, seed=0, n_steps=3,
    )
    p2, _, m2, _ = runner.run_baseline(
        X, inst.coords, inst.section_id, baseline="diffusion",
        k=6, alpha=0.5, n_prototypes=6, n_components=32, seed=0, n_steps=3,
    )
    np.testing.assert_array_equal(p1, p2)
    assert m1 == m2
