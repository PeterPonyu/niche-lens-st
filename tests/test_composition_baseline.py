"""Composition-kNN niche baseline (#320 / #339).

The natural niche baseline: describe each cell by the cell-type composition of
its local spatial neighborhood (the fraction of each cell type among the cell
and its k spatial neighbors). Clustering this composition vector is the
"raw cell-type-fraction kNN" niche that the learned model must beat. This pins:

* the contract -- rows are compositions (non-negative, sum to 1), correct shape;
* determinism given a seed; and
* the headline -- on data with spatially contiguous niches defined by distinct
  cell-type mixtures, clustering the neighborhood composition recovers the
  niches well above chance.

Pure numpy/scipy, no brand names, no external package code.
"""

from __future__ import annotations

import numpy as np

from nichelens_st import baselines as B
from nichelens_st.metrics import adjusted_rand


def test_composition_knn_rows_are_compositions():
    rng = np.random.default_rng(0)
    n = 40
    coords = rng.uniform(0, 10, size=(n, 2))
    cell_type_codes = rng.integers(0, 3, size=n)
    emb = B.composition_knn_embedding(cell_type_codes, coords, k=5, n_types=3)
    assert emb.shape == (n, 3)
    assert (emb >= 0).all()
    np.testing.assert_allclose(emb.sum(axis=1), 1.0)


def test_composition_knn_infers_n_types():
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    cell_type_codes = np.array([0, 1, 1])
    emb = B.composition_knn_embedding(cell_type_codes, coords, k=2)
    assert emb.shape == (3, 2)  # two distinct codes -> two columns


def test_composition_knn_is_deterministic():
    rng = np.random.default_rng(2)
    coords = rng.uniform(0, 10, size=(30, 2))
    codes = rng.integers(0, 4, size=30)
    a = B.composition_knn_embedding(codes, coords, k=6, n_types=4)
    b = B.composition_knn_embedding(codes, coords, k=6, n_types=4)
    np.testing.assert_array_equal(a, b)


def test_composition_knn_recovers_mixture_defined_niches():
    """Two spatial regions defined purely by cell-type mixture (not by the type
    of any single cell): left region is 80/20 type0/type1, right is 20/80.
    Per-cell type is ambiguous; the neighborhood composition is not."""
    rng = np.random.default_rng(7)
    side = 14
    xs, ys = np.meshgrid(np.arange(side), np.arange(side))
    coords = np.column_stack([xs.ravel().astype(float), ys.ravel().astype(float)])
    left = coords[:, 0] < side / 2
    true_niche = (~left).astype(int)
    p_type1 = np.where(left, 0.2, 0.8)
    cell_type_codes = (rng.uniform(size=coords.shape[0]) < p_type1).astype(int)

    emb = B.composition_knn_embedding(cell_type_codes, coords, k=8, n_types=2)
    pred = B.assign_prototypes(emb, n_clusters=2, seed=0)
    assert adjusted_rand(pred, true_niche) > 0.7
