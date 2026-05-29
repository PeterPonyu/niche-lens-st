"""Issue #102: lock determinism of the numpy paths outside the seeded encoder.

The encoder seeds torch + enables deterministic algorithms, but the downstream
k-means / separation-head / marker paths in ``model.py`` carry their own
randomness (k-means++ init). These tests run **without torch** so the
clustering-path determinism contract is verified even when the optional
``[model]`` extra is absent, and so a regression that drops the local seeded
generator (e.g. switching to global ``np.random``) is caught.
"""

from __future__ import annotations

import numpy as np

from nichelens_st.model import _compute_marker_table, _kmeans, _separation_head


def _clustered_embeddings(seed: int = 0) -> np.ndarray:
    """Three well-separated Gaussian blobs in 4-D (float32, like encoder output)."""
    rng = np.random.default_rng(seed)
    centers = np.array([[5, 5, 0, 0], [-5, -5, 0, 0], [0, 0, 5, -5]], dtype=np.float64)
    blocks = [rng.normal(c, 0.1, size=(20, 4)) for c in centers]
    return np.vstack(blocks).astype(np.float32)


def test_kmeans_identical_for_fixed_seed():
    H = _clustered_embeddings()
    a = _kmeans(H, n_clusters=3, n_iters=50, seed=7)
    b = _kmeans(H, n_clusters=3, n_iters=50, seed=7)
    np.testing.assert_array_equal(a, b)


def test_kmeans_independent_of_global_numpy_rng_state():
    """The path must use a local seeded generator, not global np.random state."""
    H = _clustered_embeddings()
    np.random.seed(1)
    a = _kmeans(H, n_clusters=3, n_iters=50, seed=7)
    np.random.seed(999)
    _ = np.random.random(123)  # perturb global state
    b = _kmeans(H, n_clusters=3, n_iters=50, seed=7)
    np.testing.assert_array_equal(a, b)


def test_separation_head_and_marker_table_deterministic():
    H = _clustered_embeddings()
    proto = _kmeans(H, n_clusters=3, n_iters=50, seed=7)
    n_protos = int(proto.max()) + 1
    section = np.array([0, 1] * (H.shape[0] // 2), dtype=np.int64)
    X = _clustered_embeddings(seed=3)
    k1 = _separation_head(proto, section, n_protos)
    k2 = _separation_head(proto, section, n_protos)
    m1 = _compute_marker_table(X, proto, n_protos, top_k=3)
    m2 = _compute_marker_table(X, proto, n_protos, top_k=3)
    assert k1 == k2
    assert m1 == m2


def test_kmeans_recovers_three_blobs():
    """Sanity: with three separated blobs the assignment has exactly 3 ids."""
    H = _clustered_embeddings()
    labels = _kmeans(H, n_clusters=3, n_iters=50, seed=0)
    assert set(labels.tolist()) == {0, 1, 2}
