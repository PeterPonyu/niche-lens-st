"""Prototype clustering must use cosine geometry for unit-sphere embeddings.

The encoder L2-normalizes embeddings and trains on a cosine objective, so the
prototype k-means must keep centroids on the unit sphere (spherical k-means).
A plain-mean update drifts centroids inside the ball and breaks the agreement
between Euclidean nearest-centroid and cosine nearest-centroid.
"""

from __future__ import annotations

import numpy as np

from nichelens_st.model import _spherical_kmeans


def _unit(M: np.ndarray) -> np.ndarray:
    return M / np.linalg.norm(M, axis=1, keepdims=True)


def test_centroids_renormalized():
    rng = np.random.default_rng(0)
    # Well-separated unit-sphere clusters so assignment is stable.
    true_centers = _unit(rng.normal(size=(5, 8)))
    H = np.vstack(
        [_unit(true_centers[c] + 0.05 * rng.normal(size=(24, 8))) for c in range(5)]
    )

    labels, centers = _spherical_kmeans(H, k=5, n_iters=100, rng=rng)

    # Centroids stay on the unit sphere (would be < 1 without renormalization).
    norms = np.linalg.norm(centers, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    # Euclidean assignment now matches cosine nearest-centroid.
    cosine_nearest = np.argmax(H @ centers.T, axis=1)
    np.testing.assert_array_equal(labels, cosine_nearest)
