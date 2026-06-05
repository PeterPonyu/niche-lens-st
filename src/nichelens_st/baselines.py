"""License-clean, neutrally-named spatial baselines for niche recovery.

These baselines exist so the learned niche model can be compared apples-to-apples
against simple, dependency-light reference methods on the *same* dataset, *same*
spatial graph, and *same* downstream clustering / metrics (see issue #152). They
are clean-room implementations of well-known ideas -- no external package code is
copied and no method brand names are used.

Three embeddings are provided:

* :func:`neighborhood_augmented_embedding` -- the standard neighborhood-
  augmentation idea: blend each cell's own expression with the average
  expression of its k spatial neighbors. On spatially contiguous niches the
  neighbor average denoises the per-cell signal, so clustering the blended
  features recovers niches better than clustering raw expression. The blend
  weight ``alpha`` and neighbor count ``k`` are explicit; ``alpha == 0`` or
  ``k == 0`` reduces the embedding to the non-augmented (self-only) features.
* :func:`spatial_diffusion_embedding` -- multi-step iterated graph smoothing:
  apply the single-step neighborhood blend ``n_steps`` times on the same
  per-section k-NN graph. The ``n_steps == 1`` case is identical to
  :func:`neighborhood_augmented_embedding`; larger ``n_steps`` produces stronger
  spatial denoising at the cost of more blur.
* :func:`pca_embedding` -- a thin, seeded PCA-of-expression baseline matching the
  existing intrinsic ``pca_baseline_silhouette`` reference.

All three feed :func:`assign_prototypes`, which reuses the model's deterministic
spherical k-means (:func:`nichelens_st.model._kmeans`) so every method shares the
identical prototype-assignment step. Everything here is pure ``numpy`` / ``scipy``
(the spatial graph is built with :func:`nichelens_st.graph.build_graph`, which
uses ``scipy.spatial``) and fully deterministic given a seed.
"""

from __future__ import annotations

import numpy as np

from nichelens_st.graph import build_graph
from nichelens_st.model import _kmeans


def neighborhood_averaged_features(
    X: np.ndarray,
    coords: np.ndarray,
    *,
    k: int,
    section_id: np.ndarray | None = None,
) -> np.ndarray:
    """Mean expression of each cell's ``k`` spatial neighbors (self-fallback).

    The spatial neighborhood is the per-section k-NN graph from
    :func:`nichelens_st.graph.build_graph` (the same graph the model uses), so
    neighbors never cross section boundaries. A cell with no neighbors -- an
    isolated node, a singleton section, or ``k == 0`` -- falls back to its own
    feature row, so the result is always well-defined and finite.

    Returns a ``(n_cells, n_genes)`` float64 array.
    """
    Xa = np.asarray(X, dtype=np.float64)
    if Xa.ndim != 2:
        raise ValueError(f"X must be (n_cells, n_genes); got {Xa.shape}")
    n = Xa.shape[0]
    if k < 0:
        raise ValueError(f"k must be non-negative; got {k}")
    if section_id is None:
        section_id = np.zeros(n, dtype=np.int64)
    section_id = np.asarray(section_id, dtype=np.int64)

    edges = build_graph(np.asarray(coords), section_id, k=int(k), method="knn")

    nbr_sum = np.zeros_like(Xa)
    deg = np.zeros(n, dtype=np.int64)
    if edges.shape[1]:
        src = edges[0]
        dst = edges[1]
        np.add.at(nbr_sum, src, Xa[dst])
        deg = np.bincount(src, minlength=n)
    has_nbr = deg > 0
    nbr_mean = Xa.copy()
    if has_nbr.any():
        nbr_mean[has_nbr] = nbr_sum[has_nbr] / deg[has_nbr][:, None]
    return nbr_mean


def neighborhood_augmented_embedding(
    X: np.ndarray,
    coords: np.ndarray,
    *,
    k: int = 8,
    alpha: float = 0.5,
    section_id: np.ndarray | None = None,
) -> np.ndarray:
    """Blend self expression with the spatial-neighborhood-averaged expression.

    ``embedding = (1 - alpha) * X + alpha * neighborhood_mean(X)``, where the
    neighborhood mean averages each cell's ``k`` nearest spatial neighbors (see
    :func:`neighborhood_averaged_features`). The output keeps the gene-space
    dimensionality, so it is directly comparable to the raw-expression / PCA
    baselines under the shared :func:`assign_prototypes` clustering.

    Reduction property (used by the tests): ``alpha == 0`` returns ``X`` exactly,
    and ``k == 0`` returns ``X`` exactly for any ``alpha`` (no neighbors -> the
    neighborhood mean is the self row), so the augmented baseline cleanly
    collapses to the non-augmented one.

    Deterministic and pure ``numpy`` / ``scipy``. Returns a ``(n_cells,
    n_genes)`` float64 array.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1]; got {alpha}")
    Xa = np.asarray(X, dtype=np.float64)
    if alpha == 0.0:
        return Xa.copy()
    nbr_mean = neighborhood_averaged_features(Xa, coords, k=k, section_id=section_id)
    return (1.0 - alpha) * Xa + alpha * nbr_mean


def spatial_diffusion_embedding(
    X: np.ndarray,
    coords: np.ndarray,
    *,
    k: int = 8,
    n_steps: int = 3,
    alpha: float = 0.5,
    section_id: np.ndarray | None = None,
) -> np.ndarray:
    """Iterated spatial-graph diffusion embedding.

    Repeatedly blends each cell's features with its k spatial-neighbor average
    ``n_steps`` times on the same per-section k-NN graph
    (:func:`nichelens_st.graph.build_graph`). This is the multi-step
    generalisation of :func:`neighborhood_augmented_embedding` (the
    ``n_steps == 1`` case) -- a stronger spatial denoiser that converges
    toward the per-niche mean as ``n_steps`` grows.

    Reduction properties (verified by the tests):

    * ``n_steps == 0`` returns ``X`` exactly (no diffusion applied).
    * ``alpha == 0`` returns ``X`` exactly (zero blend weight at every step).
    * ``k == 0`` returns ``X`` exactly (no neighbors -> neighbor mean is self).
    * ``n_steps == 1`` returns the same array as
      :func:`neighborhood_augmented_embedding` with the same ``k`` and ``alpha``.

    Validates ``alpha in [0, 1]``, ``n_steps >= 0``, and ``k >= 0`` with
    :class:`ValueError`, matching the convention of the sibling baselines.
    Pure ``numpy`` / ``scipy`` and fully deterministic. Returns a
    ``(n_cells, n_genes)`` float64 array.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1]; got {alpha}")
    if n_steps < 0:
        raise ValueError(f"n_steps must be non-negative; got {n_steps}")
    if k < 0:
        raise ValueError(f"k must be non-negative; got {k}")
    Xa = np.asarray(X, dtype=np.float64)
    if n_steps == 0 or alpha == 0.0 or k == 0:
        return Xa.copy()
    result = Xa
    for _ in range(n_steps):
        nbr_mean = neighborhood_averaged_features(
            result, coords, k=k, section_id=section_id
        )
        result = (1.0 - alpha) * result + alpha * nbr_mean
    return result


def pca_embedding(
    X: np.ndarray, *, n_components: int = 32, seed: int = 0
) -> np.ndarray:
    """PCA-of-expression embedding (non-spatial reference baseline).

    Pure-``numpy`` PCA via the economy SVD of the mean-centered matrix, so the
    baseline carries no scikit-learn dependency (it must run in the minimal CI
    matrix). The number of components is clamped to
    ``min(n_components, n_genes, n_cells - 1)``. A deterministic component-sign
    convention (largest-magnitude loading made positive) removes the SVD sign
    ambiguity so the projection is reproducible across BLAS backends. ``seed`` is
    accepted for API parity with the other baselines (the SVD path is exact and
    needs no randomness). Returns a ``(n_cells, n_comp)`` float64 array.
    """
    del seed  # exact SVD; accepted only for a uniform baseline signature
    Xa = np.asarray(X, dtype=np.float64)
    if Xa.ndim != 2:
        raise ValueError(f"X must be (n_cells, n_genes); got {Xa.shape}")
    n_comp = max(1, int(min(n_components, Xa.shape[1], Xa.shape[0] - 1)))
    Xc = Xa - Xa.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    # Deterministic sign: make each component's largest-magnitude loading positive.
    max_abs = np.argmax(np.abs(Vt), axis=1)
    signs = np.sign(Vt[np.arange(Vt.shape[0]), max_abs])
    signs[signs == 0] = 1.0
    scores = (U * signs) * S
    return np.ascontiguousarray(scores[:, :n_comp], dtype=np.float64)


def _unit_rows(M: np.ndarray) -> np.ndarray:
    """L2-normalize rows onto the unit sphere (near-zero rows stay near 0)."""
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    return M / np.maximum(norms, 1e-12)


def assign_prototypes(
    embedding: np.ndarray, *, n_clusters: int, seed: int = 0, n_iters: int = 50
) -> np.ndarray:
    """Assign niche prototypes by the model's deterministic spherical k-means.

    Rows are L2-normalized first so the cosine geometry of
    :func:`nichelens_st.model._kmeans` (spherical k-means, the exact step the
    learned model uses for prototype assignment) applies identically to every
    baseline embedding -- keeping the comparison apples-to-apples. Returns int64
    cluster ids in ``[0, n_clusters)`` (contiguous, content-ordered).
    """
    emb = _unit_rows(np.asarray(embedding, dtype=np.float64))
    return _kmeans(emb, n_clusters=int(n_clusters), n_iters=int(n_iters), seed=int(seed))


__all__ = [
    "neighborhood_averaged_features",
    "neighborhood_augmented_embedding",
    "spatial_diffusion_embedding",
    "pca_embedding",
    "assign_prototypes",
]
