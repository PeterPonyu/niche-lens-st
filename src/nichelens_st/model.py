"""Core NicheLens-ST model: contrastive encoder + prototype/separation head.

This module produces the three MVP outputs validated by
``nichelens_st.schemas.validate_outputs``:

* ``H`` ``(n_cells, d)`` float32 -- niche embeddings from the contrastive
  encoder (see ``nichelens_st.encoder``).
* ``prototype_id`` ``(n_cells,)`` int64, non-negative -- assignment to a global
  niche-prototype index via deterministic k-means over ``H``.
* ``proto_kind`` -- per-prototype tag in ``{conserved, sample_specific}`` from
  the separation head.

Dependency choice
-----------------
The encoder uses **PyTorch**, declared as the optional ``[model]`` extra (we do
not adopt ``torch-geometric`` or ``jax``; see ``nichelens_st.encoder``). The
import is gated: this module always imports, and ``TORCH_AVAILABLE`` advertises
whether the extra is installed. Calling :func:`fit_niche_model` without torch
raises a clear ``ImportError`` pointing at the extra. Prototype assignment and
the separation head are pure numpy, so they stay reproducible and
dependency-light.

Separation head
---------------
A prototype is tagged ``conserved`` iff cells assigned to it span *every*
section, and ``sample_specific`` otherwise (present in a strict subset). This
mirrors the synthetic ground-truth definition in ``docs/SYNTHETIC_BENCHMARK.md``
and the ``section_overlap_rate`` metric: conserved prototypes are shared across
all sections while sample-specific variants live in a subset.

Reproducibility
---------------
All randomness derives from ``NicheModelConfig.seed`` (encoder init/augmentation
plus k-means initialization), so a fixed seed yields identical ``prototype_id``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from nichelens_st.encoder import TORCH_AVAILABLE, EncoderConfig, train_embeddings

__all__ = [
    "TORCH_AVAILABLE",
    "NicheModelConfig",
    "NicheModelResult",
    "fit_niche_model",
]


@dataclass
class NicheModelConfig:
    """Configuration for :func:`fit_niche_model`."""

    embed_dim: int = 32
    hidden_dim: int = 64
    num_layers: int = 2
    epochs: int = 30
    lr: float = 1e-2
    tau: float = 0.2
    feat_drop: float = 0.2
    edge_drop: float = 0.2
    n_prototypes: int = 10
    kmeans_iters: int = 50
    marker_top_k: int = 5  # per-prototype markers reported on the result (#82).
    seed: int = 0

    def encoder_config(self) -> EncoderConfig:
        return EncoderConfig(
            embed_dim=self.embed_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            epochs=self.epochs,
            lr=self.lr,
            tau=self.tau,
            feat_drop=self.feat_drop,
            edge_drop=self.edge_drop,
            seed=self.seed,
        )


@dataclass
class NicheModelResult:
    """Schema-valid model outputs (see ``validate_outputs``)."""

    H: np.ndarray            # (n_cells, d) float32
    prototype_id: np.ndarray  # (n_cells,) int64 non-negative
    proto_kind: list[str]     # per-prototype tag in {conserved, sample_specific}
    # Per-prototype top-k gene indices by mean X (issue #82). Empty by
    # default for back-compat with callers that build the result directly.
    marker_table: list[list[int]] = field(default_factory=list)


def _unit_rows(M: np.ndarray) -> np.ndarray:
    """Project rows back onto the unit sphere (rows with ~0 norm stay near 0)."""
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    return M / np.maximum(norms, 1e-12)


def _spherical_kmeans(
    Hd: np.ndarray, k: int, n_iters: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Lloyd's k-means specialized for L2-normalized (unit-sphere) embeddings.

    The encoder is trained on a cosine objective and emits unit-norm rows, so
    cluster assignment must use cosine geometry. Centroids are renormalized onto
    the unit sphere after every mean update (spherical k-means): on the sphere
    squared-Euclidean nearest-centroid then agrees with cosine nearest-centroid.
    Without renormalization the mean of unit vectors has norm < 1 and drifts
    inside the ball, so Euclidean assignment no longer matches cosine. Returns
    ``(labels, centers)`` with ``centers`` on the unit sphere.
    """
    n = Hd.shape[0]

    # k-means++ initialization (seeds are sampled unit-norm rows).
    centers = np.empty((k, Hd.shape[1]), dtype=np.float64)
    first = int(rng.integers(n))
    centers[0] = Hd[first]
    closest_sq = np.sum((Hd - centers[0]) ** 2, axis=1)
    for c in range(1, k):
        total = closest_sq.sum()
        if total <= 0:
            centers[c] = Hd[int(rng.integers(n))]
        else:
            probs = closest_sq / total
            centers[c] = Hd[int(rng.choice(n, p=probs))]
        dist_sq = np.sum((Hd - centers[c]) ** 2, axis=1)
        closest_sq = np.minimum(closest_sq, dist_sq)

    labels = np.zeros(n, dtype=np.int64)
    for _ in range(n_iters):
        # Assign (Euclidean nearest == cosine nearest while centers are unit).
        dists = np.sum(
            (Hd[:, None, :] - centers[None, :, :]) ** 2, axis=2
        )
        new_labels = np.argmin(dists, axis=1).astype(np.int64)
        if np.array_equal(new_labels, labels):
            labels = new_labels
            break
        labels = new_labels
        # Update centroids, then renormalize back onto the unit sphere.
        for c in range(k):
            members = Hd[labels == c]
            if members.size:
                centers[c] = members.mean(axis=0)
        centers = _unit_rows(centers)

    return labels, centers


def _kmeans(
    H: np.ndarray, n_clusters: int, n_iters: int, seed: int
) -> np.ndarray:
    """Deterministic spherical k-means returning int64 cluster ids in ``[0, k)``.

    Uses k-means++ seeding with a fixed ``numpy`` generator so a given seed and
    embedding matrix always reproduce the same assignment. Cluster ids are
    relabeled to a contiguous, content-defined order (by first appearance over
    sorted centroids) so the catalog is stable.
    """
    n = H.shape[0]
    k = min(n_clusters, n)
    rng = np.random.default_rng(seed)
    Hd = H.astype(np.float64, copy=False)

    labels, _ = _spherical_kmeans(Hd, k, n_iters, rng)

    # Relabel to contiguous ids in order of first appearance for stability.
    _, first_idx = np.unique(labels, return_index=True)
    order = labels[np.sort(first_idx)]
    remap = {int(old): new for new, old in enumerate(order)}
    return np.array([remap[int(v)] for v in labels], dtype=np.int64)


def _separation_head(
    prototype_id: np.ndarray, section_id: np.ndarray, n_protos: int
) -> list[str]:
    """Tag each prototype conserved vs sample_specific by cross-section presence.

    Conserved iff the prototype's assigned cells span every section; otherwise
    sample_specific. Mirrors the synthetic ground-truth definition.
    """
    all_sections = set(np.asarray(section_id).tolist())
    kinds: list[str] = []
    for p in range(n_protos):
        seen = set(section_id[prototype_id == p].tolist())
        if seen and seen == all_sections:
            kinds.append("conserved")
        else:
            kinds.append("sample_specific")
    return kinds


def _compute_marker_table(
    X: np.ndarray, prototype_id: np.ndarray, n_protos: int, top_k: int
) -> list[list[int]]:
    """Per-prototype top-``top_k`` gene indices ranked by mean X (issue #82)."""
    if top_k < 1:
        raise ValueError(f"marker_top_k must be >= 1; got {top_k}")
    Xa = np.asarray(X)
    n_genes = Xa.shape[1] if Xa.ndim == 2 else 0
    k = min(top_k, n_genes)
    table: list[list[int]] = []
    for p in range(n_protos):
        members = Xa[prototype_id == p]
        if members.size == 0 or k == 0:
            table.append([])
            continue
        order = np.argsort(-members.mean(axis=0), kind="stable")[:k]
        table.append([int(g) for g in order])
    return table


def fit_niche_model(
    X: np.ndarray,
    coords: np.ndarray,
    section_id: np.ndarray,
    edges: np.ndarray,
    config: NicheModelConfig | None = None,
) -> NicheModelResult:
    """Fit the contrastive encoder and prototype/separation head end to end.

    Parameters mirror the MVP input contract (``coords`` is accepted for API
    completeness; the graph is consumed through ``edges``). Returns a
    :class:`NicheModelResult` whose fields satisfy
    :func:`nichelens_st.schemas.validate_outputs`.

    Raises ``ImportError`` (with install guidance) if the optional ``[model]``
    extra / torch is not installed.
    """
    cfg = config or NicheModelConfig()
    section_id = np.asarray(section_id)
    if section_id.ndim != 1 or section_id.shape[0] != X.shape[0]:
        raise ValueError(
            f"section_id must be (n_cells,); got shape={section_id.shape}"
        )

    # 1) Contrastive niche embeddings (torch; gated).
    H = train_embeddings(X, edges, cfg.encoder_config())

    # 2) Prototype assignment (deterministic numpy k-means).
    prototype_id = _kmeans(H, cfg.n_prototypes, cfg.kmeans_iters, cfg.seed)
    n_protos = int(prototype_id.max()) + 1 if prototype_id.size else 0

    # 3) Separation head: conserved vs sample_specific by cross-section presence.
    proto_kind = _separation_head(prototype_id, section_id, n_protos)

    # 4) Per-prototype marker table (issue #82).
    marker_table = _compute_marker_table(X, prototype_id, n_protos, cfg.marker_top_k)

    return NicheModelResult(
        H=H,
        prototype_id=prototype_id,
        proto_kind=proto_kind,
        marker_table=marker_table,
    )
