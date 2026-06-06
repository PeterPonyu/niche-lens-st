#!/usr/bin/env python
"""Run a license-clean spatial *baseline* on a niche dataset (issue #152).

This is the baseline-comparison counterpart of ``scripts/run_real_niche.py``. It
applies a simple, dependency-light reference method to the SAME dataset, the
SAME spatial graph (``build_graph``), and the SAME deterministic prototype
assignment + intrinsic metrics as the learned model, so the baseline and the
model are scored apples-to-apples under the uniform results contract.

Two baselines are provided (see ``src/nichelens_st/baselines.py``):

* ``neighborhood`` -- neighborhood-augmented expression: blend each cell's own
  features with its k spatial-neighbor average (``--k``, ``--alpha``), then
  cluster. The standard spatial-niche augmentation idea, clean-room and neutrally
  named.
* ``pca`` -- seeded PCA-of-expression, the non-spatial reference baseline.

Usage (under the project's conda env):

    conda run --no-capture-output -n dl python scripts/run_baselines_niche.py \
        --baseline neighborhood --k 8 --alpha 0.5

Results are written under ``results/baselines/<baseline>/`` so they never clobber
the learned model's ``results/niche-lens-st/`` artifacts. The metrics schema is
identical (uniform results contract), enabling a side-by-side comparison table.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

# Resolve repo root from this file so the script runs from any cwd.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_DATA_ROOT = _REPO_ROOT.parent / "data"

# Primary repointed to the CODEX protein spleen dataset (#360); see
# run_real_niche.py for the data provenance + protein/section handling. The
# baseline loader skips count-normalization for non-count input (negatives) and
# autodetects obs['section_id'] (the per-FOV-tile graph boundary).
PRIMARY_PATH = (
    _DATA_ROOT / "processed" / "codex_spleen_goltsev2018" / "anndata.h5ad"
)
FALLBACK_PATH = _DATA_ROOT / "processed" / "niche_merfish_slice" / "anndata.h5ad"

# Kept consistent with run_real_niche so the comparison is apples-to-apples.
N_PROTOTYPES = 10
N_COMPONENTS = 32
GRAPH_K = 6
SILHOUETTE_MAX = 10_000
DEFAULT_SEED = 0


def _looks_like_raw_counts(X) -> bool:
    """Heuristic: does ``X`` look like raw (unnormalized, un-logged) counts?

    Mirrors ``run_real_niche._looks_like_raw_counts`` so the baseline path
    normalizes identically to the model path (count-like input -> total-count +
    log1p): non-negative and either count-like dynamic range (max > 30) or
    integer-valued with a non-trivial range.
    """
    import scipy.sparse as sp

    sample = X[: min(X.shape[0], 2000)]
    arr = sample.toarray() if sp.issparse(sample) else np.asarray(sample)
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size == 0:
        return False
    if float(arr.min()) < 0.0:
        return False
    max_val = float(arr.max())
    integer_valued = bool(np.allclose(arr, np.round(arr)))
    return (max_val > 30.0) or (integer_valued and max_val >= 2.0)


def _to_dense(X) -> np.ndarray:
    import scipy.sparse as sp

    if sp.issparse(X):
        X = X.toarray()
    return np.ascontiguousarray(np.asarray(X), dtype=np.float64)


def _load_dataset(path: Path, already_normalized: bool):
    """Load an AnnData; conditionally normalize; return (X, coords, section_id, ...).

    Returns ``(X float64, coords float64, section_id int64, section_col,
    normalization, n_obs, n_vars)`` -- the same preprocessing the model runner
    applies, so the baseline sees identical inputs.
    """
    import scanpy as sc

    adata = sc.read_h5ad(str(path))
    n_obs, n_vars = int(adata.shape[0]), int(adata.shape[1])

    if already_normalized:
        applied, method = False, "none"
    elif _looks_like_raw_counts(adata.X):
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
        applied, method = True, "normalize_total+log1p"
    else:
        applied, method = False, "none"

    X = _to_dense(adata.X)
    if "spatial" not in adata.obsm:
        raise KeyError(f"{path} has no obsm['spatial']; cannot build graph")
    coords = np.ascontiguousarray(adata.obsm["spatial"], dtype=np.float64)

    section_col = None
    for cand in ("fov", "slice_id", "section_id", "sample", "batch"):
        if cand in adata.obs.columns:
            section_col = cand
            break
    if section_col is not None:
        codes = adata.obs[section_col].astype("category").cat.codes.to_numpy()
        section_id = np.ascontiguousarray(codes, dtype=np.int64)
    else:
        section_id = np.zeros(n_obs, dtype=np.int64)

    normalization = {"applied": applied, "method": method}
    return X, coords, section_id, section_col, normalization, n_obs, n_vars


def compute_embedding(
    X: np.ndarray,
    coords: np.ndarray,
    section_id: np.ndarray,
    *,
    baseline: str,
    k: int,
    alpha: float,
    n_components: int,
    seed: int,
    n_steps: int = 3,
) -> np.ndarray:
    """Dispatch to the requested baseline embedding (pure numpy/scipy)."""
    from nichelens_st import baselines

    if baseline == "neighborhood":
        return baselines.neighborhood_augmented_embedding(
            X, coords, k=k, alpha=alpha, section_id=section_id
        )
    if baseline == "pca":
        return baselines.pca_embedding(X, n_components=n_components, seed=seed)
    if baseline == "diffusion":
        return baselines.spatial_diffusion_embedding(
            X, coords, k=k, n_steps=n_steps, alpha=alpha, section_id=section_id
        )
    raise ValueError(
        f"unknown baseline {baseline!r}; expected 'neighborhood', 'pca', or 'diffusion'"
    )


def baseline_intrinsic_metrics(
    embedding: np.ndarray,
    prototype_id: np.ndarray,
    edges: np.ndarray,
    seed: int,
) -> tuple[dict[str, float | None], list[str]]:
    """Intrinsic-only metrics matching ``run_real_niche._intrinsic_metrics``.

    Emits the same metric keys the model run emits (prototype-size structure,
    niche Moran's I, neighbor label-agreement, embedding silhouette) so the
    baseline and model rows line up column-for-column.
    """
    from nichelens_st.metrics import morans_i, neighbor_label_agreement

    metrics: dict[str, float | None] = {}
    notes_extra: list[str] = []

    proto = np.asarray(prototype_id, dtype=np.int64)
    n_protos = int(proto.max()) + 1 if proto.size else 0
    metrics["n_prototypes"] = float(n_protos)

    counts = np.bincount(proto, minlength=n_protos).astype(np.float64)
    counts = counts[counts > 0]
    metrics["prototype_size_min"] = float(counts.min()) if counts.size else None
    metrics["prototype_size_median"] = float(np.median(counts)) if counts.size else None
    metrics["prototype_size_max"] = float(counts.max()) if counts.size else None
    if counts.size:
        p = counts / counts.sum()
        metrics["prototype_size_entropy"] = float(-np.sum(p * np.log(p)))
    else:
        metrics["prototype_size_entropy"] = None

    metrics["niche_morans_i"] = float(morans_i(proto.astype(np.float64), edges))
    metrics["niche_label_agreement"] = float(neighbor_label_agreement(proto, edges))

    H = np.asarray(embedding)
    n = H.shape[0]
    rng = np.random.default_rng(seed)
    if n > SILHOUETTE_MAX:
        sub_idx = rng.choice(n, size=SILHOUETTE_MAX, replace=False)
        sub_n = SILHOUETTE_MAX
    else:
        sub_idx = np.arange(n)
        sub_n = n
    metrics["embedding_silhouette_n_subsample"] = float(sub_n)
    sub_labels = proto[sub_idx]
    if np.unique(sub_labels).size >= 2 and sub_n >= 2:
        try:
            from sklearn.metrics import silhouette_score

            metrics["embedding_silhouette"] = float(
                silhouette_score(H[sub_idx], sub_labels)
            )
        except Exception as exc:  # noqa: BLE001
            metrics["embedding_silhouette"] = None
            notes_extra.append(f"silhouette failed: {exc}")
    else:
        metrics["embedding_silhouette"] = None
        notes_extra.append("silhouette needs >=2 prototypes in subsample")

    return metrics, notes_extra


def run_baseline(
    X: np.ndarray,
    coords: np.ndarray,
    section_id: np.ndarray,
    *,
    baseline: str,
    k: int,
    alpha: float,
    n_prototypes: int,
    n_components: int,
    seed: int,
    n_steps: int = 3,
):
    """Compute embedding -> prototypes -> intrinsic metrics for one baseline.

    Returns ``(prototype_id, embedding, metrics, notes_extra)``. Pure numpy/scipy
    and deterministic given ``seed``; takes in-memory arrays so it is unit-testable
    without scanpy / on-disk data.
    """
    from nichelens_st import baselines
    from nichelens_st.graph import build_graph

    embedding = compute_embedding(
        X,
        coords,
        section_id,
        baseline=baseline,
        k=k,
        alpha=alpha,
        n_components=n_components,
        seed=seed,
        n_steps=n_steps,
    )
    prototype_id = baselines.assign_prototypes(
        embedding, n_clusters=n_prototypes, seed=seed
    )
    edges = build_graph(np.asarray(coords), np.asarray(section_id), k=GRAPH_K, method="knn")
    metrics, notes_extra = baseline_intrinsic_metrics(
        embedding, prototype_id, edges, seed
    )
    return prototype_id, embedding, metrics, notes_extra


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--baseline",
        choices=("neighborhood", "pca", "diffusion"),
        default="neighborhood",
    )
    parser.add_argument(
        "--k", type=int, default=8, help="Spatial neighbors for the neighborhood blend."
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Neighborhood blend weight in [0, 1]; 0 -> non-augmented self features.",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=3,
        help="Diffusion steps (ignored for neighborhood/pca baselines).",
    )
    parser.add_argument("--n-prototypes", type=int, default=N_PROTOTYPES)
    parser.add_argument("--n-components", type=int, default=N_COMPONENTS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--already-normalized", action="store_true")
    parser.add_argument(
        "--dataset", choices=("auto", "primary", "fallback"), default="auto"
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    sys.path.insert(0, str(_REPO_ROOT / "src"))
    from nichelens_st import results_contract

    if args.dataset == "fallback":
        used_path = FALLBACK_PATH
    elif args.dataset == "primary":
        used_path = PRIMARY_PATH
    else:  # auto: prefer primary if present, else fallback
        used_path = PRIMARY_PATH if PRIMARY_PATH.exists() else FALLBACK_PATH

    started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    t0 = time.time()

    (
        X,
        coords,
        section_id,
        section_col,
        normalization,
        n_obs,
        n_vars,
    ) = _load_dataset(used_path, args.already_normalized)

    prototype_id, embedding, metrics, notes_extra = run_baseline(
        X,
        coords,
        section_id,
        baseline=args.baseline,
        k=args.k,
        alpha=args.alpha,
        n_prototypes=args.n_prototypes,
        n_components=args.n_components,
        seed=args.seed,
        n_steps=args.n_steps,
    )
    elapsed = time.time() - t0

    notes = list(notes_extra)
    notes.append(f"baseline={args.baseline}")
    if args.baseline == "neighborhood":
        notes.append(f"k={args.k}")
        notes.append(f"alpha={args.alpha}")
    elif args.baseline == "pca":
        notes.append(f"n_components={args.n_components}")
    elif args.baseline == "diffusion":
        notes.append(f"k={args.k}")
        notes.append(f"alpha={args.alpha}")
        notes.append(f"n_steps={args.n_steps}")
    notes.append(f"section_id source: {section_col or 'single-section zeros'}")
    notes.append(f"runtime_s={elapsed:.2f}")

    card_id = results_contract.dataset_card_id([str(used_path)])
    results_dir = _REPO_ROOT / "results" / "baselines" / args.baseline
    paths = results_contract.write_results(
        project="niche-lens-st",
        dataset_card_id=card_id,
        metrics=metrics,
        outputs={"baseline_niche": "outputs/baseline_niche.npz"},
        run_metadata={
            "dataset_paths": [str(used_path)],
            "n_obs": n_obs,
            "n_vars": n_vars,
            "seed": args.seed,
            "runtime_s": elapsed,
            "started_utc": started_utc,
            "device": "cpu",
            "deterministic": True,
            "num_threads": max(1, os.cpu_count() or 1),
            "reproducibility_level": "seeded",
            "normalization": normalization,
            "interpretability": {
                "model_is_learned": False,
                "baseline": args.baseline,
                "method": (
                    "neighborhood-augmented expression (self + k-NN neighbor "
                    "average), spherical k-means prototypes"
                    if args.baseline == "neighborhood"
                    else (
                        f"iterated spatial-graph diffusion (n_steps={args.n_steps} "
                        "k-NN blends), spherical k-means prototypes"
                        if args.baseline == "diffusion"
                        else "PCA-of-expression, spherical k-means prototypes"
                    )
                ),
                "blend_alpha": (
                    float(args.alpha)
                    if args.baseline in ("neighborhood", "diffusion")
                    else None
                ),
                "neighbors_k": (
                    int(args.k)
                    if args.baseline in ("neighborhood", "diffusion")
                    else None
                ),
                "n_diffusion_steps": (
                    int(args.n_steps) if args.baseline == "diffusion" else None
                ),
                "caveats": [
                    "license-clean clean-room baseline; not an external package"
                ],
            },
            "notes": "; ".join(notes),
        },
        results_dir=str(results_dir),
    )

    outputs_dir = Path(paths["outputs_dir"])
    np.savez_compressed(
        outputs_dir / "baseline_niche.npz",
        embedding=np.asarray(embedding),
        prototype_id=np.asarray(prototype_id),
    )

    print(f"baseline={args.baseline} dataset={used_path}")
    print(f"n_obs={n_obs} n_vars={n_vars}")
    print(f"metrics.json -> {paths['metrics']}")
    print(f"run_metadata.json -> {paths['run_metadata']}")
    print(f"outputs -> {outputs_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
