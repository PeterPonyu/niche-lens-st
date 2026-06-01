#!/usr/bin/env python
"""Run the fitted NicheLens-ST model on a real on-disk MERSCOPE/MERFISH dataset.

This is the real-data results path for NicheLens-ST (consensus plan §4.4). It
runs the FITTED contrastive niche model (``fit_niche_model``) -- NOT a
truth-vs-truth scoring -- and emits intrinsic/unsupervised metrics only (no
ARI, no marker-recall-vs-truth) via the vendored uniform results contract.

Primary dataset: ``data/processed/niche_GSE282124/anndata.h5ad`` (124938 x 315).
Feasibility fallback: ``data/processed/niche_merfish_slice/anndata.h5ad``
(5488 x 155) -- triggered on EITHER the encoder wall OR the post-encoder
``_kmeans`` dense ``(n, k, d)`` materialization wall.

Usage (under the project's conda env):

    conda run --no-capture-output -n dl python scripts/run_real_niche.py

Flags:
    --already-normalized   Skip the conditional log1p (treat X as normalized).
    --dataset {auto,primary,fallback}
                           Force a dataset; ``auto`` (default) tries primary
                           then falls back on a memory/time wall.
    --max-seconds N        Wall-time budget for the primary fit before fallback.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

# Resolve repo root from this file so the script runs from any cwd.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_DATA_ROOT = _REPO_ROOT.parent / "data"

PRIMARY_PATH = _DATA_ROOT / "processed" / "niche_GSE282124" / "anndata.h5ad"
FALLBACK_PATH = _DATA_ROOT / "processed" / "niche_merfish_slice" / "anndata.h5ad"

# Prototype/embedding sizing kept modest to bound the _kmeans (n,k,d) tensor.
N_PROTOTYPES = 10
EMBED_DIM = 32
SILHOUETTE_MAX = 10_000
DEFAULT_MAX_SECONDS = 1800.0


def _looks_like_raw_counts(X) -> bool:
    """Heuristic: integer-valued and/or large max => raw counts (apply log1p)."""
    import scipy.sparse as sp

    sample = X[: min(X.shape[0], 2000)]
    arr = sample.toarray() if sp.issparse(sample) else np.asarray(sample)
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size == 0:
        return False
    integer_valued = bool(np.allclose(arr, np.round(arr)))
    large_max = float(arr.max()) > 50.0
    # Raw counts are integer-valued; a large max alone (non-integer, normalized)
    # should NOT trigger log1p. Require integer-valued counts.
    return integer_valued and (large_max or float(arr.max()) >= 1.0)


def _to_dense_float32(X) -> np.ndarray:
    import scipy.sparse as sp

    if sp.issparse(X):
        X = X.toarray()
    return np.ascontiguousarray(np.asarray(X), dtype=np.float32)


def _load_dataset(path: Path, already_normalized: bool):
    """Load an AnnData, cast X to float32, conditionally log1p, build inputs.

    Returns (X float32 dense, coords float32, section_id int64, normalization
    dict, n_obs, n_vars).
    """
    import scanpy as sc

    adata = sc.read_h5ad(str(path))
    n_obs, n_vars = int(adata.shape[0]), int(adata.shape[1])

    # Conditional log1p (plan §4.4 step 1): apply only on a raw-counts heuristic
    # unless explicitly told the matrix is already normalized.
    if already_normalized:
        applied, method = False, "none"
    elif _looks_like_raw_counts(adata.X):
        sc.pp.log1p(adata)
        applied, method = True, "log1p"
    else:
        applied, method = False, "none"

    X = _to_dense_float32(adata.X)

    if "spatial" not in adata.obsm:
        raise KeyError(f"{path} has no obsm['spatial']; cannot build graph")
    coords = np.ascontiguousarray(adata.obsm["spatial"], dtype=np.float32)

    # section_id from a sample/section column if present, else single section.
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


def _intrinsic_metrics(result, edges, section_id, seed):
    """Intrinsic-only metrics: prototype structure, niche Moran's I, silhouette."""
    from nichelens_st.metrics import morans_i

    metrics: dict[str, float | None] = {}
    notes_extra: list[str] = []

    proto = np.asarray(result.prototype_id)
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

    # Niche spatial coherence: Moran's I of the (integer) prototype assignment.
    metrics["niche_morans_i"] = float(morans_i(proto.astype(np.float64), edges))

    # Embedding silhouette (subsample if large; O(n^2) otherwise infeasible).
    H = np.asarray(result.H)
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

    # Conserved fraction (needs >=2 sections to be meaningful).
    n_sections = int(np.unique(section_id).size)
    proto_kind = list(result.proto_kind)
    if proto_kind:
        conserved = sum(1 for k in proto_kind if k == "conserved")
        metrics["conserved_fraction"] = float(conserved / len(proto_kind))
    else:
        metrics["conserved_fraction"] = None
    if n_sections < 2:
        notes_extra.append("single_section=True (conserved_fraction degenerate)")

    return metrics, notes_extra


def _fit_with_walls(X, coords, section_id, max_seconds, device, num_threads):
    """Run fit_niche_model, guarding the encoder + _kmeans (n,k,d) walls.

    Returns (result, elapsed_s). Raises on OOM / over-budget so the caller can
    fall back to the smaller dataset.
    """
    from nichelens_st.graph import build_graph
    from nichelens_st.model import NicheModelConfig, fit_niche_model

    edges = build_graph(coords, section_id, k=6, method="knn")

    cfg = NicheModelConfig(
        embed_dim=EMBED_DIM,
        n_prototypes=N_PROTOTYPES,
        seed=0,
        num_threads=num_threads,
        device=device,
        deterministic=False,
    )
    t0 = time.time()
    result = fit_niche_model(
        X=X, coords=coords, section_id=section_id, edges=edges, config=cfg
    )
    elapsed = time.time() - t0
    if elapsed > max_seconds:
        raise TimeoutError(
            f"fit exceeded budget: {elapsed:.1f}s > {max_seconds:.1f}s"
        )
    return result, edges, elapsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--already-normalized", action="store_true")
    parser.add_argument(
        "--dataset", choices=("auto", "primary", "fallback"), default="auto"
    )
    parser.add_argument("--max-seconds", type=float, default=DEFAULT_MAX_SECONDS)
    args = parser.parse_args()

    # Ensure the package is importable when running from a source checkout.
    sys.path.insert(0, str(_REPO_ROOT / "src"))

    from nichelens_st import results_contract

    # Resolve device/threads for the relaxed real path.
    try:
        import torch

        cuda = torch.cuda.is_available()
    except Exception:
        cuda = False
    device = "cuda" if cuda else "cpu"
    num_threads = max(1, os.cpu_count() or 1)
    reproducibility_level = "seeded"

    notes: list[str] = []
    started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    seed = 0

    def run_on(path: Path, label: str):
        X, coords, section_id, section_col, normalization, n_obs, n_vars = (
            _load_dataset(path, args.already_normalized)
        )
        result, edges, elapsed = _fit_with_walls(
            X, coords, section_id, args.max_seconds, device, num_threads
        )
        metrics, notes_extra = _intrinsic_metrics(result, edges, section_id, seed)
        return dict(
            path=path,
            label=label,
            section_col=section_col,
            normalization=normalization,
            n_obs=n_obs,
            n_vars=n_vars,
            result=result,
            elapsed=elapsed,
            metrics=metrics,
            notes_extra=notes_extra,
        )

    run = None
    used_path = None

    if args.dataset == "fallback":
        used_path = FALLBACK_PATH
        run = run_on(FALLBACK_PATH, "fallback")
    elif args.dataset == "primary":
        used_path = PRIMARY_PATH
        run = run_on(PRIMARY_PATH, "primary")
    else:  # auto
        try:
            used_path = PRIMARY_PATH
            run = run_on(PRIMARY_PATH, "primary")
        except (MemoryError, TimeoutError, RuntimeError) as exc:
            notes.append(
                f"fell back to 5488-cell slice from primary 124k wall: "
                f"{type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            used_path = FALLBACK_PATH
            run = run_on(FALLBACK_PATH, "fallback")

    notes.extend(run["notes_extra"])
    if run["label"] == "fallback" and args.dataset != "fallback":
        notes.append("dataset=fallback (5488 cells)")
    else:
        notes.append(f"dataset={run['label']} ({run['n_obs']} cells)")
    notes.append(f"section_id source: {run['section_col'] or 'single-section zeros'}")
    notes.append(f"fit_runtime_s={run['elapsed']:.2f}")

    # Write outputs/ artifacts: niche.npz (H, prototype_id) + proto_kind.json.
    card_id = results_contract.dataset_card_id([str(used_path)])
    results_dir = _REPO_ROOT / "results"
    paths = results_contract.write_results(
        project="niche-lens-st",
        dataset_card_id=card_id,
        metrics=run["metrics"],
        outputs={
            "niche": "outputs/niche.npz",
            "proto_kind": "outputs/proto_kind.json",
        },
        run_metadata={
            "dataset_paths": [str(used_path)],
            "n_obs": run["n_obs"],
            "n_vars": run["n_vars"],
            "seed": seed,
            "runtime_s": run["elapsed"],
            "started_utc": started_utc,
            "device": device,
            "deterministic": False,
            "num_threads": num_threads,
            "reproducibility_level": reproducibility_level,
            "normalization": run["normalization"],
            "interpretability": {
                "model_is_learned": True,
                "encoder": (
                    "contrastive GraphSAGE-mean niche encoder (InfoNCE) trained "
                    "on cell-centered subgraphs"
                ),
                "domain_assignment": (
                    "deterministic k-means over learned embeddings H -> prototype_id"
                ),
                "caveats": [],
            },
            "notes": "; ".join(notes),
        },
        results_dir=str(results_dir),
    )

    outputs_dir = Path(paths["outputs_dir"])
    np.savez_compressed(
        outputs_dir / "niche.npz",
        H=np.asarray(run["result"].H),
        prototype_id=np.asarray(run["result"].prototype_id),
    )
    import json

    with open(outputs_dir / "proto_kind.json", "w", encoding="utf-8") as fh:
        json.dump(list(run["result"].proto_kind), fh, indent=2)

    print(f"dataset used: {run['label']} ({used_path})")
    print(f"n_obs={run['n_obs']} n_vars={run['n_vars']} device={device}")
    print(f"metrics.json -> {paths['metrics']}")
    print(f"run_metadata.json -> {paths['run_metadata']}")
    print(f"outputs -> {outputs_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
