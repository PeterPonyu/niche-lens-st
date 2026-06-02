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
    --already-normalized   Skip conditional normalize_total+log1p (treat X as
                           already normalized).
    --dataset {auto,primary,fallback}
                           Force a dataset; ``auto`` (default) tries primary
                           then falls back on a memory/time wall.
    --max-seconds N        Wall-time budget for the primary fit before fallback.
    --batch-size N         Minibatch InfoNCE size (#61). 0 (default) keeps the
                           exact full-batch loss; a positive value bounds the
                           contrastive matrix to O(batch^2), enabling the 124k
                           primary dataset without the ~232 GiB OOM.
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


class _NeverRaised(Exception):
    """Sentinel exception type that is never raised (torch-absent fallback)."""


def _cuda_oom_error():
    """Return the narrow CUDA OOM exception type to catch.

    Narrows the previously broad ``except RuntimeError`` to torch's dedicated
    ``torch.cuda.OutOfMemoryError`` (#61). If torch is unavailable, return a
    sentinel type so the ``except`` clause never matches anything else.
    """
    try:
        import torch

        return torch.cuda.OutOfMemoryError
    except Exception:  # noqa: BLE001
        return _NeverRaised


def _looks_like_raw_counts(X) -> bool:
    """Heuristic: does ``X`` look like raw (unnormalized, un-logged) counts?

    Detect count-like matrices regardless of integer-vs-float dtype. MERSCOPE /
    MERFISH high-plex panels frequently store whole-number (or near-whole-number,
    e.g. volume-normalized) transcript counts as ``float32``; the old
    ``integer_valued AND max>50`` rule silently skipped normalization on that
    float storage, leaking raw counts into the encoder (#154).

    A matrix looks like raw counts when:

    * it is non-negative (log/scaled data may be negative), and
    * its dynamic range is count-like rather than already log-compressed --
      i.e. the max value is larger than what log1p of a typical library would
      produce. log1p-normalized expression rarely exceeds ~12-15, whereas raw
      high-plex counts routinely exceed that. We use ``max > 30`` as the
      count-vs-log discriminator (well above any plausible log1p value, well
      below typical raw maxima).

    Integer-valued data is additionally treated as counts whenever it is not
    obviously a tiny binarized/log range, so legacy integer-count inputs keep
    being normalized.
    """
    import scipy.sparse as sp

    sample = X[: min(X.shape[0], 2000)]
    arr = sample.toarray() if sp.issparse(sample) else np.asarray(sample)
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size == 0:
        return False

    # Negative values => not raw counts (scaled / centered / PCA'd).
    if float(arr.min()) < 0.0:
        return False

    max_val = float(arr.max())
    integer_valued = bool(np.allclose(arr, np.round(arr)))

    # Count-like dynamic range, independent of dtype. log1p expression maxes out
    # well below this; raw counts (int OR float-stored) exceed it.
    count_like_range = max_val > 30.0

    # Integer-valued, non-trivial-range data is also counts even if the max is
    # modest (e.g. a low-depth panel), matching the legacy integer behavior.
    integer_counts = integer_valued and max_val >= 2.0

    return count_like_range or integer_counts


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

    # Conditional normalization (plan §4.4 step 1): for count-like input
    # (integer OR float-stored, e.g. MERSCOPE 315-plex, #154) total-count
    # normalize then log1p, so raw counts never reach the encoder. Skipped when
    # the caller asserts the matrix is already normalized.
    if already_normalized:
        applied, method = False, "none"
    elif _looks_like_raw_counts(adata.X):
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
        applied, method = True, "normalize_total+log1p"
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
    return (
        X,
        coords,
        section_id,
        section_col,
        normalization,
        n_obs,
        n_vars,
        adata,
    )


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


def _fit_with_walls(
    X,
    coords,
    section_id,
    max_seconds,
    device,
    num_threads,
    batch_size=0,
    compute_interaction_summary=False,
    adata=None,
):
    """Run fit_niche_model, guarding the encoder + _kmeans (n,k,d) walls.

    Returns (result, edges, elapsed_s). Raises on OOM / over-budget so the
    caller can fall back to the smaller dataset. ``batch_size`` (#61) is
    threaded into the model config to bound the InfoNCE matrix to O(batch^2).

    When ``compute_interaction_summary`` is True (#151), the fitted
    ``prototype_id`` is used to score ligand-receptor enrichment via the gated
    ``[data]`` extra (squidpy/OmniPath); ``adata`` (an AnnData with named genes)
    must be supplied. The default (False) leaves the fit path squidpy-free.
    """
    from nichelens_st import model as _model
    from nichelens_st.graph import build_graph

    edges = build_graph(coords, section_id, k=6, method="knn")

    cfg = _model.NicheModelConfig(
        embed_dim=EMBED_DIM,
        n_prototypes=N_PROTOTYPES,
        seed=0,
        num_threads=num_threads,
        device=device,
        deterministic=False,
        batch_size=int(batch_size),
    )
    t0 = time.time()
    try:
        result = _model.fit_niche_model(
            X=X,
            coords=coords,
            section_id=section_id,
            edges=edges,
            config=cfg,
            compute_interaction_summary=compute_interaction_summary,
            adata=adata,
        )
    except _cuda_oom_error() as exc:  # narrow: CUDA OOM only
        raise RuntimeError(
            "CUDA out of memory during niche fit; rerun with a smaller "
            "--batch-size (e.g. --batch-size 4096) to bound the InfoNCE "
            f"matrix to O(batch^2). Original error: {exc}"
        ) from exc
    elapsed = time.time() - t0
    if elapsed > max_seconds:
        raise TimeoutError(
            f"fit exceeded budget: {elapsed:.1f}s > {max_seconds:.1f}s"
        )
    return result, edges, elapsed


def _interaction_output_rel(summary_df) -> str:
    """Pick the interaction_summary output relative path (parquet if available).

    Prefers ``outputs/interaction_summary.parquet`` when a parquet engine
    (pyarrow/fastparquet) is importable; otherwise falls back to CSV. Only the
    relative path string is returned here -- the file is written later, once the
    results-contract ``outputs/`` dir exists.
    """
    try:
        import importlib

        importlib.import_module("pyarrow")
        return "outputs/interaction_summary.parquet"
    except Exception:  # noqa: BLE001
        try:
            import importlib

            importlib.import_module("fastparquet")
            return "outputs/interaction_summary.parquet"
        except Exception:  # noqa: BLE001
            return "outputs/interaction_summary.csv"


def _write_interaction_summary(summary_df, outputs_dir: Path, rel: str) -> None:
    """Persist the interaction_summary DataFrame to ``outputs_dir`` as parquet/csv."""
    dest = outputs_dir / Path(rel).name
    if dest.suffix == ".parquet":
        summary_df.to_parquet(dest, index=False)
    else:
        summary_df.to_csv(dest, index=False)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--already-normalized", action="store_true")
    parser.add_argument(
        "--dataset", choices=("auto", "primary", "fallback"), default="auto"
    )
    parser.add_argument("--max-seconds", type=float, default=DEFAULT_MAX_SECONDS)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help=(
            "Minibatch InfoNCE size (#61). 0 keeps the exact full-batch loss; "
            ">0 bounds the contrastive matrix to O(batch^2) for large datasets."
        ),
    )
    parser.add_argument(
        "--interaction-summary",
        action="store_true",
        help=(
            "Score ligand-receptor enrichment per prototype pair (#151) and "
            "write outputs/interaction_summary.{parquet,csv}. OFF by default; "
            "requires the optional [data] extra (squidpy/OmniPath) -- the run "
            "errors actionably if invoked without it."
        ),
    )
    return parser


def main() -> int:
    parser = _build_parser()
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
        (
            X,
            coords,
            section_id,
            section_col,
            normalization,
            n_obs,
            n_vars,
            adata,
        ) = _load_dataset(path, args.already_normalized)
        result, edges, elapsed = _fit_with_walls(
            X,
            coords,
            section_id,
            args.max_seconds,
            device,
            num_threads,
            batch_size=args.batch_size,
            compute_interaction_summary=args.interaction_summary,
            adata=adata if args.interaction_summary else None,
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

    # Optionally persist the ligand-receptor interaction_summary (#151). Written
    # only when --interaction-summary is set AND the fit produced a table; the
    # output path is recorded in run_metadata.outputs for provenance.
    interaction_rel = None
    if args.interaction_summary:
        summary_df = getattr(run["result"], "interaction_summary", None)
        if summary_df is not None:
            interaction_rel = _interaction_output_rel(summary_df)
            notes.append(f"interaction_summary rows={len(summary_df)}")
        else:
            notes.append("interaction_summary requested but result was empty")

    # Write outputs/ artifacts: niche.npz (H, prototype_id) + proto_kind.json.
    card_id = results_contract.dataset_card_id([str(used_path)])
    results_dir = _REPO_ROOT / "results"
    outputs_manifest = {
        "niche": "outputs/niche.npz",
        "proto_kind": "outputs/proto_kind.json",
    }
    if interaction_rel is not None:
        outputs_manifest["interaction_summary"] = interaction_rel
    paths = results_contract.write_results(
        project="niche-lens-st",
        dataset_card_id=card_id,
        metrics=run["metrics"],
        outputs=outputs_manifest,
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
            "batch_size": int(args.batch_size),
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

    if interaction_rel is not None:
        _write_interaction_summary(
            run["result"].interaction_summary, outputs_dir, interaction_rel
        )
        print(f"interaction_summary -> {outputs_dir / Path(interaction_rel).name}")

    print(f"dataset used: {run['label']} ({used_path})")
    print(f"n_obs={run['n_obs']} n_vars={run['n_vars']} device={device}")
    print(f"metrics.json -> {paths['metrics']}")
    print(f"run_metadata.json -> {paths['run_metadata']}")
    print(f"outputs -> {outputs_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
