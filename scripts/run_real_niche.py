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
import inspect
import os
import sys
import time
import traceback
import warnings
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


# Conservative MERFISH count-QC floor (operator-overridable via the CLI). The
# primary processed GSE282124 AnnData had NO QC filtering -- 76.18% of its cells
# carry 0 transcript counts, so ``normalize_total`` divides by zero on those rows
# -> NaN, which poisons the contrastive fit. We drop low-quality cells from the
# RAW counts BEFORE normalization so the divide-by-zero never happens.
DEFAULT_MIN_COUNTS = 10
DEFAULT_MIN_GENES = 5
# Refuse to fit on near-nothing; warn loudly when QC removes most of the input.
QC_MIN_CELLS = 100
QC_LOUD_DROP_FRACTION = 0.5


def qc_cell_mask(counts_matrix, *, min_counts: int, min_genes: int) -> np.ndarray:
    """Boolean keep-mask for count-based per-cell QC (pure; scanpy-free).

    Computes per-cell total counts and per-cell n_genes-detected from the RAW
    counts ``counts_matrix`` (scipy.sparse OR dense; int-like or float-stored
    counts) and keeps a cell iff BOTH:

    * ``total_counts >= min_counts`` and
    * ``n_genes_detected >= min_genes``

    ``>=`` semantics: a cell exactly AT a threshold is KEPT; only strictly-below
    is dropped. An all-zero matrix returns an all-``False`` mask (the caller then
    aborts rather than fitting on nothing). Returns a 1-D ``bool`` ndarray of
    length ``counts_matrix.shape[0]``.
    """
    import scipy.sparse as sp

    if sp.issparse(counts_matrix):
        csr = counts_matrix.tocsr()
        total_counts = np.asarray(csr.sum(axis=1)).reshape(-1)
        # n_genes-detected = nonzeros per row (works on the sparse structure).
        n_genes = np.asarray((csr != 0).sum(axis=1)).reshape(-1)
    else:
        arr = np.asarray(counts_matrix)
        total_counts = np.asarray(arr.sum(axis=1)).reshape(-1)
        n_genes = np.asarray((arr != 0).sum(axis=1)).reshape(-1)

    keep = (total_counts >= float(min_counts)) & (n_genes >= int(min_genes))
    return np.ascontiguousarray(keep, dtype=bool).reshape(-1)


def qc_note_string(
    *, min_counts: int, min_genes: int, n_before: int, n_after: int
) -> str:
    """Render the run_metadata NOTES fragment recording the QC outcome.

    The contract passes the free-text ``notes`` string through verbatim, so this
    is how the QC provenance (thresholds + before/after/dropped counts) reaches
    the manifest. Format::

        qc: min_counts=<x>, min_genes=<y>, n_before=<N0>, n_after=<N1>,
        n_dropped=<d> (<pct>% dropped)
    """
    n_before = int(n_before)
    n_after = int(n_after)
    n_dropped = n_before - n_after
    pct = (100.0 * n_dropped / n_before) if n_before else 0.0
    return (
        f"qc: min_counts={int(min_counts)}, min_genes={int(min_genes)}, "
        f"n_before={n_before}, n_after={n_after}, n_dropped={n_dropped} "
        f"({pct:.1f}% dropped)"
    )


# Section/sample column autodetect order + the tiling-artifact heuristic (#343).
# A "section" is a biological tissue section over which a kNN niche graph is
# coherent. MERSCOPE stores obs['fov'] (microscope field-of-view tiles, ~tens of
# cells each) FIRST in this order, so naive autodetect would shatter the graph
# into per-tile patches. We still pick the column (record-and-warn, never crash)
# but flag it loudly when its level count / cells-per-level look tile-sized.
SECTION_CANDIDATES = ("fov", "slice_id", "section_id", "sample", "batch")
TILING_MAX_SECTIONS = 50
TILING_MIN_CELLS_PER_SECTION = 200


def _column_codes(obs, col) -> np.ndarray:
    """Integer group codes for obs column ``col``.

    Works for a pandas DataFrame/Series (uses categorical codes) and for a plain
    dict-of-arrays obs (factorizes via ``np.unique``), so the resolver is testable
    without scanpy/anndata. Returns an int64 ndarray of group codes.
    """
    column = obs[col]
    # pandas Series: categorical codes preserve label identity and map NaN -> -1.
    if hasattr(column, "astype") and hasattr(column, "to_numpy"):
        try:
            codes = column.astype("category").cat.codes.to_numpy()
            return np.ascontiguousarray(codes, dtype=np.int64).reshape(-1)
        except Exception:  # noqa: BLE001
            pass
    values = np.asarray(column)
    _uniques, codes = np.unique(values, return_inverse=True)
    return np.ascontiguousarray(codes, dtype=np.int64).reshape(-1)


def resolve_section_id(obs_columns, obs, n_obs, *, section_col_arg="auto"):
    """Resolve per-cell ``section_id`` codes for the per-section kNN graph.

    Pure (scanpy-free) and testable. ``obs`` may be a pandas DataFrame or a plain
    dict-of-arrays; ``obs_columns`` is its column-name collection.

    ``section_col_arg`` semantics:

    * ``"auto"`` (default): pick the first present of :data:`SECTION_CANDIDATES`.
      If the chosen column has an implausibly high section count
      (``n_sections > TILING_MAX_SECTIONS`` OR mean cells/section
      ``< TILING_MIN_CELLS_PER_SECTION``) it is almost certainly a microscope
      tiling artifact (e.g. MERSCOPE ``fov``), not biological sections -- emit a
      prominent ``warnings.warn`` and record the same text as ``note`` (so the
      run manifest captures it), but STILL return the codes (record-and-warn,
      never silently fragment, never crash). If no candidate column is present,
      fall back to a single section (zeros).
    * ``"none"`` / ``"single"`` (case-insensitive): force a single section over
      the global coords (``section_id`` all zeros, ``section_col_used=None``).
    * any other value: treat as an explicit obs column name; raise ``KeyError``
      naming the column if it is absent. No heuristic warning -- an explicit
      column is the operator's deliberate choice.

    Returns ``(section_id int64 ndarray[n_obs], section_col_used, note)`` where
    ``note`` is ``None`` when there is nothing to record.
    """
    n_obs = int(n_obs)
    arg = (section_col_arg or "auto").strip()
    arg_lower = arg.lower()

    if arg_lower in ("none", "single"):
        note = (
            "section_col=none -> forced single section "
            "(zeros over global coords)"
        )
        return np.zeros(n_obs, dtype=np.int64), None, note

    columns = list(obs_columns)

    if arg_lower == "auto":
        section_col = next((c for c in SECTION_CANDIDATES if c in columns), None)
        if section_col is None:
            return np.zeros(n_obs, dtype=np.int64), None, None
        heuristic = True
    else:
        if arg not in columns:
            raise KeyError(
                f"--section-col '{arg}' not found in obs columns {sorted(columns)}"
            )
        section_col = arg
        heuristic = False  # explicit column: operator's choice, no tiling warn

    section_id = _column_codes(obs, section_col)
    n_sections = int(np.unique(section_id).size)
    mean_cells = float(n_obs / n_sections) if n_sections else 0.0

    note = None
    if heuristic and (
        n_sections > TILING_MAX_SECTIONS
        or mean_cells < TILING_MIN_CELLS_PER_SECTION
    ):
        note = (
            f"column '{section_col}' has {n_sections} levels "
            f"(~{mean_cells:.0f} cells/level); likely a microscope tiling "
            "artifact (e.g. MERSCOPE fov), not biological sections -- the "
            "per-section kNN graph will be fragmented. Pass --section-col none "
            "to treat as a single section."
        )
        warnings.warn(note, UserWarning, stacklevel=2)

    return section_id, section_col, note


def _peak_rss_bytes():
    """Peak resident set size of this process in bytes, or ``None`` (#343).

    Wraps ``resource.getrusage(RUSAGE_SELF).ru_maxrss``. The unit of ``ru_maxrss``
    is platform-dependent -- **bytes on macOS, KiB on Linux/BSD** -- so we
    normalize to bytes. Never raises: returns ``None`` if ``resource`` is
    unavailable (e.g. Windows) or the call fails. This feeds the exact
    ``run_metadata["peak_rss_bytes"]`` key the N-F2 scalability figure consumes.
    """
    try:
        import resource

        ru_maxrss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if sys.platform == "darwin":
            return ru_maxrss  # already bytes on macOS
        return ru_maxrss * 1024  # KiB -> bytes on Linux/BSD
    except Exception:  # noqa: BLE001
        return None


def _load_dataset(
    path: Path,
    already_normalized: bool,
    section_col_arg="auto",
    *,
    min_counts: int = DEFAULT_MIN_COUNTS,
    min_genes: int = DEFAULT_MIN_GENES,
):
    """Load an AnnData, QC-filter, cast X to float32, conditionally log1p.

    Returns (X float32 dense, coords float32, section_id int64, section_col_used,
    normalization dict, n_obs, n_vars, adata). The section-resolution note (if
    any -- e.g. the tiling-artifact warning) is stashed on
    ``adata.uns['_section_note']`` so callers can record it without widening this
    return tuple; the QC summary is likewise stashed on ``adata.uns['_qc_note']``.

    Count-based QC (computed from the RAW counts, BEFORE normalization) drops
    cells with ``total_counts < min_counts`` OR ``n_genes_detected < min_genes``.
    This removes the zero-count cells whose ``normalize_total`` divide-by-zero
    would otherwise produce NaN rows that poison the fit. The same keep-mask is
    applied to the AnnData (and therefore X, coords, and section_id, which are
    derived AFTER filtering), and ``n_obs`` is re-derived from the filtered data.
    """
    import scanpy as sc

    adata = sc.read_h5ad(str(path))
    n_before = int(adata.shape[0])

    # Count-based QC on the RAW counts, BEFORE normalization (so the
    # divide-by-zero on empty cells never happens). The mask is applied to the
    # AnnData itself, so X / coords / section_id below are all derived from the
    # filtered cells and stay aligned.
    keep = qc_cell_mask(adata.X, min_counts=min_counts, min_genes=min_genes)
    n_after = int(keep.sum())
    qc_note = qc_note_string(
        min_counts=min_counts,
        min_genes=min_genes,
        n_before=n_before,
        n_after=n_after,
    )
    if n_before and (n_before - n_after) > QC_LOUD_DROP_FRACTION * n_before:
        warnings.warn(
            f"QC dropped >{int(QC_LOUD_DROP_FRACTION * 100)}% of cells "
            f"({qc_note}); the input is mostly empty cells -- inspect the "
            "upstream processing.",
            UserWarning,
            stacklevel=2,
        )
    # Abort only when QC actually removed cells and the survivors are too few to
    # fit (don't penalize a legitimately small input that passes QC untouched).
    if n_after < n_before and n_after < QC_MIN_CELLS:
        raise ValueError(
            f"QC left {n_after} cells (< {QC_MIN_CELLS}); refusing to fit on "
            f"near-nothing. {qc_note}. Loosen --min-counts / --min-genes or "
            "check the upstream processing for this dataset."
        )
    adata = adata[keep].copy()
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

    # section_id via the operator-overridable resolver + tiling-artifact guard.
    section_id, section_col, section_note = resolve_section_id(
        list(adata.obs.columns),
        adata.obs,
        n_obs,
        section_col_arg=section_col_arg,
    )
    # Carry the notes out without widening the return tuple (#343).
    adata.uns["_section_note"] = section_note
    adata.uns["_qc_note"] = qc_note

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


def _conserved_fraction(proto_kind, n_sections: int):
    """Fraction of *scorable* prototypes tagged conserved, or ``None`` (issue #150).

    Returns ``None`` when the value is undefined: a single-section run (the
    conserved/sample_specific distinction is degenerate) or when no prototype
    carries a conserved/sample_specific tag (e.g. all ``"unknown"``). "unknown"
    tags are excluded from the denominator so they never deflate the fraction.
    """
    if int(n_sections) < 2:
        return None
    scorable = [k for k in proto_kind if k in ("conserved", "sample_specific")]
    if not scorable:
        return None
    conserved = sum(1 for k in scorable if k == "conserved")
    return float(conserved / len(scorable))


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

    # Conserved fraction (undefined for single-section runs; issue #150).
    n_sections = int(np.unique(section_id).size)
    proto_kind = list(result.proto_kind)
    metrics["conserved_fraction"] = _conserved_fraction(proto_kind, n_sections)
    if n_sections < 2:
        notes_extra.append("single_section=True (conserved_fraction undefined)")

    return metrics, notes_extra


# Auto-engage minibatch InfoNCE above this cell count when the user leaves
# --batch-size at 0 (#302). Below the threshold the exact full-batch loss is
# kept so existing small-scale numerics stay bitwise-identical; above it a
# full-batch (2n, 2n) similarity matrix would OOM (~232 GiB at n=124938), so we
# bound it to O(batch^2) instead of silently downgrading to the 5k slice.
AUTO_MINIBATCH_THRESHOLD = 50_000
AUTO_MINIBATCH_SIZE = 4096


def _effective_batch_size(requested: int, n_cells: int) -> int:
    """Resolve the InfoNCE minibatch size actually used.

    A positive ``requested`` is honored verbatim. ``requested <= 0`` (the
    default) keeps the exact full-batch loss for small inputs but auto-engages a
    bounded minibatch once ``n_cells`` exceeds :data:`AUTO_MINIBATCH_THRESHOLD`,
    so atlas-scale datasets fit (O(batch^2)) instead of OOMing then silently
    downgrading to the fallback slice (#302).
    """
    requested = int(requested)
    if requested > 0:
        return requested
    if n_cells > AUTO_MINIBATCH_THRESHOLD:
        return AUTO_MINIBATCH_SIZE
    return 0


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

    Returns (result, edges, elapsed_s, effective_batch_size). Raises on OOM /
    over-budget so the caller can fall back to the smaller dataset.
    ``batch_size`` (#61) is threaded into the model config to bound the InfoNCE
    matrix to O(batch^2); when left at 0 it auto-engages for large ``n`` (#302).

    When ``compute_interaction_summary`` is True (#151), the fitted
    ``prototype_id`` is used to score ligand-receptor enrichment via the gated
    ``[data]`` extra (squidpy/OmniPath); ``adata`` (an AnnData with named genes)
    must be supplied. The default (False) leaves the fit path squidpy-free.
    """
    from nichelens_st import model as _model
    from nichelens_st.graph import build_graph

    edges = build_graph(coords, section_id, k=6, method="knn")

    eff_batch = _effective_batch_size(batch_size, int(X.shape[0]))

    cfg = _model.NicheModelConfig(
        embed_dim=EMBED_DIM,
        n_prototypes=N_PROTOTYPES,
        seed=0,
        num_threads=num_threads,
        device=device,
        deterministic=False,
        batch_size=eff_batch,
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
    except MemoryError as exc:  # host (CPU) OOM: same actionable hint (#302)
        raise MemoryError(
            f"Host out of memory during niche fit on {int(X.shape[0])} cells; "
            "rerun with --batch-size 4096 (or smaller) to bound the InfoNCE "
            "matrix to O(batch^2) rather than the full-batch (2n, 2n) "
            f"similarity. Original error: {exc}"
        ) from exc
    elapsed = time.time() - t0
    if elapsed > max_seconds:
        raise TimeoutError(
            f"fit exceeded budget: {elapsed:.1f}s > {max_seconds:.1f}s"
        )
    return result, edges, elapsed, eff_batch


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
    parser.add_argument(
        "--section-col",
        default="auto",
        help=(
            "Section/sample column for the per-section kNN graph (#343). 'auto' "
            "(default) autodetects from (fov, slice_id, section_id, sample, "
            "batch) and WARNS if the chosen column looks like a microscope "
            "tiling artifact (e.g. MERSCOPE fov: thousands of ~tile-sized "
            "levels that would fragment the graph). 'none' (or 'single') forces "
            "a single section over global coords. Any other value is used as an "
            "explicit obs column name (errors if absent)."
        ),
    )
    parser.add_argument(
        "--min-counts",
        type=int,
        default=DEFAULT_MIN_COUNTS,
        help=(
            "Per-cell total-count QC floor: cells with fewer total transcript "
            f"counts are dropped BEFORE normalization (default {DEFAULT_MIN_COUNTS}, "
            "a conservative MERFISH floor). The primary GSE282124 input had 76% "
            "zero-count cells whose normalize_total divide-by-zero produced NaN "
            "rows that poisoned the fit; this removes them."
        ),
    )
    parser.add_argument(
        "--min-genes",
        type=int,
        default=DEFAULT_MIN_GENES,
        help=(
            "Per-cell n_genes-detected QC floor: cells expressing fewer distinct "
            f"genes are dropped BEFORE normalization (default {DEFAULT_MIN_GENES})."
        ),
    )
    parser.add_argument("--max-seconds", type=float, default=DEFAULT_MAX_SECONDS)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help=(
            "Minibatch InfoNCE size (#61). 0 (default) keeps the exact "
            "full-batch loss for small inputs but auto-engages a 4096 minibatch "
            f"above {AUTO_MINIBATCH_THRESHOLD} cells (#302); >0 forces that size "
            "and bounds the contrastive matrix to O(batch^2)."
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
        # Thread the QC thresholds only when the (possibly monkeypatched)
        # _load_dataset accepts them, so legacy test stubs with the older
        # 3-arg signature keep working without change.
        load_kwargs = {}
        try:
            _params = inspect.signature(_load_dataset).parameters
        except (TypeError, ValueError):
            _params = {}
        if "min_counts" in _params:
            load_kwargs["min_counts"] = args.min_counts
        if "min_genes" in _params:
            load_kwargs["min_genes"] = args.min_genes
        (
            X,
            coords,
            section_id,
            section_col,
            normalization,
            n_obs,
            n_vars,
            adata,
        ) = _load_dataset(
            path,
            args.already_normalized,
            args.section_col,
            **load_kwargs,
        )
        section_note = adata.uns.get("_section_note")
        qc_note = adata.uns.get("_qc_note")
        result, edges, elapsed, eff_batch = _fit_with_walls(
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
            section_note=section_note,
            qc_note=qc_note,
            normalization=normalization,
            n_obs=n_obs,
            n_vars=n_vars,
            result=result,
            elapsed=elapsed,
            effective_batch=eff_batch,
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
            # OOM-as-OOM honesty (#344): when the primary fit ran out of memory,
            # record it AS an OOM (the exception text already carries the real
            # attempted-allocation/cell-count + the minibatch hint -- we do NOT
            # fabricate any number) rather than logging a generic "wall".
            is_oom = isinstance(exc, MemoryError) or (
                "out of memory" in str(exc).lower()
            )
            if is_oom:
                notes.append(
                    f"oom=True; {type(exc).__name__}: {exc}; "
                    "minibatch fallback engaged -> switched to fallback dataset"
                )
            else:
                notes.append(
                    f"fell back to 5488-cell slice from primary 124k wall: "
                    f"{type(exc).__name__}: {exc}"
                )
            traceback.print_exc()
            used_path = FALLBACK_PATH
            run = run_on(FALLBACK_PATH, "fallback")

    notes.extend(run["notes_extra"])
    # A REAL fresh fallback = the auto path downshifted to the 5488-cell single
    # MERFISH section (an explicit --dataset fallback is an operator choice, not
    # an atlas-scale claim being silently downsized, so it is not flagged here).
    is_fallback = run["label"] == "fallback" and args.dataset != "fallback"
    if is_fallback:
        notes.append("dataset=fallback (5488 cells)")
    else:
        notes.append(f"dataset={run['label']} ({run['n_obs']} cells)")
    notes.append(f"section_id source: {run['section_col'] or 'single-section zeros'}")
    if run.get("qc_note"):
        notes.append(run["qc_note"])
    if run.get("section_note"):
        notes.append(run["section_note"])
    notes.append(f"fit_runtime_s={run['elapsed']:.2f}")

    # Peak resident memory of the run (#343) -- the field the N-F2 scalability
    # figure consumes. None (with a note) when the platform can't report it.
    peak_rss_bytes = _peak_rss_bytes()
    if peak_rss_bytes is None:
        notes.append("peak_rss_bytes unavailable (resource.getrusage failed)")
    eff_batch = int(run["effective_batch"])
    if eff_batch > 0 and args.batch_size <= 0:
        notes.append(
            f"minibatch InfoNCE auto-engaged: batch_size={eff_batch} "
            f"(n_obs={run['n_obs']} > {AUTO_MINIBATCH_THRESHOLD}; #302)"
        )
    elif eff_batch > 0:
        notes.append(f"minibatch InfoNCE: batch_size={eff_batch}")
    else:
        notes.append(f"full-batch InfoNCE (batch_size=0, n_obs={run['n_obs']})")

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
    run_metadata = {
        "dataset_paths": [str(used_path)],
        "n_obs": run["n_obs"],
        "n_vars": run["n_vars"],
        "seed": seed,
        "runtime_s": run["elapsed"],
        "started_utc": started_utc,
        "device": device,
        "deterministic": False,
        "num_threads": num_threads,
        "batch_size": int(run["effective_batch"]),
        "batch_size_requested": int(args.batch_size),
        "peak_rss_bytes": peak_rss_bytes,
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
    }
    # Structured anti-overclaim signal: the emitters (emit_figures.py /
    # emit_results_tables.py) detect the single-section fallback EXCLUSIVELY via
    # this key and only then set paper_claim_ready=False. Free-text notes alone
    # would leave that safeguard silently inert on a real fresh fallback run.
    if is_fallback:
        run_metadata["_fallback_note"] = (
            "DOWNSIZED SINGLE-SECTION FALLBACK, NOT THE ATLAS-SCALE RUN "
            "(5488-cell single MERFISH section; the conserved/sample-specific "
            "distinction is degenerate and the scale is not representative of "
            "the target dataset). Do NOT cite as atlas-scale results."
        )
    paths = results_contract.write_results(
        project="niche-lens-st",
        dataset_card_id=card_id,
        metrics=run["metrics"],
        outputs=outputs_manifest,
        run_metadata=run_metadata,
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
