"""Marker-based reference annotation scaffold (issue #350).

The primary GSE282124 MERSCOPE panel is **unsupervised** -- it ships no
cell-type labels -- yet ``scripts/compute_further_niche_metrics.py`` requires
``obs['cell_class']`` (a categorical column) for its composition,
co-localization, and annotation-agreement metrics. Without a label column those
supervised metrics are empty/blocked on the primary.

This module provides a dependency-light, deterministic primitive that *produces*
a ``cell_class`` assignment from a marker-gene dictionary, so those metrics
become non-empty once real data lands. It is the **scaffold** only: wiring it
into ``run_real_niche.py`` (writing ``adata.obs['cell_class']``) is a deliberate
FOLLOW-UP and out of scope here.

Consumer contract (mirror of ``compute_further_niche_metrics.py`` ~ll. 320-323)::

    cell_class = pd.Series(adata.obs['cell_class']).astype('category')
    cell_class_codes = cell_class.cat.codes.to_numpy()   # (n_cells,) int
    cell_class_names = list(cell_class.cat.categories)    # code -> name map

:class:`AnnotationResult` exposes exactly this shape: ``cell_class_codes`` are
non-negative ints indexing into ``cell_class_names``. To populate the runner one
would do ``adata.obs['cell_class'] = [res.cell_class_names[c] for c in
res.cell_class_codes]``.

Pure ``numpy``; deterministic; no sklearn/scanpy/external deps (keeps the
minimal-CI base import-light, consistent with the rest of the package).

Issue #83 discipline: undefined inputs (empty marker dict, no usable types,
shape mismatch) raise ``ValueError`` rather than silently returning an
all-``unassigned`` field that would read as a (fake) clean result.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

_METHODS = ("mean_zscore", "mean")


@dataclass
class AnnotationResult:
    cell_class_codes: np.ndarray        # (n_cells,) int64, indexes cell_class_names
    cell_class_names: list[str]         # code -> name map (incl. unassigned iff used)
    scores: np.ndarray                  # (n_cells, n_used_types) float64
    used_types: list[str]               # scored types, marker_dict insertion order
    dropped_genes: dict[str, list[str]] # type -> marker genes absent from var_names
    excluded_types: list[str]           # types dropped for < min_markers usable genes


def marker_score_annotate(
    X: np.ndarray,
    var_names,
    marker_dict: dict[str, list[str]],
    *,
    method: str = "mean_zscore",
    min_markers: int = 1,
    unassigned_label: str = "unassigned",
    assign_threshold: float | None = None,
) -> AnnotationResult:
    """Assign each cell a ``cell_class`` by marker-panel scoring.

    Each cell type is scored per cell by aggregating that type's marker columns;
    the per-cell ``cell_class`` is the argmax type. This is intentionally simple
    and label-free -- a reference scaffold, not a trained classifier.

    Parameters
    ----------
    X : (n_cells, n_genes) array
        Expression matrix assumed **already normalized / log-transformed**
        (the scoring does not re-normalize beyond optional z-scoring). Cast to
        float64 internally.
    var_names : sequence of str
        Gene names aligned to the columns of ``X``; ``len(var_names)`` must equal
        ``X.shape[1]``. The first occurrence wins for any duplicated name.
    marker_dict : {cell_type_name: [marker_gene_names]}
        Marker panel per candidate cell type. Iteration order of this mapping
        sets the type ordering (and thus the deterministic tie-break).
    method : {"mean_zscore", "mean"}
        ``"mean_zscore"`` z-scores each gene across cells first (so a single
        high-variance gene cannot dominate), then averages a type's usable marker
        columns. ``"mean"`` averages the marker columns of ``X`` directly (relies
        on the input already being comparably scaled).
    min_markers : int
        A type with fewer than this many *usable* markers (present in
        ``var_names``) is excluded and recorded in ``excluded_types`` -- never
        silently scored on an empty/partial panel.
    unassigned_label : str
        Name used for cells whose top score falls below ``assign_threshold``.
        Added to ``cell_class_names`` only when at least one cell is unassigned.
    assign_threshold : float or None
        If not ``None``, cells whose top type score is ``< assign_threshold`` get
        ``unassigned_label``. ``None`` (default) assigns every cell to its argmax.

    Returns
    -------
    AnnotationResult

    Tie-break
    ---------
    On equal top scores the **lowest used-type index wins** -- i.e. the type
    appearing earliest in ``marker_dict`` iteration order (``np.argmax`` returns
    the first maximum).

    Raises
    ------
    ValueError
        If ``method`` is unknown; ``X`` is not 2-D; ``len(var_names) !=
        X.shape[1]``; ``marker_dict`` is empty; or no type retains
        ``>= min_markers`` usable markers (issue #83 -- undefined, not a free
        all-``unassigned`` result).
    """
    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}; got {method!r}")

    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D (n_cells, n_genes); got shape {X.shape}")
    n_cells, n_genes = X.shape

    var_names = [str(v) for v in var_names]
    if len(var_names) != n_genes:
        raise ValueError(
            f"var_names length {len(var_names)} != X.shape[1] {n_genes}"
        )
    if not marker_dict:
        raise ValueError("marker_dict is empty; nothing to score (issue #83)")
    if min_markers < 1:
        raise ValueError(f"min_markers must be >= 1; got {min_markers}")

    # First-occurrence gene-name -> column index map.
    gene_index: dict[str, int] = {}
    for col, name in enumerate(var_names):
        gene_index.setdefault(name, col)

    used_types: list[str] = []
    used_cols: list[list[int]] = []
    dropped_genes: dict[str, list[str]] = {}
    excluded_types: list[str] = []
    for type_name, markers in marker_dict.items():
        cols: list[int] = []
        dropped: list[str] = []
        for g in markers:
            col = gene_index.get(str(g))
            if col is None:
                dropped.append(str(g))
            else:
                cols.append(col)
        if dropped:
            dropped_genes[type_name] = dropped
        if len(cols) < min_markers:
            excluded_types.append(type_name)
            continue
        used_types.append(type_name)
        used_cols.append(cols)

    if not used_types:
        raise ValueError(
            "no cell type retains >= min_markers usable markers; refusing to "
            "return an all-unassigned field (issue #83)"
        )
    if excluded_types:
        warnings.warn(
            f"excluded {len(excluded_types)} type(s) with < {min_markers} usable "
            f"markers: {excluded_types}",
            stacklevel=2,
        )

    if method == "mean_zscore":
        mean = X.mean(axis=0, keepdims=True)
        std = X.std(axis=0, keepdims=True)
        # Zero-variance genes contribute 0 (no information), not NaN/inf.
        scored = np.divide(
            X - mean, std, out=np.zeros_like(X), where=std > 0.0
        )
    else:  # "mean"
        scored = X

    scores = np.empty((n_cells, len(used_types)), dtype=np.float64)
    for j, cols in enumerate(used_cols):
        scores[:, j] = scored[:, cols].mean(axis=1)

    codes = np.argmax(scores, axis=1).astype(np.int64)
    cell_class_names = list(used_types)

    if assign_threshold is not None:
        top = scores[np.arange(n_cells), codes]
        below = top < assign_threshold
        if np.any(below):
            unassigned_code = len(cell_class_names)
            cell_class_names.append(unassigned_label)
            codes = codes.copy()
            codes[below] = unassigned_code

    return AnnotationResult(
        cell_class_codes=codes,
        cell_class_names=cell_class_names,
        scores=scores,
        used_types=used_types,
        dropped_genes=dropped_genes,
        excluded_types=excluded_types,
    )
