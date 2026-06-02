"""Cell-cell communication (ligand-receptor) scoring for NicheLens-ST.

This module produces the ``interaction_summary`` MVP output: a per-prototype
(prototype-pair) ligand-receptor enrichment table. Scoring is delegated to
third-party tooling -- :func:`squidpy.gr.ligrec` with the OmniPath
ligand-receptor reference (a superset of CellPhoneDB). Both are legitimate
external dependencies shipped via the optional ``[data]`` extra; no reference
implementation is vendored.

Dependency gating
-----------------
``squidpy`` (and its OmniPath backend) are heavy and live behind the ``[data]``
extra, so they are **not** imported at module import time -- this module always
imports cleanly. The import is gated inside :func:`_import_squidpy`, which is
only reached when :func:`compute_interaction_summary` is actually invoked. If
the extra is missing, a single clear, actionable :class:`ImportError` is raised
pointing at ``pip install 'nichelens-st[data]'``.

Output schema
-------------
:func:`compute_interaction_summary` returns a tidy :class:`pandas.DataFrame`
with exactly the columns in :data:`INTERACTION_SUMMARY_COLUMNS`:

``ligand``, ``receptor`` (str), ``source``, ``target`` (str cluster/prototype
labels), ``score`` (float mean L-R co-expression), ``pvalue`` (float
permutation p-value). One row per (ligand, receptor, source, target).
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "INTERACTION_SUMMARY_COLUMNS",
    "compute_interaction_summary",
]

#: Column order of the tidy ``interaction_summary`` table.
INTERACTION_SUMMARY_COLUMNS = (
    "ligand",
    "receptor",
    "source",
    "target",
    "score",
    "pvalue",
)

_MISSING_SQUIDPY_MSG = (
    "Cell-cell communication scoring requires `squidpy` (with its OmniPath "
    "ligand-receptor backend), which ships in the optional [data] extra. "
    "Install it with: pip install 'nichelens-st[data]'"
)


def _import_squidpy() -> Any:
    """Import ``squidpy`` lazily, raising a clear error if the extra is absent.

    Isolated into its own function so the gating is trivially testable (the
    failure path can be exercised by monkeypatching this symbol) and so the
    actionable message is emitted from a single place.
    """
    try:
        import squidpy as sq  # noqa: PLC0415 (intentional gated import)
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
        raise ImportError(_MISSING_SQUIDPY_MSG) from exc
    return sq


def compute_interaction_summary(
    adata: Any,
    *,
    cluster_key: str,
    n_perms: int = 100,
    threshold: float = 0.01,
    seed: int = 0,
    **ligrec_kwargs: Any,
):
    """Score ligand-receptor interactions per prototype pair via squidpy.

    Parameters
    ----------
    adata
        An :class:`anndata.AnnData` with expression in ``X`` and a categorical
        cluster / prototype label column in ``obs[cluster_key]`` (e.g. the
        ``prototype_id`` from the contrastive encoder, stringified). Genes must
        be named in ``var_names`` so OmniPath can resolve ligand/receptor
        symbols.
    cluster_key
        Name of the ``obs`` column holding the prototype / cell-type labels.
    n_perms
        Number of permutations for the p-value null (forwarded to ``ligrec``).
    threshold
        Minimum fraction of expressing cells for a gene to be considered
        (forwarded to ``ligrec``).
    seed
        Seed for the permutation test (forwarded to ``ligrec`` as ``seed``).
    **ligrec_kwargs
        Extra keyword arguments forwarded verbatim to ``squidpy.gr.ligrec``.

    Returns
    -------
    pandas.DataFrame
        Tidy table with columns :data:`INTERACTION_SUMMARY_COLUMNS`. Pairs whose
        mean co-expression is undefined (NaN, i.e. screened out by ``threshold``)
        are dropped so every row is a scored interaction.

    Raises
    ------
    ImportError
        If ``squidpy`` / the ``[data]`` extra is not installed.
    """
    sq = _import_squidpy()
    import pandas as pd  # anndata pulls pandas; safe here (helper already gated)

    res = sq.gr.ligrec(
        adata,
        cluster_key=cluster_key,
        n_perms=n_perms,
        threshold=threshold,
        seed=seed,
        copy=True,
        **ligrec_kwargs,
    )
    return _tidy_ligrec_result(res, pd)


def _tidy_ligrec_result(res: Any, pd: Any):
    """Reshape a ``squidpy.gr.ligrec`` result dict into the tidy schema.

    ``ligrec`` returns ``means`` and ``pvalues`` DataFrames indexed by a
    ``(source, target)`` ligand-receptor MultiIndex on the rows and a
    ``(cluster_1, cluster_2)`` MultiIndex on the columns. We reshape both into
    long form, join on the (ligand, receptor, source, target) key, and drop
    pairs with no defined mean (filtered out by ``threshold``).
    """
    means = res["means"]
    pvalues = res["pvalues"]

    long_means = _melt_ligrec_frame(means, "score", pd)
    long_pvals = _melt_ligrec_frame(pvalues, "pvalue", pd)

    key = ["ligand", "receptor", "source", "target"]
    df = long_means.merge(long_pvals, on=key, how="left")

    # `threshold` screens pairs out -> NaN mean; keep only scored interactions.
    df = df.dropna(subset=["score"]).reset_index(drop=True)
    df["score"] = df["score"].astype(float)
    df["pvalue"] = df["pvalue"].astype(float)
    for col in ("ligand", "receptor", "source", "target"):
        df[col] = df[col].astype(str)
    return df[list(INTERACTION_SUMMARY_COLUMNS)]


def _melt_ligrec_frame(frame: Any, value_name: str, pd: Any):
    """Reshape one ligrec wide frame (L-R rows x cluster-pair cols) to long form.

    Uses ``stack`` on the column MultiIndex rather than ``reset_index().melt()``:
    on a frame with a 2-level column MultiIndex, ``reset_index()`` promotes the
    row-index labels to *tuples* (e.g. ``('ligand', '')``), which breaks
    ``melt(id_vars=["ligand", "receptor"])``. Stacking references the levels by
    name and is robust to level ordering. NaN cells (threshold-screened pairs)
    are preserved here and dropped later in ``_tidy_ligrec_result``.
    """
    flat = frame.copy()
    # Row MultiIndex: (ligand source gene, receptor target gene).
    flat.index = flat.index.set_names(["ligand", "receptor"])
    # Column MultiIndex: (source cluster, target cluster).
    flat.columns = flat.columns.set_names(["source", "target"])
    long = (
        flat.stack(["source", "target"], future_stack=True)
        .rename(value_name)
        .reset_index()
    )
    return long[["ligand", "receptor", "source", "target", value_name]]
