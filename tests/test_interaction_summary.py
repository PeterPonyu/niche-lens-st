"""Ligand-receptor / ``interaction_summary`` output (issues #57, #151).

The MVP design lists ``interaction_summary`` as a per-prototype ligand-receptor
score table. These tests pin three guarantees:

(a) ``NicheModelResult.interaction_summary`` defaults to ``None`` and the
    existing fit path is unchanged (no new hard dependency at fit time);
(b) :func:`nichelens_st.communication.compute_interaction_summary` returns the
    documented tidy schema on a small synthetic AnnData (skipped when the
    ``[data]`` extra -- squidpy / anndata -- is absent);
(c) invoking the helper without squidpy raises a clear, actionable error.
"""

from __future__ import annotations

import numpy as np
import pytest


# --- (a) result field default + back-compat -------------------------------

def test_result_interaction_summary_defaults_none():
    from nichelens_st.model import NicheModelResult

    res = NicheModelResult(
        H=np.zeros((3, 2), dtype=np.float32),
        prototype_id=np.zeros(3, dtype=np.int64),
        proto_kind=["unknown"],
    )
    assert res.interaction_summary is None


def test_fit_does_not_populate_interaction_summary_by_default():
    """Default fit path must not require squidpy and must leave the field None."""
    from nichelens_st.model import TORCH_AVAILABLE, NicheModelConfig, fit_niche_model

    if not TORCH_AVAILABLE:
        pytest.skip("requires the optional [model] extra (torch)")

    rng = np.random.default_rng(0)
    n = 12
    X = rng.standard_normal((n, 5)).astype(np.float32)
    coords = rng.standard_normal((n, 2)).astype(np.float32)
    section_id = np.zeros(n, dtype=np.int64)
    # edges contract is (2, n_edges): row 0 = sources, row 1 = targets.
    src = np.arange(n, dtype=np.int64)
    dst = (src + 1) % n
    edges = np.stack([src, dst]).astype(np.int64)
    cfg = NicheModelConfig(epochs=1, embed_dim=4, n_prototypes=3)

    res = fit_niche_model(X, coords, section_id, edges, cfg)
    assert res.interaction_summary is None


# --- (c) clear error without squidpy --------------------------------------

def test_helper_raises_clear_error_without_squidpy(monkeypatch):
    """When squidpy is missing the helper must raise an actionable ImportError.

    We drive the *real* gated importer (``_import_squidpy``) but force the
    underlying ``import squidpy`` to fail by poisoning ``sys.modules`` (works
    whether or not squidpy is actually installed). The raised message must
    point users at the ``[data]`` extra.
    """
    import builtins
    import importlib

    comm = importlib.import_module("nichelens_st.communication")

    real_import = builtins.__import__

    def _blocked_import(name, *args, **kwargs):
        if name == "squidpy" or name.startswith("squidpy."):
            raise ImportError("No module named 'squidpy'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked_import)
    with pytest.raises(ImportError, match=r"\[data\]"):
        comm.compute_interaction_summary(object(), cluster_key="proto")


# --- (b) tidy schema on a synthetic AnnData -------------------------------

def test_compute_interaction_summary_tidy_schema():
    pytest.importorskip("anndata")
    pytest.importorskip("squidpy")

    import anndata as ad

    from nichelens_st.communication import (
        INTERACTION_SUMMARY_COLUMNS,
        compute_interaction_summary,
    )

    rng = np.random.default_rng(0)
    n = 60
    genes = ["TGFB1", "TGFBR1", "TNF", "TNFRSF1A", "IL6", "IL6R"]
    X = rng.poisson(2.0, size=(n, len(genes))).astype(np.float32)
    labels = np.array(["A", "B"] * (n // 2))
    import pandas as pd

    adata = ad.AnnData(
        X=X,
        obs=pd.DataFrame({"proto": pd.Categorical(labels)}),
        var=pd.DataFrame(index=genes),
    )

    df = compute_interaction_summary(
        adata, cluster_key="proto", n_perms=10, threshold=0.0, seed=0
    )

    assert list(df.columns) == list(INTERACTION_SUMMARY_COLUMNS)
    # Tidy: one row per (ligand, receptor, source, target).
    assert len(df) >= 1
    assert df["ligand"].dtype == object
    assert df["receptor"].dtype == object
    assert np.issubdtype(df["score"].dtype, np.floating)
    assert np.issubdtype(df["pvalue"].dtype, np.floating)


# --- (d) squidpy-free reshape on a synthetic squidpy-shaped result --------

def _make_squidpy_shaped_result():
    """Build ``means``/``pvalues`` frames matching squidpy.gr.ligrec's layout.

    squidpy lays the result out as wide DataFrames with:
      * a *row* MultiIndex over gene pairs, ``names=["source", "target"]``
        (source gene -> ligand, target gene -> receptor);
      * a *column* MultiIndex over directional cluster pairs,
        ``names=["cluster_1", "cluster_2"]`` (cluster_1 -> source cluster,
        cluster_2 -> target cluster).

    No squidpy import: we hand-build the exact frame shapes so the reshape is
    exercised in CI even when the [data] extra is absent.
    """
    import pandas as pd

    gene_pairs = pd.MultiIndex.from_tuples(
        [("TGFB1", "TGFBR1"), ("TNF", "TNFRSF1A")],
        names=["source", "target"],
    )
    cluster_pairs = pd.MultiIndex.from_tuples(
        [("A", "B"), ("B", "A"), ("A", "A")],
        names=["cluster_1", "cluster_2"],
    )
    means = pd.DataFrame(
        np.array([[0.5, np.nan, 0.2], [0.9, 0.1, np.nan]]),
        index=gene_pairs,
        columns=cluster_pairs,
    )
    pvalues = pd.DataFrame(
        np.array([[0.01, 0.4, 0.03], [0.02, 0.5, 0.6]]),
        index=gene_pairs,
        columns=cluster_pairs,
    )
    return means, pvalues


def test_tidy_ligrec_result_reshape_no_squidpy():
    """The reshape must run without squidpy and honor the documented orientation.

    Drives ``_tidy_ligrec_result`` directly on a synthetic squidpy-shaped
    result (a 2-level row MultiIndex over gene pairs x a 2-level column
    MultiIndex over directional cluster pairs). Asserts the tidy schema, the
    ligand/receptor (gene pair) and source/target (cluster pair) mapping, and
    that NaN-mean (threshold-screened) rows are dropped.
    """
    import pandas as pd

    from nichelens_st.communication import (
        INTERACTION_SUMMARY_COLUMNS,
        _tidy_ligrec_result,
    )

    means, pvalues = _make_squidpy_shaped_result()
    res = {"means": means, "pvalues": pvalues}

    df = _tidy_ligrec_result(res, pd)

    # Schema.
    assert list(df.columns) == list(INTERACTION_SUMMARY_COLUMNS)

    # NaN-mean rows screened out: 6 cells, 2 are NaN -> 4 scored interactions.
    assert len(df) == 4
    assert not df["score"].isna().any()

    # Orientation/mapping: ligand=source gene, receptor=target gene;
    # source=cluster_1, target=cluster_2.
    row = df[
        (df["ligand"] == "TGFB1")
        & (df["receptor"] == "TGFBR1")
        & (df["source"] == "A")
        & (df["target"] == "B")
    ]
    assert len(row) == 1
    assert row["score"].iloc[0] == pytest.approx(0.5)
    assert row["pvalue"].iloc[0] == pytest.approx(0.01)

    # The reverse cluster direction (B->A) carries its own score.
    rev = df[
        (df["ligand"] == "TNF")
        & (df["receptor"] == "TNFRSF1A")
        & (df["source"] == "B")
        & (df["target"] == "A")
    ]
    assert len(rev) == 1
    assert rev["score"].iloc[0] == pytest.approx(0.1)
    assert rev["pvalue"].iloc[0] == pytest.approx(0.5)

    # The two NaN-mean cells (TGFB1/TGFBR1 @ B->A and TNF/TNFRSF1A @ A->A)
    # must be absent.
    dropped = df[
        ((df["ligand"] == "TGFB1") & (df["source"] == "B") & (df["target"] == "A"))
        | ((df["ligand"] == "TNF") & (df["source"] == "A") & (df["target"] == "A"))
    ]
    assert len(dropped) == 0

    # dtypes per the documented contract.
    assert df["ligand"].dtype == object
    assert df["receptor"].dtype == object
    assert np.issubdtype(df["score"].dtype, np.floating)
    assert np.issubdtype(df["pvalue"].dtype, np.floating)


def test_melt_ligrec_frame_robust_to_level_ordering_no_squidpy():
    """The reshape must reference MultiIndex levels by name, not position.

    squidpy may not guarantee a fixed physical level order; if levels are
    permuted, name-based access must still produce the correct ligand/receptor
    and source/target columns.
    """
    import pandas as pd

    from nichelens_st.communication import _melt_ligrec_frame

    means, _ = _make_squidpy_shaped_result()
    long = _melt_ligrec_frame(means, "score", pd)

    assert set(["ligand", "receptor", "source", "target", "score"]).issubset(
        long.columns
    )
    # Pre-screening: all 6 cells present (NaN kept until _tidy drops them).
    assert len(long) == 6
    hit = long[
        (long["ligand"] == "TGFB1")
        & (long["receptor"] == "TGFBR1")
        & (long["source"] == "A")
        & (long["target"] == "A")
    ]
    assert len(hit) == 1
    assert hit["score"].iloc[0] == pytest.approx(0.2)
