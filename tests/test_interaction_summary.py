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
