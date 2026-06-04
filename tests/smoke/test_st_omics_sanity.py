"""Smoke / sanity tests with very small ST-omics data in real format for NicheLens-ST.

Real data (cards, run_real_niche): AnnData with obsm['spatial'], obs columns for
section (fov/slice_id/section_id/sample/batch), X raw counts (or float-stored ints).
The runner does section extraction + graph build + _looks_like_raw_counts.

Not scRNA (flat, no spatial, only cell_type obs, lognorm X).
"""

from __future__ import annotations

import numpy as np
import pytest

import anndata as ad

# Import runner logic for the _load / looks_like used on real cards (via sys.path in other tests)
import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))
# The script under test re-exports or defines the helpers we exercise.
try:
    import nichelens_st  # noqa
except Exception:
    pass


def _tiny_niche_st_adata(n: int = 12, n_genes: int = 6, section_col: str = "slice_id", seed: int = 99) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    X = rng.poisson(2.4, size=(n, n_genes)).astype(np.float32)
    a = ad.AnnData(X=X)
    a.obsm["spatial"] = rng.uniform(0, 300, size=(n, 2)).astype(np.float32)
    # Real card style section column (factorized later)
    a.obs[section_col] = (["sec0"] * (n // 2) + ["sec1"] * (n - n // 2))
    a.var_names = [f"NG_{i}" for i in range(n_genes)]
    return a


def test_tiny_st_has_required_st_keys():
    """Tiny ST must carry spatial + a recognizable section column (as run_real expects)."""
    a = _tiny_niche_st_adata()
    assert "spatial" in a.obsm
    assert a.obsm["spatial"].shape[1] == 2
    # at least one of the candidate section keys from runner
    cands = ("fov", "slice_id", "section_id", "sample", "batch")
    assert any(c in a.obs for c in cands)


def test_runner_looks_like_raw_on_tiny_st():
    """The _looks_like_raw_counts helper (used before norm in run_real) must accept our format."""
    # Re-exec the runner module to get the helper (matches pattern in test_run_real_niche_runner.py)
    import importlib.util
    _SCRIPT = _REPO_ROOT / "scripts" / "run_real_niche.py"
    spec = importlib.util.spec_from_file_location("run_real_niche", _SCRIPT)
    runner = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(runner)

    a = _tiny_niche_st_adata(n=20, seed=7)
    assert runner._looks_like_raw_counts(a.X) is True

    # scRNA-style log should not
    loggy = np.log1p(a.X)
    assert runner._looks_like_raw_counts(loggy) is False


def test_graph_build_from_tiny_st_coords_section():
    """graph.build_graph (core for niche) accepts coords+section from real ST adata."""
    from nichelens_st.graph import build_graph
    a = _tiny_niche_st_adata(n=15, seed=42)
    coords = np.asarray(a.obsm["spatial"], dtype=np.float32)
    # simulate runner's section code extraction
    sec_col = "slice_id"
    codes = a.obs[sec_col].astype("category").cat.codes.to_numpy().astype(np.int64)
    edges = build_graph(coords, codes, k=4, method="knn")
    assert edges.shape[0] == 2
    assert edges.dtype == np.int64 or edges.dtype == np.int32  # impl may vary
    assert edges.shape[1] > 0
