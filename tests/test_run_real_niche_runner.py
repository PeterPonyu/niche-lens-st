"""Runner-level tests for ``scripts/run_real_niche.py`` (#154, #61).

These pin two runner behaviors without fitting the model:

* #154 -- count-like FLOAT matrices (e.g. MERSCOPE 315-plex counts stored as
  float) are normalized: ``sc.pp.normalize_total`` then ``sc.pp.log1p`` is
  applied and the run manifest records it. Already-normalized / log-scaled
  input is left untouched.
* #61 -- the ``--batch-size`` CLI flag is parsed and threaded into
  ``NicheModelConfig(batch_size=...)``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

# Import the script module by path (it lives under scripts/, not on sys.path).
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO_ROOT / "scripts" / "run_real_niche.py"
sys.path.insert(0, str(_REPO_ROOT / "src"))

_spec = importlib.util.spec_from_file_location("run_real_niche", _SCRIPT)
runner = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(runner)

sc = pytest.importorskip("scanpy")
ad = pytest.importorskip("anndata")


def _make_adata(X, with_spatial=True):
    obs_n = X.shape[0]
    a = ad.AnnData(X=np.asarray(X, dtype=np.float32))
    if with_spatial:
        a.obsm["spatial"] = np.zeros((obs_n, 2), dtype=np.float32)
    return a


# --------------------------------------------------------------------------
# #154 -- float-stored count detection / normalization
# --------------------------------------------------------------------------


def test_looks_like_raw_counts_float_counts_detected():
    """Float dtype but integer-valued, non-negative count magnitudes => counts."""
    rng = np.random.default_rng(0)
    # MERSCOPE-style: float32 storage of whole-number counts, max well above 1.
    X = rng.integers(0, 200, size=(100, 30)).astype(np.float32)
    assert runner._looks_like_raw_counts(X) is True


def test_looks_like_raw_counts_float_noninteger_counts_detected():
    """Count-like floats need not be exactly integer-valued (e.g. decontaminated
    / volume-normalized MERSCOPE counts) but are still raw, unlogged counts."""
    rng = np.random.default_rng(1)
    X = (rng.integers(0, 150, size=(100, 30)) + rng.random((100, 30))).astype(
        np.float32
    )
    assert runner._looks_like_raw_counts(X) is True


def test_looks_like_raw_counts_lognormalized_not_detected():
    """Already log1p-normalized data (small, fractional, max ~ a few) is NOT
    flagged as raw counts."""
    rng = np.random.default_rng(2)
    counts = rng.integers(0, 200, size=(100, 30)).astype(np.float32)
    logged = np.log1p(counts)  # typical max ~5-6
    assert runner._looks_like_raw_counts(logged) is False


def test_looks_like_raw_counts_rejects_negative():
    """Scaled/centered data with negatives is not counts."""
    rng = np.random.default_rng(3)
    X = rng.standard_normal((100, 30)).astype(np.float32)
    assert runner._looks_like_raw_counts(X) is False


def test_load_dataset_normalizes_float_counts(tmp_path):
    """A float-stored count matrix is normalize_total'd then log1p'd, the
    manifest records both steps, and pre-log1p row sums are equalized."""
    rng = np.random.default_rng(4)
    counts = rng.integers(0, 300, size=(64, 25)).astype(np.float32)
    # Ensure per-cell library sizes differ so normalize_total has an effect.
    counts[0] *= 3.0
    a = _make_adata(counts)
    path = tmp_path / "float_counts.h5ad"
    a.write_h5ad(path)

    X, coords, section_id, section_col, normalization, n_obs, n_vars = (
        runner._load_dataset(path, already_normalized=False)
    )

    assert normalization["applied"] is True
    method = normalization["method"]
    assert "normalize_total" in method
    assert "log1p" in method

    # log1p applied => no value should exceed log1p(max raw library-normed value),
    # and crucially the matrix is no longer the raw integer counts.
    assert float(X.max()) < float(counts.max())
    # Undo log1p and confirm row sums are equalized (normalize_total ran first).
    row_sums = np.expm1(X.astype(np.float64)).sum(axis=1)
    assert np.allclose(row_sums, row_sums[0], rtol=1e-3, atol=1e-2)


def test_load_dataset_respects_already_normalized(tmp_path):
    """--already-normalized override skips normalization even on count-like X."""
    rng = np.random.default_rng(5)
    counts = rng.integers(0, 300, size=(40, 20)).astype(np.float32)
    a = _make_adata(counts)
    path = tmp_path / "counts.h5ad"
    a.write_h5ad(path)

    X, *_rest, normalization, _n_obs, _n_vars = runner._load_dataset(
        path, already_normalized=True
    )
    assert normalization["applied"] is False
    # X is the untouched counts (cast to float32).
    np.testing.assert_allclose(X, counts, rtol=0, atol=0)


# --------------------------------------------------------------------------
# #61 -- --batch-size CLI flag reaches NicheModelConfig
# --------------------------------------------------------------------------


def test_batch_size_cli_parsed():
    """The runner exposes a --batch-size flag that parses to an int."""
    parser = runner._build_parser()
    args = parser.parse_args(["--batch-size", "4096"])
    assert args.batch_size == 4096


def test_batch_size_default_zero():
    parser = runner._build_parser()
    args = parser.parse_args([])
    assert args.batch_size == 0


def test_batch_size_reaches_model_config(monkeypatch):
    """--batch-size threads into NicheModelConfig(batch_size=...)."""
    from nichelens_st import model as model_mod

    captured = {}
    real_cfg = model_mod.NicheModelConfig

    def spy_cfg(*a, **kw):
        captured["batch_size"] = kw.get("batch_size")
        return real_cfg(*a, **kw)

    # Patch the symbol the runner resolves at call time.
    monkeypatch.setattr(model_mod, "NicheModelConfig", spy_cfg)

    # Stop before the expensive fit: make fit_niche_model raise so we only
    # observe the config construction.
    def boom(*a, **kw):
        raise RuntimeError("stop after config built")

    monkeypatch.setattr(model_mod, "fit_niche_model", boom)

    # build_graph is cheap on tiny input; feed a 4-cell toy.
    coords = np.zeros((4, 2), dtype=np.float32)
    X = np.ones((4, 3), dtype=np.float32)
    section_id = np.zeros(4, dtype=np.int64)

    with pytest.raises(RuntimeError, match="stop after config built"):
        runner._fit_with_walls(
            X,
            coords,
            section_id,
            max_seconds=10.0,
            device="cpu",
            num_threads=1,
            batch_size=2048,
        )
    assert captured["batch_size"] == 2048
