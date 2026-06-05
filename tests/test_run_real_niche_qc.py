"""Count-based QC filtering tests for ``scripts/run_real_niche.py``.

Context: the primary processed GSE282124 AnnData (124,938 x 315) had received NO
QC filtering -- 76.18% of cells carry 0 transcript counts. ``normalize_total``
then divides by zero on those rows -> NaN, which poisons the contrastive fit (a
124k run dumped 43.5% of cells into a single empty-cell prototype). This pins the
pure QC mask helper + the notes/CLI plumbing.

The PURE ``qc_cell_mask`` tests need NO scanpy/anndata/real-data: the mask is a
small numpy/scipy.sparse function tested directly. Only the optional
integration-style ``_load_dataset`` driver is guarded with
``importorskip("scanpy")``/``importorskip("anndata")`` (mirroring
tests/smoke/test_plot_niche_results.py), since scanpy may be absent in CI.
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


# --------------------------------------------------------------------------
# qc_cell_mask -- pure, scanpy-free count-QC mask (>= semantics)
# --------------------------------------------------------------------------


def test_qc_mask_drops_all_zero_rows():
    """Rows with zero total counts are masked out; populated rows are kept."""
    X = np.array(
        [
            [0, 0, 0, 0, 0],  # empty -> dropped
            [5, 5, 5, 0, 0],  # 15 counts, 3 genes -> kept
            [0, 0, 0, 0, 0],  # empty -> dropped
            [2, 2, 2, 2, 2],  # 10 counts, 5 genes -> kept
        ],
        dtype=np.int64,
    )
    mask = runner.qc_cell_mask(X, min_counts=10, min_genes=5)
    # row1 has only 3 genes (< 5) -> dropped by the n_genes floor.
    assert mask.tolist() == [False, False, False, True]
    assert mask.dtype == bool


def test_qc_mask_min_counts_boundary_is_kept():
    """A cell with total counts EXACTLY == min_counts is KEPT (>= semantics)."""
    # 5 genes each at 2 counts = 10 total counts, 5 genes detected.
    X = np.array(
        [
            [2, 2, 2, 2, 2],  # 10 counts == min_counts -> kept
            [1, 2, 2, 2, 2],  # 9 counts  <  min_counts -> dropped
        ],
        dtype=np.int64,
    )
    mask = runner.qc_cell_mask(X, min_counts=10, min_genes=1)
    assert mask.tolist() == [True, False]


def test_qc_mask_min_genes_boundary_is_kept():
    """A cell with n_genes-detected EXACTLY == min_genes is KEPT (>= semantics)."""
    X = np.array(
        [
            [5, 5, 5, 0, 0],  # 3 genes == min_genes -> kept (15 counts)
            [9, 5, 0, 0, 0],  # 2 genes  <  min_genes -> dropped (14 counts)
        ],
        dtype=np.int64,
    )
    mask = runner.qc_cell_mask(X, min_counts=1, min_genes=3)
    assert mask.tolist() == [True, False]


def test_qc_mask_either_threshold_drops():
    """A cell is dropped if EITHER total_counts OR n_genes is below threshold."""
    X = np.array(
        [
            [100, 0, 0, 0, 0],  # 100 counts but only 1 gene -> dropped by n_genes
            [1, 1, 1, 1, 1],  # 5 genes but only 5 counts -> dropped by min_counts
            [3, 3, 3, 3, 0],  # 12 counts, 4 genes -> kept
        ],
        dtype=np.int64,
    )
    mask = runner.qc_cell_mask(X, min_counts=10, min_genes=4)
    assert mask.tolist() == [False, False, True]


def test_qc_mask_sparse_and_dense_identical():
    """scipy.sparse and dense inputs yield identical masks."""
    sp = pytest.importorskip("scipy.sparse")
    rng = np.random.default_rng(7)
    dense = rng.integers(0, 4, size=(50, 12)).astype(np.int64)
    # Force a handful of all-zero rows so the drop path is exercised.
    dense[::7] = 0
    sparse = sp.csr_matrix(dense)

    mask_dense = runner.qc_cell_mask(dense, min_counts=10, min_genes=5)
    mask_sparse = runner.qc_cell_mask(sparse, min_counts=10, min_genes=5)
    assert mask_dense.tolist() == mask_sparse.tolist()
    assert mask_dense.dtype == bool and mask_sparse.dtype == bool


def test_qc_mask_all_zero_input_all_false():
    """An all-zero matrix masks every cell False (caller then aborts)."""
    X = np.zeros((20, 8), dtype=np.int64)
    mask = runner.qc_cell_mask(X, min_counts=10, min_genes=5)
    assert mask.shape == (20,)
    assert not mask.any()


def test_qc_mask_float_stored_counts():
    """Float-stored (MERSCOPE) whole-number counts work like integer counts."""
    X = np.array(
        [
            [0.0, 0.0, 0.0],  # empty -> dropped
            [4.0, 4.0, 4.0],  # 12 counts, 3 genes -> kept
        ],
        dtype=np.float32,
    )
    mask = runner.qc_cell_mask(X, min_counts=10, min_genes=3)
    assert mask.tolist() == [False, True]


# --------------------------------------------------------------------------
# qc_note_string -- the run_metadata notes fragment format
# --------------------------------------------------------------------------


def test_qc_note_string_format_includes_before_after_dropped():
    """The QC note records min thresholds, n_before/n_after/n_dropped + percent."""
    note = runner.qc_note_string(
        min_counts=10, min_genes=5, n_before=1000, n_after=240
    )
    assert "min_counts=10" in note
    assert "min_genes=5" in note
    assert "n_before=1000" in note
    assert "n_after=240" in note
    assert "n_dropped=760" in note
    assert "76" in note  # 76.0% dropped


# --------------------------------------------------------------------------
# CLI flags: --min-counts / --min-genes (defaults: conservative MERFISH floor)
# --------------------------------------------------------------------------


def test_min_counts_min_genes_cli_defaults():
    parser = runner._build_parser()
    args = parser.parse_args([])
    assert args.min_counts == 10
    assert args.min_genes == 5


def test_min_counts_min_genes_cli_parsed():
    parser = runner._build_parser()
    args = parser.parse_args(["--min-counts", "3", "--min-genes", "1"])
    assert args.min_counts == 3
    assert args.min_genes == 1


# --------------------------------------------------------------------------
# Integration-style: _load_dataset applies QC before normalization. Guarded
# with importorskip (scanpy/anndata) -- builds a tiny synthetic AnnData in
# tmp_path, no real data, no network.
# --------------------------------------------------------------------------


def test_load_dataset_filters_empty_cells(tmp_path):
    """_load_dataset drops zero-count cells before normalization (no NaN rows)."""
    sc = pytest.importorskip("scanpy")  # noqa: F841
    ad = pytest.importorskip("anndata")

    rng = np.random.default_rng(11)
    # > QC_MIN_CELLS good cells so the survivors clear the abort floor.
    n_good, n_empty, n_genes = 120, 80, 20
    good = rng.integers(5, 60, size=(n_good, n_genes)).astype(np.float32)
    empty = np.zeros((n_empty, n_genes), dtype=np.float32)
    X = np.vstack([good, empty])

    adata = ad.AnnData(X=X)
    adata.obsm["spatial"] = rng.uniform(0, 1000, size=(X.shape[0], 2)).astype(
        np.float32
    )
    path = tmp_path / "qc.h5ad"
    adata.write_h5ad(str(path))

    (
        out_X,
        coords,
        section_id,
        _section_col,
        normalization,
        n_obs,
        _n_vars,
        out_adata,
    ) = runner._load_dataset(
        path, already_normalized=False, min_counts=10, min_genes=5
    )

    # The 60 empty cells are gone; only the populated cells survive.
    assert n_obs == n_good
    assert out_X.shape[0] == n_good
    assert coords.shape[0] == n_good
    assert section_id.shape[0] == n_good
    # Normalization ran on filtered data -> no NaN leaked into X.
    assert np.isfinite(out_X).all()
    # The QC note is stashed for the caller to record.
    qc_note = out_adata.uns.get("_qc_note")
    assert qc_note is not None
    assert f"n_before={n_good + n_empty}" in qc_note
    assert f"n_after={n_good}" in qc_note


def test_load_dataset_aborts_on_near_empty(tmp_path):
    """_load_dataset raises when QC leaves < 100 cells (don't fit on near-nothing)."""
    sc = pytest.importorskip("scanpy")  # noqa: F841
    ad = pytest.importorskip("anndata")

    rng = np.random.default_rng(13)
    # 10 populated cells, 90 empty -> only 10 survive (< 100) -> abort.
    good = rng.integers(5, 60, size=(10, 15)).astype(np.float32)
    empty = np.zeros((90, 15), dtype=np.float32)
    X = np.vstack([good, empty])
    adata = ad.AnnData(X=X)
    adata.obsm["spatial"] = np.zeros((X.shape[0], 2), dtype=np.float32)
    path = tmp_path / "near_empty.h5ad"
    adata.write_h5ad(str(path))

    with pytest.raises((ValueError, RuntimeError), match="QC"):
        runner._load_dataset(
            path, already_normalized=False, min_counts=10, min_genes=5
        )
