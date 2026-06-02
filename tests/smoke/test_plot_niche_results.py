"""Headless smoke test for scripts/plot_niche_results.py.

Creates tiny synthetic inputs (200 cells, 16-D embeddings, fake spatial coords)
and asserts that:
  1. plot_niche() runs without error in a headless environment.
  2. umap_by_niche.png is written to the output directory.
  3. spatial_niche.png is written when spatial coords are provided.
  4. plot_niche() also works when only the npz is provided (no h5ad / spatial).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Make the scripts/ directory importable from the worktree root regardless of
# how pytest is invoked.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = _REPO_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))


# ---------------------------------------------------------------------------
# Import the module under test
# ---------------------------------------------------------------------------
_spec = importlib.util.spec_from_file_location(
    "plot_niche_results", _SCRIPTS / "plot_niche_results.py"
)
_mod = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
_spec.loader.exec_module(_mod)  # type: ignore[union-attr]
plot_niche = _mod.plot_niche


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_CELLS = 200
EMBED_DIM = 16
N_PROTOS = 8
SEED = 42


@pytest.fixture()
def synth_npz(tmp_path: Path) -> Path:
    """Write a small synthetic niche.npz to tmp_path."""
    rng = np.random.default_rng(SEED)
    H = rng.standard_normal((N_CELLS, EMBED_DIM)).astype(np.float32)
    prototype_id = rng.integers(0, N_PROTOS, size=N_CELLS).astype(np.int64)
    npz_path = tmp_path / "niche.npz"
    np.savez_compressed(npz_path, H=H, prototype_id=prototype_id)
    return npz_path


@pytest.fixture()
def synth_h5ad(tmp_path: Path) -> Path:
    """Write a minimal AnnData .h5ad with obsm['spatial'] to tmp_path."""
    pytest.importorskip("anndata")
    import anndata

    rng = np.random.default_rng(SEED + 1)
    X = rng.standard_normal((N_CELLS, 10)).astype(np.float32)
    spatial = rng.uniform(0, 1000, size=(N_CELLS, 2)).astype(np.float32)

    adata = anndata.AnnData(X=X)
    adata.obsm["spatial"] = spatial

    h5ad_path = tmp_path / "anndata.h5ad"
    adata.write_h5ad(str(h5ad_path))
    return h5ad_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_umap_and_spatial_pngs_written(
    tmp_path: Path, synth_npz: Path, synth_h5ad: Path
) -> None:
    """Both umap_by_niche.png and spatial_niche.png must be created."""
    out_dir = tmp_path / "plots"
    outputs = plot_niche(
        npz_path=synth_npz,
        h5ad_path=synth_h5ad,
        out_dir=out_dir,
        seed=SEED,
    )

    assert "umap" in outputs, "umap key missing from return value"
    assert "spatial" in outputs, "spatial key missing from return value"

    umap_png = outputs["umap"]
    spatial_png = outputs["spatial"]

    assert umap_png.exists(), f"umap_by_niche.png not written at {umap_png}"
    assert spatial_png.exists(), f"spatial_niche.png not written at {spatial_png}"

    # Sanity: files are non-trivially large (> 1 KiB each).
    assert umap_png.stat().st_size > 1024, "umap_by_niche.png suspiciously small"
    assert spatial_png.stat().st_size > 1024, "spatial_niche.png suspiciously small"


def test_npz_only_no_spatial(tmp_path: Path, synth_npz: Path) -> None:
    """plot_niche works without an h5ad (no spatial plot expected)."""
    out_dir = tmp_path / "plots_no_spatial"
    outputs = plot_niche(npz_path=synth_npz, out_dir=out_dir, seed=SEED)

    assert "umap" in outputs, "umap key missing when no h5ad provided"
    assert outputs["umap"].exists(), "umap_by_niche.png not written"
    # No spatial coords => no spatial PNG.
    assert "spatial" not in outputs, "spatial key should be absent when no h5ad"


def test_output_dir_created_automatically(tmp_path: Path, synth_npz: Path) -> None:
    """plot_niche creates the output directory if it does not exist."""
    out_dir = tmp_path / "deep" / "nested" / "dir"
    assert not out_dir.exists()
    outputs = plot_niche(npz_path=synth_npz, out_dir=out_dir, seed=SEED)
    assert out_dir.exists()
    assert outputs["umap"].exists()
