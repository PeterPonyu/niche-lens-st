"""Tests for the config-driven niche sweep harness (#101, closes #153).

These pin the sweep harness behavior:

* grid expansion: a parameter grid expands to the cartesian product of points,
  and a fixed seed/base config is carried into every run.
* end-to-end on tiny synthetic data: a 2-3 point grid runs through
  ``fit_niche_model`` and writes, per run, a ``manifest.json`` (config + seed +
  git_sha + timing), a ``metrics.json`` (the intrinsic metric dict) and the
  fitted artifacts (``niche.npz`` with H + prototype_id). A top-level
  ``sweep_summary.json`` aggregates the runs.

The harness REUSES the existing intrinsic-metric computation and the uniform
results-contract writer rather than duplicating logic.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

# Import the script module by path (it lives under scripts/, not on sys.path).
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO_ROOT / "scripts" / "sweep_niche.py"
sys.path.insert(0, str(_REPO_ROOT / "src"))

_spec = importlib.util.spec_from_file_location("sweep_niche", _SCRIPT)
sweep = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(sweep)

# The end-to-end sweep needs the torch [model] extra; the grid-expansion test
# is pure python and runs unconditionally.
from nichelens_st.model import TORCH_AVAILABLE  # noqa: E402


# --------------------------------------------------------------------------
# grid expansion (pure python, no torch)
# --------------------------------------------------------------------------


def test_expand_grid_cartesian_product():
    grid = {"embed_dim": [4, 8], "tau": [0.1, 0.2]}
    points = sweep.expand_grid(grid)
    assert len(points) == 4
    # Each point is a dict carrying every swept key.
    for pt in points:
        assert set(pt) == {"embed_dim", "tau"}
    # Cartesian product covers all combinations.
    seen = {(p["embed_dim"], p["tau"]) for p in points}
    assert seen == {(4, 0.1), (4, 0.2), (8, 0.1), (8, 0.2)}


def test_expand_grid_single_point():
    points = sweep.expand_grid({"embed_dim": [16]})
    assert points == [{"embed_dim": 16}]


def test_expand_grid_empty_is_one_default_point():
    # An empty grid yields a single run using all defaults.
    assert sweep.expand_grid({}) == [{}]


def test_parser_accepts_grid_overrides():
    parser = sweep.build_parser()
    args = parser.parse_args(
        ["--embed-dim", "4", "8", "--epochs", "1", "--out", "/tmp/x"]
    )
    assert args.embed_dim == [4, 8]
    assert args.epochs == [1]
    assert str(args.out) == "/tmp/x"


# --------------------------------------------------------------------------
# end-to-end tiny sweep on synthetic data (needs torch)
# --------------------------------------------------------------------------


@pytest.mark.skipif(
    not TORCH_AVAILABLE, reason="requires the optional [model] extra (torch)"
)
def test_run_sweep_writes_per_run_and_summary(tmp_path):
    from nichelens_st.synth.generator import generate_instance

    # Tiny instance so the sweep finishes in seconds.
    inst = generate_instance(
        n_sections=2,
        n_cells_per_section=40,
        n_genes=20,
        K_conserved=2,
        J_specific=1,
        k_nn=4,
        n_markers=3,
        seed=0,
    )

    grid = {"embed_dim": [4, 8], "epochs": [1]}  # 2-point grid
    summary = sweep.run_sweep(
        X=inst.X,
        coords=inst.coords,
        section_id=inst.section_id,
        edges=inst.edges,
        grid=grid,
        out_dir=tmp_path,
        seed=0,
        dataset_paths=["synthetic/tiny.h5ad"],
        base_config={"n_prototypes": 3, "kmeans_iters": 5},
    )

    # Two runs were executed and aggregated.
    assert len(summary["runs"]) == 2

    summary_path = tmp_path / "sweep_summary.json"
    assert summary_path.is_file()
    loaded = json.loads(summary_path.read_text())
    assert loaded["n_runs"] == 2
    assert len(loaded["runs"]) == 2

    for run in loaded["runs"]:
        run_dir = tmp_path / run["run_id"]
        assert run_dir.is_dir()

        manifest = json.loads((run_dir / "manifest.json").read_text())
        assert "config" in manifest
        assert manifest["seed"] == 0
        assert "git_sha" in manifest
        assert "runtime_s" in manifest and manifest["runtime_s"] >= 0.0
        # The swept value is present in the persisted config.
        assert manifest["config"]["embed_dim"] in (4, 8)

        metrics = json.loads((run_dir / "metrics.json").read_text())
        # metrics.json from the uniform contract carries a "metrics" block.
        assert "metrics" in metrics
        assert "niche_morans_i" in metrics["metrics"]

        # Fitted artifacts persisted.
        npz_path = run_dir / "outputs" / "niche.npz"
        assert npz_path.is_file()
        with np.load(npz_path) as data:
            assert "H" in data
            assert "prototype_id" in data
            assert data["H"].shape[0] == inst.X.shape[0]
            assert data["prototype_id"].shape[0] == inst.X.shape[0]
