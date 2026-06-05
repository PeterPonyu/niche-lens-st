"""TDD tests for scripts/emit_figures.py (N-F1 niche-map + N-F2 scalability).

All tests use synthetic fixtures in tmp_path — NEVER touch real
results/niche-lens-st/** files (gitignored, absent in CI).
Any test referencing a real committed-absent path is guarded with
pytest.mark.skipif(not path.exists(), ...).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = REPO_ROOT / "scripts" / "emit_figures.py"
sys.path.insert(0, str(REPO_ROOT / "scripts"))

# emit_figures.py imports matplotlib at module load (to pin the Agg backend).
# matplotlib is an optional `viz` extra and is absent in the minimal CI matrix,
# so skip this whole module when it is unavailable -- mirroring the
# importorskip("matplotlib") guard in tests/smoke/test_plot_niche_results.py.
pytest.importorskip("matplotlib")

_spec = importlib.util.spec_from_file_location("emit_figures", _SCRIPT)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)

emit_niche_map = _mod.emit_niche_map
emit_scalability = _mod.emit_scalability

# Real paths — gitignored, absent in CI.
_REAL_NPZ = REPO_ROOT / "results" / "niche-lens-st" / "outputs" / "niche.npz"
_REAL_META = REPO_ROOT / "results" / "niche-lens-st" / "run_metadata.json"
_REAL_K_SWEEP = REPO_ROOT / "results" / "niche-lens-st" / "outputs" / "k_stability_sweep.json"


# ---------------------------------------------------------------------------
# Synthetic fixture helpers
# ---------------------------------------------------------------------------


def _make_synthetic_npz(
    tmp_path: Path, n_cells: int = 60, n_features: int = 8, n_niches: int = 4
) -> Path:
    """Write a minimal synthetic niche.npz (keys: H, prototype_id)."""
    rng = np.random.default_rng(42)
    H = rng.random((n_cells, n_features)).astype(np.float32)
    prototype_id = np.array([i % n_niches for i in range(n_cells)], dtype=np.int64)
    npz_path = tmp_path / "niche.npz"
    np.savez(str(npz_path), H=H, prototype_id=prototype_id)
    return npz_path


def _make_run_metadata(
    base_dir: Path,
    n_obs: int = 5000,
    runtime_s: float = 100.0,
    *,
    peak_rss_bytes: int | None = 2_000_000_000,
    fallback: bool = False,
) -> Path:
    """Write a synthetic run_metadata.json in *base_dir*."""
    meta: dict = {
        "schema_version": "1.0.0",
        "n_obs": n_obs,
        "runtime_s": runtime_s,
    }
    if peak_rss_bytes is not None:
        meta["peak_rss_bytes"] = peak_rss_bytes
    if fallback:
        meta["_fallback_note"] = (
            "DOWNSIZED SINGLE-SECTION FALLBACK, NOT THE ATLAS-SCALE RUN."
        )
    base_dir.mkdir(parents=True, exist_ok=True)
    p = base_dir / "run_metadata.json"
    p.write_text(json.dumps(meta))
    return p


def _make_nf1_stub(stubs_dir: Path) -> Path:
    stubs_dir.mkdir(parents=True, exist_ok=True)
    stub: dict = {
        "asset_id": "N-F1",
        "project": "NicheLens-ST",
        "role": "niche map",
        "readiness_status": "planned",
        "source_metric_reference": "planned",
        "source_script_reference": "planned",
        "source_exists": False,
        "metric_source_exists": False,
        "artifact_exists": False,
        "supports_safe_prose": False,
        "paper_claim_ready": False,
        "notes": "stub",
    }
    p = stubs_dir / "n-f1_metrics_stub.json"
    p.write_text(json.dumps(stub))
    return p


def _make_nf2_stub(stubs_dir: Path) -> Path:
    stubs_dir.mkdir(parents=True, exist_ok=True)
    stub: dict = {
        "asset_id": "N-F2",
        "project": "NicheLens-ST",
        "role": "scalability",
        "readiness_status": "planned",
        "source_metric_reference": "planned",
        "source_script_reference": "planned",
        "source_exists": False,
        "metric_source_exists": False,
        "artifact_exists": False,
        "supports_safe_prose": False,
        "paper_claim_ready": False,
        "notes": "stub",
    }
    p = stubs_dir / "n-f2_metrics_stub.json"
    p.write_text(json.dumps(stub))
    return p


# ---------------------------------------------------------------------------
# N-F1: niche-map PNG tests
# ---------------------------------------------------------------------------


class TestNF1NicheMap:
    """N-F1: emit_niche_map() produces PNGs from synthetic fixtures."""

    def test_renders_umap_png(self, tmp_path: Path) -> None:
        npz = _make_synthetic_npz(tmp_path)
        out = emit_niche_map(npz_path=npz, out_dir=tmp_path / "out")
        assert "umap" in out, "Expected 'umap' key in output dict"
        assert out["umap"].exists(), f"umap PNG not written: {out['umap']}"

    def test_umap_png_is_nonempty(self, tmp_path: Path) -> None:
        npz = _make_synthetic_npz(tmp_path)
        out = emit_niche_map(npz_path=npz, out_dir=tmp_path / "out")
        assert out["umap"].stat().st_size > 0, "umap PNG is empty"

    def test_returns_dict(self, tmp_path: Path) -> None:
        npz = _make_synthetic_npz(tmp_path)
        out = emit_niche_map(npz_path=npz, out_dir=tmp_path / "out")
        assert isinstance(out, dict)

    def test_output_dir_created(self, tmp_path: Path) -> None:
        npz = _make_synthetic_npz(tmp_path)
        out_dir = tmp_path / "nested" / "plots"
        emit_niche_map(npz_path=npz, out_dir=out_dir)
        assert out_dir.exists()

    def test_stub_artifact_exists_set(self, tmp_path: Path) -> None:
        npz = _make_synthetic_npz(tmp_path)
        stub_p = _make_nf1_stub(tmp_path / "stubs")
        emit_niche_map(npz_path=npz, out_dir=tmp_path / "out", stub_path=stub_p)
        data = json.loads(stub_p.read_text())
        assert data["artifact_exists"] is True

    def test_stub_source_exists_set(self, tmp_path: Path) -> None:
        npz = _make_synthetic_npz(tmp_path)
        stub_p = _make_nf1_stub(tmp_path / "stubs")
        emit_niche_map(npz_path=npz, out_dir=tmp_path / "out", stub_path=stub_p)
        data = json.loads(stub_p.read_text())
        assert data["source_exists"] is True

    def test_stub_script_reference_set(self, tmp_path: Path) -> None:
        npz = _make_synthetic_npz(tmp_path)
        stub_p = _make_nf1_stub(tmp_path / "stubs")
        emit_niche_map(npz_path=npz, out_dir=tmp_path / "out", stub_path=stub_p)
        data = json.loads(stub_p.read_text())
        assert "emit_figures" in data["source_script_reference"]

    def test_stub_paper_claim_ready_stays_false_on_fallback(self, tmp_path: Path) -> None:
        npz = _make_synthetic_npz(tmp_path)
        meta_p = _make_run_metadata(tmp_path / "meta", fallback=True)
        stub_p = _make_nf1_stub(tmp_path / "stubs")
        emit_niche_map(
            npz_path=npz,
            out_dir=tmp_path / "out",
            run_metadata_path=meta_p,
            stub_path=stub_p,
        )
        data = json.loads(stub_p.read_text())
        assert data["paper_claim_ready"] is False

    def test_stub_paper_claim_ready_stays_false_without_fallback(
        self, tmp_path: Path
    ) -> None:
        npz = _make_synthetic_npz(tmp_path)
        stub_p = _make_nf1_stub(tmp_path / "stubs")
        emit_niche_map(npz_path=npz, out_dir=tmp_path / "out", stub_path=stub_p)
        data = json.loads(stub_p.read_text())
        assert data["paper_claim_ready"] is False

    def test_no_error_without_stub(self, tmp_path: Path) -> None:
        npz = _make_synthetic_npz(tmp_path)
        out = emit_niche_map(npz_path=npz, out_dir=tmp_path / "out")
        assert out

    def test_stub_not_mutated_when_not_provided(self, tmp_path: Path) -> None:
        npz = _make_synthetic_npz(tmp_path)
        absent = tmp_path / "nonexistent" / "stub.json"
        # Should not raise even when stub_path points to nonexistent file.
        out = emit_niche_map(npz_path=npz, out_dir=tmp_path / "out", stub_path=absent)
        assert not absent.exists()
        assert out["umap"].exists()


# ---------------------------------------------------------------------------
# N-F2: scalability figure + JSON tests
# ---------------------------------------------------------------------------


class TestNF2Scalability:
    """N-F2: emit_scalability() — figure + JSON from synthetic metadata."""

    def test_single_run_emits_png(self, tmp_path: Path) -> None:
        meta_p = _make_run_metadata(tmp_path / "run0")
        out = emit_scalability([meta_p], out_dir=tmp_path / "out")
        assert "figure" in out
        assert out["figure"].exists()

    def test_single_run_emits_json(self, tmp_path: Path) -> None:
        meta_p = _make_run_metadata(tmp_path / "run0")
        out = emit_scalability([meta_p], out_dir=tmp_path / "out")
        assert "json" in out
        assert out["json"].exists()

    def test_figure_is_nonempty(self, tmp_path: Path) -> None:
        meta_p = _make_run_metadata(tmp_path / "run0")
        out = emit_scalability([meta_p], out_dir=tmp_path / "out")
        assert out["figure"].stat().st_size > 0

    def test_json_has_required_columns(self, tmp_path: Path) -> None:
        meta_p = _make_run_metadata(tmp_path / "run0")
        out = emit_scalability([meta_p], out_dir=tmp_path / "out")
        data = json.loads(out["json"].read_text())
        assert "columns" in data
        for col in ("n_obs", "runtime_s", "peak_rss_bytes"):
            assert col in data["columns"], f"Missing column {col!r} in scalability.json"

    def test_json_has_one_row_for_single_run(self, tmp_path: Path) -> None:
        meta_p = _make_run_metadata(tmp_path / "run0")
        out = emit_scalability([meta_p], out_dir=tmp_path / "out")
        data = json.loads(out["json"].read_text())
        assert len(data["rows"]) == 1

    def test_json_row_has_correct_values(self, tmp_path: Path) -> None:
        meta_p = _make_run_metadata(
            tmp_path / "run0", n_obs=5000, runtime_s=100.0, peak_rss_bytes=2_000_000_000
        )
        out = emit_scalability([meta_p], out_dir=tmp_path / "out")
        data = json.loads(out["json"].read_text())
        row = data["rows"][0]
        assert row["n_obs"] == 5000
        assert abs(row["runtime_s"] - 100.0) < 1e-9
        assert row["peak_rss_bytes"] == 2_000_000_000

    def test_missing_peak_rss_bytes_is_none(self, tmp_path: Path) -> None:
        """Run without peak_rss_bytes -> None — exact key contract."""
        meta_p = _make_run_metadata(tmp_path / "run0", peak_rss_bytes=None)
        out = emit_scalability([meta_p], out_dir=tmp_path / "out")
        data = json.loads(out["json"].read_text())
        row = data["rows"][0]
        assert row["peak_rss_bytes"] is None, (
            f"peak_rss_bytes must be None when absent, got {row['peak_rss_bytes']!r}"
        )

    def test_missing_peak_rss_bytes_not_zero(self, tmp_path: Path) -> None:
        meta_p = _make_run_metadata(tmp_path / "run0", peak_rss_bytes=None)
        out = emit_scalability([meta_p], out_dir=tmp_path / "out")
        data = json.loads(out["json"].read_text())
        row = data["rows"][0]
        assert row["peak_rss_bytes"] != 0, "peak_rss_bytes must not be 0 when absent"

    def test_multiple_runs_sorted_by_n_obs(self, tmp_path: Path) -> None:
        meta_paths = [
            _make_run_metadata(tmp_path / "run0", n_obs=10000, runtime_s=200.0),
            _make_run_metadata(tmp_path / "run1", n_obs=2000, runtime_s=50.0),
            _make_run_metadata(tmp_path / "run2", n_obs=5000, runtime_s=100.0),
        ]
        out = emit_scalability(meta_paths, out_dir=tmp_path / "out")
        data = json.loads(out["json"].read_text())
        n_obs_list = [r["n_obs"] for r in data["rows"]]
        assert n_obs_list == sorted(n_obs_list), f"Rows not sorted by n_obs: {n_obs_list}"

    def test_multiple_runs_all_rows_present(self, tmp_path: Path) -> None:
        meta_paths = [
            _make_run_metadata(tmp_path / f"run{i}", n_obs=(i + 1) * 1000)
            for i in range(3)
        ]
        out = emit_scalability(meta_paths, out_dir=tmp_path / "out")
        data = json.loads(out["json"].read_text())
        assert len(data["rows"]) == 3

    def test_determinism_json_byte_identical(self, tmp_path: Path) -> None:
        meta_p = _make_run_metadata(tmp_path / "run0")
        out1 = emit_scalability([meta_p], out_dir=tmp_path / "out1")
        out2 = emit_scalability([meta_p], out_dir=tmp_path / "out2")
        assert out1["json"].read_bytes() == out2["json"].read_bytes()

    def test_json_paper_claim_ready_always_false(self, tmp_path: Path) -> None:
        meta_p = _make_run_metadata(tmp_path / "run0")
        out = emit_scalability([meta_p], out_dir=tmp_path / "out")
        data = json.loads(out["json"].read_text())
        assert data["paper_claim_ready"] is False

    def test_stub_paper_claim_ready_stays_false_on_fallback(
        self, tmp_path: Path
    ) -> None:
        meta_p = _make_run_metadata(tmp_path / "run0", fallback=True)
        stub_p = _make_nf2_stub(tmp_path / "stubs")
        emit_scalability([meta_p], out_dir=tmp_path / "out", stub_path=stub_p)
        data = json.loads(stub_p.read_text())
        assert data["paper_claim_ready"] is False

    def test_stub_artifact_exists_set(self, tmp_path: Path) -> None:
        meta_p = _make_run_metadata(tmp_path / "run0")
        stub_p = _make_nf2_stub(tmp_path / "stubs")
        emit_scalability([meta_p], out_dir=tmp_path / "out", stub_path=stub_p)
        data = json.loads(stub_p.read_text())
        assert data["artifact_exists"] is True

    def test_stub_source_exists_set(self, tmp_path: Path) -> None:
        meta_p = _make_run_metadata(tmp_path / "run0")
        stub_p = _make_nf2_stub(tmp_path / "stubs")
        emit_scalability([meta_p], out_dir=tmp_path / "out", stub_path=stub_p)
        data = json.loads(stub_p.read_text())
        assert data["source_exists"] is True

    def test_stub_script_reference_set(self, tmp_path: Path) -> None:
        meta_p = _make_run_metadata(tmp_path / "run0")
        stub_p = _make_nf2_stub(tmp_path / "stubs")
        emit_scalability([meta_p], out_dir=tmp_path / "out", stub_path=stub_p)
        data = json.loads(stub_p.read_text())
        assert "emit_figures" in data["source_script_reference"]

    def test_k_sweep_absent_no_error(self, tmp_path: Path) -> None:
        meta_p = _make_run_metadata(tmp_path / "run0")
        absent_sweep = tmp_path / "nonexistent_k_sweep.json"
        out = emit_scalability(
            [meta_p], out_dir=tmp_path / "out", k_sweep_path=absent_sweep
        )
        assert out["figure"].exists()

    def test_k_sweep_none_no_error(self, tmp_path: Path) -> None:
        meta_p = _make_run_metadata(tmp_path / "run0")
        out = emit_scalability([meta_p], out_dir=tmp_path / "out", k_sweep_path=None)
        assert out["figure"].exists()

    def test_k_sweep_present_generates_figure(self, tmp_path: Path) -> None:
        meta_p = _make_run_metadata(tmp_path / "run0")
        sweep: dict = {
            "fitted_k": 8,
            "per_k": {
                str(k): {"silhouette": 0.1 + k * 0.01, "calinski_harabasz": 400.0}
                for k in range(5, 11)
            },
        }
        sweep_p = tmp_path / "k_sweep.json"
        sweep_p.write_text(json.dumps(sweep))
        out = emit_scalability(
            [meta_p], out_dir=tmp_path / "out", k_sweep_path=sweep_p
        )
        assert out["figure"].exists()
        assert out["figure"].stat().st_size > 0

    def test_empty_run_metadata_list_emits_zero_rows(self, tmp_path: Path) -> None:
        """Empty list is valid — emits 0-row scalability.json."""
        out = emit_scalability([], out_dir=tmp_path / "out")
        data = json.loads(out["json"].read_text())
        assert data["rows"] == []
        assert out["figure"].exists()

    def test_mixed_peak_rss_present_and_absent(self, tmp_path: Path) -> None:
        """Some runs have peak_rss_bytes, some don't — both appear correctly."""
        meta_with = _make_run_metadata(
            tmp_path / "run0", n_obs=1000, peak_rss_bytes=1_000_000
        )
        meta_without = _make_run_metadata(
            tmp_path / "run1", n_obs=2000, peak_rss_bytes=None
        )
        out = emit_scalability([meta_with, meta_without], out_dir=tmp_path / "out")
        data = json.loads(out["json"].read_text())
        rows_by_n_obs = {r["n_obs"]: r for r in data["rows"]}
        assert rows_by_n_obs[1000]["peak_rss_bytes"] == 1_000_000
        assert rows_by_n_obs[2000]["peak_rss_bytes"] is None


# ---------------------------------------------------------------------------
# CI-absence guard: real paths are gitignored / absent
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _REAL_NPZ.exists(),
    reason=(
        "results/niche-lens-st/outputs/niche.npz is gitignored (local-run artifact); "
        "absent in CI"
    ),
)
class TestNF1RealData:
    """Smoke test against real niche.npz (local only, skipped in CI)."""

    def test_renders_png_from_real_npz(self, tmp_path: Path) -> None:
        out = emit_niche_map(npz_path=_REAL_NPZ, out_dir=tmp_path / "out")
        assert out["umap"].exists()


@pytest.mark.skipif(
    not _REAL_META.exists(),
    reason=(
        "results/niche-lens-st/run_metadata.json is gitignored (local-run artifact); "
        "absent in CI"
    ),
)
class TestNF2RealData:
    """Smoke test against real run_metadata.json (local only, skipped in CI)."""

    def test_emits_figure_from_real_metadata(self, tmp_path: Path) -> None:
        out = emit_scalability(
            [_REAL_META], out_dir=tmp_path / "out", k_sweep_path=_REAL_K_SWEEP
        )
        assert out["figure"].exists()
        assert out["json"].exists()

    def test_real_peak_rss_bytes_is_none_or_int(self, tmp_path: Path) -> None:
        """Real run_metadata may lack peak_rss_bytes (added by worker-runner)."""
        out = emit_scalability([_REAL_META], out_dir=tmp_path / "out")
        data = json.loads(out["json"].read_text())
        val = data["rows"][0]["peak_rss_bytes"]
        assert val is None or isinstance(val, int), (
            f"peak_rss_bytes must be None or int, got {type(val).__name__}: {val!r}"
        )
