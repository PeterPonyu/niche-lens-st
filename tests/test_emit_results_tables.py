"""TDD tests for scripts/emit_results_tables.py (issues #314 / #341).

Covers:
- model-only (no baselines dir) still emits a valid 1-row table
- 2 fake baselines -> 3 rows in stable order
- missing metric -> 'NA' in CSV/MD, null in JSON; never 0
- determinism (same inputs -> byte-identical output, run twice)
- fallback note is surfaced in artifacts
- paper_metrics stub gating (fallback -> paper_claim_ready stays false)
- all three artifact types (CSV, MD, JSON) are emitted
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = REPO_ROOT / "scripts" / "emit_results_tables.py"
sys.path.insert(0, str(REPO_ROOT / "scripts"))

_spec = importlib.util.spec_from_file_location("emit_results_tables", _SCRIPT)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)

emit_leaderboard = _mod.emit_leaderboard
LEADERBOARD_COLS = _mod.LEADERBOARD_COLS

# Fixture: the committed model run metrics.json
_MODEL_METRICS = REPO_ROOT / "results" / "niche-lens-st" / "metrics.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_metrics_json(base_dir: Path, method_name: str, overrides: dict | None = None) -> Path:
    """Write a minimal valid metrics.json under *base_dir*/*method_name*/."""
    content: dict = {
        "schema_version": "1.0.0",
        "project": method_name,
        "dataset_card_id": "test_dataset",
        "metrics": {
            "domain_ari": 0.10,
            "domain_ami": 0.20,
            "domain_nmi": 0.30,
            "domain_macro_f1": 0.40,
            "domain_homogeneity": 0.50,
            "domain_accuracy": 0.60,
            "embedding_silhouette": 0.70,
            "niche_morans_i": 0.80,
        },
        "n_obs": 100,
        "n_vars": 50,
        "seed": 0,
        "runtime_s": 1.0,
        "git_sha": "abc1234",
    }
    if overrides:
        metrics_patch = overrides.pop("metrics", None)
        if metrics_patch is not None:
            content["metrics"].update(metrics_patch)
        content.update(overrides)

    d = base_dir / method_name
    d.mkdir(parents=True, exist_ok=True)
    p = d / "metrics.json"
    p.write_text(json.dumps(content))
    return p


def _make_fallback_run_metadata(metrics_path: Path) -> Path:
    """Write run_metadata.json with _fallback_note alongside *metrics_path*."""
    meta = {
        "_fallback_note": (
            "DOWNSIZED SINGLE-SECTION FALLBACK, NOT THE ATLAS-SCALE RUN. "
            "5,488-cell MERFISH section only."
        ),
        "dataset_card_id": "niche_merfish_slice",
    }
    p = metrics_path.parent / "run_metadata.json"
    p.write_text(json.dumps(meta))
    return p


def _csv_data_lines(csv_text: str) -> list[str]:
    """Return non-empty, non-comment lines from CSV text."""
    return [ln for ln in csv_text.splitlines() if ln.strip() and not ln.startswith("#")]


def _make_stub(stubs_dir: Path) -> Path:
    """Create a minimal n-t1_metrics_stub.json in *stubs_dir*."""
    stubs_dir.mkdir(parents=True, exist_ok=True)
    stub: dict = {
        "asset_id": "N-T1",
        "project": "NicheLens-ST",
        "role": "multi-method comparison table",
        "readiness_status": "contract-only",
        "source_metric_reference": "planned",
        "source_script_reference": "planned",
        "source_exists": False,
        "metric_source_exists": False,
        "artifact_exists": False,
        "supports_safe_prose": False,
        "paper_claim_ready": False,
        "notes": "stub",
    }
    p = stubs_dir / "n-t1_metrics_stub.json"
    p.write_text(json.dumps(stub))
    return p


# ---------------------------------------------------------------------------
# TestModelOnly
# ---------------------------------------------------------------------------


class TestModelOnly:
    """Model-only run (no baselines dir) must emit a valid 1-row table."""

    def test_emits_csv(self, tmp_path: Path) -> None:
        model_p = _make_metrics_json(tmp_path / "model", "niche-lens-st")
        out = emit_leaderboard(model_p, baselines_glob="", out_dir=tmp_path / "out")
        assert out["csv"].exists()

    def test_emits_md(self, tmp_path: Path) -> None:
        model_p = _make_metrics_json(tmp_path / "model", "niche-lens-st")
        out = emit_leaderboard(model_p, baselines_glob="", out_dir=tmp_path / "out")
        assert out["md"].exists()

    def test_emits_json(self, tmp_path: Path) -> None:
        model_p = _make_metrics_json(tmp_path / "model", "niche-lens-st")
        out = emit_leaderboard(model_p, baselines_glob="", out_dir=tmp_path / "out")
        assert out["json"].exists()

    def test_csv_has_one_data_row(self, tmp_path: Path) -> None:
        model_p = _make_metrics_json(tmp_path / "model", "niche-lens-st")
        out = emit_leaderboard(model_p, baselines_glob="", out_dir=tmp_path / "out")
        lines = _csv_data_lines(out["csv"].read_text())
        assert len(lines) == 2, f"Expected header+1 row, got {len(lines)} lines"

    def test_md_has_one_data_row(self, tmp_path: Path) -> None:
        model_p = _make_metrics_json(tmp_path / "model", "niche-lens-st")
        out = emit_leaderboard(model_p, baselines_glob="", out_dir=tmp_path / "out")
        # MD: header | sep | data row (plus optional prose lines)
        table_lines = [
            ln
            for ln in out["md"].read_text().splitlines()
            if ln.strip().startswith("|")
        ]
        assert len(table_lines) == 3, f"Expected header+sep+1 data row, got {table_lines}"

    def test_csv_header_starts_with_method(self, tmp_path: Path) -> None:
        model_p = _make_metrics_json(tmp_path / "model", "niche-lens-st")
        out = emit_leaderboard(model_p, baselines_glob="", out_dir=tmp_path / "out")
        header = _csv_data_lines(out["csv"].read_text())[0]
        assert header.split(",")[0] == "method"

    def test_model_method_name_is_niche_lens_st(self, tmp_path: Path) -> None:
        model_p = _make_metrics_json(tmp_path / "model", "niche-lens-st")
        out = emit_leaderboard(model_p, baselines_glob="", out_dir=tmp_path / "out")
        data_row = _csv_data_lines(out["csv"].read_text())[1]
        assert data_row.split(",")[0] == "niche-lens-st"


# ---------------------------------------------------------------------------
# TestMultipleBaselines
# ---------------------------------------------------------------------------


class TestMultipleBaselines:
    """2 fake baselines -> 3 rows in stable, deterministic order."""

    def test_three_rows_in_csv(self, tmp_path: Path) -> None:
        bd = tmp_path / "baselines"
        _make_metrics_json(bd, "baseline_a")
        _make_metrics_json(bd, "baseline_b")
        model_p = _make_metrics_json(tmp_path / "model", "niche-lens-st")
        out = emit_leaderboard(
            model_p,
            baselines_glob=str(bd / "*/metrics.json"),
            out_dir=tmp_path / "out",
        )
        lines = _csv_data_lines(out["csv"].read_text())
        assert len(lines) == 4, f"Expected header+3 rows, got {len(lines)}"

    def test_model_row_is_first(self, tmp_path: Path) -> None:
        bd = tmp_path / "baselines"
        _make_metrics_json(bd, "baseline_a")
        _make_metrics_json(bd, "baseline_b")
        model_p = _make_metrics_json(tmp_path / "model", "niche-lens-st")
        out = emit_leaderboard(
            model_p,
            baselines_glob=str(bd / "*/metrics.json"),
            out_dir=tmp_path / "out",
        )
        data = json.loads(out["json"].read_text())
        assert data["rows"][0]["method"] == "niche-lens-st"

    def test_baselines_sorted_alphabetically(self, tmp_path: Path) -> None:
        bd = tmp_path / "baselines"
        _make_metrics_json(bd, "z_baseline")
        _make_metrics_json(bd, "a_baseline")
        model_p = _make_metrics_json(tmp_path / "model", "niche-lens-st")
        out = emit_leaderboard(
            model_p,
            baselines_glob=str(bd / "*/metrics.json"),
            out_dir=tmp_path / "out",
        )
        data = json.loads(out["json"].read_text())
        methods = [r["method"] for r in data["rows"]]
        assert methods[0] == "niche-lens-st"
        assert methods[1] < methods[2], f"Baselines not sorted: {methods[1:]}"

    def test_three_rows_in_md(self, tmp_path: Path) -> None:
        bd = tmp_path / "baselines"
        _make_metrics_json(bd, "baseline_a")
        _make_metrics_json(bd, "baseline_b")
        model_p = _make_metrics_json(tmp_path / "model", "niche-lens-st")
        out = emit_leaderboard(
            model_p,
            baselines_glob=str(bd / "*/metrics.json"),
            out_dir=tmp_path / "out",
        )
        table_lines = [
            ln
            for ln in out["md"].read_text().splitlines()
            if ln.strip().startswith("|")
        ]
        assert len(table_lines) == 5, f"Expected header+sep+3 rows, got {table_lines}"


# ---------------------------------------------------------------------------
# TestMissingMetrics
# ---------------------------------------------------------------------------


class TestMissingMetrics:
    """Missing metric -> 'NA' in CSV/MD and null in JSON; never 0."""

    def test_null_metric_is_na_in_csv(self, tmp_path: Path) -> None:
        model_p = _make_metrics_json(
            tmp_path / "model",
            "niche-lens-st",
            overrides={"metrics": {"domain_ari": None}},
        )
        out = emit_leaderboard(model_p, baselines_glob="", out_dir=tmp_path / "out")
        header_line, data_line = _csv_data_lines(out["csv"].read_text())[:2]
        headers = header_line.split(",")
        values = data_line.split(",")
        ari_idx = headers.index("domain_ari")
        assert values[ari_idx] == "NA", f"Expected NA, got {values[ari_idx]!r}"

    def test_null_metric_not_zero_in_csv(self, tmp_path: Path) -> None:
        model_p = _make_metrics_json(
            tmp_path / "model",
            "niche-lens-st",
            overrides={"metrics": {"domain_ari": None}},
        )
        out = emit_leaderboard(model_p, baselines_glob="", out_dir=tmp_path / "out")
        header_line, data_line = _csv_data_lines(out["csv"].read_text())[:2]
        headers = header_line.split(",")
        values = data_line.split(",")
        ari_idx = headers.index("domain_ari")
        assert values[ari_idx] != "0", "Missing metric must not be '0'"
        assert values[ari_idx] != "0.0000", "Missing metric must not be '0.0000'"

    def test_null_metric_is_null_in_json(self, tmp_path: Path) -> None:
        model_p = _make_metrics_json(
            tmp_path / "model",
            "niche-lens-st",
            overrides={"metrics": {"domain_ari": None}},
        )
        out = emit_leaderboard(model_p, baselines_glob="", out_dir=tmp_path / "out")
        data = json.loads(out["json"].read_text())
        assert data["rows"][0]["metrics"]["domain_ari"] is None

    def test_absent_key_is_na_in_csv(self, tmp_path: Path) -> None:
        """Metric key entirely absent from metrics dict -> NA everywhere."""
        content = {
            "schema_version": "1.0.0",
            "project": "niche-lens-st",
            "dataset_card_id": "test_dataset",
            "metrics": {},
            "git_sha": "abc1234",
        }
        d = tmp_path / "model" / "niche-lens-st"
        d.mkdir(parents=True, exist_ok=True)
        p = d / "metrics.json"
        p.write_text(json.dumps(content))
        out = emit_leaderboard(p, baselines_glob="", out_dir=tmp_path / "out")
        header_line, data_line = _csv_data_lines(out["csv"].read_text())[:2]
        headers = header_line.split(",")
        values = data_line.split(",")
        # Every metric column should be NA
        for col in LEADERBOARD_COLS:
            idx = headers.index(col)
            assert values[idx] == "NA", f"col {col!r}: expected NA, got {values[idx]!r}"

    def test_absent_key_is_null_in_json(self, tmp_path: Path) -> None:
        content = {
            "schema_version": "1.0.0",
            "project": "niche-lens-st",
            "dataset_card_id": "test_dataset",
            "metrics": {},
            "git_sha": "abc1234",
        }
        d = tmp_path / "model" / "niche-lens-st"
        d.mkdir(parents=True, exist_ok=True)
        p = d / "metrics.json"
        p.write_text(json.dumps(content))
        out = emit_leaderboard(p, baselines_glob="", out_dir=tmp_path / "out")
        data = json.loads(out["json"].read_text())
        for col in LEADERBOARD_COLS:
            assert data["rows"][0]["metrics"][col] is None, f"col {col!r} should be null"


# ---------------------------------------------------------------------------
# TestDeterminism
# ---------------------------------------------------------------------------


class TestDeterminism:
    """Same inputs must produce byte-identical output on two consecutive runs."""

    def test_csv_byte_identical(self, tmp_path: Path) -> None:
        model_p = _make_metrics_json(tmp_path / "model", "niche-lens-st")
        out1 = emit_leaderboard(model_p, baselines_glob="", out_dir=tmp_path / "run1")
        out2 = emit_leaderboard(model_p, baselines_glob="", out_dir=tmp_path / "run2")
        assert out1["csv"].read_bytes() == out2["csv"].read_bytes()

    def test_md_byte_identical(self, tmp_path: Path) -> None:
        model_p = _make_metrics_json(tmp_path / "model", "niche-lens-st")
        out1 = emit_leaderboard(model_p, baselines_glob="", out_dir=tmp_path / "run1")
        out2 = emit_leaderboard(model_p, baselines_glob="", out_dir=tmp_path / "run2")
        assert out1["md"].read_bytes() == out2["md"].read_bytes()

    def test_json_byte_identical(self, tmp_path: Path) -> None:
        model_p = _make_metrics_json(tmp_path / "model", "niche-lens-st")
        out1 = emit_leaderboard(model_p, baselines_glob="", out_dir=tmp_path / "run1")
        out2 = emit_leaderboard(model_p, baselines_glob="", out_dir=tmp_path / "run2")
        assert out1["json"].read_bytes() == out2["json"].read_bytes()

    def test_deterministic_with_two_baselines(self, tmp_path: Path) -> None:
        bd = tmp_path / "baselines"
        _make_metrics_json(bd, "baseline_a")
        _make_metrics_json(bd, "baseline_b")
        model_p = _make_metrics_json(tmp_path / "model", "niche-lens-st")
        g = str(bd / "*/metrics.json")
        out1 = emit_leaderboard(model_p, baselines_glob=g, out_dir=tmp_path / "run1")
        out2 = emit_leaderboard(model_p, baselines_glob=g, out_dir=tmp_path / "run2")
        assert out1["csv"].read_bytes() == out2["csv"].read_bytes()
        assert out1["json"].read_bytes() == out2["json"].read_bytes()


# ---------------------------------------------------------------------------
# TestFallbackNote
# ---------------------------------------------------------------------------


class TestFallbackNote:
    """Fallback note must be surfaced in all artifacts; absence is also tested."""

    def test_fallback_warning_in_md(self, tmp_path: Path) -> None:
        model_p = _make_metrics_json(tmp_path / "model", "niche-lens-st")
        _make_fallback_run_metadata(model_p)
        out = emit_leaderboard(model_p, baselines_glob="", out_dir=tmp_path / "out")
        md_text = out["md"].read_text()
        assert "fallback" in md_text.lower(), "MD must mention fallback"

    def test_fallback_warning_in_csv(self, tmp_path: Path) -> None:
        model_p = _make_metrics_json(tmp_path / "model", "niche-lens-st")
        _make_fallback_run_metadata(model_p)
        out = emit_leaderboard(model_p, baselines_glob="", out_dir=tmp_path / "out")
        csv_text = out["csv"].read_text()
        assert "fallback" in csv_text.lower(), "CSV must surface fallback warning"

    def test_fallback_warning_in_json(self, tmp_path: Path) -> None:
        model_p = _make_metrics_json(tmp_path / "model", "niche-lens-st")
        _make_fallback_run_metadata(model_p)
        out = emit_leaderboard(model_p, baselines_glob="", out_dir=tmp_path / "out")
        data = json.loads(out["json"].read_text())
        assert data.get("fallback_warning"), "JSON must have non-empty fallback_warning"

    def test_no_fallback_warning_when_absent(self, tmp_path: Path) -> None:
        """No run_metadata.json -> no fallback warning anywhere."""
        model_p = _make_metrics_json(tmp_path / "model", "niche-lens-st")
        # Deliberately do NOT write run_metadata.json
        out = emit_leaderboard(model_p, baselines_glob="", out_dir=tmp_path / "out")
        data = json.loads(out["json"].read_text())
        assert not data.get("fallback_warning"), "Should have no fallback_warning"

    def test_fallback_not_paper_claim_ready_in_json(self, tmp_path: Path) -> None:
        """JSON fallback_warning implies the table is not paper-claim-ready."""
        model_p = _make_metrics_json(tmp_path / "model", "niche-lens-st")
        _make_fallback_run_metadata(model_p)
        out = emit_leaderboard(model_p, baselines_glob="", out_dir=tmp_path / "out")
        data = json.loads(out["json"].read_text())
        # The JSON must not claim to be paper-ready when fallback
        assert data.get("paper_claim_ready") is not True


# ---------------------------------------------------------------------------
# TestStubGating
# ---------------------------------------------------------------------------


class TestStubGating:
    """paper_metrics stub gating: fallback -> paper_claim_ready stays false."""

    def test_artifact_exists_flipped_after_emit(self, tmp_path: Path) -> None:
        model_p = _make_metrics_json(tmp_path / "model", "niche-lens-st")
        stub_p = _make_stub(tmp_path / "stubs")
        emit_leaderboard(
            model_p,
            baselines_glob="",
            out_dir=tmp_path / "out",
            stub_path=stub_p,
        )
        data = json.loads(stub_p.read_text())
        assert data["artifact_exists"] is True

    def test_source_exists_flipped_after_emit(self, tmp_path: Path) -> None:
        model_p = _make_metrics_json(tmp_path / "model", "niche-lens-st")
        stub_p = _make_stub(tmp_path / "stubs")
        emit_leaderboard(
            model_p,
            baselines_glob="",
            out_dir=tmp_path / "out",
            stub_path=stub_p,
        )
        data = json.loads(stub_p.read_text())
        assert data["source_exists"] is True

    def test_source_script_reference_set(self, tmp_path: Path) -> None:
        model_p = _make_metrics_json(tmp_path / "model", "niche-lens-st")
        stub_p = _make_stub(tmp_path / "stubs")
        emit_leaderboard(
            model_p,
            baselines_glob="",
            out_dir=tmp_path / "out",
            stub_path=stub_p,
        )
        data = json.loads(stub_p.read_text())
        assert "emit_results_tables" in data["source_script_reference"]

    def test_fallback_keeps_paper_claim_ready_false(self, tmp_path: Path) -> None:
        model_p = _make_metrics_json(tmp_path / "model", "niche-lens-st")
        _make_fallback_run_metadata(model_p)
        stub_p = _make_stub(tmp_path / "stubs")
        emit_leaderboard(
            model_p,
            baselines_glob="",
            out_dir=tmp_path / "out",
            stub_path=stub_p,
        )
        data = json.loads(stub_p.read_text())
        assert data["paper_claim_ready"] is False

    def test_non_fallback_keeps_paper_claim_ready_false(self, tmp_path: Path) -> None:
        """Even non-fallback single-method runs must not auto-promote to claim-ready."""
        model_p = _make_metrics_json(tmp_path / "model", "niche-lens-st")
        stub_p = _make_stub(tmp_path / "stubs")
        emit_leaderboard(
            model_p,
            baselines_glob="",
            out_dir=tmp_path / "out",
            stub_path=stub_p,
        )
        data = json.loads(stub_p.read_text())
        assert data["paper_claim_ready"] is False

    def test_stub_not_modified_when_not_provided(self, tmp_path: Path) -> None:
        """When stub_path is None and no canonical stub exists, no file is written."""
        model_p = _make_metrics_json(tmp_path / "model", "niche-lens-st")
        # Pass a non-existent stub_path so the function doesn't find the real repo stub
        nonexistent = tmp_path / "does_not_exist" / "n-t1_metrics_stub.json"
        out = emit_leaderboard(
            model_p,
            baselines_glob="",
            out_dir=tmp_path / "out",
            stub_path=nonexistent,  # doesn't exist -> silently skipped
        )
        assert not nonexistent.exists()
        assert out["csv"].exists()


# ---------------------------------------------------------------------------
# TestColumnContract
# ---------------------------------------------------------------------------


class TestColumnContract:
    """LEADERBOARD_COLS must be present in the header and in JSON rows."""

    def test_csv_header_has_all_cols(self, tmp_path: Path) -> None:
        model_p = _make_metrics_json(tmp_path / "model", "niche-lens-st")
        out = emit_leaderboard(model_p, baselines_glob="", out_dir=tmp_path / "out")
        header = _csv_data_lines(out["csv"].read_text())[0]
        cols = header.split(",")
        for col in LEADERBOARD_COLS:
            assert col in cols, f"Expected column {col!r} in CSV header"

    def test_json_has_columns_field(self, tmp_path: Path) -> None:
        model_p = _make_metrics_json(tmp_path / "model", "niche-lens-st")
        out = emit_leaderboard(model_p, baselines_glob="", out_dir=tmp_path / "out")
        data = json.loads(out["json"].read_text())
        assert "columns" in data
        for col in LEADERBOARD_COLS:
            assert col in data["columns"], f"Expected {col!r} in JSON columns"

    def test_json_rows_have_metrics_dict(self, tmp_path: Path) -> None:
        model_p = _make_metrics_json(tmp_path / "model", "niche-lens-st")
        out = emit_leaderboard(model_p, baselines_glob="", out_dir=tmp_path / "out")
        data = json.loads(out["json"].read_text())
        row = data["rows"][0]
        assert "metrics" in row
        for col in LEADERBOARD_COLS:
            assert col in row["metrics"], f"Expected {col!r} in row metrics"

    def test_json_has_provenance(self, tmp_path: Path) -> None:
        model_p = _make_metrics_json(tmp_path / "model", "niche-lens-st")
        out = emit_leaderboard(model_p, baselines_glob="", out_dir=tmp_path / "out")
        data = json.loads(out["json"].read_text())
        assert "provenance" in data
        assert "niche-lens-st" in data["provenance"]
        prov = data["provenance"]["niche-lens-st"]
        assert "source_path" in prov
        assert "git_sha" in prov


# ---------------------------------------------------------------------------
# TestRealMetricsFixture
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _MODEL_METRICS.exists(),
    reason=(
        "results/niche-lens-st/metrics.json is gitignored (local-run artifact); "
        "absent in CI"
    ),
)
class TestRealMetricsFixture:
    """Smoke test against the committed results/niche-lens-st/metrics.json.

    stub_path is omitted (defaults to None) so emit_leaderboard never touches
    the canonical results/paper_metrics/n-t1_metrics_stub.json during tests.
    The real stub is updated only when the CLI is run explicitly (VERIFY step 4).
    Entire class is skipped in CI where the gitignored file is absent.
    """

    def test_runs_on_committed_metrics(self, tmp_path: Path) -> None:
        assert _MODEL_METRICS.exists(), f"committed metrics.json not found: {_MODEL_METRICS}"
        out = emit_leaderboard(_MODEL_METRICS, baselines_glob="", out_dir=tmp_path / "out")
        assert out["csv"].exists()
        assert out["md"].exists()
        assert out["json"].exists()

    def test_committed_metrics_one_row(self, tmp_path: Path) -> None:
        out = emit_leaderboard(_MODEL_METRICS, baselines_glob="", out_dir=tmp_path / "out")
        data = json.loads(out["json"].read_text())
        assert len(data["rows"]) == 1
        assert data["rows"][0]["method"] == "niche-lens-st"

    def test_committed_metrics_has_all_leaderboard_cols(self, tmp_path: Path) -> None:
        out = emit_leaderboard(_MODEL_METRICS, baselines_glob="", out_dir=tmp_path / "out")
        header = _csv_data_lines(out["csv"].read_text())[0]
        cols = header.split(",")
        assert cols[0] == "method"
        for col in LEADERBOARD_COLS:
            assert col in cols, f"Expected column {col!r} in CSV header"

    def test_committed_metrics_detected_as_fallback(self, tmp_path: Path) -> None:
        """The committed run is a fallback; JSON must carry fallback_warning."""
        out = emit_leaderboard(_MODEL_METRICS, baselines_glob="", out_dir=tmp_path / "out")
        data = json.loads(out["json"].read_text())
        assert data.get("fallback_warning"), (
            "Committed metrics.json is a single-section fallback run; "
            "fallback_warning must be set in the JSON output"
        )

    def test_schema_version_present(self, tmp_path: Path) -> None:
        out = emit_leaderboard(_MODEL_METRICS, baselines_glob="", out_dir=tmp_path / "out")
        data = json.loads(out["json"].read_text())
        assert "schema_version" in data
