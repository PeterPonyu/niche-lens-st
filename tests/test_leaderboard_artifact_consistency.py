"""CI guard for the committed results/paper_metrics/leaderboard.json artifact.

These tests load the committed leaderboard.json (which IS tracked in git) and
assert internal self-consistency.  They run in CI where metrics.json is absent.
The single drift-check test (test_source_sha256_matches_live_metrics) is
skip-guarded for CI and runs only when metrics.json is present locally.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
_LEADERBOARD_JSON = REPO_ROOT / "results" / "paper_metrics" / "leaderboard.json"
_MODEL_METRICS = REPO_ROOT / "results" / "niche-lens-st" / "metrics.json"

# Load LEADERBOARD_COLS from the emit script (single source of truth).
_SCRIPT = REPO_ROOT / "scripts" / "emit_results_tables.py"
sys.path.insert(0, str(REPO_ROOT / "scripts"))
_spec = importlib.util.spec_from_file_location("emit_results_tables", _SCRIPT)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)
LEADERBOARD_COLS = _mod.LEADERBOARD_COLS

_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load() -> dict:
    assert _LEADERBOARD_JSON.exists(), f"committed leaderboard.json missing: {_LEADERBOARD_JSON}"
    return json.loads(_LEADERBOARD_JSON.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Internal self-consistency tests (CI-safe — no metrics.json needed)
# ---------------------------------------------------------------------------


def test_leaderboard_json_is_valid_json() -> None:
    assert _LEADERBOARD_JSON.exists(), f"committed leaderboard.json not found: {_LEADERBOARD_JSON}"
    data = json.loads(_LEADERBOARD_JSON.read_text(encoding="utf-8"))
    assert isinstance(data, dict)


def test_schema_version_present() -> None:
    data = _load()
    assert "schema_version" in data, "schema_version field missing"
    assert isinstance(data["schema_version"], str)


def test_columns_match_leaderboard_cols() -> None:
    data = _load()
    assert "columns" in data, "columns field missing"
    assert tuple(data["columns"]) == LEADERBOARD_COLS, (
        f"columns mismatch: got {data['columns']!r}, expected {list(LEADERBOARD_COLS)!r}"
    )


def test_paper_claim_ready_is_false() -> None:
    data = _load()
    assert data.get("paper_claim_ready") is False, (
        f"paper_claim_ready must be False, got {data.get('paper_claim_ready')!r}"
    )


def test_fallback_warning_is_nonempty_string() -> None:
    data = _load()
    fw = data.get("fallback_warning")
    assert isinstance(fw, str) and fw, (
        "fallback_warning must be a non-empty string (committed run is a fallback)"
    )


def test_dataset_card_id_is_niche_merfish_slice() -> None:
    data = _load()
    assert data.get("dataset_card_id") == "niche_merfish_slice", (
        f"dataset_card_id mismatch: {data.get('dataset_card_id')!r}"
    )


def test_source_sha256_is_64hex() -> None:
    data = _load()
    sha = data.get("source_sha256", "")
    assert _HEX64_RE.match(str(sha)), (
        f"source_sha256 must be a 64-character hex string, got {sha!r}"
    )


def test_rows_present() -> None:
    data = _load()
    assert "rows" in data and isinstance(data["rows"], list), "rows must be a list"
    assert len(data["rows"]) >= 1, "at least one row (the model) must be present"


def test_model_row_is_first() -> None:
    data = _load()
    assert data["rows"][0]["method"] == "niche-lens-st", (
        f"first row must be 'niche-lens-st', got {data['rows'][0]['method']!r}"
    )


def test_all_rows_have_all_leaderboard_columns() -> None:
    data = _load()
    for row in data["rows"]:
        method = row.get("method", "<unknown>")
        metrics = row.get("metrics", {})
        for col in LEADERBOARD_COLS:
            assert col in metrics, f"row {method!r} missing column {col!r}"


def test_metric_values_are_float_or_null_not_zero_string() -> None:
    """Values must be float, int, or None/null — never the string '0' or similar."""
    data = _load()
    for row in data["rows"]:
        method = row.get("method", "<unknown>")
        for col, val in row.get("metrics", {}).items():
            assert not isinstance(val, str), (
                f"row {method!r} col {col!r}: value is a string {val!r}; "
                "must be float/int/null"
            )
            # null (None) is fine; floats/ints are fine; strings are not.


def test_provenance_present_for_model() -> None:
    data = _load()
    prov = data.get("provenance", {})
    assert "niche-lens-st" in prov, "provenance must have 'niche-lens-st' entry"
    model_prov = prov["niche-lens-st"]
    assert "source_path" in model_prov
    assert "git_sha" in model_prov
    assert "source_sha256" in model_prov, (
        "model provenance must include source_sha256"
    )
    assert _HEX64_RE.match(str(model_prov["source_sha256"])), (
        f"provenance source_sha256 must be 64-hex, got {model_prov['source_sha256']!r}"
    )


# ---------------------------------------------------------------------------
# Drift-check test (local-only — skipped in CI where metrics.json is absent)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _MODEL_METRICS.exists(),
    reason=(
        "results/niche-lens-st/metrics.json is gitignored (local-run artifact); "
        "absent in CI — drift check skipped"
    ),
)
def test_source_sha256_matches_live_metrics() -> None:
    """Detect if metrics.json was edited without re-running emit_results_tables.py."""
    actual_sha = hashlib.sha256(_MODEL_METRICS.read_bytes()).hexdigest()
    data = _load()
    committed_sha = data.get("source_sha256", "")
    assert actual_sha == committed_sha, (
        "metrics.json has been modified without re-emitting the leaderboard. "
        f"Run: python scripts/emit_results_tables.py\n"
        f"  committed sha256 = {committed_sha!r}\n"
        f"  live metrics sha256 = {actual_sha!r}"
    )
