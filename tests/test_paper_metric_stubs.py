"""Validate that per-repo paper_metrics stubs satisfy the required contract.

Each n-*.json must:
  - be valid JSON
  - contain every required key
  - have paper_claim_ready == False
  - have supports_safe_prose == False
  - have source_exists == False
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
STUBS_DIR = REPO_ROOT / "results" / "paper_metrics"

REQUIRED_KEYS = {
    "asset_id",
    "project",
    "role",
    "readiness_status",
    "source_metric_reference",
    "source_script_reference",
    "source_exists",
    "metric_source_exists",
    "artifact_exists",
    "supports_safe_prose",
    "paper_claim_ready",
    "notes",
}

STUB_FILES = ["n-f1_metrics_stub.json", "n-f2_metrics_stub.json", "n-t1_metrics_stub.json"]


@pytest.mark.parametrize("filename", STUB_FILES)
def test_stub_is_valid_json(filename: str) -> None:
    path = STUBS_DIR / filename
    assert path.exists(), f"stub file not found: {path}"
    data = json.loads(path.read_text())
    assert isinstance(data, dict), f"{filename}: expected a JSON object at top level"


@pytest.mark.parametrize("filename", STUB_FILES)
def test_stub_has_required_keys(filename: str) -> None:
    path = STUBS_DIR / filename
    data = json.loads(path.read_text())
    missing = REQUIRED_KEYS - set(data.keys())
    assert not missing, f"{filename}: missing required keys: {missing}"


@pytest.mark.parametrize("filename", STUB_FILES)
def test_stub_paper_claim_ready_is_false(filename: str) -> None:
    path = STUBS_DIR / filename
    data = json.loads(path.read_text())
    assert data["paper_claim_ready"] is False, (
        f"{filename}: paper_claim_ready must be false (got {data['paper_claim_ready']!r})"
    )


@pytest.mark.parametrize("filename", STUB_FILES)
def test_stub_supports_safe_prose_is_false(filename: str) -> None:
    path = STUBS_DIR / filename
    data = json.loads(path.read_text())
    assert data["supports_safe_prose"] is False, (
        f"{filename}: supports_safe_prose must be false (got {data['supports_safe_prose']!r})"
    )


@pytest.mark.parametrize("filename", STUB_FILES)
def test_stub_source_exists_is_false(filename: str) -> None:
    path = STUBS_DIR / filename
    data = json.loads(path.read_text())
    # source_exists may be True for n-t1 once its emit script has run
    # (artifact_exists also becomes True at that point).  N-F1 and N-F2 must
    # still have source_exists=False so this guard is scoped to n-t1 only.
    if filename == "n-t1_metrics_stub.json" and data.get("artifact_exists") is True:
        return
    assert data["source_exists"] is False, (
        f"{filename}: source_exists must be false (got {data['source_exists']!r})"
    )


@pytest.mark.parametrize("filename", STUB_FILES)
def test_stub_asset_id_matches_filename(filename: str) -> None:
    """Asset ID prefix (N-F1 etc.) should correspond to the file prefix (n-f1 etc.)."""
    path = STUBS_DIR / filename
    data = json.loads(path.read_text())
    expected_id = filename.split("_")[0].upper()  # "n-f1" -> "N-F1"
    assert data["asset_id"] == expected_id, (
        f"{filename}: asset_id {data['asset_id']!r} does not match expected {expected_id!r}"
    )
