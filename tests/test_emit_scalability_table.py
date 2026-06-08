"""Tests for the #315 cohort-scale scalability TABLE emitter.

Distinct from the N-F2 *figure* (#289/test_emit_figures): this is the numeric
table (csv + md + json) with explicit InfoNCE-mode and OOM/fallback status
columns. Honesty contract: a missing peak_rss / runtime renders as an em-dash,
NEVER a fabricated 0, and blocked regimes (OOM, download-blocked) carry their
declared status with no invented numbers.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

est = pytest.importorskip("emit_scalability_table")


def _write_meta(tmp_path: Path, name: str, **fields) -> Path:
    p = tmp_path / f"{name}.json"
    p.write_text(json.dumps(fields), encoding="utf-8")
    return p


# --------------------------------------------------------------------------
# Pure helpers
# --------------------------------------------------------------------------
def test_infonce_mode_full_batch_when_no_batch_size():
    assert est._infonce_mode({"batch_size": None}) == "full-batch"
    assert est._infonce_mode({"batch_size": 0}) == "full-batch"
    assert est._infonce_mode({}) == "full-batch"


def test_infonce_mode_minibatch_reports_size():
    assert est._infonce_mode({"batch_size": 4096}) == "minibatch(bs=4096)"


def test_status_ok_vs_fallback():
    assert est._status({"notes": "dataset=primary"}) == "ok"
    assert est._status({"_fallback_note": "DOWNSIZED ..."}) == "fallback-to-slice"
    assert est._status({"notes": "dataset=fallback (5488 cells)"}) == "fallback-to-slice"


def test_row_from_metadata_extracts_and_computes_throughput():
    meta = {
        "dataset_card_id": "niche_merfish_slice",
        "n_obs": 5488,
        "runtime_s": 2.0,
        "peak_rss_bytes": 2_000_000_000,
        "batch_size": None,
        "notes": "dataset=fallback",
    }
    row = est.row_from_metadata(meta)
    assert row["dataset"] == "niche_merfish_slice"
    assert row["n_cells"] == 5488
    assert row["wall_clock_s"] == 2.0
    assert row["peak_rss_gb"] == pytest.approx(2.0)  # 2e9 bytes -> 2.0 GB
    assert row["throughput_cells_per_s"] == pytest.approx(2744.0)
    assert row["infonce_mode"] == "full-batch"
    assert row["status"] == "fallback-to-slice"


def test_row_missing_peak_rss_is_none_never_zero():
    meta = {"dataset_card_id": "x", "n_obs": 100, "runtime_s": 1.0}
    row = est.row_from_metadata(meta)
    assert row["peak_rss_gb"] is None  # NOT 0.0


# --------------------------------------------------------------------------
# Aggregation + sort + declared (blocked) regimes
# --------------------------------------------------------------------------
def test_build_rows_sorted_ascending_by_n_cells(tmp_path):
    a = _write_meta(tmp_path, "big", dataset_card_id="big", n_obs=734101, runtime_s=120.0,
                    peak_rss_bytes=8_000_000_000, batch_size=4096, notes="dataset=primary")
    b = _write_meta(tmp_path, "small", dataset_card_id="small", n_obs=5488, runtime_s=1.3,
                    peak_rss_bytes=2_000_000_000, batch_size=None, notes="dataset=fallback")
    rows = est.build_scalability_rows([a, b])
    assert [r["n_cells"] for r in rows] == [5488, 734101]
    assert rows[1]["infonce_mode"] == "minibatch(bs=4096)"


def test_declared_blocked_regime_carries_status_no_fabricated_numbers(tmp_path):
    run = _write_meta(tmp_path, "slice", dataset_card_id="slice", n_obs=5488, runtime_s=1.3,
                      peak_rss_bytes=2_000_000_000, batch_size=None, notes="dataset=fallback")
    declared = [
        {"dataset": "niche_GSE282124_high_plex", "n_cells": 124938,
         "status": "OOM (full-batch)", "infonce_mode": "full-batch",
         "note": "232 GiB full-batch InfoNCE"},
    ]
    rows = est.build_scalability_rows([run], declared_regimes=declared)
    oom = [r for r in rows if r["n_cells"] == 124938][0]
    assert oom["status"] == "OOM (full-batch)"
    assert oom["wall_clock_s"] is None
    assert oom["peak_rss_gb"] is None
    assert oom["throughput_cells_per_s"] is None
    # still sorted ascending with the real row first
    assert [r["n_cells"] for r in rows] == [5488, 124938]


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------
def test_render_markdown_uses_emdash_for_missing_and_lists_columns():
    rows = [
        {"dataset": "slice", "n_cells": 5488, "wall_clock_s": 1.3, "peak_rss_gb": 2.0,
         "throughput_cells_per_s": 4221.0, "infonce_mode": "full-batch", "status": "fallback-to-slice"},
        {"dataset": "atlas", "n_cells": 124938, "wall_clock_s": None, "peak_rss_gb": None,
         "throughput_cells_per_s": None, "infonce_mode": "full-batch", "status": "OOM (full-batch)"},
    ]
    md = est.render_markdown(rows, fallback=True)
    assert "| dataset |" in md
    assert "OOM (full-batch)" in md
    assert "—" in md  # missing cells rendered as em-dash, not 0
    assert "FALLBACK" in md.upper()  # fallback banner present


def test_emit_writes_csv_md_json_with_gate(tmp_path):
    run = _write_meta(tmp_path, "slice", dataset_card_id="slice", n_obs=5488, runtime_s=1.3,
                      peak_rss_bytes=2_000_000_000, batch_size=None, notes="dataset=fallback")
    out = est.emit_scalability_table([run], out_dir=tmp_path)
    for key in ("csv", "md", "json"):
        assert out[key].exists()
    payload = json.loads(out["json"].read_text())
    assert payload["paper_claim_ready"] is False
    assert payload["columns"][0] == "dataset"
    assert payload["rows"][0]["n_cells"] == 5488
    # csv has header + 1 data row
    lines = [ln for ln in out["csv"].read_text().splitlines() if ln and not ln.startswith("#")]
    assert lines[0].startswith("dataset,")
    assert len(lines) == 2
