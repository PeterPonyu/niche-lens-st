"""In-lane fallback detection via run_metadata.notes.

The structured ``_fallback_note`` key is dropped by the byte-locked
results_contract until META adds it, but the runner records the downsized-
fallback marker (``dataset=fallback``) in the free-text ``notes`` field, which
the contract passes through. Both emitters must detect fallback there so a
freshly-regenerated artifact still carries the "not the atlas run" warning.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_tables = _load("emit_results_tables")


def _write_meta(tmp_path: Path, **fields) -> Path:
    p = tmp_path / "run_metadata.json"
    p.write_text(json.dumps(fields), encoding="utf-8")
    return p


# --- emit_results_tables._detect_fallback (keyed on metrics_path's sibling) ---


def test_tables_detects_fallback_from_notes(tmp_path: Path) -> None:
    _write_meta(
        tmp_path,
        notes="single_section=True; dataset=fallback (5488 cells); fit_runtime_s=1.4",
    )
    is_fb, warning = _tables._detect_fallback(tmp_path / "metrics.json")
    assert is_fb is True
    assert warning and "ATLAS" in warning.upper()


def test_tables_structured_note_takes_precedence(tmp_path: Path) -> None:
    _write_meta(tmp_path, _fallback_note="STRUCTURED NOTE", notes="dataset=fallback")
    is_fb, warning = _tables._detect_fallback(tmp_path / "metrics.json")
    assert is_fb is True
    assert warning == "STRUCTURED NOTE"


def test_tables_non_fallback_notes_not_flagged(tmp_path: Path) -> None:
    _write_meta(tmp_path, notes="dataset=primary (124938 cells); single_section=True")
    is_fb, warning = _tables._detect_fallback(tmp_path / "metrics.json")
    assert is_fb is False
    assert warning == ""


# --- emit_figures._detect_fallback (keyed on the run_metadata path directly) ---


def test_figures_detects_fallback_from_notes(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    figs = _load("emit_figures")
    meta = _write_meta(tmp_path, notes="dataset=fallback (5488 cells)")
    is_fb, warning = figs._detect_fallback(meta)
    assert is_fb is True
    assert warning and "ATLAS" in warning.upper()


def test_figures_non_fallback_not_flagged(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    figs = _load("emit_figures")
    meta = _write_meta(tmp_path, notes="dataset=primary (5488 cells)")
    is_fb, _ = figs._detect_fallback(meta)
    assert is_fb is False
