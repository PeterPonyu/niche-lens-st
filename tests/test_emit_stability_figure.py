"""Smoke tests for the #296 stability figure emitter (renders a 3-panel PNG)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

pytest.importorskip("matplotlib")
esf = pytest.importorskip("emit_stability_figure")


def _write(tmp_path: Path, payload: dict) -> Path:
    p = tmp_path / "niche_stability.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


def test_renders_multisection_figure(tmp_path):
    payload = {
        "n_sections": 5,
        "n_seeds": 3,
        "seed_stability_ari": 0.62,
        "seed_stability_ari_sd": 0.05,
        "ari_matrix": [[1.0, 0.6, 0.65], [0.6, 1.0, 0.61], [0.65, 0.61, 1.0]],
        "coverage_sweep": [
            {"min_section_coverage": 1.0, "conserved_fraction": 0.3, "section_overlap_rate": 0.8,
             "n_conserved": 3, "n_prototypes": 10},
            {"min_section_coverage": 0.8, "conserved_fraction": 0.6, "section_overlap_rate": 0.7,
             "n_conserved": 6, "n_prototypes": 10},
        ],
        "prototype_matching_seed0_seed1": [
            {"proto_a": 0, "proto_b": 1, "overlap": 50},
            {"proto_a": 1, "proto_b": 0, "overlap": 40},
        ],
        "paper_claim_ready": False,
    }
    out = esf.emit_stability_figure(_write(tmp_path, payload), out_dir=tmp_path)
    assert out["figure"].exists()
    assert out["figure"].stat().st_size > 0


def test_single_section_degeneracy_does_not_crash(tmp_path):
    payload = {
        "n_sections": 1,
        "n_seeds": 5,
        "seed_stability_ari": 0.35,
        "seed_stability_ari_sd": 0.12,
        "ari_matrix": [[1.0, 0.3], [0.3, 1.0]],
        "coverage_sweep": [
            {"min_section_coverage": 1.0, "conserved_fraction": None,
             "section_overlap_rate": float("nan"), "n_conserved": 0, "n_prototypes": 10},
        ],
        "prototype_matching_seed0_seed1": [{"proto_a": 0, "proto_b": 0, "overlap": 30}],
        "paper_claim_ready": False,
    }
    out = esf.emit_stability_figure(_write(tmp_path, payload), out_dir=tmp_path)
    assert out["figure"].exists()
