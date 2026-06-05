"""Baseline discovery for the N-T1 leaderboard.

run_baselines_niche writes each baseline under
``results/baselines/<baseline>/<project>/metrics.json`` (write_results nests a
``<project>/`` subdir), and records the baseline tag in run_metadata's
``interpretability.baseline``. The leaderboard must discover those nested files
(recursive glob) and name each row by the recorded tag, not the parent dir
(which is the project name and would collide across baselines).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = REPO_ROOT / "scripts" / "emit_results_tables.py"
sys.path.insert(0, str(REPO_ROOT / "scripts"))

_spec = importlib.util.spec_from_file_location("emit_results_tables", _SCRIPT)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)

emit_leaderboard = _mod.emit_leaderboard
_baseline_method = _mod._baseline_method


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _metrics(project: str, **vals) -> dict:
    base = {
        "domain_ari": None,
        "embedding_silhouette": 0.2,
        "niche_morans_i": 0.4,
    }
    base.update(vals)
    return {
        "schema_version": "1.0.0",
        "project": project,
        "dataset_card_id": "niche_merfish_slice",
        "metrics": base,
        "git_sha": "deadbeef",
    }


def _make_nested_baseline(root: Path, baseline: str, **vals) -> Path:
    """Mirror run_baselines_niche: results/baselines/<baseline>/<project>/metrics.json."""
    d = root / "baselines" / baseline / "niche-lens-st"
    _write(d / "metrics.json", _metrics("niche-lens-st", **vals))
    _write(
        d / "run_metadata.json",
        {"interpretability": {"model_is_learned": False, "baseline": baseline}},
    )
    return d / "metrics.json"


def test_discovers_nested_baselines_and_names_by_tag(tmp_path: Path) -> None:
    model = tmp_path / "model" / "metrics.json"
    _write(model, _metrics("niche-lens-st", embedding_silhouette=0.24, niche_morans_i=0.42))
    _make_nested_baseline(tmp_path, "neighborhood", embedding_silhouette=0.10)
    _make_nested_baseline(tmp_path, "pca", embedding_silhouette=0.05)
    _make_nested_baseline(tmp_path, "diffusion", embedding_silhouette=0.12)

    out = emit_leaderboard(
        model,
        baselines_glob=str(tmp_path / "baselines" / "**" / "metrics.json"),
        out_dir=tmp_path / "out",
    )
    data = json.loads(out["json"].read_text())
    methods = [r["method"] for r in data["rows"]]
    # model first, then baselines sorted lexicographically — named by tag, no collision.
    assert methods == ["niche-lens-st", "diffusion", "neighborhood", "pca"]
    assert len({m for m in methods if m != "niche-lens-st"}) == 3


def test_baseline_method_prefers_recorded_tag(tmp_path: Path) -> None:
    mp = _make_nested_baseline(tmp_path, "diffusion")
    assert _baseline_method(mp) == "diffusion"  # tag, not the "niche-lens-st" parent dir


def test_baseline_method_falls_back_to_dir_under_baselines(tmp_path: Path) -> None:
    # No run_metadata sidecar -> use the first path component under baselines/.
    mp = tmp_path / "baselines" / "pca" / "niche-lens-st" / "metrics.json"
    _write(mp, _metrics("niche-lens-st"))
    assert _baseline_method(mp) == "pca"


def test_model_only_when_no_baselines(tmp_path: Path) -> None:
    model = tmp_path / "model" / "metrics.json"
    _write(model, _metrics("niche-lens-st"))
    out = emit_leaderboard(
        model, baselines_glob=str(tmp_path / "baselines" / "**" / "metrics.json"),
        out_dir=tmp_path / "out",
    )
    data = json.loads(out["json"].read_text())
    assert [r["method"] for r in data["rows"]] == ["niche-lens-st"]
