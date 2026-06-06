"""Secondary (caveated) supervised niche-recovery metrics for ``run_real_niche.py``
(#288).

The CODEX spleen primary ships author dual ground truth (``obs['niche']`` 101
clusters incl. ``nan``; ``obs['cell_type']`` 58 clusters). Both are
clustering-derived, so per the project's intrinsic-metrics-first / GT-skeptic
rule these ARI/NMI/AMI/macro-F1 scores are a **secondary** artifact written to
``outputs/supervised_metrics.json`` -- the headline ``metrics.json`` stays
intrinsic-only.

The pure scoring helpers (``_unlabeled_mask`` / ``supervised_scores`` /
``compute_supervised_table``) need NO scanpy/real-data; only the integration
driver is scanpy-guarded.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO_ROOT / "scripts" / "run_real_niche.py"
sys.path.insert(0, str(_REPO_ROOT / "src"))

_spec = importlib.util.spec_from_file_location("run_real_niche", _SCRIPT)
runner = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(runner)


# --------------------------------------------------------------------------
# _unlabeled_mask -- NaN / None / 'nan' string / '' are unlabeled
# --------------------------------------------------------------------------


def test_unlabeled_mask_variants():
    vals = np.array(["6.0", "nan", "29.0", "", "NaN", "96.0"], dtype=object)
    mask = runner._unlabeled_mask(vals)
    assert mask.tolist() == [False, True, False, True, True, False]


def test_unlabeled_mask_real_nan_and_none():
    vals = np.array([1.0, np.nan, 2.0, None, 3.0], dtype=object)
    mask = runner._unlabeled_mask(vals)
    assert mask.tolist() == [False, True, False, True, False]


# --------------------------------------------------------------------------
# supervised_scores -- excludes unlabeled, factorizes ref, returns metrics
# --------------------------------------------------------------------------


def test_supervised_scores_perfect_agreement():
    pred = np.array([0, 0, 1, 1, 2, 2])
    ref = np.array(["a", "a", "b", "b", "c", "c"], dtype=object)
    sc = runner.supervised_scores(pred, ref)
    assert sc["ari"] == pytest.approx(1.0)
    assert sc["nmi"] == pytest.approx(1.0)
    assert sc["ami"] == pytest.approx(1.0)
    assert sc["macro_f1"] == pytest.approx(1.0)
    assert sc["n_scored"] == 6
    assert sc["n_unlabeled_excluded"] == 0
    assert sc["n_ref_classes"] == 3


def test_supervised_scores_excludes_unlabeled_from_score_only():
    # Two 'nan' cells must be dropped from the SUPERVISED score (n_scored=6).
    pred = np.array([0, 0, 1, 1, 2, 2, 9, 9])
    ref = np.array(["a", "a", "b", "b", "c", "c", "nan", "nan"], dtype=object)
    sc = runner.supervised_scores(pred, ref)
    assert sc["n_scored"] == 6
    assert sc["n_unlabeled_excluded"] == 2
    assert sc["n_ref_classes"] == 3
    assert sc["ari"] == pytest.approx(1.0)


def test_supervised_scores_none_when_too_few_classes():
    pred = np.array([0, 1, 2])
    ref = np.array(["a", "a", "nan"], dtype=object)  # one class after exclusion
    assert runner.supervised_scores(pred, ref) is None


# --------------------------------------------------------------------------
# compute_supervised_table -- model prototypes + matched-K k-means rows
# --------------------------------------------------------------------------


def test_compute_supervised_table_rows():
    rng = np.random.default_rng(0)
    # 3 well-separated clusters in embedding space, matching the ref labels.
    centers = np.array([[5, 0], [-5, 0], [0, 5]], dtype=np.float64)
    ref_ids = np.repeat([0, 1, 2], 40)
    H = (centers[ref_ids] + 0.1 * rng.standard_normal((120, 2))).astype(np.float32)
    proto = ref_ids.astype(np.int64)  # model prototypes == truth here
    ref = np.array([["a", "b", "c"][i] for i in ref_ids], dtype=object)

    rows = runner.compute_supervised_table(
        H, proto, {"niche": ref}, seeds=(0, 1), kmeans_iters=20
    )
    assert len(rows) >= 1
    assigns = {r["assignment"] for r in rows}
    assert "model_prototypes" in assigns
    assert "kmeans_matched_k" in assigns
    # matched-K rows use k == n_ref_classes (3) and carry a seed.
    km = [r for r in rows if r["assignment"] == "kmeans_matched_k"]
    assert {r["seed"] for r in km} == {0, 1}
    assert all(r["k"] == 3 and r["reference"] == "niche" for r in km)
    # near-perfect recovery on cleanly separated clusters
    proto_row = next(r for r in rows if r["assignment"] == "model_prototypes")
    assert proto_row["ari"] == pytest.approx(1.0)


# --------------------------------------------------------------------------
# CLI flag
# --------------------------------------------------------------------------


def test_supervised_metrics_cli_flag():
    parser = runner._build_parser()
    assert parser.parse_args([]).supervised_metrics is False
    assert parser.parse_args(["--supervised-metrics"]).supervised_metrics is True
