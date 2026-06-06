"""Niche-abundance to per-sample outcome association wired into the runner
(#320 / #339).

``run_real_niche.compute_outcome_table`` is the pure bridge between a per-cell
prototype assignment and the per-sample association test in
:mod:`nichelens_st.outcome`: it reduces per-cell ``(sample_id, prototype_id)`` to
a per-sample abundance matrix, pulls the per-sample outcome (which must be
constant within a sample), and runs the test. Like the supervised table (#288)
it is a SECONDARY artifact -- written to its own ``outputs/`` file, never folded
into the intrinsic headline ``metrics.json`` -- and the real biological claim is
data-gated. These tests need no scanpy/real data.
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


def _planted_per_cell(seed: int = 0, cells_per_sample: int = 80):
    """Per-cell (sample_id, prototype_id, outcome) for 12 samples x 4 prototypes.

    Prototype 0 is enriched in the 6 outcome=1 samples and depleted in the 6
    outcome=0 samples; the outcome is constant within each sample (it is a
    per-sample label, expanded onto that sample's cells)."""
    rng = np.random.default_rng(seed)
    sample_ids, proto_ids, outcomes = [], [], []
    for s in range(12):
        positive = s < 6
        # planted prototype proportions; proto 0 high iff positive
        p = np.array([0.5, 0.2, 0.15, 0.15]) if positive else np.array(
            [0.1, 0.4, 0.25, 0.25]
        )
        protos = rng.choice(4, size=cells_per_sample, p=p)
        sample_ids.append(np.full(cells_per_sample, s))
        proto_ids.append(protos)
        outcomes.append(np.full(cells_per_sample, 1 if positive else 0))
    return (
        np.concatenate(sample_ids),
        np.concatenate(proto_ids),
        np.concatenate(outcomes),
    )


def test_compute_outcome_table_recovers_planted_association():
    sample_id, prototype_id, outcome = _planted_per_cell()
    payload = runner.compute_outcome_table(
        prototype_id, sample_id, outcome, label_kind="binary"
    )
    assert payload is not None
    proto = {r["prototype"]: r for r in payload["prototypes"]}
    assert payload["n_samples"] == 12
    assert proto[0]["q_value"] < 0.05
    assert proto[0]["effect_size"] > 0.0


def test_compute_outcome_table_auto_detects_binary():
    sample_id, prototype_id, outcome = _planted_per_cell()
    payload = runner.compute_outcome_table(
        prototype_id, sample_id, outcome, label_kind="auto"
    )
    assert payload is not None
    assert payload["label_kind"] == "binary"


def test_compute_outcome_table_none_with_single_sample():
    # one sample -> no between-sample contrast possible
    prototype_id = np.array([0, 1, 1, 0, 2])
    sample_id = np.zeros(5, dtype=int)
    outcome = np.ones(5, dtype=int)
    assert (
        runner.compute_outcome_table(prototype_id, sample_id, outcome) is None
    )


def test_compute_outcome_table_none_when_label_has_one_group():
    sample_id, prototype_id, _ = _planted_per_cell()
    outcome = np.ones_like(sample_id)  # every sample the same label
    assert (
        runner.compute_outcome_table(
            prototype_id, sample_id, outcome, label_kind="binary"
        )
        is None
    )


def test_compute_outcome_table_rejects_inconsistent_within_sample_outcome():
    # sample 0 has two different outcome values -> a data bug, not a silent guess
    prototype_id = np.array([0, 1, 0, 1])
    sample_id = np.array([0, 0, 1, 1])
    outcome = np.array([1, 0, 1, 1])  # sample 0: {1, 0}
    with pytest.raises(ValueError):
        runner.compute_outcome_table(prototype_id, sample_id, outcome)


def test_compute_outcome_table_payload_is_json_serialisable():
    import json

    sample_id, prototype_id, outcome = _planted_per_cell()
    payload = runner.compute_outcome_table(prototype_id, sample_id, outcome)
    json.dumps(payload)
