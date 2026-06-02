"""conserved_fraction is undefined (None) for single-section runs (issue #150).

The real-data runner previously reported a misleading numeric conserved_fraction
for single-section inputs (every prototype is tagged "unknown" by the separation
head, so the old denominator made it 0.0). It must be ``None`` instead, and
"unknown" tags must be excluded from the denominator in the multi-section case.

`scripts/run_real_niche.py` only imports numpy + stdlib at module scope (scanpy /
nichelens_st imports are lazy), so it loads cheaply here.
"""

import importlib.util
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[1]


def _load_runner():
    path = REPO / "scripts" / "run_real_niche.py"
    spec = importlib.util.spec_from_file_location("run_real_niche", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_single_section_returns_none():
    m = _load_runner()
    assert m._conserved_fraction(["unknown", "unknown"], 1) is None


def test_multi_section_excludes_unknown():
    m = _load_runner()
    # 1 conserved of 2 scorable -> 0.5; the "unknown" tag is excluded.
    assert m._conserved_fraction(["conserved", "sample_specific", "unknown"], 3) == 0.5


def test_all_unknown_multi_section_none():
    m = _load_runner()
    assert m._conserved_fraction(["unknown", "unknown"], 3) is None


def test_empty_proto_kind_none():
    m = _load_runner()
    assert m._conserved_fraction([], 3) is None
