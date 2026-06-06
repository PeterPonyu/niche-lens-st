"""Section-column resolution + at-scale instrumentation tests (#343/#344).

These pin the PURE, scanpy-free pieces of ``scripts/run_real_niche.py``:

* ``resolve_section_id`` -- operator override + tiling-artifact guard. The
  primary GSE282124 MERSCOPE stores ``obs['fov']`` with 2239 microscope-tile
  levels (~56 cells each); auto-detect must still pick a column but WARN loudly
  that a per-tile kNN graph would be fragmented, never silently proceed. ``none``
  forces a single section; an explicit column name is honored (error if absent).
* ``_peak_rss_bytes`` -- peak-RSS helper (#343) that returns an int or None and
  never raises. Its value is recorded under ``run_metadata['peak_rss_bytes']``,
  the exact key the N-F2 figure consumes.

All tests run WITHOUT scanpy / real data: ``resolve_section_id`` is fed a plain
dict obs + numpy arrays.
"""

from __future__ import annotations

import importlib.util
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

# Import the script module by path (it lives under scripts/, not on sys.path).
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO_ROOT / "scripts" / "run_real_niche.py"
sys.path.insert(0, str(_REPO_ROOT / "src"))

_spec = importlib.util.spec_from_file_location("run_real_niche", _SCRIPT)
runner = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(runner)


def _obs(**cols):
    """A minimal dict obs: column name -> 1-D numpy array of labels."""
    return {k: np.asarray(v) for k, v in cols.items()}


# --------------------------------------------------------------------------
# resolve_section_id -- auto detection (normal column => no warning)
# --------------------------------------------------------------------------


def test_resolve_auto_normal_column_no_warning():
    """3 plausible biological sections of 1000 cells: used, NO warning."""
    labels = np.repeat(["s0", "s1", "s2"], 1000)
    obs = _obs(section_id=labels)
    n_obs = labels.size
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning fails the test
        section_id, col_used, note = runner.resolve_section_id(
            list(obs.keys()), obs, n_obs, section_col_arg="auto"
        )
    assert col_used == "section_id"
    assert section_id.shape == (n_obs,)
    assert int(np.unique(section_id).size) == 3
    assert note is None


def test_resolve_auto_no_candidate_column_single_section():
    """No candidate column present: single section zeros, no warning, no error."""
    obs = _obs(other=np.arange(120))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        section_id, col_used, note = runner.resolve_section_id(
            list(obs.keys()), obs, 120, section_col_arg="auto"
        )
    assert col_used is None
    assert np.array_equal(section_id, np.zeros(120, dtype=np.int64))
    assert note is None


# --------------------------------------------------------------------------
# resolve_section_id -- tiling-artifact guard (record AND warn, never crash)
# --------------------------------------------------------------------------


def test_resolve_auto_tiling_artifact_low_cells_per_level_warns():
    """50 levels over 500 cells (~10/level) => warn AND still return codes."""
    labels = np.repeat(np.arange(50), 10)  # 50 fov-like tiles, 10 cells each
    obs = _obs(fov=labels)
    n_obs = labels.size
    with pytest.warns(UserWarning, match="tiling artifact"):
        section_id, col_used, note = runner.resolve_section_id(
            list(obs.keys()), obs, n_obs, section_col_arg="auto"
        )
    # record-and-warn, NOT crash: codes for every cell are returned.
    assert col_used == "fov"
    assert section_id.shape == (n_obs,)
    assert int(np.unique(section_id).size) == 50
    assert note is not None and "fov" in note


def test_resolve_auto_tiling_artifact_many_levels_warns():
    """>50 levels (even with healthy cells/level) trips the level-count branch."""
    labels = np.repeat(np.arange(60), 300)  # 60 levels, 300 cells each
    obs = _obs(fov=labels)
    n_obs = labels.size
    with pytest.warns(UserWarning, match="tiling artifact"):
        _section_id, col_used, note = runner.resolve_section_id(
            list(obs.keys()), obs, n_obs, section_col_arg="auto"
        )
    assert col_used == "fov"
    assert note is not None and "60 levels" in note


def test_resolve_explicit_tiling_like_column_does_not_warn():
    """An EXPLICIT column is the operator's deliberate choice: no heuristic warn."""
    labels = np.repeat(np.arange(50), 10)
    obs = _obs(fov=labels)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        section_id, col_used, note = runner.resolve_section_id(
            list(obs.keys()), obs, labels.size, section_col_arg="fov"
        )
    assert col_used == "fov"
    assert int(np.unique(section_id).size) == 50
    assert note is None


# --------------------------------------------------------------------------
# resolve_section_id -- coordinate-aware tiling guard (#360). Many sections that
# share a coordinate frame (overlapping per-section bounding boxes, e.g. CODEX
# per-FOV-tile-LOCAL coords) are independent local frames that MUST stay split --
# NOT a sub-region tiling artifact -- so the "pass --section-col none" advice (and
# its warning) must be SUPPRESSED, while the codes are still returned.
# --------------------------------------------------------------------------


def _tiled_coords(section_id, *, local: bool, box=(1000.0, 1000.0), seed=0):
    """Coordinates for a sectioned dataset.

    ``local=True``  -> every section spans the SAME box (per-section-local frames,
    overlapping bounding boxes; the CODEX case).
    ``local=False`` -> each section occupies a DISJOINT sub-region of one global
    frame (true microscope tiling; boxes do not overlap).
    """
    rng = np.random.default_rng(seed)
    sid = np.asarray(section_id)
    coords = np.zeros((sid.size, 2), dtype=np.float64)
    for code in np.unique(sid):
        m = sid == code
        local_xy = rng.uniform(0, box, size=(int(m.sum()), 2))
        if local:
            coords[m] = local_xy
        else:
            coords[m] = local_xy + np.array([code * box[0], 0.0])  # shifted tiles
    return coords


def test_resolve_overlapping_local_frames_suppresses_tiling_warn():
    """100 sections sharing one coordinate box: no warn, no 'none' advice."""
    labels = np.repeat(np.arange(100), 50)  # 100 levels (> TILING_MAX_SECTIONS)
    obs = _obs(section_id=labels)
    coords = _tiled_coords(labels, local=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning fails the test
        section_id, col_used, note = runner.resolve_section_id(
            list(obs.keys()), obs, labels.size, section_col_arg="auto",
            coords=coords,
        )
    assert col_used == "section_id"
    assert int(np.unique(section_id).size) == 100
    # No misleading "pass none" tiling advice for per-section-local frames.
    if note is not None:
        assert "tiling artifact" not in note
        assert "none" not in note


def test_resolve_disjoint_tiles_still_warn_with_coords():
    """Disjoint sub-region tiles (non-overlapping boxes) still trip the warn."""
    labels = np.repeat(np.arange(60), 300)  # 60 tiles, healthy cells/tile
    obs = _obs(fov=labels)
    coords = _tiled_coords(labels, local=False)
    with pytest.warns(UserWarning, match="tiling artifact"):
        _sid, col_used, note = runner.resolve_section_id(
            list(obs.keys()), obs, labels.size, section_col_arg="auto",
            coords=coords,
        )
    assert col_used == "fov"
    assert note is not None and "tiling artifact" in note


def test_resolve_no_coords_keeps_legacy_warn():
    """Without coords (legacy callers) the level-count warn is unchanged."""
    labels = np.repeat(np.arange(60), 300)
    obs = _obs(fov=labels)
    with pytest.warns(UserWarning, match="tiling artifact"):
        runner.resolve_section_id(
            list(obs.keys()), obs, labels.size, section_col_arg="auto"
        )


# --------------------------------------------------------------------------
# resolve_section_id -- none/single override
# --------------------------------------------------------------------------


@pytest.mark.parametrize("arg", ["none", "single", "NONE", "Single"])
def test_resolve_none_forces_single_section(arg):
    """'none'/'single' (case-insensitive): all-zero section_id, col_used None."""
    obs = _obs(fov=np.repeat(np.arange(50), 10))
    n_obs = 500
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # the override must never warn
        section_id, col_used, note = runner.resolve_section_id(
            list(obs.keys()), obs, n_obs, section_col_arg=arg
        )
    assert col_used is None
    assert np.array_equal(section_id, np.zeros(n_obs, dtype=np.int64))
    assert section_id.dtype == np.int64
    assert note is not None  # records that single-section was forced


# --------------------------------------------------------------------------
# resolve_section_id -- explicit column name
# --------------------------------------------------------------------------


def test_resolve_explicit_existing_column_used():
    """An explicit, present column is used verbatim."""
    labels = np.array(["a", "a", "b", "c", "c", "c"])
    obs = _obs(my_section=labels, fov=np.arange(6))
    section_id, col_used, _note = runner.resolve_section_id(
        list(obs.keys()), obs, labels.size, section_col_arg="my_section"
    )
    assert col_used == "my_section"
    assert int(np.unique(section_id).size) == 3


def test_resolve_explicit_missing_column_errors_with_name():
    """An explicit, absent column raises a clear error naming the column."""
    obs = _obs(fov=np.arange(6))
    with pytest.raises((KeyError, ValueError), match="absent_col"):
        runner.resolve_section_id(
            list(obs.keys()), obs, 6, section_col_arg="absent_col"
        )


# --------------------------------------------------------------------------
# peak-RSS helper (#343) -- int or None, never raises; key name pinned
# --------------------------------------------------------------------------


def test_peak_rss_bytes_is_int_or_none_and_never_raises():
    """Helper returns a non-negative int (bytes) or None, and does not raise."""
    val = runner._peak_rss_bytes()
    assert val is None or (isinstance(val, int) and val >= 0)


def test_peak_rss_key_name_is_exact():
    """N-F2 (worker-figures) consumes the exact key 'peak_rss_bytes'.

    Pin the literal so a rename can't silently break the downstream figure.
    """
    src = _SCRIPT.read_text(encoding="utf-8")
    assert '"peak_rss_bytes"' in src


# --------------------------------------------------------------------------
# --section-col CLI flag
# --------------------------------------------------------------------------


def test_section_col_cli_default_auto():
    parser = runner._build_parser()
    args = parser.parse_args([])
    assert args.section_col == "auto"


def test_section_col_cli_parsed():
    parser = runner._build_parser()
    assert parser.parse_args(["--section-col", "none"]).section_col == "none"
    assert parser.parse_args(["--section-col", "my_col"]).section_col == "my_col"


# --------------------------------------------------------------------------
# Producer side of the fallback-detection contract (#344): a REAL fresh
# (auto) fallback run MUST write the STRUCTURED run_metadata['_fallback_note']
# key -- the emitters (emit_figures.py / emit_results_tables.py) gate
# paper_claim_ready on that exact key, so free-text notes alone would leave the
# anti-overclaim safeguard silently inert. The whole fit is stubbed so this runs
# without scanpy/torch/real data.
# --------------------------------------------------------------------------


class _FakeResult:
    H = np.zeros((2, 2), dtype=np.float32)
    prototype_id = np.zeros(2, dtype=np.int64)
    proto_kind = ("conserved", "sample_specific")
    interaction_summary = None


def _drive_main(monkeypatch, tmp_path, *, primary_oom):
    """Run ``runner.main()`` with the fit stubbed; return the captured run_metadata.

    ``primary_oom=True`` makes the primary load raise a host OOM so the auto path
    downshifts to the fallback; ``False`` lets the primary succeed.
    """
    from nichelens_st import results_contract

    def fake_load(path, already_normalized, section_col_arg="auto"):
        if primary_oom and path == runner.PRIMARY_PATH:
            raise MemoryError("Host out of memory during niche fit on 124938 cells")
        n = 5488 if path == runner.FALLBACK_PATH else 124938
        adata = type("A", (), {"uns": {"_section_note": None}})()
        return (
            np.zeros((2, 2), dtype=np.float32),  # X
            np.zeros((2, 2), dtype=np.float32),  # coords
            np.zeros(2, dtype=np.int64),  # section_id
            None,  # section_col
            {"applied": False, "method": "none"},  # normalization
            n,  # n_obs
            3,  # n_vars
            adata,
        )

    def fake_fit(X, coords, section_id, max_seconds, device, num_threads, **kw):
        return _FakeResult(), object(), 1.0, 0

    def fake_metrics(result, edges, section_id, seed):
        return {"n_prototypes": 2.0}, []

    captured = {}

    def fake_write(*, project, dataset_card_id, metrics, outputs, run_metadata,
                   results_dir):
        captured["run_metadata"] = run_metadata
        outdir = tmp_path / "outputs"
        outdir.mkdir(exist_ok=True)
        return {
            "outputs_dir": str(outdir),
            "metrics": str(tmp_path / "metrics.json"),
            "run_metadata": str(tmp_path / "run_metadata.json"),
        }

    monkeypatch.setattr(runner, "_load_dataset", fake_load)
    monkeypatch.setattr(runner, "_fit_with_walls", fake_fit)
    monkeypatch.setattr(runner, "_intrinsic_metrics", fake_metrics)
    monkeypatch.setattr(results_contract, "write_results", fake_write)
    monkeypatch.setattr(results_contract, "dataset_card_id", lambda paths: "card-x")
    monkeypatch.setattr(sys, "argv", ["run_real_niche.py"])  # default --dataset auto

    assert runner.main() == 0
    return captured["run_metadata"]


def test_fallback_run_writes_structured_fallback_note(monkeypatch, tmp_path):
    """A real auto->fallback run carries the structured _fallback_note key."""
    run_metadata = _drive_main(monkeypatch, tmp_path, primary_oom=True)
    assert "_fallback_note" in run_metadata
    note = run_metadata["_fallback_note"]
    assert note  # non-empty: the consumer treats any truthy value as fallback
    assert "DOWNSIZED SINGLE-SECTION FALLBACK" in note
    assert "5488" in note


def test_primary_success_omits_fallback_note(monkeypatch, tmp_path):
    """A successful primary run must NOT carry _fallback_note (no false flag)."""
    run_metadata = _drive_main(monkeypatch, tmp_path, primary_oom=False)
    assert "_fallback_note" not in run_metadata
