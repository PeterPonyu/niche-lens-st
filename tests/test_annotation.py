"""Marker-based reference annotation scaffold (issue #350).

The primary GSE282124 is unsupervised (no cell-type labels), yet
``scripts/compute_further_niche_metrics.py`` needs ``obs['cell_class']`` for the
composition / co-localization / annotation-agreement metrics. These tests pin a
dependency-light, deterministic marker-scoring primitive that *produces* a
``cell_class`` assignment whose codes index into a name list -- exactly the
consumer contract (``cell_class.cat.codes`` / ``cell_class.cat.categories``).

The core recovery test plants an INDEPENDENT ground truth (each cell type owns
distinct high-expression marker genes) and asserts the primitive recovers it --
it does not read the answer back out of its own output (issue #83 discipline).
"""

import numpy as np
import pytest

from nichelens_st.annotation import AnnotationResult, marker_score_annotate


# --- toy fixture with an independent planted ground truth --------------------

def _planted_dataset(n_per_type=80, noise=0.05, seed=0):
    """Three cell types, each with two DISTINCT high-expression marker genes.

    Returns ``(X, var_names, marker_dict, true_codes, type_names)``. The truth is
    the block structure of the design matrix, not anything the annotator emits.
    """
    rng = np.random.default_rng(seed)
    type_names = ["TypeA", "TypeB", "TypeC"]
    var_names = [f"g{j}" for j in range(6)]
    marker_dict = {
        "TypeA": ["g0", "g1"],
        "TypeB": ["g2", "g3"],
        "TypeC": ["g4", "g5"],
    }
    blocks = []
    true_codes = []
    for code, _name in enumerate(type_names):
        base = rng.normal(0.0, noise, size=(n_per_type, 6))
        base[:, 2 * code] += 5.0
        base[:, 2 * code + 1] += 5.0
        blocks.append(base)
        true_codes.extend([code] * n_per_type)
    X = np.vstack(blocks).astype(np.float64)
    return X, var_names, marker_dict, np.asarray(true_codes, dtype=np.int64), type_names


# --- core: known-answer recovery (independent ground truth) ------------------

@pytest.mark.parametrize("method", ["mean_zscore", "mean"])
def test_recovers_planted_cell_class(method):
    X, var_names, marker_dict, true_codes, type_names = _planted_dataset()
    res = marker_score_annotate(X, var_names, marker_dict, method=method)
    # used_types order mirrors marker_dict insertion order, so codes line up.
    assert res.used_types == type_names
    acc = float(np.mean(res.cell_class_codes == true_codes))
    assert acc > 0.9, f"{method}: recovery accuracy {acc} too low"


def test_returns_dataclass_instance():
    X, var_names, marker_dict, _, _ = _planted_dataset()
    res = marker_score_annotate(X, var_names, marker_dict)
    assert isinstance(res, AnnotationResult)


# --- determinism -------------------------------------------------------------

def test_deterministic_identical_inputs():
    X, var_names, marker_dict, _, _ = _planted_dataset()
    a = marker_score_annotate(X, var_names, marker_dict, method="mean_zscore")
    b = marker_score_annotate(X, var_names, marker_dict, method="mean_zscore")
    assert np.array_equal(a.cell_class_codes, b.cell_class_codes)
    assert np.array_equal(a.scores, b.scores)


# --- missing markers: dropped + excluded, never crashed ----------------------

def test_missing_marker_genes_are_dropped_and_recorded():
    X, var_names, marker_dict, _, _ = _planted_dataset()
    marker_dict = dict(marker_dict)
    marker_dict["TypeA"] = ["g0", "g1", "absent_gene"]
    res = marker_score_annotate(X, var_names, marker_dict)
    assert res.dropped_genes.get("TypeA") == ["absent_gene"]
    assert "TypeA" in res.used_types  # still has 2 usable markers


def test_type_with_no_usable_markers_is_excluded_not_crashed():
    X, var_names, marker_dict, _, _ = _planted_dataset()
    marker_dict = dict(marker_dict)
    marker_dict["Ghost"] = ["nope1", "nope2"]
    res = marker_score_annotate(X, var_names, marker_dict)
    assert "Ghost" in res.excluded_types
    assert "Ghost" not in res.used_types
    assert res.dropped_genes.get("Ghost") == ["nope1", "nope2"]


def test_min_markers_excludes_underspecified_type():
    X, var_names, marker_dict, _, _ = _planted_dataset()
    marker_dict = dict(marker_dict)
    marker_dict["TypeC"] = ["g4"]  # only one usable marker
    res = marker_score_annotate(X, var_names, marker_dict, min_markers=2)
    assert "TypeC" in res.excluded_types
    assert res.used_types == ["TypeA", "TypeB"]


# --- assign_threshold --------------------------------------------------------

def test_threshold_marks_low_score_cells_unassigned():
    X, var_names, marker_dict, _, _ = _planted_dataset()
    res = marker_score_annotate(
        X, var_names, marker_dict, method="mean_zscore",
        assign_threshold=1e9, unassigned_label="unassigned",
    )
    assert "unassigned" in res.cell_class_names
    unassigned_code = res.cell_class_names.index("unassigned")
    assert np.all(res.cell_class_codes == unassigned_code)


def test_no_threshold_assigns_every_cell():
    X, var_names, marker_dict, _, _ = _planted_dataset()
    res = marker_score_annotate(X, var_names, marker_dict)
    assert "unassigned" not in res.cell_class_names
    assert res.cell_class_codes.max() < len(res.used_types)


def test_threshold_without_trigger_does_not_add_unassigned():
    X, var_names, marker_dict, _, _ = _planted_dataset()
    # Very negative threshold: every cell clears it -> no unassigned label added.
    res = marker_score_annotate(
        X, var_names, marker_dict, method="mean_zscore", assign_threshold=-1e9
    )
    assert "unassigned" not in res.cell_class_names


# --- errors (issue #83 family) -----------------------------------------------

def test_empty_marker_dict_raises():
    X, var_names, _, _, _ = _planted_dataset()
    with pytest.raises(ValueError):
        marker_score_annotate(X, var_names, {})


def test_var_names_length_mismatch_raises():
    X, _, marker_dict, _, _ = _planted_dataset()
    with pytest.raises(ValueError):
        marker_score_annotate(X, [f"g{j}" for j in range(5)], marker_dict)


def test_all_types_unusable_raises():
    X, var_names, _, _, _ = _planted_dataset()
    with pytest.raises(ValueError):
        marker_score_annotate(X, var_names, {"X": ["absent"], "Y": ["alsoabsent"]})


def test_non_2d_X_raises():
    _, var_names, marker_dict, _, _ = _planted_dataset()
    with pytest.raises(ValueError):
        marker_score_annotate(np.zeros(6), var_names, marker_dict)


def test_unknown_method_raises():
    X, var_names, marker_dict, _, _ = _planted_dataset()
    with pytest.raises(ValueError):
        marker_score_annotate(X, var_names, marker_dict, method="bogus")


# --- output contract ---------------------------------------------------------

def test_codes_index_into_names_and_score_shape():
    X, var_names, marker_dict, _, _ = _planted_dataset()
    res = marker_score_annotate(X, var_names, marker_dict)
    n_cells = X.shape[0]
    assert res.cell_class_codes.dtype == np.int64
    assert res.cell_class_codes.shape == (n_cells,)
    assert res.cell_class_codes.min() >= 0
    assert res.cell_class_codes.max() < len(res.cell_class_names)
    assert res.scores.shape == (n_cells, len(res.used_types))
    assert res.scores.dtype == np.float64
    # Every assigned name resolvable through the code->name map.
    names = [res.cell_class_names[c] for c in res.cell_class_codes]
    assert len(names) == n_cells
