import numpy as np
import pytest

from nichelens_st.schemas import SchemaError, validate_inputs


def _ok():
    n = 6
    return dict(
        X=np.zeros((n, 4), dtype=np.float32),
        coords=np.zeros((n, 2), dtype=np.float32),
        section_id=np.zeros(n, dtype=np.int64),
        edges=np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64),
    )


def test_valid_inputs_pass():
    validate_inputs(**_ok())


def test_valid_3d_coords_pass():
    kw = _ok()
    kw["coords"] = np.zeros((6, 3), dtype=np.float32)
    validate_inputs(**kw)


def test_X_must_be_2d():
    kw = _ok()
    kw["X"] = np.zeros(10, dtype=np.float32)
    with pytest.raises(SchemaError, match="X must be 2D"):
        validate_inputs(**kw)


def test_coords_wrong_inner_dim():
    kw = _ok()
    kw["coords"] = np.zeros((6, 5), dtype=np.float32)
    with pytest.raises(SchemaError, match="coords must be"):
        validate_inputs(**kw)


def test_coords_n_cells_mismatch():
    kw = _ok()
    kw["coords"] = np.zeros((5, 2), dtype=np.float32)
    with pytest.raises(SchemaError, match="coords n_cells"):
        validate_inputs(**kw)


def test_X_wrong_dtype():
    kw = _ok()
    kw["X"] = np.zeros((6, 4), dtype=np.float64)
    with pytest.raises(SchemaError, match="X must be float32"):
        validate_inputs(**kw)

def test_coords_wrong_dtype():
    kw = _ok()
    kw["coords"] = np.zeros((6, 2), dtype=np.float64)
    with pytest.raises(SchemaError, match="coords must be float32"):
        validate_inputs(**kw)

def test_section_id_wrong_dtype():
    kw = _ok()
    kw["section_id"] = np.zeros(6, dtype=np.float32)
    with pytest.raises(SchemaError, match="section_id must be integer"):
        validate_inputs(**kw)


def test_edges_wrong_outer_dim():
    kw = _ok()
    kw["edges"] = np.zeros((3, 6), dtype=np.int64)
    with pytest.raises(SchemaError, match="edges must be"):
        validate_inputs(**kw)


def test_edges_out_of_range():
    kw = _ok()
    kw["edges"] = np.array([[0, 10], [1, 1]], dtype=np.int64)
    with pytest.raises(SchemaError, match="outside"):
        validate_inputs(**kw)


def test_edges_wrong_dtype():
    kw = _ok()
    kw["edges"] = np.zeros((2, 3), dtype=np.float32)
    with pytest.raises(SchemaError, match="edges must be int64"):
        validate_inputs(**kw)

def test_edges_int32_rejected():
    kw = _ok()
    kw["edges"] = np.zeros((2, 3), dtype=np.int32)
    with pytest.raises(SchemaError, match="edges must be int64"):
        validate_inputs(**kw)


def test_empty_edges_pass():
    kw = _ok()
    kw["edges"] = np.zeros((2, 0), dtype=np.int64)
    validate_inputs(**kw)


def test_cross_section_edge_rejected():
    kw = _ok()
    kw["section_id"] = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    kw["edges"] = np.array([[0, 1], [4, 5]], dtype=np.int64)
    with pytest.raises(SchemaError, match="within section"):
        validate_inputs(**kw)


def test_nonfinite_X_rejected():
    kw = _ok()
    kw["X"][0, 0] = np.nan
    with pytest.raises(SchemaError, match="NaN or Inf"):
        validate_inputs(**kw)


def test_nonfinite_coords_rejected():
    kw = _ok()
    kw["coords"][0, 0] = np.inf
    with pytest.raises(SchemaError, match="NaN or Inf"):
        validate_inputs(**kw)
