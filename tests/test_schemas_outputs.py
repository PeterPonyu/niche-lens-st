import numpy as np
import pytest

from nichelens_st.schemas import SchemaError, validate_outputs


def test_valid_outputs_pass():
    n_cells = 6
    H = np.zeros((n_cells, 4), dtype=np.float32)
    prototype_id = np.array([0, 0, 1, 1, 1, 1], dtype=np.int64)
    proto_kind = ["conserved", "sample_specific"]
    validate_outputs(H, prototype_id, proto_kind, n_cells)


def test_H_wrong_dtype():
    n_cells = 6
    H = np.zeros((n_cells, 4), dtype=np.float64)
    proto = np.zeros(n_cells, dtype=np.int64)
    with pytest.raises(SchemaError, match="H must be float32"):
        validate_outputs(H, proto, ["conserved"], n_cells)

def test_H_wrong_n_cells():
    n_cells = 6
    H = np.zeros((5, 4), dtype=np.float32)
    proto = np.zeros(n_cells, dtype=np.int64)
    with pytest.raises(SchemaError, match="H must be"):
        validate_outputs(H, proto, ["conserved"], n_cells)


def test_prototype_id_wrong_dtype():
    n = 3
    H = np.zeros((n, 2), dtype=np.float32)
    proto = np.array([0.0, 1.0, 1.0])
    with pytest.raises(SchemaError, match="prototype_id must be int64"):
        validate_outputs(H, proto, ["conserved", "sample_specific"], n)

def test_prototype_id_int32_rejected():
    n = 3
    H = np.zeros((n, 2), dtype=np.float32)
    proto = np.array([0, 1, 1], dtype=np.int32)
    with pytest.raises(SchemaError, match="prototype_id must be int64"):
        validate_outputs(H, proto, ["conserved", "sample_specific"], n)


def test_prototype_id_negative_fails():
    n = 3
    H = np.zeros((n, 2), dtype=np.float32)
    proto = np.array([-1, 0, 0], dtype=np.int64)
    with pytest.raises(SchemaError, match="non-negative"):
        validate_outputs(H, proto, ["conserved"], n)


def test_proto_kind_invalid_value():
    n = 3
    H = np.zeros((n, 2), dtype=np.float32)
    proto = np.array([0, 1, 1], dtype=np.int64)
    with pytest.raises(SchemaError, match="invalid values"):
        validate_outputs(H, proto, ["conserved", "bad_label"], n)


def test_proto_kind_length_mismatch():
    n = 3
    H = np.zeros((n, 2), dtype=np.float32)
    proto = np.array([0, 1, 2], dtype=np.int64)
    with pytest.raises(SchemaError, match="proto_kind length"):
        validate_outputs(H, proto, ["conserved"], n)


def test_proto_kind_catalog_larger_than_observed_passes():
    n = 4
    H = np.zeros((n, 2), dtype=np.float32)
    proto = np.array([0, 1, 1, 0], dtype=np.int64)
    validate_outputs(H, proto, ["conserved", "conserved", "sample_specific"], n)


def test_holey_prototype_indexing_passes_when_catalog_covers_max_id():
    n = 3
    H = np.zeros((n, 2), dtype=np.float32)
    proto = np.array([0, 2, 2], dtype=np.int64)
    validate_outputs(H, proto, ["conserved", "sample_specific", "sample_specific"], n)


def test_observed_prototype_id_out_of_catalog_fails():
    n = 3
    H = np.zeros((n, 2), dtype=np.float32)
    proto = np.array([0, 2, 2], dtype=np.int64)
    with pytest.raises(SchemaError, match="does not cover"):
        validate_outputs(H, proto, ["conserved", "sample_specific"], n)


def test_zero_cell_outputs_empty_catalog_pass():
    H = np.zeros((0, 2), dtype=np.float32)
    proto = np.zeros(0, dtype=np.int64)
    validate_outputs(H, proto, [], 0)
