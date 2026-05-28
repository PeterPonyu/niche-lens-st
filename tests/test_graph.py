import numpy as np
import pytest

from nichelens_st.graph import GraphError, build_graph, extract_subgraph, extract_subgraphs
from nichelens_st.schemas import validate_inputs


def test_build_graph_shape_dtype_and_per_section():
    coords = np.array([[0, 0], [1, 0], [3, 0], [0, 0], [0, 1]], dtype=np.float32)
    section_id = np.array([0, 0, 0, 1, 1], dtype=np.int64)
    edges = build_graph(coords, section_id, k=2)
    assert edges.dtype == np.int64
    assert edges.shape == (2, 8)  # section sizes clamp to 2 and 1 neighbors
    X = np.zeros((5, 2), dtype=np.float32)
    validate_inputs(X, coords, section_id, edges)


def test_build_graph_tiny_knn_correctness_and_singleton():
    coords = np.array([[0, 0], [1, 0], [5, 0], [0, 0]], dtype=np.float32)
    section_id = np.array([0, 0, 0, 1], dtype=np.int64)
    edges = build_graph(coords, section_id, k=1)
    assert set(map(tuple, edges.T.tolist())) == {(0, 1), (1, 0), (2, 1)}


def test_build_graph_rejects_unknown_method():
    with pytest.raises(GraphError, match="unsupported"):
        build_graph(np.zeros((1, 2), np.float32), np.zeros(1, np.int64), method="bad")


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_build_graph_rejects_non_finite_coords(bad):
    coords = np.array([[0, 0], [1, 0], [3, 0]], dtype=np.float32)
    section_id = np.zeros(3, dtype=np.int64)
    coords[1, 0] = bad
    with pytest.raises(GraphError, match="NaN or Inf"):
        build_graph(coords, section_id, k=1)


def test_extract_subgraph_k_hops_and_isolated():
    edges = np.array([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.int64)
    nodes0, sub0 = extract_subgraph(edges, center=2, k_hop=0)
    np.testing.assert_array_equal(nodes0, np.array([2], dtype=np.int64))
    assert sub0.shape == (2, 0)

    nodes1, sub1 = extract_subgraph(edges, center=2, k_hop=1)
    np.testing.assert_array_equal(nodes1, np.array([1, 2, 3], dtype=np.int64))
    assert set(map(tuple, sub1.T.tolist())) == {(1, 2), (2, 3)}

    nodes2, _ = extract_subgraph(edges, center=2, k_hop=2)
    np.testing.assert_array_equal(nodes2, np.array([0, 1, 2, 3, 4], dtype=np.int64))

    nodes_iso, sub_iso = extract_subgraph(edges, center=99, k_hop=2)
    np.testing.assert_array_equal(nodes_iso, np.array([99], dtype=np.int64))
    assert sub_iso.shape == (2, 0)


def test_extract_subgraphs_batches_with_shared_adjacency():
    edges = np.array([[0, 1], [1, 2]], dtype=np.int64)
    got = extract_subgraphs(edges, [0, 2], k_hop=1)
    np.testing.assert_array_equal(got[0][0], np.array([0, 1], dtype=np.int64))
    np.testing.assert_array_equal(got[1][0], np.array([1, 2], dtype=np.int64))

