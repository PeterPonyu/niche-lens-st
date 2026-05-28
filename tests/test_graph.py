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


def test_coincident_coords_no_self_loop():
    """Regression for #60: coincident/tied coordinates must not produce
    self-loops or drop a real neighbor. ``cKDTree.query`` can return the
    duplicate instead of self in column 0 when distances tie, so a
    positional ``nn_local[:, 1:]`` slice silently keeps ``(i, i)`` and
    drops cell i's real nearest neighbor.
    """
    coords = np.array([[0, 0], [0, 0], [1, 0], [5, 0]], dtype=np.float32)
    section_id = np.zeros(4, dtype=np.int64)
    edges = build_graph(coords, section_id, k=1)
    pairs = edges.T.tolist()
    # No self-loops anywhere.
    assert not any(s == d for s, d in pairs), (
        f"self-loop emitted: {[(s, d) for s, d in pairs if s == d]}"
    )
    # Coincident cells 0 and 1 must each get a real neighbor (each other).
    pair_set = set(map(tuple, pairs))
    assert (0, 1) in pair_set, f"cell 0 lost its real neighbor; got {pair_set}"
    assert (1, 0) in pair_set, f"cell 1 lost its real neighbor; got {pair_set}"


def test_coincident_coords_full_k_neighbors():
    """All four cells share coords with k=2: each cell must still get
    k=2 real (non-self) neighbors. Without the self-mask fix, ties on
    column 0 caused k-1 retained neighbors plus a self-loop.
    """
    coords = np.zeros((4, 2), dtype=np.float32)  # all coincident
    section_id = np.zeros(4, dtype=np.int64)
    edges = build_graph(coords, section_id, k=2)
    assert edges.shape == (2, 4 * 2)
    pairs = edges.T.tolist()
    assert not any(s == d for s, d in pairs), "self-loop with coincident coords"
    # Each source should have exactly k=2 distinct neighbors.
    from collections import Counter

    counts = Counter(s for s, _ in pairs)
    for src, cnt in counts.items():
        assert cnt == 2, f"cell {src} has {cnt} neighbors; expected k=2"


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
