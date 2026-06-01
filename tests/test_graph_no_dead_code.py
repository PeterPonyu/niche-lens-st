"""Regression guard for issue #76: no defined-but-unused public symbol in graph.

The ``Subgraph`` frozen dataclass was dead code -- defined but never
instantiated, returned, imported, or referenced. The extraction helpers return
bare ``tuple[np.ndarray, np.ndarray]``. This test locks the deletion so the
unused container does not silently rot back in.
"""

from __future__ import annotations

import numpy as np

import nichelens_st.graph as graph
from nichelens_st.graph import extract_subgraph


def test_subgraph_symbol_removed():
    assert not hasattr(graph, "Subgraph")


def test_extract_subgraph_returns_plain_tuple():
    edges = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int64)
    result = extract_subgraph(edges, center=1, k_hop=1)
    assert isinstance(result, tuple)
    node_ids, sub_edges = result
    assert isinstance(node_ids, np.ndarray)
    assert isinstance(sub_edges, np.ndarray)
