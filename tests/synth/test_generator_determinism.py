import numpy as np

from nichelens_st.synth import generate_instance

KW = dict(
    n_sections=2,
    n_cells_per_section=20,
    n_genes=10,
    K_conserved=2,
    J_specific=1,
    k_nn=3,
)


def test_same_seed_same_output():
    a = generate_instance(**KW, seed=0)
    b = generate_instance(**KW, seed=0)
    np.testing.assert_array_equal(a.X, b.X)
    np.testing.assert_array_equal(a.coords, b.coords)
    np.testing.assert_array_equal(a.section_id, b.section_id)
    np.testing.assert_array_equal(a.edges, b.edges)
    np.testing.assert_array_equal(a.prototype_id, b.prototype_id)
    assert a.proto_kind == b.proto_kind
    assert a.marker_genes == b.marker_genes


def test_different_seed_diverges():
    a = generate_instance(**KW, seed=0)
    b = generate_instance(**KW, seed=1)
    assert not np.array_equal(a.X, b.X)
