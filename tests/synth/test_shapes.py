import numpy as np

from nichelens_st.synth import generate_instance


def test_shapes_match_documented_schema():
    inst = generate_instance(
        n_sections=3,
        n_cells_per_section=50,
        n_genes=20,
        K_conserved=4,
        J_specific=2,
        k_nn=4,
        seed=42,
    )
    n_cells = 3 * 50
    assert inst.X.shape == (n_cells, 20)
    assert inst.X.dtype == np.float32
    assert inst.coords.shape == (n_cells, 2)
    assert inst.section_id.shape == (n_cells,)
    assert inst.section_id.dtype == np.int64
    assert inst.prototype_id.shape == (n_cells,)
    assert inst.prototype_id.dtype == np.int64
    assert inst.edges.shape == (2, n_cells * 4)
    assert inst.edges.dtype == np.int64
    assert len(inst.proto_kind) == 4 + 2
    assert len(inst.marker_genes) == 4 + 2


def test_edges_point_to_valid_cells():
    inst = generate_instance(
        n_sections=2,
        n_cells_per_section=30,
        n_genes=10,
        K_conserved=2,
        J_specific=1,
        k_nn=3,
        seed=7,
    )
    n_cells = 60
    assert inst.edges.min() >= 0
    assert inst.edges.max() < n_cells


def test_zero_j_specific_skips_specific_protos():
    inst = generate_instance(
        n_sections=2,
        n_cells_per_section=20,
        n_genes=10,
        K_conserved=3,
        J_specific=0,
        k_nn=3,
        seed=0,
    )
    assert len(inst.proto_kind) == 3
    assert "sample_specific" not in inst.proto_kind
