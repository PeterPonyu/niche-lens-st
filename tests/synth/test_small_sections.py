import numpy as np

from nichelens_st.synth import generate_instance


def test_small_section_clamps_when_m_less_than_k_nn():
    inst = generate_instance(n_sections=1, n_cells_per_section=5, n_genes=4, K_conserved=2, J_specific=0, k_nn=8, seed=0)
    assert inst.edges.shape == (2, 5 * 4)


def test_small_section_clamps_when_m_equals_k_nn():
    inst = generate_instance(n_sections=1, n_cells_per_section=8, n_genes=4, K_conserved=2, J_specific=0, k_nn=8, seed=0)
    assert inst.edges.shape == (2, 8 * 7)


def test_singleton_section_has_zero_edges():
    inst = generate_instance(n_sections=1, n_cells_per_section=1, n_genes=4, K_conserved=1, J_specific=0, k_nn=8, seed=0)
    assert inst.edges.shape == (2, 0)
    assert inst.edges.dtype == np.int64
