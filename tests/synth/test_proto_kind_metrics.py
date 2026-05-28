from nichelens_st.metrics import section_overlap_rate
from nichelens_st.synth import generate_instance


def test_section_overlap_rate_classifies_truth():
    inst = generate_instance(
        n_sections=4,
        n_cells_per_section=40,
        n_genes=8,
        K_conserved=2,
        J_specific=2,
        seed=0,
    )
    assert section_overlap_rate(inst.prototype_id, inst.section_id, inst.proto_kind) == 1.0
