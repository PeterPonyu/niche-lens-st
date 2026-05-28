from nichelens_st.synth import generate_instance


def test_proto_kind_counts_match_params():
    inst = generate_instance(
        n_sections=2,
        n_cells_per_section=20,
        n_genes=10,
        K_conserved=3,
        J_specific=2,
        k_nn=3,
        seed=0,
    )
    assert inst.proto_kind.count("conserved") == 3
    assert inst.proto_kind.count("sample_specific") == 2


def test_sample_specific_lives_in_strict_subset_of_sections():
    n_sections = 4
    inst = generate_instance(
        n_sections=n_sections,
        n_cells_per_section=80,
        n_genes=10,
        K_conserved=2,
        J_specific=2,
        k_nn=3,
        seed=0,
    )
    for proto_idx, kind in enumerate(inst.proto_kind):
        mask = inst.prototype_id == proto_idx
        sections_seen = set(inst.section_id[mask].tolist()) if mask.any() else set()
        if kind == "sample_specific":
            assert len(sections_seen) < n_sections, (
                f"sample_specific proto {proto_idx} reached every section: {sections_seen}"
            )


def test_conserved_prototypes_appear_in_every_section():
    n_sections = 4
    inst = generate_instance(
        n_sections=n_sections,
        n_cells_per_section=200,
        n_genes=10,
        K_conserved=2,
        J_specific=2,
        k_nn=3,
        seed=0,
    )
    for proto_idx, kind in enumerate(inst.proto_kind):
        if kind == "conserved":
            mask = inst.prototype_id == proto_idx
            sections_seen = set(inst.section_id[mask].tolist())
            assert sections_seen == set(range(n_sections))
