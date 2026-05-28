"""Regression tests for the synthetic generator's prototype catalog (issue #72).

When `J_specific > n_sections`, the indices
`K_conserved + n_sections .. K_conserved + J_specific - 1` were never assigned
to any cell but still appeared in `proto_kind` and `marker_genes`, silently
inflating `section_overlap_rate` (every unassigned `sample_specific` prototype
counts as "correct"). The fix clips `J_specific = min(J_specific, n_sections)`
with a warning so the catalog and the assignments stay consistent.
"""

from __future__ import annotations

import warnings

from nichelens_st.synth import generate_instance


def test_J_specific_clipped():
    """With J_specific > n_sections, the realised catalog has at most n_sections specific protos."""
    n_sections = 4
    K_conserved = 3
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        inst = generate_instance(
            n_sections=n_sections,
            n_cells_per_section=20,
            n_genes=8,
            K_conserved=K_conserved,
            J_specific=10,
            noise_sigma=0.1,
            k_nn=4,
            seed=0,
        )

    specific_count = sum(1 for k in inst.proto_kind if k == "sample_specific")
    assert specific_count <= n_sections, (
        f"expected at most {n_sections} sample_specific prototypes; got {specific_count}"
    )

    # No phantom prototypes: every catalog entry is assigned to ≥1 cell.
    assigned_ids = set(int(p) for p in inst.prototype_id)
    catalog_ids = set(range(len(inst.proto_kind)))
    assert catalog_ids.issubset(assigned_ids), (
        f"phantom prototypes in catalog: {catalog_ids - assigned_ids}"
    )

    # marker_genes must mirror the realised catalog.
    assert len(inst.marker_genes) == len(inst.proto_kind)

    # The clip should warn the caller about the silent reduction.
    assert any("J_specific" in str(w.message) for w in caught), (
        "expected a warning that J_specific was clipped"
    )
