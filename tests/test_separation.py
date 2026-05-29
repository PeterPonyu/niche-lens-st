"""Regression tests for the separation head's single-section behavior (issue #85).

`_separation_head` previously tagged every populated prototype `conserved` when
`section_id` had exactly one unique value, because `seen == all_sections` is
trivially true. The conserved/sample_specific distinction is undefined with one
section; we now tag every prototype `'unknown'` to flag the degeneracy without
silently asserting a meaningless biological "everything is shared" result.
"""

from __future__ import annotations

import numpy as np
import pytest

from nichelens_st.model import TORCH_AVAILABLE, NicheModelConfig, fit_niche_model
from nichelens_st.synth import generate_instance

pytestmark = pytest.mark.skipif(
    not TORCH_AVAILABLE, reason="requires the optional [model] extra (torch)"
)


def test_single_section_no_false_conserved():
    """With n_sections=1, no prototype is silently labelled 'conserved'."""
    inst = generate_instance(
        n_sections=1,
        n_cells_per_section=60,
        n_genes=12,
        K_conserved=3,
        J_specific=0,
        noise_sigma=0.2,
        k_nn=4,
        seed=0,
    )
    cfg = NicheModelConfig(embed_dim=8, epochs=4, n_prototypes=4, seed=0)
    result = fit_niche_model(
        X=inst.X,
        coords=inst.coords,
        section_id=inst.section_id,
        edges=inst.edges,
        config=cfg,
    )

    assert "conserved" not in result.proto_kind, (
        f"single-section input must not produce 'conserved' tags; "
        f"got proto_kind={result.proto_kind}"
    )


def test_multi_section_separation_unchanged():
    """Multi-section runs still produce at least one 'conserved' prototype."""
    inst = generate_instance(
        n_sections=3,
        n_cells_per_section=60,
        n_genes=24,
        K_conserved=3,
        J_specific=2,
        noise_sigma=0.3,
        k_nn=4,
        seed=0,
    )
    cfg = NicheModelConfig(embed_dim=16, epochs=8, n_prototypes=8, seed=0)
    result = fit_niche_model(
        X=inst.X,
        coords=inst.coords,
        section_id=inst.section_id,
        edges=inst.edges,
        config=cfg,
    )
    assigned = set(result.prototype_id.tolist())
    kinds = {result.proto_kind[p] for p in assigned}
    assert "conserved" in kinds
