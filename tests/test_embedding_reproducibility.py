"""Reproducibility hook from docs/MVP_DESIGN.md acceptance matrix.

A fixed seed must yield identical ``prototype_id`` across runs (deterministic
niche prototypes). Skipped when the optional ``[model]`` extra is absent.
"""

from __future__ import annotations

import numpy as np
import pytest

from nichelens_st.model import TORCH_AVAILABLE, NicheModelConfig, fit_niche_model
from nichelens_st.synth import generate_instance

pytestmark = pytest.mark.skipif(
    not TORCH_AVAILABLE, reason="requires the optional [model] extra (torch)"
)


def _run(seed: int):
    inst = generate_instance(
        n_sections=2,
        n_cells_per_section=50,
        n_genes=20,
        K_conserved=3,
        J_specific=1,
        noise_sigma=0.3,
        k_nn=4,
        seed=0,
    )
    cfg = NicheModelConfig(embed_dim=12, epochs=6, n_prototypes=6, seed=seed)
    return fit_niche_model(
        X=inst.X,
        coords=inst.coords,
        section_id=inst.section_id,
        edges=inst.edges,
        config=cfg,
    )


def test_fixed_seed_gives_identical_prototype_id():
    a = _run(seed=7)
    b = _run(seed=7)
    np.testing.assert_array_equal(a.prototype_id, b.prototype_id)
    # Embeddings should be bitwise reproducible too.
    np.testing.assert_array_equal(a.H, b.H)
    assert a.proto_kind == b.proto_kind


def test_different_seed_is_allowed_to_differ():
    a = _run(seed=1)
    b = _run(seed=2)
    # Not a hard requirement that they differ, but shapes must still match.
    assert a.prototype_id.shape == b.prototype_id.shape
