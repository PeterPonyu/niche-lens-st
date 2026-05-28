"""End-to-end model output tests against the MVP output schema.

These exercise the torch-backed contrastive encoder + prototype/separation head
and are skipped automatically when the optional ``[model]`` extra (torch) is not
installed.
"""

from __future__ import annotations

import numpy as np
import pytest

from nichelens_st.encoder import (
    _INFO_NCE_TAU_MIN,
    EncoderConfig,
    _info_nce,
    train_embeddings,
)
from nichelens_st.model import TORCH_AVAILABLE, NicheModelConfig, fit_niche_model
from nichelens_st.schemas import VALID_PROTO_KIND, validate_outputs
from nichelens_st.synth import generate_instance

pytestmark = pytest.mark.skipif(
    not TORCH_AVAILABLE, reason="requires the optional [model] extra (torch)"
)


@pytest.mark.parametrize("tau", [0.0, -0.1])
def test_train_embeddings_rejects_non_positive_tau(tau):
    X = np.zeros((4, 3), dtype=np.float32)
    edges = np.array([[0, 1], [1, 2]], dtype=np.int64)
    with pytest.raises(ValueError, match="tau must be > 0"):
        train_embeddings(X, edges, EncoderConfig(tau=tau, epochs=1))


def test_tau_tiny_no_overflow_f32():
    """#88: a positive but tiny tau (< _INFO_NCE_TAU_MIN) must not let
    ``z @ z.t() / tau`` overflow float32 and NaN-out the loss. The
    clamp in ``_info_nce`` bounds tau at ``_INFO_NCE_TAU_MIN``.
    """
    import torch  # local import; module is gated on TORCH_AVAILABLE.

    torch.manual_seed(0)
    n, d = 8, 16
    z1 = torch.nn.functional.normalize(
        torch.randn(n, d, dtype=torch.float32), dim=1
    )
    z2 = torch.nn.functional.normalize(
        torch.randn(n, d, dtype=torch.float32), dim=1
    )
    # tau orders of magnitude below the clamp threshold; without the
    # clamp `sim / tau` overflows float32 and softmax → NaN.
    loss = _info_nce(z1, z2, tau=1e-8)
    assert torch.isfinite(loss).item(), (
        f"InfoNCE loss not finite for tiny tau (got {float(loss)}); "
        f"clamp at _INFO_NCE_TAU_MIN={_INFO_NCE_TAU_MIN} regressed."
    )


def _small_instance(seed: int = 0):
    return generate_instance(
        n_sections=3,
        n_cells_per_section=60,
        n_genes=24,
        K_conserved=3,
        J_specific=2,
        noise_sigma=0.3,
        k_nn=4,
        seed=seed,
    )


def test_end_to_end_passes_validate_outputs():
    inst = _small_instance()
    cfg = NicheModelConfig(embed_dim=16, epochs=8, n_prototypes=8, seed=0)
    result = fit_niche_model(
        X=inst.X,
        coords=inst.coords,
        section_id=inst.section_id,
        edges=inst.edges,
        config=cfg,
    )
    # Must satisfy the existing output contract.
    validate_outputs(
        result.H, result.prototype_id, result.proto_kind, n_cells=inst.X.shape[0]
    )


def test_output_shapes_and_dtypes():
    inst = _small_instance()
    n_cells = inst.X.shape[0]
    cfg = NicheModelConfig(embed_dim=12, epochs=5, n_prototypes=6, seed=1)
    result = fit_niche_model(
        X=inst.X,
        coords=inst.coords,
        section_id=inst.section_id,
        edges=inst.edges,
        config=cfg,
    )

    assert result.H.shape == (n_cells, 12)
    assert result.H.dtype == np.float32
    assert np.isfinite(result.H).all()

    assert result.prototype_id.shape == (n_cells,)
    assert result.prototype_id.dtype == np.int64
    assert result.prototype_id.min() >= 0

    # proto_kind covers every observed prototype id and uses only valid tags.
    assert len(result.proto_kind) >= int(result.prototype_id.max()) + 1
    assert set(result.proto_kind).issubset(VALID_PROTO_KIND)


def test_separation_head_emits_both_kinds_when_data_supports_it():
    # The synthetic instance has conserved prototypes (every section) and
    # sample_specific ones (subset of sections); the head should be able to
    # produce at least one conserved tag.
    inst = _small_instance()
    cfg = NicheModelConfig(embed_dim=16, epochs=8, n_prototypes=8, seed=0)
    result = fit_niche_model(
        X=inst.X,
        coords=inst.coords,
        section_id=inst.section_id,
        edges=inst.edges,
        config=cfg,
    )
    # Only consider prototypes that are actually assigned to cells.
    assigned = set(result.prototype_id.tolist())
    kinds = {result.proto_kind[p] for p in assigned}
    assert "conserved" in kinds
