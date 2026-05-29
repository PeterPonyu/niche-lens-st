"""Encoder numerics: L2 normalization must be float32-safe.

The contrastive readout L2-normalizes embeddings onto the unit sphere. With
torch's default ``eps=1e-12`` a near-zero-norm row (isolated node with
zero/dropped features) is divided by a tiny norm and blown up into a
noise-dominated unit vector that distorts the InfoNCE similarity matrix. An
explicit float32-appropriate ``eps`` keeps such degenerate rows bounded.
"""

from __future__ import annotations

import pytest

from nichelens_st.encoder import TORCH_AVAILABLE

pytestmark = pytest.mark.skipif(
    not TORCH_AVAILABLE, reason="requires the optional [model] extra (torch)"
)


def test_l2norm_eps_float32_safe():
    import torch

    from nichelens_st.encoder import _l2_normalize

    # Row 0: near-zero norm (~1e-7), far below a meaningful float32 magnitude.
    # Row 1: healthy unit-scale embedding.
    h = torch.tensor(
        [[1e-7, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]], dtype=torch.float32
    )
    out = _l2_normalize(h)

    assert torch.isfinite(out).all()

    norms = out.norm(dim=1)
    # Healthy row sits on the unit sphere.
    assert norms[1].item() == pytest.approx(1.0, abs=1e-5)
    # Degenerate row stays bounded well below 1: it is NOT amplified to a unit
    # vector. With eps=1e-6 its norm is ~0.1; torch's default eps=1e-12 would
    # divide 1e-7 by 1e-7 and blow it up to ~1.0.
    assert norms[0].item() < 0.5
