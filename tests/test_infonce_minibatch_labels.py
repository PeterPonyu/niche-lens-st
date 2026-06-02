"""Minibatch InfoNCE (#61/#148) composed with group-aware masking (#92/#103).

The dataset-integration merge reconciled two changes to the contrastive loss:
the minibatched NT-Xent (bounds memory to O(batch^2)) and the optional
same-group negative masking. ``_info_nce_minibatch`` now accepts ``labels``;
``labels=None`` preserves the existing minibatch numerics, and the masking is
applied within each minibatch.
"""

from __future__ import annotations

import pytest

from nichelens_st.encoder import TORCH_AVAILABLE

pytestmark = pytest.mark.skipif(
    not TORCH_AVAILABLE, reason="requires the optional [model] extra (torch)"
)

import torch  # noqa: E402

from nichelens_st.encoder import _info_nce, _info_nce_minibatch  # noqa: E402


def _norm(t):
    return torch.nn.functional.normalize(t, dim=1)


def test_minibatch_labels_none_matches_no_labels():
    torch.manual_seed(0)
    z1 = _norm(torch.randn(8, 6))
    z2 = _norm(torch.randn(8, 6))
    g1 = torch.Generator().manual_seed(0)
    g2 = torch.Generator().manual_seed(0)
    a = _info_nce_minibatch(z1, z2, 0.2, 4, g1)
    b = _info_nce_minibatch(z1, z2, 0.2, 4, g2, labels=None)
    assert torch.allclose(a, b)


def test_minibatch_fullbatch_fallback_applies_labels():
    # batch_size <= 0 degenerates to the full-batch loss, which must still honor
    # labels masking (all-same-group -> only positives remain -> loss ~ 0).
    torch.manual_seed(0)
    z1 = _norm(torch.randn(6, 6))
    z2 = _norm(torch.randn(6, 6))
    labels = torch.zeros(6, dtype=torch.long)
    g = torch.Generator().manual_seed(0)
    mb = _info_nce_minibatch(z1, z2, 0.2, 0, g, labels=labels)
    fb = _info_nce(z1, z2, 0.2, labels=labels)
    assert torch.allclose(mb, fb)
    assert mb.item() == pytest.approx(0.0, abs=1e-5)


def test_minibatch_labels_mask_lowers_loss():
    # All cells share a group: masking same-group negatives within each
    # minibatch removes the false-negative repulsion -> strictly lower loss.
    torch.manual_seed(0)
    z1 = _norm(torch.randn(8, 6))
    z2 = _norm(torch.randn(8, 6))
    labels = torch.zeros(8, dtype=torch.long)
    g1 = torch.Generator().manual_seed(0)
    g2 = torch.Generator().manual_seed(0)
    masked = _info_nce_minibatch(z1, z2, 0.2, 4, g1, labels=labels)
    plain = _info_nce_minibatch(z1, z2, 0.2, 4, g2)
    assert masked.item() < plain.item()
