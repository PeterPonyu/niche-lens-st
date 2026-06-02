"""Group-aware negative masking in InfoNCE (issues #92 / #103).

Vanilla NT-Xent treats every non-twin cell as a negative, so same-prototype
cells are pushed apart (false negatives). Passing per-node group ``labels`` masks
those same-group off-diagonal entries from the softmax denominator. ``None``
reproduces vanilla NT-Xent exactly.
"""

from __future__ import annotations

import numpy as np
import pytest

from nichelens_st.encoder import TORCH_AVAILABLE

pytestmark = pytest.mark.skipif(
    not TORCH_AVAILABLE, reason="requires the optional [model] extra (torch)"
)

import torch  # noqa: E402

from nichelens_st.encoder import (  # noqa: E402
    EncoderConfig,
    _info_nce,
    train_embeddings,
)


def _norm(t):
    return torch.nn.functional.normalize(t, dim=1)


def test_labels_none_matches_vanilla():
    torch.manual_seed(0)
    z1 = _norm(torch.randn(5, 8))
    z2 = _norm(torch.randn(5, 8))
    assert torch.allclose(_info_nce(z1, z2, 0.2), _info_nce(z1, z2, 0.2, labels=None))


def test_all_same_label_masks_every_negative():
    # Every cell shares one prototype: masking leaves only the positive twin in
    # each row's denominator, so the loss collapses to ~0 -- and is strictly
    # below the vanilla loss, which still has real negatives.
    torch.manual_seed(0)
    z1 = _norm(torch.randn(6, 8))
    z2 = _norm(torch.randn(6, 8))
    labels = torch.zeros(6, dtype=torch.long)
    masked = _info_nce(z1, z2, 0.2, labels=labels)
    vanilla = _info_nce(z1, z2, 0.2)
    assert masked.item() < vanilla.item()
    assert masked.item() == pytest.approx(0.0, abs=1e-5)


def test_distinct_labels_keep_all_negatives():
    # Every cell in its own group -> no same-group off-diagonal -> == vanilla.
    torch.manual_seed(0)
    z1 = _norm(torch.randn(4, 8))
    z2 = _norm(torch.randn(4, 8))
    labels = torch.arange(4)
    assert torch.allclose(_info_nce(z1, z2, 0.2, labels=labels), _info_nce(z1, z2, 0.2))


def test_labels_length_validated():
    z1 = _norm(torch.randn(3, 8))
    z2 = _norm(torch.randn(3, 8))
    with pytest.raises(ValueError):
        _info_nce(z1, z2, 0.2, labels=torch.zeros(2, dtype=torch.long))


def test_train_embeddings_accepts_labels():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((12, 6)).astype(np.float32)
    edges = np.array([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]], dtype=np.int64)
    labels = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1])
    cfg = EncoderConfig(embed_dim=4, hidden_dim=8, epochs=3, seed=0)
    H = train_embeddings(X, edges, cfg, labels=labels)
    assert H.shape == (12, 4)
    assert np.isfinite(H).all()


def test_train_embeddings_labels_none_unchanged():
    # labels=None must reproduce the default-path embeddings bit-for-bit.
    rng = np.random.default_rng(1)
    X = rng.standard_normal((10, 5)).astype(np.float32)
    edges = np.array([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.int64)
    cfg = EncoderConfig(embed_dim=4, hidden_dim=8, epochs=4, seed=0)
    h_default = train_embeddings(X, edges, cfg)
    h_none = train_embeddings(X, edges, cfg, labels=None)
    assert np.array_equal(h_default, h_none)
