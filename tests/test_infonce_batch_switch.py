"""train_embeddings selects the contrastive loss by batch_size (option B).

Reconciles #61/#148 (minibatch, OOM-safe) with #92/#103 (group-aware negative
masking): the full-batch path applies labels masking; the minibatch path is the
large-scale path (masking within minibatches is a deferred enhancement). The
default (batch_size=0, labels=None) is unchanged.
"""

from __future__ import annotations

import numpy as np
import pytest

from nichelens_st.encoder import TORCH_AVAILABLE

pytestmark = pytest.mark.skipif(
    not TORCH_AVAILABLE, reason="requires the optional [model] extra (torch)"
)

import pytest as _pytest  # noqa: E402

from nichelens_st import encoder as enc  # noqa: E402
from nichelens_st.encoder import EncoderConfig, train_embeddings  # noqa: E402


def _data():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((12, 6)).astype(np.float32)
    edges = np.array([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]], dtype=np.int64)
    return X, edges


def test_fullbatch_path_forwards_labels():
    X, edges = _data()
    labels = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1])
    seen: dict = {}
    real = enc._info_nce

    def spy(z1, z2, tau, labels=None):
        seen["called"] = True
        seen["labels_not_none"] = labels is not None
        return real(z1, z2, tau, labels)

    mp = _pytest.MonkeyPatch()
    mp.setattr(enc, "_info_nce", spy)
    try:
        train_embeddings(
            X, edges,
            EncoderConfig(embed_dim=4, hidden_dim=8, epochs=1, batch_size=0, seed=0),
            labels=labels,
        )
    finally:
        mp.undo()
    assert seen.get("called") and seen.get("labels_not_none")


def test_minibatch_path_used_when_batch_size_positive():
    X, edges = _data()
    seen: dict = {}
    real = enc._info_nce_minibatch

    def spy(z1, z2, tau, batch_size, generator):
        seen["called"] = True
        return real(z1, z2, tau, batch_size, generator)

    mp = _pytest.MonkeyPatch()
    mp.setattr(enc, "_info_nce_minibatch", spy)
    try:
        train_embeddings(
            X, edges,
            EncoderConfig(embed_dim=4, hidden_dim=8, epochs=1, batch_size=4, seed=0),
        )
    finally:
        mp.undo()
    assert seen.get("called")


def test_labels_none_default_unchanged():
    X, edges = _data()
    cfg = EncoderConfig(embed_dim=4, hidden_dim=8, epochs=3, batch_size=0, seed=0)
    assert np.array_equal(train_embeddings(X, edges, cfg),
                          train_embeddings(X, edges, cfg, labels=None))
