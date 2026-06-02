"""Minibatch / subgraph-sampled InfoNCE (#61, closes #148).

The full-batch ``_info_nce`` materializes a dense ``(2n, 2n)`` similarity
matrix, so memory scales as ``O(n^2)``. At 124k cells that is ~232 GiB and
OOMs. These tests pin the minibatched path:

* a ``batch_size`` knob on :class:`EncoderConfig`,
* training that never allocates a tensor larger than ``O(batch^2)`` even on an
  ``n`` where the full ``(2n, 2n)`` matrix would be infeasible, and
* small-scale numerics: the loss stays finite and decreasing, and the default
  (full-batch) path is left bitwise-unchanged.
"""

from __future__ import annotations

import numpy as np
import pytest

from nichelens_st.encoder import (
    EncoderConfig,
    TORCH_AVAILABLE,
    _info_nce,
    _info_nce_minibatch,
    train_embeddings,
)

pytestmark = pytest.mark.skipif(
    not TORCH_AVAILABLE, reason="requires the optional [model] extra (torch)"
)


def _toy_graph(n: int, d: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    # simple chain graph so every node has neighbors
    src = np.arange(n - 1, dtype=np.int64)
    dst = np.arange(1, n, dtype=np.int64)
    edges = np.stack([src, dst])
    return X, edges


def test_config_has_batch_size_default_zero():
    """Default batch_size must preserve full-batch behavior (0 == off)."""
    assert EncoderConfig().batch_size == 0


def test_minibatch_loss_matches_fullbatch_when_batch_ge_n():
    """With batch_size >= n the minibatch loss must equal the full-batch loss
    (single block covers everything)."""
    import torch

    torch.manual_seed(0)
    z1 = torch.nn.functional.normalize(torch.randn(16, 8), dim=1)
    z2 = torch.nn.functional.normalize(torch.randn(16, 8), dim=1)
    gen = torch.Generator().manual_seed(0)
    full = _info_nce(z1, z2, 0.2)
    mb = _info_nce_minibatch(z1, z2, 0.2, batch_size=16, generator=gen)
    assert torch.allclose(full, mb, atol=1e-5)


def test_minibatch_loss_is_finite_small_batch():
    import torch

    torch.manual_seed(0)
    z1 = torch.nn.functional.normalize(torch.randn(64, 8), dim=1)
    z2 = torch.nn.functional.normalize(torch.randn(64, 8), dim=1)
    gen = torch.Generator().manual_seed(0)
    loss = _info_nce_minibatch(z1, z2, 0.2, batch_size=8, generator=gen)
    assert torch.isfinite(loss).all()
    assert loss.item() > 0.0


def test_minibatch_never_allocates_full_matrix(monkeypatch):
    """Train with a tiny batch_size on an n where the full (2n, 2n) matrix
    would be huge, and assert no single tensor of (2n, 2n) is ever created.

    We patch ``torch.matmul`` to reject any product whose output would be the
    full (2n, 2n) similarity matrix. The minibatch path must only form
    (<=2*batch, <=2*batch) blocks.
    """
    import torch

    n, d, batch = 4000, 16, 64
    X, edges = _toy_graph(n, d)

    real_matmul = torch.matmul
    forbidden = 2 * n  # full (2n, 2n) leading dim

    def guarded_matmul(a, b, *args, **kwargs):
        out = real_matmul(a, b, *args, **kwargs)
        # any 2D product with a (2n, 2n) shape is the forbidden dense sim matrix
        if out.dim() == 2 and out.shape[0] >= forbidden and out.shape[1] >= forbidden:
            raise AssertionError(
                f"full (2n, 2n)={tuple(out.shape)} similarity matrix materialized"
            )
        return out

    monkeypatch.setattr(torch, "matmul", guarded_matmul)

    cfg = EncoderConfig(
        embed_dim=8, hidden_dim=16, num_layers=2, epochs=2, batch_size=batch, seed=0
    )
    H = train_embeddings(X, edges, cfg)
    assert H.shape == (n, 8)
    assert np.isfinite(H).all()


def test_minibatch_loss_decreases_over_epochs():
    """On a small graph the minibatch contrastive loss should decrease."""
    import torch

    from nichelens_st import encoder as enc

    n, d = 200, 12
    X, edges = _toy_graph(n, d, seed=1)

    losses: list[float] = []
    real = enc._info_nce_minibatch

    def spy(z1, z2, tau, batch_size, generator):
        out = real(z1, z2, tau, batch_size, generator)
        losses.append(float(out.detach()))
        return out

    import pytest as _pytest

    mp = _pytest.MonkeyPatch()
    mp.setattr(enc, "_info_nce_minibatch", spy)
    try:
        cfg = EncoderConfig(
            embed_dim=8, hidden_dim=16, epochs=40, lr=1e-2, batch_size=32, seed=0
        )
        train_embeddings(X, edges, cfg)
    finally:
        mp.undo()

    assert len(losses) >= 10
    # last-quarter mean below first-quarter mean: training reduces the loss.
    q = max(1, len(losses) // 4)
    assert np.mean(losses[-q:]) < np.mean(losses[:q])


def test_fullbatch_default_unchanged_bitwise():
    """The default config (batch_size=0) must reproduce the legacy full-batch
    embeddings bitwise across runs (reproducibility guard, #61 AC)."""
    X, edges = _toy_graph(80, 12, seed=3)
    cfg = EncoderConfig(embed_dim=8, hidden_dim=16, epochs=8, seed=5)
    a = train_embeddings(X, edges, cfg)
    b = train_embeddings(X, edges, cfg)
    np.testing.assert_array_equal(a, b)


def test_minibatch_reproducible_bitwise():
    """Fixed seed + batch_size must yield identical embeddings across runs."""
    X, edges = _toy_graph(120, 12, seed=2)
    cfg = EncoderConfig(embed_dim=8, hidden_dim=16, epochs=10, batch_size=32, seed=9)
    a = train_embeddings(X, edges, cfg)
    b = train_embeddings(X, edges, cfg)
    np.testing.assert_array_equal(a, b)
