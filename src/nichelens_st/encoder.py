"""Contrastive niche-token encoder over cell-centered subgraphs.

Dependency choice
-----------------
The encoder is implemented in **plain PyTorch** (``torch``), declared as the
optional ``[model]`` extra in ``pyproject.toml``. The base package stays
dependency-light (numpy/scipy); ``torch`` is imported lazily so importing this
module never requires it. We deliberately avoid ``torch-geometric`` and ``jax``:
message passing over the cell-centered subgraph edge lists is expressed with
``index_add_`` (scatter) on plain tensors, which keeps the dependency surface
small and the receptive field explicit.

Message passing
---------------
For cell ``i`` the *cell-centered subgraph* is the induced subgraph over the
``k``-hop neighborhood of ``i`` (``docs/MVP_DESIGN.md``). Stacking ``L`` mean-
aggregation layers over the per-section graph yields, for each center node, an
embedding whose receptive field is exactly the ``L``-hop subgraph -- so running
``L = k_hop`` layers on the full (per-section) edge list and reading out the
center node is equivalent to encoding each cell's subgraph, without materializing
one tensor per cell. Each layer computes

    h_v <- ReLU( W [ x_v ; mean_{u in N(v)} h_u ] )

where the neighbor mean is a scatter (``index_add_``) over the COO edge list,
treated as undirected. The final layer output is L2-normalized to live on the
unit sphere for the contrastive objective.

Contrastive objective (InfoNCE)
-------------------------------
Positives are two stochastic *augmented views* of the same neighborhood:
per-view feature dropout on ``X`` and edge dropout on the graph. Both views are
encoded; for each anchor cell the matching cell in the other view is the single
positive and all other cells in the batch are negatives (NT-Xent / InfoNCE with
temperature ``tau``). This trains the encoder so that two augmentations of the
same niche map to nearby embeddings while distinct niches separate.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

import numpy as np

# Optional ``[model]`` extra. The names below are deliberately typed ``Any`` so
# this module imports without torch; torch-backed code paths call
# ``_require_torch`` first and raise a clear error when the extra is absent.
torch: Any = None
nn: Any = None
Tensor: Any = object
TORCH_AVAILABLE: bool = False

try:  # pragma: no cover - the import-failure branch needs torch uninstalled
    torch = importlib.import_module("torch")
    nn = torch.nn
    Tensor = torch.Tensor
    TORCH_AVAILABLE = True
except ImportError:
    pass


_NO_TORCH_MSG = (
    "The contrastive niche encoder requires PyTorch. Install the optional "
    "model extra: `pip install nichelens-st[model]`."
)


def _require_torch() -> None:
    if not TORCH_AVAILABLE:
        raise ImportError(_NO_TORCH_MSG)


def _undirected_edge_index(edges: np.ndarray) -> "Tensor":
    """Symmetrize a (2, n_edges) int64 COO array into a torch edge index.

    Self-loops are added so isolated/center nodes always aggregate their own
    feature during message passing.
    """
    src = edges[0]
    dst = edges[1]
    sym_src = np.concatenate([src, dst])
    sym_dst = np.concatenate([dst, src])
    return torch.from_numpy(np.stack([sym_src, sym_dst]).astype(np.int64))


if TORCH_AVAILABLE:

    def _l2_normalize(h: "Tensor") -> "Tensor":
        """L2-normalize rows onto the unit sphere with a float32-safe ``eps``.

        torch's default ``eps=1e-12`` is far below the smallest meaningful
        float32 norm. For a near-zero-norm row (e.g. an isolated node with
        zero/dropped features), dividing by a tiny norm amplifies it into a
        noise-dominated unit vector that then dominates the InfoNCE similarity
        matrix. ``eps=1e-6`` keeps such degenerate rows bounded.
        """
        return torch.nn.functional.normalize(h, dim=1, eps=1e-6)

    class _MeanAggLayer(nn.Module):
        """One GraphSAGE-mean message-passing layer using scatter add."""

        def __init__(self, in_dim: int, out_dim: int) -> None:
            super().__init__()
            self.lin = nn.Linear(in_dim * 2, out_dim)

        def forward(self, h: Tensor, edge_index: Tensor, n_nodes: int) -> Tensor:
            src, dst = edge_index[0], edge_index[1]
            # Mean of neighbor features: sum messages into dst, divide by degree.
            agg = h.new_zeros((n_nodes, h.shape[1]))
            agg.index_add_(0, dst, h[src])
            deg = h.new_zeros((n_nodes, 1))
            ones = h.new_ones((src.shape[0], 1))
            deg.index_add_(0, dst, ones)
            agg = agg / deg.clamp_min(1.0)
            return self.lin(torch.cat([h, agg], dim=1))

    class NicheEncoder(nn.Module):
        """Stacked mean-aggregation encoder mapping cells to niche embeddings.

        ``num_layers`` controls the receptive field (k-hop subgraph depth).
        """

        def __init__(
            self,
            in_dim: int,
            hidden_dim: int,
            embed_dim: int,
            num_layers: int = 2,
        ) -> None:
            super().__init__()
            if num_layers < 1:
                raise ValueError(f"num_layers must be >= 1; got {num_layers}")
            self.layers = nn.ModuleList()
            dims = [in_dim] + [hidden_dim] * (num_layers - 1)
            for i in range(num_layers - 1):
                self.layers.append(_MeanAggLayer(dims[i], dims[i + 1]))
            self.layers.append(_MeanAggLayer(dims[-1], embed_dim))
            self.act = nn.ReLU()

        def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
            n_nodes = x.shape[0]
            h = x
            for i, layer in enumerate(self.layers):
                h = layer(h, edge_index, n_nodes)
                if i < len(self.layers) - 1:
                    h = self.act(h)
            return _l2_normalize(h)


def _augment(
    x: "Tensor",
    edge_index: "Tensor",
    feat_drop: float,
    edge_drop: float,
    generator: "torch.Generator",
) -> tuple["Tensor", "Tensor"]:
    """Produce one stochastic view: feature dropout + edge dropout."""
    if feat_drop > 0.0:
        mask = (
            torch.rand(x.shape, generator=generator, device=x.device) >= feat_drop
        ).to(x.dtype)
        x_aug = x * mask / (1.0 - feat_drop)
    else:
        x_aug = x
    n_edges = edge_index.shape[1]
    if edge_drop > 0.0 and n_edges > 0:
        keep = torch.rand(n_edges, generator=generator, device=x.device) >= edge_drop
        edge_aug = edge_index[:, keep]
    else:
        edge_aug = edge_index
    return x_aug, edge_aug


_INFO_NCE_TAU_MIN: float = 1e-4
"""Lower bound for InfoNCE ``tau`` (issue #88). Below this value
``z @ z.t() / tau`` overflows float32 and softmax in cross-entropy
degenerates to NaN. ``train_embeddings`` rejects ``tau <= 0`` with a
``ValueError``; this constant bounds the positive range against silent
underflow corruption."""


def _info_nce(z1: "Tensor", z2: "Tensor", tau: float) -> "Tensor":
    """NT-Xent / InfoNCE loss over two aligned views of the same nodes.

    ``tau`` is clamped to ``[_INFO_NCE_TAU_MIN, +inf)`` so a tiny user
    setting cannot make ``sim = z @ z.t() / tau`` overflow float32 and
    NaN-out the softmax/cross-entropy. ``tau <= 0`` is rejected upstream
    in :func:`train_embeddings` (issue #88).
    """
    safe_tau = max(float(tau), _INFO_NCE_TAU_MIN)
    n = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)  # (2n, d), already L2-normalized
    sim = z @ z.t() / safe_tau
    # Mask self-similarity.
    diag = torch.eye(2 * n, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(diag, float("-inf"))
    # Positive for row i in [0, n) is i + n, and vice versa.
    targets = torch.cat(
        [torch.arange(n, 2 * n, device=z.device), torch.arange(0, n, device=z.device)]
    )
    return torch.nn.functional.cross_entropy(sim, targets)


@dataclass
class EncoderConfig:
    embed_dim: int = 32
    hidden_dim: int = 64
    num_layers: int = 2
    epochs: int = 30
    lr: float = 1e-2
    tau: float = 0.2
    feat_drop: float = 0.2
    edge_drop: float = 0.2
    seed: int = 0


def train_embeddings(
    X: np.ndarray,
    edges: np.ndarray,
    config: EncoderConfig,
) -> np.ndarray:
    """Train the contrastive encoder and return float32 niche embeddings ``H``.

    Determinism: all randomness (parameter init, augmentation, optimizer) is
    seeded from ``config.seed`` on CPU, so a fixed seed yields identical ``H``.
    """
    _require_torch()
    if X.ndim != 2:
        raise ValueError(f"X must be 2D; got ndim={X.ndim}")
    if not config.tau > 0:
        raise ValueError(f"tau must be > 0; got {config.tau}")

    # Bit-reproducible training: single-threaded CPU + deterministic reductions
    # so ``index_add_`` (scatter) yields identical results across runs of the
    # same seed. This makes both H and prototype_id reproducible.
    torch.manual_seed(config.seed)
    _prev_det = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    _prev_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    device = torch.device("cpu")

    try:
        x = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32)).to(device)
        edge_index = _undirected_edge_index(edges).to(device)

        model = NicheEncoder(
            in_dim=x.shape[1],
            hidden_dim=config.hidden_dim,
            embed_dim=config.embed_dim,
            num_layers=config.num_layers,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        aug_gen = torch.Generator(device=device)
        aug_gen.manual_seed(config.seed)

        model.train()
        for _ in range(config.epochs):
            optimizer.zero_grad()
            x1, e1 = _augment(
                x, edge_index, config.feat_drop, config.edge_drop, aug_gen
            )
            x2, e2 = _augment(
                x, edge_index, config.feat_drop, config.edge_drop, aug_gen
            )
            z1 = model(x1, e1)
            z2 = model(x2, e2)
            loss = _info_nce(z1, z2, config.tau)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            h = model(x, edge_index)
        return h.detach().cpu().numpy().astype(np.float32)
    finally:
        torch.set_num_threads(_prev_threads)
        torch.use_deterministic_algorithms(_prev_det)
