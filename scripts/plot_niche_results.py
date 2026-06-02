#!/usr/bin/env python
"""Render niche results: UMAP of cell embeddings + spatial niche overlay.

Produces two PNG files per dataset:
  1. ``umap_by_niche.png``  – UMAP of learned cell embeddings (H) coloured by
     prototype_id / niche assignment.
  2. ``spatial_niche.png``  – Spatial scatter of tissue coordinates coloured by
     prototype_id.

Accepts either:
  * An on-disk ``niche.npz`` (keys: ``H`` float32, ``prototype_id`` int64) plus
    an AnnData ``.h5ad`` with ``obsm['spatial']``.
  * An AnnData ``.h5ad`` that already carries ``obsm['H']`` / ``obsm['X_niche']``
    and ``obs['prototype_id']``.

UMAP is computed with **scanpy** (prefers umap-learn; falls back to PCA via
``sc.pp.pca`` when umap is unavailable or fails).

Usage
-----
    python scripts/plot_niche_results.py \\
        --npz results/niche-lens-st/outputs/niche.npz \\
        --h5ad data/processed/niche_merfish_slice/anndata.h5ad \\
        --out-dir results/niche-lens-st/plots

Importable entry point
----------------------
    from scripts.plot_niche_results import plot_niche
    plot_niche(npz_path="...", h5ad_path="...", out_dir="plots/")
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# matplotlib is an optional viz dependency. Gate it (like the torch/squidpy
# extras) so importing this module / pytest collection never requires it; a
# non-interactive backend is selected before pyplot is imported. Public entry
# points call ``_require_matplotlib`` and raise an actionable message when it is
# absent.
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402

    _MPL_AVAILABLE = True
except ImportError:  # pragma: no cover - optional viz dep
    matplotlib = None  # type: ignore[assignment]
    plt = None  # type: ignore[assignment]
    _MPL_AVAILABLE = False

_NO_MPL_MSG = (
    "Rendering niche plots requires matplotlib (optional viz dependency). "
    "Install it with `pip install matplotlib` (or the project's [viz] extra)."
)


def _require_matplotlib() -> None:
    if not _MPL_AVAILABLE:
        raise ImportError(_NO_MPL_MSG)

# Resolve repo root so the script works from any cwd.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))


# ---------------------------------------------------------------------------
# Core plotting helpers
# ---------------------------------------------------------------------------

def _discrete_cmap(n: int):
    """Return a list of n colours from a qualitative colormap."""
    cmap = plt.get_cmap("tab20" if n <= 20 else "hsv")
    return [cmap(i / max(n - 1, 1)) for i in range(n)]


def _scatter_colored(
    ax: plt.Axes,
    xy: np.ndarray,
    labels: np.ndarray,
    title: str,
    xlabel: str = "dim 1",
    ylabel: str = "dim 2",
    point_size: float = 4.0,
) -> None:
    """Draw a scatter plot with one colour per unique integer label."""
    unique = np.unique(labels)
    colors = _discrete_cmap(len(unique))
    for idx, uid in enumerate(unique):
        mask = labels == uid
        ax.scatter(
            xy[mask, 0],
            xy[mask, 1],
            s=point_size,
            c=[colors[idx]],
            label=f"niche {uid}",
            alpha=0.7,
            linewidths=0,
            rasterized=True,
        )
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=7)
    # Legend outside when ≤ 20 niches; skip otherwise to avoid clutter.
    if len(unique) <= 20:
        ax.legend(
            markerscale=3,
            fontsize=6,
            loc="upper right",
            framealpha=0.6,
            ncol=max(1, len(unique) // 10),
        )


# ---------------------------------------------------------------------------
# UMAP / dimensionality reduction
# ---------------------------------------------------------------------------

def _compute_embedding(H: np.ndarray, seed: int = 0) -> np.ndarray:
    """Return 2-D embedding of H using scanpy/UMAP with PCA fallback.

    Parameters
    ----------
    H:
        Cell embedding matrix, shape ``(n_cells, embed_dim)`` float32/64.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Shape ``(n_cells, 2)``.
    """
    try:
        import anndata
        import scanpy as sc

        adata_emb = anndata.AnnData(X=np.asarray(H, dtype=np.float32))
        # scanpy.pp.neighbors needs a minimum of 2 cells and sensible n_pcs.
        n_cells, d = adata_emb.shape
        # arpack solver requires n_comps < min(n_samples, n_features) strictly.
        n_pcs = min(d, 50, n_cells) - 1
        if n_pcs < 2:
            raise ValueError("too few dimensions for PCA")

        sc.pp.pca(adata_emb, n_comps=n_pcs, random_state=seed)
        # Use PCA-precomputed neighbors; n_neighbors capped for small datasets.
        n_neighbors = min(15, n_cells - 1)
        sc.pp.neighbors(
            adata_emb,
            n_neighbors=n_neighbors,
            n_pcs=n_pcs,
            random_state=seed,
        )
        try:
            sc.tl.umap(adata_emb, random_state=seed)
            return np.asarray(adata_emb.obsm["X_umap"], dtype=np.float64)
        except Exception:  # umap-learn not installed or fails – fall back to PCA
            return np.asarray(adata_emb.obsm["X_pca"][:, :2], dtype=np.float64)

    except ImportError:
        # Neither scanpy nor anndata: pure PCA via numpy SVD.
        Hc = np.asarray(H, dtype=np.float64)
        Hc -= Hc.mean(axis=0)
        _, _, Vt = np.linalg.svd(Hc, full_matrices=False)
        return (Hc @ Vt[:2].T)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_niche(
    npz_path: str | Path | None = None,
    h5ad_path: str | Path | None = None,
    out_dir: str | Path = ".",
    seed: int = 0,
    point_size: float = 4.0,
    dpi: int = 150,
) -> dict[str, Path]:
    """Render niche results to PNG files.

    Parameters
    ----------
    npz_path:
        Path to ``niche.npz`` with keys ``H`` (float32, shape ``(n, d)``) and
        ``prototype_id`` (int64, shape ``(n,)``).  Required unless the h5ad
        already contains ``obsm['H']`` or ``obsm['X_niche']`` and
        ``obs['prototype_id']``.
    h5ad_path:
        Path to AnnData ``.h5ad`` with ``obsm['spatial']`` for the spatial
        scatter.  Optional when only the UMAP plot is desired.
    out_dir:
        Directory where PNGs are written (created if absent).
    seed:
        Random seed forwarded to the UMAP computation.
    point_size:
        Marker size for scatter plots.
    dpi:
        Resolution for saved figures.

    Returns
    -------
    dict
        ``{"umap": Path, "spatial": Path}`` for each output that was written.
        Keys are absent when the corresponding plot could not be produced (e.g.
        no spatial coords available).
    """
    _require_matplotlib()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load H and prototype_id
    # ------------------------------------------------------------------
    H: np.ndarray | None = None
    prototype_id: np.ndarray | None = None
    spatial: np.ndarray | None = None

    if npz_path is not None:
        data = np.load(str(npz_path))
        H = np.asarray(data["H"], dtype=np.float32)
        prototype_id = np.asarray(data["prototype_id"], dtype=np.int64)

    if h5ad_path is not None:
        try:
            import anndata as _ad
            adata = _ad.read_h5ad(str(h5ad_path))
        except ImportError:
            try:
                import scanpy as sc
                adata = sc.read_h5ad(str(h5ad_path))
            except ImportError:
                adata = None

        if adata is not None:
            # Try to get H from obsm if not supplied via npz.
            if H is None:
                for key in ("H", "X_niche", "X_embedding"):
                    if key in adata.obsm:
                        H = np.asarray(adata.obsm[key], dtype=np.float32)
                        break
            if prototype_id is None and "prototype_id" in adata.obs.columns:
                prototype_id = np.asarray(
                    adata.obs["prototype_id"].to_numpy(), dtype=np.int64
                )
            if "spatial" in adata.obsm:
                spatial = np.asarray(adata.obsm["spatial"], dtype=np.float64)

    if H is None:
        raise ValueError(
            "H embedding not found: supply --npz with 'H' key or --h5ad with "
            "obsm['H'] / obsm['X_niche']."
        )
    if prototype_id is None:
        raise ValueError(
            "prototype_id not found: supply --npz with 'prototype_id' key or "
            "--h5ad with obs['prototype_id']."
        )

    outputs: dict[str, Path] = {}

    # ------------------------------------------------------------------
    # Plot 1: UMAP coloured by prototype_id
    # ------------------------------------------------------------------
    emb2d = _compute_embedding(H, seed=seed)

    fig, ax = plt.subplots(figsize=(6, 5))
    _scatter_colored(
        ax,
        emb2d,
        prototype_id,
        title="Cell embeddings (UMAP) coloured by niche / prototype",
        xlabel="UMAP 1",
        ylabel="UMAP 2",
        point_size=point_size,
    )
    fig.tight_layout()
    umap_path = out_dir / "umap_by_niche.png"
    fig.savefig(umap_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    outputs["umap"] = umap_path

    # ------------------------------------------------------------------
    # Plot 2: Spatial scatter coloured by prototype_id
    # ------------------------------------------------------------------
    if spatial is not None:
        fig, ax = plt.subplots(figsize=(6, 5))
        _scatter_colored(
            ax,
            spatial[:, :2],
            prototype_id,
            title="Spatial niche assignment",
            xlabel="x",
            ylabel="y",
            point_size=point_size,
        )
        ax.set_aspect("equal", adjustable="datalim")
        fig.tight_layout()
        spatial_path = out_dir / "spatial_niche.png"
        fig.savefig(spatial_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        outputs["spatial"] = spatial_path

    return outputs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--npz",
        metavar="PATH",
        default=None,
        help="Path to niche.npz (keys: H, prototype_id).",
    )
    p.add_argument(
        "--h5ad",
        metavar="PATH",
        default=None,
        help="Path to AnnData .h5ad with obsm['spatial'].",
    )
    p.add_argument(
        "--out-dir",
        metavar="DIR",
        default="results/niche-lens-st/plots",
        help="Output directory for PNGs (default: results/niche-lens-st/plots).",
    )
    p.add_argument("--seed", type=int, default=0, help="Random seed (default: 0).")
    p.add_argument(
        "--point-size", type=float, default=4.0, help="Scatter point size (default: 4)."
    )
    p.add_argument("--dpi", type=int, default=150, help="PNG resolution (default: 150).")
    return p


def main() -> int:
    args = _build_parser().parse_args()
    if args.npz is None and args.h5ad is None:
        print("ERROR: supply at least --npz or --h5ad", file=sys.stderr)
        return 1
    outputs = plot_niche(
        npz_path=args.npz,
        h5ad_path=args.h5ad,
        out_dir=args.out_dir,
        seed=args.seed,
        point_size=args.point_size,
        dpi=args.dpi,
    )
    for kind, path in outputs.items():
        print(f"{kind}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
