#!/usr/bin/env python
"""Emit N-F1 (spatial niche map) and N-F2 (scalability) paper figure assets.

N-F1 — spatial niche map:
    Calls ``plot_niche()`` from ``plot_niche_results.py`` to produce a spatial
    scatter of tissue coordinates coloured by prototype_id (niche assignment).
    This is the primary N-F1 figure ("niche map — spatial co-localisation of
    cell-type niches overlaid on tissue section").

N-F2 — scalability figure + JSON:
    Reads one or more ``run_metadata.json`` files and plots ``runtime_s`` and
    ``peak_rss_bytes`` vs ``n_obs``.  An optional k-sweep panel is added when a
    ``k_stability_sweep.json`` file is provided.  Also emits a deterministic
    ``scalability.json`` table with columns ``n_obs``, ``runtime_s``,
    ``peak_rss_bytes`` (``None``/null when absent — never fabricated).

CONTRACT: ``peak_rss_bytes`` is read from ``run_metadata["peak_rss_bytes"]``
(exact key, int bytes).  Missing field → ``None`` in all artifacts; never 0.

Usage::

    # N-F1
    python scripts/emit_figures.py nf1 \\
        --npz results/niche-lens-st/outputs/niche.npz \\
        --h5ad data/processed/niche_merfish_slice/anndata.h5ad \\
        --out-dir results/paper_metrics

    # N-F2
    python scripts/emit_figures.py nf2 \\
        --run-metadata results/niche-lens-st/run_metadata.json \\
        --k-sweep results/niche-lens-st/outputs/k_stability_sweep.json \\
        --out-dir results/paper_metrics

Importable entry points::

    from scripts.emit_figures import emit_niche_map, emit_scalability
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional

# Non-interactive backend must be set before any matplotlib import.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from plot_niche_results import plot_niche  # noqa: E402

_N_F1_STUB_REL = Path("results") / "paper_metrics" / "n-f1_metrics_stub.json"
_N_F2_STUB_REL = Path("results") / "paper_metrics" / "n-f2_metrics_stub.json"

SCHEMA_VERSION = "1.0.0"
_SCRIPT_REF = "scripts/emit_figures.py"


# ---------------------------------------------------------------------------
# JSON I/O helpers (mirrored from emit_results_tables.py)
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> dict:
    """Load JSON; raise RuntimeError with context on decode failure."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON in {path}: {exc}") from exc


def _atomic_write(path: Path, text: str) -> None:
    """Write *text* to *path* atomically via a temp-file + os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# Fallback detection
# ---------------------------------------------------------------------------


def _detect_fallback(run_metadata_path: Optional[Path]) -> tuple[bool, str]:
    """Return ``(is_fallback, note)`` from *run_metadata_path*.

    Returns ``(False, "")`` when the file is absent or has no fallback note.
    """
    if run_metadata_path is None or not run_metadata_path.exists():
        return False, ""
    try:
        meta = _load_json(run_metadata_path)
        note = meta.get("_fallback_note", "")
        if note:
            return True, str(note)
    except RuntimeError:
        pass
    return False, ""


# ---------------------------------------------------------------------------
# Stub updater (mirrors emit_results_tables._update_stub pattern)
# ---------------------------------------------------------------------------


def _update_stub(stub_path: Path, is_fallback: bool) -> None:
    """Update a paper-metrics stub after a successful artifact emit.

    Sets ``source_exists``, ``metric_source_exists``, ``artifact_exists``, and
    ``source_script_reference``.  ``paper_claim_ready`` is NEVER promoted to
    ``True`` here — only manual review can do that.
    """
    if not stub_path.exists():
        return
    try:
        data = _load_json(stub_path)
    except RuntimeError:
        return
    data["source_exists"] = True
    data["metric_source_exists"] = True
    data["artifact_exists"] = True
    data["source_script_reference"] = _SCRIPT_REF
    data["readiness_status"] = "script-ready"
    # Gate: fallback data must not become paper-claim-ready.
    if is_fallback:
        data["paper_claim_ready"] = False
        data["supports_safe_prose"] = False
    # We never auto-promote paper_claim_ready; that requires manual review.
    _atomic_write(stub_path, json.dumps(data, indent=2) + "\n")


# ---------------------------------------------------------------------------
# N-F1: spatial niche map
# ---------------------------------------------------------------------------


def emit_niche_map(
    npz_path: "str | Path",
    h5ad_path: "Optional[str | Path]" = None,
    out_dir: "str | Path" = ".",
    run_metadata_path: "Optional[str | Path]" = None,
    stub_path: "Optional[str | Path]" = None,
    seed: int = 0,
    dpi: int = 150,
) -> "dict[str, Path]":
    """Emit N-F1: spatial niche-map PNG (reuses ``plot_niche``).

    Parameters
    ----------
    npz_path:
        Path to ``niche.npz`` (keys: ``H`` float32, ``prototype_id`` int64).
    h5ad_path:
        Optional AnnData ``.h5ad`` with ``obsm['spatial']`` for the spatial
        scatter (the primary N-F1 figure panel).
    out_dir:
        Output directory.  Created if absent.
    run_metadata_path:
        Optional path to ``run_metadata.json`` to detect fallback status for
        stub gating.
    stub_path:
        Optional path to ``n-f1_metrics_stub.json`` to update after emit.
        Silently skipped when ``None`` or the file does not exist.
    seed:
        Random seed forwarded to the UMAP computation.
    dpi:
        PNG resolution.

    Returns
    -------
    dict
        Keys from ``plot_niche``: ``"umap"`` always present; ``"spatial"``
        present when spatial coordinates are available.
    """
    out_dir = Path(out_dir)
    outputs = plot_niche(
        npz_path=npz_path,
        h5ad_path=h5ad_path,
        out_dir=out_dir,
        seed=seed,
        dpi=dpi,
    )

    run_meta_path = Path(run_metadata_path) if run_metadata_path is not None else None
    is_fallback, _ = _detect_fallback(run_meta_path)

    if stub_path is not None:
        _update_stub(Path(stub_path), is_fallback)

    return outputs


# ---------------------------------------------------------------------------
# N-F2: scalability figure + JSON
# ---------------------------------------------------------------------------


def emit_scalability(
    run_metadata_paths: "list[str | Path]",
    out_dir: "str | Path" = ".",
    k_sweep_path: "Optional[str | Path]" = None,
    stub_path: "Optional[str | Path]" = None,
) -> "dict[str, Path]":
    """Emit N-F2: scalability figure (PNG) + machine-readable ``scalability.json``.

    Parameters
    ----------
    run_metadata_paths:
        List of paths to ``run_metadata.json`` files (one per run).  Rows are
        sorted by ``n_obs`` in the output.  Empty list → 0-row table.
    out_dir:
        Output directory.  Created if absent.
    k_sweep_path:
        Optional path to ``k_stability_sweep.json`` for a hyperparameter-sweep
        panel.  Absent or missing file → panel omitted, no error.
    stub_path:
        Optional path to ``n-f2_metrics_stub.json`` to update after emit.
        Silently skipped when ``None`` or the file does not exist.

    Returns
    -------
    dict
        ``{"figure": Path, "json": Path}`` — scalability.png + scalability.json.

    Notes
    -----
    ``peak_rss_bytes`` is read from ``run_metadata["peak_rss_bytes"]`` (exact
    key, int bytes).  Missing field → ``None`` / null in all outputs; NEVER 0
    or any fabricated value.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load run rows
    # ------------------------------------------------------------------
    rows: list[dict] = []
    is_fallback = False

    for raw_p in run_metadata_paths:
        p = Path(raw_p)
        if not p.exists():
            continue
        try:
            meta = _load_json(p)
        except RuntimeError:
            continue

        n_obs = meta.get("n_obs")
        if n_obs is None:
            # Try nested shapes dict (older schema).
            n_obs = (meta.get("shapes") or {}).get("n_obs")

        runtime_s = meta.get("runtime_s")
        # Exact key contract: missing field → None (NEVER fabricate 0).
        peak_rss_bytes = meta.get("peak_rss_bytes")

        if meta.get("_fallback_note"):
            is_fallback = True

        rows.append(
            {
                "n_obs": n_obs,
                "runtime_s": runtime_s,
                "peak_rss_bytes": peak_rss_bytes,
            }
        )

    # Sort by n_obs ascending; rows with n_obs=None sort last.
    rows.sort(key=lambda r: (r["n_obs"] is None, r["n_obs"] or 0))

    # ------------------------------------------------------------------
    # Load k-sweep (optional)
    # ------------------------------------------------------------------
    k_sweep: Optional[dict] = None
    if k_sweep_path is not None:
        kp = Path(k_sweep_path)
        if kp.exists():
            try:
                k_sweep = _load_json(kp)
            except RuntimeError:
                k_sweep = None

    # ------------------------------------------------------------------
    # Build figure
    # ------------------------------------------------------------------
    has_memory = any(r["peak_rss_bytes"] is not None for r in rows)
    has_sweep = k_sweep is not None and "per_k" in k_sweep
    n_panels = 1 + int(has_memory) + int(has_sweep)

    fig, axes_2d = plt.subplots(
        1, n_panels, figsize=(4 * n_panels, 4), squeeze=False
    )
    axes = list(axes_2d[0])

    panel_idx = 0

    # Panel 1: runtime_s vs n_obs
    ax = axes[panel_idx]
    rt_x = [r["n_obs"] for r in rows if r["n_obs"] is not None and r["runtime_s"] is not None]
    rt_y = [r["runtime_s"] for r in rows if r["n_obs"] is not None and r["runtime_s"] is not None]
    if rt_x:
        ax.plot(rt_x, rt_y, "o-", markersize=6)
    ax.set_xlabel("n_obs", fontsize=9)
    ax.set_ylabel("runtime (s)", fontsize=9)
    ax.set_title("Runtime vs dataset size", fontsize=10)
    ax.tick_params(labelsize=8)
    panel_idx += 1

    # Panel 2: peak_rss_bytes vs n_obs (GiB), only when present.
    if has_memory:
        ax = axes[panel_idx]
        mem_x = [
            r["n_obs"]
            for r in rows
            if r["n_obs"] is not None and r["peak_rss_bytes"] is not None
        ]
        mem_y = [
            r["peak_rss_bytes"] / (1024 ** 3)
            for r in rows
            if r["n_obs"] is not None and r["peak_rss_bytes"] is not None
        ]
        if mem_x:
            ax.plot(mem_x, mem_y, "s-", color="tab:orange", markersize=6)
        ax.set_xlabel("n_obs", fontsize=9)
        ax.set_ylabel("peak RSS (GiB)", fontsize=9)
        ax.set_title("Peak memory vs dataset size", fontsize=10)
        ax.tick_params(labelsize=8)
        panel_idx += 1

    # Panel 3: k-sweep silhouette vs k.
    if has_sweep and k_sweep is not None:
        ax = axes[panel_idx]
        per_k: dict = k_sweep["per_k"]
        ks = sorted(int(k) for k in per_k)
        silhouettes = [per_k[str(k)]["silhouette"] for k in ks]
        fitted_k = k_sweep.get("fitted_k")
        ax.plot(ks, silhouettes, "o-", color="tab:green", markersize=5)
        if fitted_k is not None:
            ax.axvline(
                x=fitted_k,
                color="red",
                linestyle="--",
                linewidth=1.0,
                label=f"k={fitted_k}",
            )
            ax.legend(fontsize=8)
        ax.set_xlabel("k (n_niches)", fontsize=9)
        ax.set_ylabel("silhouette score", fontsize=9)
        ax.set_title("Hyperparameter sweep: k", fontsize=10)
        ax.tick_params(labelsize=8)

    fig.tight_layout()
    figure_path = out_dir / "scalability.png"
    fig.savefig(figure_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Emit scalability.json
    # ------------------------------------------------------------------
    payload: dict = {
        "schema_version": SCHEMA_VERSION,
        "columns": ["n_obs", "runtime_s", "peak_rss_bytes"],
        "paper_claim_ready": False,
        "rows": rows,
    }
    json_path = out_dir / "scalability.json"
    _atomic_write(json_path, json.dumps(payload, indent=2, sort_keys=False) + "\n")

    # ------------------------------------------------------------------
    # Update stub
    # ------------------------------------------------------------------
    if stub_path is not None:
        _update_stub(Path(stub_path), is_fallback)

    return {"figure": figure_path, "json": json_path}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="figure", required=True)

    # --- nf1 ---------------------------------------------------------------
    nf1 = sub.add_parser("nf1", help="Emit N-F1 spatial niche-map PNG.")
    nf1.add_argument("--npz", required=True, metavar="PATH", help="Path to niche.npz.")
    nf1.add_argument(
        "--h5ad",
        default=None,
        metavar="PATH",
        help="AnnData .h5ad with obsm['spatial'] (optional).",
    )
    nf1.add_argument(
        "--run-metadata",
        default=None,
        metavar="PATH",
        help="run_metadata.json for fallback detection (optional).",
    )
    nf1.add_argument(
        "--out-dir",
        default=str(_REPO_ROOT / "results" / "paper_metrics"),
        metavar="DIR",
        help="Output directory (default: results/paper_metrics).",
    )
    nf1.add_argument("--seed", type=int, default=0, help="Random seed (default: 0).")
    nf1.add_argument("--dpi", type=int, default=150, help="PNG DPI (default: 150).")
    nf1.add_argument(
        "--stub",
        default=None,
        metavar="PATH",
        help="Path to n-f1_metrics_stub.json to update (optional).",
    )

    # --- nf2 ---------------------------------------------------------------
    nf2 = sub.add_parser("nf2", help="Emit N-F2 scalability figure + JSON.")
    nf2.add_argument(
        "--run-metadata",
        nargs="+",
        required=True,
        metavar="PATH",
        help="One or more run_metadata.json paths.",
    )
    nf2.add_argument(
        "--k-sweep",
        default=None,
        metavar="PATH",
        help="Path to k_stability_sweep.json (optional).",
    )
    nf2.add_argument(
        "--out-dir",
        default=str(_REPO_ROOT / "results" / "paper_metrics"),
        metavar="DIR",
        help="Output directory (default: results/paper_metrics).",
    )
    nf2.add_argument(
        "--stub",
        default=None,
        metavar="PATH",
        help="Path to n-f2_metrics_stub.json to update (optional).",
    )
    return p


def main() -> int:
    args = _build_parser().parse_args()

    if args.figure == "nf1":
        outputs = emit_niche_map(
            npz_path=args.npz,
            h5ad_path=args.h5ad,
            out_dir=args.out_dir,
            run_metadata_path=args.run_metadata,
            stub_path=args.stub or (_REPO_ROOT / _N_F1_STUB_REL),
            seed=args.seed,
            dpi=args.dpi,
        )
        for kind, path in outputs.items():
            print(f"{kind}: {path}")

    elif args.figure == "nf2":
        outputs = emit_scalability(
            run_metadata_paths=args.run_metadata,
            out_dir=args.out_dir,
            k_sweep_path=args.k_sweep,
            stub_path=args.stub or (_REPO_ROOT / _N_F2_STUB_REL),
        )
        print(f"figure: {outputs['figure']}")
        print(f"json:   {outputs['json']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
