"""#296 figure: niche stability across seeds + cohort reproducibility.

Renders the three panels from a ``niche_stability.json`` (produced by
``compute_niche_stability.py``):

(a) **Seed-stability ARI matrix** — pairwise ARI of ``prototype_id`` across
    seeds; the off-diagonal mean ± sd is the stability score.
(b) **Coverage sweep** — conserved fraction and section-overlap-rate vs
    ``min_section_coverage`` (#105). On single-section input the conserved
    distinction is degenerate, so the panel says so instead of plotting nothing.
(c) **Prototype-matching** — Hungarian seed0↔seed1 prototype correspondence
    (bipartite edges weighted by shared-cell overlap): the reproducibility
    "Sankey" content.

NEVER paper-claim-ready: fallback / single-section inputs are surfaced, not
hidden.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _panel_ari_matrix(ax, payload: dict) -> None:
    M = np.asarray(payload.get("ari_matrix", []), dtype=float)
    if M.ndim != 2 or M.size == 0:
        ax.text(0.5, 0.5, "no ARI matrix", ha="center", va="center")
        ax.set_axis_off()
        return
    im = ax.imshow(M, vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_xticks(range(M.shape[0]))
    ax.set_yticks(range(M.shape[0]))
    ax.set_xlabel("seed")
    ax.set_ylabel("seed")
    mean = payload.get("seed_stability_ari")
    sd = payload.get("seed_stability_ari_sd")
    mean_s = "—" if mean is None else f"{mean:.3f}"
    sd_s = "" if sd is None else f" ± {sd:.3f}"
    ax.set_title(f"(a) Seed-stability ARI\nmean off-diag={mean_s}{sd_s}", fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def _panel_coverage(ax, payload: dict) -> None:
    rows = payload.get("coverage_sweep", [])
    n_sections = payload.get("n_sections", 0)
    if n_sections < 2 or not rows or all(r.get("conserved_fraction") is None for r in rows):
        ax.text(
            0.5, 0.5,
            f"single section (n={n_sections})\nconserved/sample-specific\ndistinction degenerate",
            ha="center", va="center", fontsize=9,
        )
        ax.set_title("(b) Coverage sweep", fontsize=9)
        ax.set_axis_off()
        return
    xs = [r["min_section_coverage"] for r in rows]
    cf = [r["conserved_fraction"] for r in rows]
    ax.plot(xs, cf, "o-", color="tab:blue", label="conserved fraction")
    sor = [r.get("section_overlap_rate") for r in rows]
    if any(v is not None and not (isinstance(v, float) and np.isnan(v)) for v in sor):
        ax.plot(xs, sor, "s--", color="tab:orange", label="section-overlap rate")
    ax.set_xlabel("min_section_coverage")
    ax.set_ylabel("fraction")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=7)
    ax.set_title("(b) Coverage sweep", fontsize=9)


def _panel_matching(ax, payload: dict) -> None:
    matches = payload.get("prototype_matching_seed0_seed1", [])
    if not matches:
        ax.text(0.5, 0.5, "no matching\n(need >=2 seeds)", ha="center", va="center", fontsize=9)
        ax.set_title("(c) Prototype matching", fontsize=9)
        ax.set_axis_off()
        return
    max_ov = max((m["overlap"] for m in matches), default=1) or 1
    a_ids = sorted({m["proto_a"] for m in matches})
    b_ids = sorted({m["proto_b"] for m in matches})
    a_y = {p: i for i, p in enumerate(a_ids)}
    b_y = {p: i for i, p in enumerate(b_ids)}
    for m in matches:
        y0, y1 = a_y[m["proto_a"]], b_y[m["proto_b"]]
        lw = 0.4 + 3.5 * (m["overlap"] / max_ov)
        ax.plot([0, 1], [y0, y1], "-", color="tab:gray", linewidth=lw, alpha=0.7)
    ax.scatter([0] * len(a_ids), list(a_y.values()), color="tab:blue", s=30, zorder=3)
    ax.scatter([1] * len(b_ids), list(b_y.values()), color="tab:green", s=30, zorder=3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["seed 0", "seed 1"])
    ax.set_yticks([])
    ax.set_title("(c) Prototype matching\n(edge width ∝ shared cells)", fontsize=9)


def emit_stability_figure(
    stability_json: "str | Path",
    out_dir: "str | Path" = ".",
) -> "dict[str, Path]":
    """Render the 3-panel #296 stability figure from a niche_stability.json."""
    payload = json.loads(Path(stability_json).read_text(encoding="utf-8"))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    _panel_ari_matrix(axes[0], payload)
    _panel_coverage(axes[1], payload)
    _panel_matching(axes[2], payload)
    fallback = not payload.get("paper_claim_ready", False)
    suptitle = "Niche stability across seeds & cohort reproducibility (N-F3)"
    if fallback:
        suptitle += "  —  NOT paper-claim-ready"
    fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    fig_path = out_dir / "niche_stability.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return {"figure": fig_path}


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--stability-json", required=True, metavar="PATH")
    p.add_argument(
        "--out-dir", default=str(_REPO_ROOT / "results" / "paper_metrics"), metavar="DIR"
    )
    return p


def main(argv: "Optional[list[str]]" = None) -> int:
    args = _build_parser().parse_args(argv)
    out = emit_stability_figure(args.stability_json, out_dir=args.out_dir)
    print(f"niche_stability.png -> {out['figure']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
