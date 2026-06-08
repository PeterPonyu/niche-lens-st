"""#296 driver: niche stability across seeds + cohort reproducibility.

Refits the contrastive niche model with >=5 distinct seeds on one dataset and
measures (a) seed-to-seed agreement (pairwise ARI of ``prototype_id``), (b)
conserved-fraction / section-overlap across a ``min_section_coverage`` sweep,
and (c) Hungarian prototype correspondence between two seeds (the Sankey
content). Writes ``niche_stability.json`` + ``niche_stability.csv``.

The data load / graph build / model config are reused verbatim from
``run_real_niche.py`` (same ``_load_dataset``, ``build_graph(k=6)`` and
``NicheModelConfig``), so the refit prototypes are directly comparable to the
headline run; only the seed varies.

Usage::

    conda run -n dl python scripts/compute_niche_stability.py \\
        --dataset fallback --n-seeds 5 --out-dir results/niche-lens-st/outputs

The seed-stability ARI and conserved fraction are the headline numbers (#296
acceptance); they are NOT paper-claim-ready on the fallback slice.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from nichelens_st import stability  # noqa: E402

_DEFAULT_THRESHOLDS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]


def _load_runner():
    """Import run_real_niche.py as a module (it is a script, not a package)."""
    spec = importlib.util.spec_from_file_location(
        "run_real_niche", _REPO_ROOT / "scripts" / "run_real_niche.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def compute_stability(
    path: Path,
    n_seeds: int,
    section_col: str,
    thresholds: "list[float]",
    device: str = "cpu",
    num_threads: int = 8,
) -> dict:
    """Refit ``n_seeds`` times and assemble the stability payload."""
    runner = _load_runner()
    from nichelens_st import model as _model
    from nichelens_st.graph import build_graph

    X, coords, section_id, section_col_used, normalization, n_obs, _n_vars, _adata = (
        runner._load_dataset(path, False, section_col)
    )
    edges = build_graph(coords, section_id, k=6, method="knn")
    eff_batch = runner._effective_batch_size(0, int(X.shape[0]))
    n_sections = int(np.unique(section_id).size)

    labelings: list[np.ndarray] = []
    for seed in range(n_seeds):
        cfg = _model.NicheModelConfig(
            embed_dim=runner.EMBED_DIM,
            n_prototypes=runner.N_PROTOTYPES,
            seed=seed,
            num_threads=num_threads,
            device=device,
            deterministic=False,
            batch_size=eff_batch,
        )
        result = _model.fit_niche_model(X, coords, section_id, edges, cfg)
        labelings.append(np.asarray(result.prototype_id, dtype=np.int64))

    ari_matrix = stability.pairwise_ari_matrix(labelings)
    seed_summary = stability.seed_stability_summary(ari_matrix)

    n_protos = int(max(int(lab.max()) + 1 for lab in labelings)) if labelings else 0
    coverage = stability.coverage_sweep(labelings[0], section_id, n_protos, thresholds)

    # Panel (c): prototype correspondence between the two most stable seeds
    # (seed 0 vs seed 1) over the SAME cells — a well-defined per-cell Hungarian
    # match. (Cross-section matching over disjoint cells is left to a follow-up;
    # the seed-pair match is the reproducibility story this figure makes.)
    matching = (
        stability.prototype_matching(labelings[0], labelings[1])
        if len(labelings) >= 2
        else []
    )

    conserved_fraction = coverage[0]["conserved_fraction"] if coverage else None

    return {
        "dataset_path": str(path),
        "n_obs": int(n_obs),
        "n_sections": n_sections,
        "section_col": section_col_used,
        "n_seeds": int(n_seeds),
        "n_prototypes": n_protos,
        "seed_stability_ari": seed_summary["mean_offdiag_ari"],
        "seed_stability_ari_sd": seed_summary["sd_offdiag_ari"],
        "seed_stability_ari_min": seed_summary["min_offdiag_ari"],
        "ari_matrix": ari_matrix.tolist(),
        "conserved_fraction": conserved_fraction,
        "coverage_sweep": coverage,
        "prototype_matching_seed0_seed1": matching,
        "normalization": normalization,
        "paper_claim_ready": False,
    }


def write_outputs(payload: dict, out_dir: Path) -> "dict[str, Path]":
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "niche_stability.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    # niche_stability.csv = the min_section_coverage sweep (acceptance #2).
    csv_path = out_dir / "niche_stability.csv"
    lines = ["min_section_coverage,conserved_fraction,section_overlap_rate,n_conserved,n_prototypes"]
    for r in payload["coverage_sweep"]:
        cf = "" if r["conserved_fraction"] is None else f"{r['conserved_fraction']:.4f}"
        sor = "" if r["section_overlap_rate"] is None or (
            isinstance(r["section_overlap_rate"], float) and np.isnan(r["section_overlap_rate"])
        ) else f"{r['section_overlap_rate']:.4f}"
        lines.append(
            f"{r['min_section_coverage']:.2f},{cf},{sor},{r['n_conserved']},{r['n_prototypes']}"
        )
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": json_path, "csv": csv_path}


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--dataset", choices=("primary", "fallback"), default="fallback")
    p.add_argument("--path", default=None, help="Explicit h5ad path (overrides --dataset).")
    p.add_argument("--n-seeds", type=int, default=5)
    p.add_argument("--section-col", default="auto")
    p.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    p.add_argument("--num-threads", type=int, default=8)
    p.add_argument(
        "--out-dir",
        default=str(_REPO_ROOT / "results" / "niche-lens-st" / "outputs"),
    )
    return p


def main(argv: "list[str] | None" = None) -> int:
    args = _build_parser().parse_args(argv)
    runner = _load_runner()
    if args.path is not None:
        path = Path(args.path)
    else:
        path = runner.PRIMARY_PATH if args.dataset == "primary" else runner.FALLBACK_PATH

    payload = compute_stability(
        path,
        n_seeds=args.n_seeds,
        section_col=args.section_col,
        thresholds=_DEFAULT_THRESHOLDS,
        device=args.device,
        num_threads=args.num_threads,
    )
    out = write_outputs(payload, Path(args.out_dir))
    print(f"niche_stability.json -> {out['json']}")
    print(f"niche_stability.csv  -> {out['csv']}")
    print(
        f"seed_stability_ari={payload['seed_stability_ari']} "
        f"(sd={payload['seed_stability_ari_sd']}, min={payload['seed_stability_ari_min']}, "
        f"n_seeds={payload['n_seeds']}) | conserved_fraction={payload['conserved_fraction']} "
        f"| n_sections={payload['n_sections']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
