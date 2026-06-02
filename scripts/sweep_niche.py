#!/usr/bin/env python
"""Config-driven sweep harness for NicheLens-ST (#101, closes #153).

The single-config real-data path lives in ``run_real_niche.py``; this script
adds a **grid sweep** on top of it without duplicating the fit or metric logic.
For each point in a hyper-parameter grid it:

* fits the contrastive niche model (``fit_niche_model``) on a dataset,
* computes the intrinsic/unsupervised metrics (reusing
  ``run_real_niche._intrinsic_metrics``),
* persists a per-run directory containing:
    - ``manifest.json``  -- swept config + seed + git_sha + timing,
    - ``metrics.json``   -- the uniform results-contract metrics block,
    - ``run_metadata.json`` -- the uniform results-contract provenance block,
    - ``outputs/niche.npz`` -- fitted artifacts (H + prototype_id),
    - ``outputs/proto_kind.json``,
* and a top-level ``sweep_summary.json`` aggregating every run.

The per-run ``metrics.json`` / ``run_metadata.json`` / ``outputs/`` directory are
produced by the SAME vendored uniform results-contract writer
(``nichelens_st.results_contract.write_results``) the single-config runner uses,
so a swept run is contract-compatible with a single run.

Swept knobs (all optional; each accepts a space-separated list of values):
``--embed-dim``, ``--num-layers``, ``--tau``, ``--epochs``, ``--batch-size``,
``--n-prototypes`` (the clustering choice exposed by the model config). Omitted
knobs fall back to the model defaults. The cartesian product of the supplied
lists defines the run grid.

Usage (under the project's conda env):

    conda run --no-capture-output -n dl python scripts/sweep_niche.py \
        --embed-dim 16 32 --tau 0.1 0.2 --epochs 30 --out results/niche_sweep

Heavy deps (torch via the ``[model]`` extra; scanpy/anndata via ``[data]``) are
gated exactly as in ``run_real_niche.py`` -- importing this module is cheap and
side-effect free.
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# Resolve repo root from this file so the script runs from any cwd.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

# Reuse the single-config runner's dataset loader, fit-with-walls guard and
# intrinsic-metric computation rather than re-implementing them (#101).
_RUNNER_SCRIPT = _REPO_ROOT / "scripts" / "run_real_niche.py"
_spec = importlib.util.spec_from_file_location("run_real_niche", _RUNNER_SCRIPT)
runner = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(runner)


# --------------------------------------------------------------------------- #
# grid expansion
# --------------------------------------------------------------------------- #
#: Model-config knobs exposed to the sweep (maps CLI dest -> NicheModelConfig
#: field). Each is a list-valued CLI arg whose cartesian product forms the grid.
SWEEP_KEYS = (
    "embed_dim",
    "num_layers",
    "tau",
    "epochs",
    "batch_size",
    "n_prototypes",
)


def expand_grid(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Expand a ``{key: [values...]}`` grid to a list of point dicts.

    The cartesian product of every key's value list. An empty grid yields a
    single empty point (one run using all model defaults).
    """
    keys = [k for k, v in grid.items() if v]
    if not keys:
        return [{}]
    value_lists = [grid[k] for k in keys]
    points = []
    for combo in itertools.product(*value_lists):
        points.append({k: v for k, v in zip(keys, combo)})
    return points


# --------------------------------------------------------------------------- #
# single run
# --------------------------------------------------------------------------- #
def _run_id(index: int, point: dict[str, Any]) -> str:
    """Stable, filesystem-safe per-run id from the swept point."""
    if not point:
        parts = "default"
    else:
        parts = "_".join(f"{k}-{point[k]}" for k in sorted(point))
    return f"run{index:03d}_{parts}"


def run_one(
    X: np.ndarray,
    coords: np.ndarray,
    section_id: np.ndarray,
    edges: np.ndarray,
    point: dict[str, Any],
    *,
    run_id: str,
    out_dir: Path,
    seed: int,
    dataset_paths: list[str],
    base_config: dict[str, Any] | None = None,
    max_seconds: float = float("inf"),
    device: str = "cpu",
    num_threads: int = 1,
) -> dict[str, Any]:
    """Fit one grid point, compute metrics, and persist the run directory.

    Returns the per-run summary record (config + metrics + paths + timing).
    """
    from nichelens_st import model as _model
    from nichelens_st import results_contract

    cfg_kwargs: dict[str, Any] = dict(base_config or {})
    cfg_kwargs.update(point)
    cfg_kwargs.setdefault("seed", seed)
    cfg_kwargs.setdefault("device", device)
    cfg_kwargs.setdefault("num_threads", num_threads)
    cfg_kwargs.setdefault("deterministic", False)

    cfg = _model.NicheModelConfig(**cfg_kwargs)

    t0 = time.time()
    result = _model.fit_niche_model(
        X=X, coords=coords, section_id=section_id, edges=edges, config=cfg
    )
    elapsed = time.time() - t0

    metrics, notes_extra = runner._intrinsic_metrics(
        result, edges, section_id, seed
    )

    # Persist via the uniform results contract (project=run_id => per-run dir).
    paths = results_contract.write_results(
        project=run_id,
        dataset_card_id=results_contract.dataset_card_id(dataset_paths),
        metrics=metrics,
        outputs={
            "niche": "outputs/niche.npz",
            "proto_kind": "outputs/proto_kind.json",
        },
        run_metadata={
            "dataset_paths": dataset_paths,
            "n_obs": int(X.shape[0]),
            "n_vars": int(X.shape[1]),
            "seed": seed,
            "runtime_s": elapsed,
            "device": device,
            "deterministic": False,
            "num_threads": num_threads,
            "reproducibility_level": "seeded",
            "notes": "; ".join(["sweep run", *notes_extra]),
        },
        results_dir=str(out_dir),
    )
    run_dir = Path(paths["results_dir"])
    outputs_dir = Path(paths["outputs_dir"])

    # Fitted artifacts.
    np.savez_compressed(
        outputs_dir / "niche.npz",
        H=np.asarray(result.H),
        prototype_id=np.asarray(result.prototype_id),
    )
    with open(outputs_dir / "proto_kind.json", "w", encoding="utf-8") as fh:
        json.dump(list(result.proto_kind), fh, indent=2)

    # Per-run manifest: the swept config + seed + git_sha + timing.
    manifest = {
        "run_id": run_id,
        "point": point,
        "config": {
            **cfg_kwargs,
        },
        "seed": seed,
        "git_sha": results_contract.git_sha(),
        "runtime_s": elapsed,
        "dataset_paths": dataset_paths,
    }
    with open(run_dir / "manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    return {
        "run_id": run_id,
        "point": point,
        "runtime_s": elapsed,
        "metrics": metrics,
        "run_dir": str(run_dir),
        "manifest": str(run_dir / "manifest.json"),
        "metrics_path": str(paths["metrics"]),
    }


def run_sweep(
    X: np.ndarray,
    coords: np.ndarray,
    section_id: np.ndarray,
    edges: np.ndarray,
    grid: dict[str, list[Any]],
    out_dir: Path | str,
    *,
    seed: int = 0,
    dataset_paths: list[str] | None = None,
    base_config: dict[str, Any] | None = None,
    max_seconds: float = float("inf"),
    device: str = "cpu",
    num_threads: int = 1,
) -> dict[str, Any]:
    """Run the full grid and write a top-level ``sweep_summary.json``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_paths = list(dataset_paths or [])

    points = expand_grid(grid)
    runs: list[dict[str, Any]] = []
    for i, point in enumerate(points):
        run_id = _run_id(i, point)
        record = run_one(
            X,
            coords,
            section_id,
            edges,
            point,
            run_id=run_id,
            out_dir=out_dir,
            seed=seed,
            dataset_paths=dataset_paths,
            base_config=base_config,
            max_seconds=max_seconds,
            device=device,
            num_threads=num_threads,
        )
        runs.append(record)

    summary = {
        "n_runs": len(runs),
        "grid": grid,
        "seed": seed,
        "git_sha": _git_sha(),
        "dataset_paths": dataset_paths,
        "runs": runs,
    }
    with open(out_dir / "sweep_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    return summary


def _git_sha() -> str:
    from nichelens_st import results_contract

    return results_contract.git_sha()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--embed-dim", type=int, nargs="+", default=None)
    parser.add_argument("--num-layers", type=int, nargs="+", default=None)
    parser.add_argument("--tau", type=float, nargs="+", default=None)
    parser.add_argument("--epochs", type=int, nargs="+", default=None)
    parser.add_argument("--batch-size", type=int, nargs="+", default=None)
    parser.add_argument("--n-prototypes", type=int, nargs="+", default=None)
    parser.add_argument(
        "--out",
        type=Path,
        default=_REPO_ROOT / "results" / "niche_sweep",
        help="Output directory for per-run dirs + sweep_summary.json.",
    )
    parser.add_argument(
        "--dataset", choices=("auto", "primary", "fallback"), default="auto"
    )
    parser.add_argument("--already-normalized", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-seconds", type=float, default=runner.DEFAULT_MAX_SECONDS)
    return parser


def _grid_from_args(args: argparse.Namespace) -> dict[str, list[Any]]:
    grid: dict[str, list[Any]] = {}
    for key in SWEEP_KEYS:
        val = getattr(args, key, None)
        if val is not None:
            grid[key] = list(val)
    return grid


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        import torch

        cuda = torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        cuda = False
    device = "cuda" if cuda else "cpu"
    num_threads = max(1, os.cpu_count() or 1)

    # Resolve the dataset exactly as the single-config runner does.
    if args.dataset == "fallback":
        path = runner.FALLBACK_PATH
    else:
        # primary or auto -> try primary; the loader raises if it is absent.
        path = runner.PRIMARY_PATH
    (
        X,
        coords,
        section_id,
        _section_col,
        _normalization,
        _n_obs,
        _n_vars,
        _adata,
    ) = runner._load_dataset(path, args.already_normalized)

    from nichelens_st.graph import build_graph

    edges = build_graph(coords, section_id, k=6, method="knn")

    grid = _grid_from_args(args)
    summary = run_sweep(
        X=X,
        coords=coords,
        section_id=section_id,
        edges=edges,
        grid=grid,
        out_dir=args.out,
        seed=args.seed,
        dataset_paths=[str(path)],
        max_seconds=args.max_seconds,
        device=device,
        num_threads=num_threads,
    )

    print(f"sweep complete: {summary['n_runs']} runs -> {args.out}")
    print(f"sweep_summary.json -> {Path(args.out) / 'sweep_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
