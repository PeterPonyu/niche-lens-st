"""Emit the #315 cohort-scale runtime & peak-memory SCALABILITY TABLE.

This is the numeric table the #289/N-F2 runtime/peak-memory *curve* reads from:
rows by ascending cell-count regime, columns = wall-clock (s), peak RSS (GB),
throughput (cells/s), InfoNCE mode (full-batch vs minibatch ``bs``), and an
explicit status (``ok`` / ``fallback-to-slice`` / declared ``OOM`` /
download-blocked). It backs the "linear in cells, no O(n^2) blow-up" claim with
hard numbers and honest OOM markers.

Honesty contract (matches emit_figures N-F2):
- ``peak_rss_bytes`` / ``runtime_s`` are read from the exact run_metadata keys;
  a missing value is ``None`` -> em-dash in markdown / empty in csv. NEVER 0.
- Genuinely un-runnable regimes (the 124k full-batch OOM, million-cell
  download-blocked rows) are passed in as *declared* rows with their status and
  NO fabricated numbers, so the table tells the full scaling story without
  inventing data.
- ``paper_claim_ready`` is always False here; fallback-slice rows flip the
  table-level fallback banner on.

Usage::

    python scripts/emit_scalability_table.py \\
        --run-metadata results/niche-lens-st/run_metadata.json other/run_metadata.json \\
        --declared-regimes data/scalability_declared_regimes.json \\
        --out-dir results/paper_metrics
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parents[1]

SCHEMA_VERSION = "1.0.0"
_SCRIPT_REF = "scripts/emit_scalability_table.py"

#: Output columns, in render order.
COLUMNS = [
    "dataset",
    "n_cells",
    "wall_clock_s",
    "peak_rss_gb",
    "throughput_cells_per_s",
    "infonce_mode",
    "status",
]

_FALLBACK_BANNER = (
    "the fallback-to-slice row is the downsized 5,488-cell MERFISH section, NOT "
    "an atlas-scale run. Do NOT cite its scale as representative."
)


# ---------------------------------------------------------------------------
# JSON I/O helpers (mirrored from emit_figures.py / emit_results_tables.py)
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
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
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
# Row extraction
# ---------------------------------------------------------------------------
def _n_obs(meta: dict) -> Optional[int]:
    n = meta.get("n_obs")
    if n is None:
        n = (meta.get("shapes") or {}).get("n_obs")
    return n


def _infonce_mode(meta: dict) -> str:
    """Full-batch when batch_size is unset/0; otherwise minibatch(bs=N) (#302)."""
    bs = meta.get("batch_size")
    if not bs:  # None or 0
        return "full-batch"
    return f"minibatch(bs={int(bs)})"


def _status(meta: dict) -> str:
    """``fallback-to-slice`` on the downsized-fallback signal, else ``ok``.

    Dual detection mirrors emit_figures: the structured ``_fallback_note`` key
    (dropped by the byte-locked contract until META adds it) OR the free-text
    ``dataset=fallback`` marker the runner writes (contract passthrough).
    """
    if meta.get("_fallback_note") or "dataset=fallback" in str(meta.get("notes", "")):
        return "fallback-to-slice"
    return "ok"


def row_from_metadata(meta: dict) -> dict:
    """Build one scalability row from a run_metadata dict.

    Missing runtime/peak_rss stay ``None`` (em-dash downstream), never 0.
    peak RSS is reported in GB (1e9 bytes) to match the issue's column.
    """
    n_cells = _n_obs(meta)
    runtime_s = meta.get("runtime_s")
    peak_bytes = meta.get("peak_rss_bytes")
    peak_gb = (peak_bytes / 1e9) if isinstance(peak_bytes, (int, float)) else None
    throughput = None
    if n_cells is not None and isinstance(runtime_s, (int, float)) and runtime_s > 0:
        throughput = n_cells / runtime_s
    dataset = meta.get("dataset_card_id") or meta.get("project") or "unknown"
    return {
        "dataset": dataset,
        "n_cells": n_cells,
        "wall_clock_s": runtime_s,
        "peak_rss_gb": peak_gb,
        "throughput_cells_per_s": throughput,
        "infonce_mode": _infonce_mode(meta),
        "status": _status(meta),
    }


def _declared_row(decl: dict) -> dict:
    """Normalise a declared (blocked/OOM) regime to a full row with no numbers."""
    return {
        "dataset": decl.get("dataset", "unknown"),
        "n_cells": decl.get("n_cells"),
        "wall_clock_s": decl.get("wall_clock_s"),  # normally None for blocked rows
        "peak_rss_gb": decl.get("peak_rss_gb"),
        "throughput_cells_per_s": decl.get("throughput_cells_per_s"),
        "infonce_mode": decl.get("infonce_mode", "full-batch"),
        "status": decl.get("status", "blocked"),
    }


def build_scalability_rows(
    run_metadata_paths: "list[str | Path]",
    declared_regimes: "Optional[list[dict]]" = None,
) -> "list[dict]":
    """Aggregate run_metadata files (+ declared blocked rows) into sorted rows.

    Rows are sorted ascending by ``n_cells`` (None sorts last). Unreadable or
    missing metadata files are skipped silently (mirrors emit_figures).
    """
    rows: list[dict] = []
    for raw_p in run_metadata_paths:
        p = Path(raw_p)
        if not p.exists():
            continue
        try:
            meta = _load_json(p)
        except RuntimeError:
            continue
        rows.append(row_from_metadata(meta))
    for decl in declared_regimes or []:
        rows.append(_declared_row(decl))
    rows.sort(key=lambda r: (r["n_cells"] is None, r["n_cells"] or 0))
    return rows


def has_fallback(rows: "list[dict]") -> bool:
    return any(r["status"] == "fallback-to-slice" for r in rows)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
_HEADER = [
    "dataset",
    "n_cells",
    "wall_clock_s",
    "peak_rss_gb",
    "throughput_cells_per_s",
    "infonce_mode",
    "status",
]
_MD_HEADER = [
    "dataset",
    "n_cells",
    "wall-clock (s)",
    "peak RSS (GB)",
    "throughput (cells/s)",
    "InfoNCE mode",
    "status",
]


def _fmt(value, ndigits: Optional[int] = None) -> str:
    if value is None:
        return "—"
    if isinstance(value, float) and ndigits is not None:
        return f"{value:.{ndigits}f}"
    return str(value)


def render_markdown(rows: "list[dict]", fallback: bool = False) -> str:
    """Render the scalability table as GitHub-flavoured markdown."""
    lines = ["## N-T2 Cohort-scale Runtime & Peak-memory Scalability", ""]
    if fallback:
        lines += [f"> ⚠️ **FALLBACK ROW PRESENT** — {_FALLBACK_BANNER}", ""]
    lines.append("| " + " | ".join(_MD_HEADER) + " |")
    lines.append("| " + " | ".join("---" for _ in _MD_HEADER) + " |")
    for r in rows:
        cells = [
            _fmt(r["dataset"]),
            _fmt(r["n_cells"]),
            _fmt(r["wall_clock_s"], 2),
            _fmt(r["peak_rss_gb"], 2),
            _fmt(r["throughput_cells_per_s"], 0),
            _fmt(r["infonce_mode"]),
            _fmt(r["status"]),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(
        "*peak RSS in GB (1e9 bytes); — = not measured / not applicable; "
        "blocked regimes (OOM, download-blocked) carry status with no fabricated "
        "numbers. NOT paper-claim-ready.*"
    )
    return "\n".join(lines) + "\n"


def render_csv(rows: "list[dict]", fallback: bool = False) -> str:
    """Render the scalability table as CSV (with a leading comment banner)."""
    out: list[str] = []
    if fallback:
        out.append("# WARNING: " + _FALLBACK_BANNER)
    out.append(",".join(_HEADER))
    for r in rows:
        vals = [
            str(r["dataset"]),
            "" if r["n_cells"] is None else str(r["n_cells"]),
            "" if r["wall_clock_s"] is None else f"{r['wall_clock_s']:.4f}",
            "" if r["peak_rss_gb"] is None else f"{r['peak_rss_gb']:.4f}",
            "" if r["throughput_cells_per_s"] is None else f"{r['throughput_cells_per_s']:.2f}",
            str(r["infonce_mode"]),
            str(r["status"]),
        ]
        out.append(",".join(vals))
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# Emit
# ---------------------------------------------------------------------------
def emit_scalability_table(
    run_metadata_paths: "list[str | Path]",
    out_dir: "str | Path" = ".",
    declared_path: "Optional[str | Path]" = None,
) -> "dict[str, Path]":
    """Emit ``scalability_table.{csv,md,json}`` to *out_dir*.

    Returns ``{"csv": Path, "md": Path, "json": Path}``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    declared: list[dict] = []
    if declared_path is not None:
        dp = Path(declared_path)
        if dp.exists():
            try:
                loaded = _load_json(dp)
                declared = loaded if isinstance(loaded, list) else loaded.get("regimes", [])
            except RuntimeError:
                declared = []

    rows = build_scalability_rows(run_metadata_paths, declared_regimes=declared)
    fallback = has_fallback(rows)

    csv_path = out_dir / "scalability_table.csv"
    md_path = out_dir / "scalability_table.md"
    json_path = out_dir / "scalability_table.json"

    _atomic_write(csv_path, render_csv(rows, fallback=fallback))
    _atomic_write(md_path, render_markdown(rows, fallback=fallback))
    payload = {
        "schema_version": SCHEMA_VERSION,
        "source_script_reference": _SCRIPT_REF,
        "columns": COLUMNS,
        "paper_claim_ready": False,
        "fallback": fallback,
        "rows": rows,
    }
    _atomic_write(json_path, json.dumps(payload, indent=2, sort_keys=False) + "\n")
    return {"csv": csv_path, "md": md_path, "json": json_path}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--run-metadata",
        nargs="*",
        default=[],
        metavar="PATH",
        help="One or more run_metadata.json paths (the 'ok'/'fallback' rows).",
    )
    p.add_argument(
        "--declared-regimes",
        default=None,
        metavar="PATH",
        help="Optional JSON list of blocked/OOM regimes (status + n_cells, no "
        "fabricated numbers) to render alongside the real runs.",
    )
    p.add_argument(
        "--out-dir",
        default=str(_REPO_ROOT / "results" / "paper_metrics"),
        metavar="DIR",
        help="Output directory (default: results/paper_metrics).",
    )
    return p


def main(argv: "Optional[list[str]]" = None) -> int:
    args = _build_parser().parse_args(argv)
    out = emit_scalability_table(
        args.run_metadata, out_dir=args.out_dir, declared_path=args.declared_regimes
    )
    print(f"scalability_table.csv  -> {out['csv']}")
    print(f"scalability_table.md   -> {out['md']}")
    print(f"scalability_table.json -> {out['json']}")
    print()
    print(out["md"].read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
