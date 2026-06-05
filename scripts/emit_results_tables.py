#!/usr/bin/env python
"""Aggregate per-run metrics.json files into the N-T1 multi-method leaderboard.

Emits three deterministic artifacts into *out_dir*:
  - leaderboard.csv    (standard CSV, NA for missing values)
  - leaderboard.md     (GitHub-flavored Markdown table)
  - leaderboard.json   (machine-readable with per-row provenance)

IMPORTANT: the single-section fallback note (``_fallback_note`` in
``run_metadata.json``) is surfaced in ALL three artifacts.  This table is
NEVER paper-claim-ready on fallback data — do NOT cite these numbers as the
124k-cell atlas result.

Fixed column set (documented, stable across runs):
    domain_ari, domain_ami, domain_nmi, domain_macro_f1,
    domain_homogeneity, domain_accuracy,
    embedding_silhouette, niche_morans_i

Usage::

    python scripts/emit_results_tables.py \\
        --model-metrics results/niche-lens-st/metrics.json \\
        --baselines-glob "results/baselines/*/metrics.json" \\
        --out-dir results/paper_metrics
"""

from __future__ import annotations

import argparse
import glob as glob_module
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Optional

# Resolve repo root from this file so the script runs from any cwd.
_REPO_ROOT = Path(__file__).resolve().parents[1]

# Canonical path of the N-T1 stub relative to repo root.
_N_T1_STUB_REL = Path("results") / "paper_metrics" / "n-t1_metrics_stub.json"

SCHEMA_VERSION = "1.0.0"

#: Fixed, documented column set drawn from annotation-agreement + intrinsic metrics.
LEADERBOARD_COLS: tuple[str, ...] = (
    "domain_ari",
    "domain_ami",
    "domain_nmi",
    "domain_macro_f1",
    "domain_homogeneity",
    "domain_accuracy",
    "embedding_silhouette",
    "niche_morans_i",
)

_FLOAT_FMT = ".4f"
_NA = "NA"


# ---------------------------------------------------------------------------
# JSON I/O helpers
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> dict[str, Any]:
    """Load JSON; raise RuntimeError with context on decode failure."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON in {path}: {exc}") from exc


def _atomic_write(path: Path, text: str) -> None:
    """Write *text* to *path* atomically (temp file + os.replace)."""
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


def _detect_fallback(metrics_path: Path) -> tuple[bool, str]:
    """Return ``(is_fallback, note)`` for the run at *metrics_path*.

    The authoritative signal is ``_fallback_note`` in the sibling
    ``run_metadata.json``.  Returns ``(False, "")`` when no fallback is
    detected.
    """
    meta_path = metrics_path.parent / "run_metadata.json"
    if meta_path.exists():
        try:
            meta = _load_json(meta_path)
            note = meta.get("_fallback_note", "")
            if note:
                return True, str(note)
        except RuntimeError:
            pass
    return False, ""


# ---------------------------------------------------------------------------
# Git SHA
# ---------------------------------------------------------------------------


def _resolve_git_sha(metrics_path: Path) -> str:
    """Best-effort git SHA: prefer the ``git_sha`` field inside metrics.json.

    NOTE: byte-identical reproducibility relies on metrics.json carrying a
    stable ``git_sha`` field.  The ``git rev-parse`` fallback below is
    non-deterministic across commits and exists only for runs that lack an
    embedded sha (e.g. hand-crafted test fixtures).
    """
    try:
        data = _load_json(metrics_path)
        sha = data.get("git_sha", "")
        if sha and sha not in ("unknown", "ambiguous-parent-checkout"):
            return str(sha)
    except RuntimeError:
        pass
    try:
        result = subprocess.run(
            ["git", "-C", str(_REPO_ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


# ---------------------------------------------------------------------------
# Row extraction
# ---------------------------------------------------------------------------


def _extract_row(metrics_path: Path, method: str) -> dict[str, Any]:
    """Extract a leaderboard row dict from *metrics_path*.

    Returns ``{"method", "metrics", "source_path", "git_sha", "dataset_card_id"}``.
    Missing or None-valued metrics map to ``None`` (never coerced to 0).
    """
    data = _load_json(metrics_path)
    raw: dict[str, Any] = data.get("metrics", {})

    metric_vals: dict[str, Optional[float]] = {}
    for col in LEADERBOARD_COLS:
        raw_val = raw.get(col)
        if raw_val is None:
            metric_vals[col] = None
        else:
            try:
                metric_vals[col] = float(raw_val)
            except (TypeError, ValueError):
                metric_vals[col] = None

    return {
        "method": method,
        "metrics": metric_vals,
        "source_path": str(metrics_path),
        "git_sha": _resolve_git_sha(metrics_path),
        "dataset_card_id": str(data.get("dataset_card_id", "unknown")),
    }


# ---------------------------------------------------------------------------
# Format helpers
# ---------------------------------------------------------------------------


def _fmt_val(val: Optional[float]) -> str:
    """Format *val* for CSV/MD: float to 4 d.p., None -> 'NA'."""
    if val is None:
        return _NA
    return format(val, _FLOAT_FMT)


# ---------------------------------------------------------------------------
# Emitters
# ---------------------------------------------------------------------------


def _emit_csv(
    rows: list[dict[str, Any]],
    path: Path,
    dataset_card_id: str,
    fallback_warning: str,
) -> None:
    lines: list[str] = []
    # Fallback warning surfaced as comment lines (hard requirement).
    lines.append(f"# dataset_card_id: {dataset_card_id}")
    if fallback_warning:
        lines.append(
            "# WARNING: SINGLE-SECTION FALLBACK DATA — NOT the 124k-cell atlas run. "
            "Do NOT cite as atlas-scale result."
        )
    # Header + data rows.
    lines.append(",".join(["method"] + list(LEADERBOARD_COLS)))
    for row in rows:
        fields = [row["method"]] + [_fmt_val(row["metrics"][c]) for c in LEADERBOARD_COLS]
        lines.append(",".join(fields))
    _atomic_write(path, "\n".join(lines) + "\n")


def _emit_md(
    rows: list[dict[str, Any]],
    path: Path,
    dataset_card_id: str,
    fallback_warning: str,
) -> None:
    cols = ["method"] + list(LEADERBOARD_COLS)
    header_row = "| " + " | ".join(cols) + " |"
    sep_row = "| " + " | ".join(["---"] * len(cols)) + " |"

    parts: list[str] = [
        "## N-T1 Multi-Method Niche Leaderboard",
        "",
        f"**Dataset:** `{dataset_card_id}`",
    ]
    if fallback_warning:
        parts += [
            "",
            "> ⚠️ **SINGLE-SECTION FALLBACK DATA** — results are from the downsized "
            "5,488-cell MERFISH section, **NOT** the 124k-cell atlas run. "
            "Do NOT cite as atlas-scale results.",
            "",
            f"> *Fallback note: {fallback_warning[:300]}*",
        ]
    parts += [
        "",
        header_row,
        sep_row,
    ]
    for row in rows:
        fields = [row["method"]] + [_fmt_val(row["metrics"][c]) for c in LEADERBOARD_COLS]
        parts.append("| " + " | ".join(fields) + " |")
    parts += [
        "",
        f"*Columns: {', '.join(LEADERBOARD_COLS)}. "
        "NA = metric absent or not applicable for this method.*",
    ]
    _atomic_write(path, "\n".join(parts) + "\n")


def _emit_json(
    rows: list[dict[str, Any]],
    path: Path,
    dataset_card_id: str,
    fallback_warning: str,
    source_sha256: str,
) -> None:
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "dataset_card_id": dataset_card_id,
        "source_sha256": source_sha256,
        "columns": list(LEADERBOARD_COLS),
        "paper_claim_ready": False,
        "fallback_warning": fallback_warning if fallback_warning else None,
        "rows": rows,
        "provenance": {
            row["method"]: {
                "source_path": row["source_path"],
                "git_sha": row["git_sha"],
                "dataset_card_id": row["dataset_card_id"],
                **({"source_sha256": source_sha256} if row["method"] == "niche-lens-st" else {}),
            }
            for row in rows
        },
    }
    _atomic_write(path, json.dumps(payload, indent=2, sort_keys=False) + "\n")


# ---------------------------------------------------------------------------
# Stub updater
# ---------------------------------------------------------------------------


def _update_stub(stub_path: Path, is_fallback: bool) -> None:
    """Update the n-t1_metrics_stub.json after successful artifact emit.

    Flips ``source_exists``, ``metric_source_exists``, ``artifact_exists`` to
    ``True`` and records the script reference.  ``paper_claim_ready`` is
    NEVER set to ``True`` here — only manual promotion is allowed.
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
    data["source_script_reference"] = "scripts/emit_results_tables.py"
    data["readiness_status"] = "script-ready"
    # Gate: fallback data must not become paper-claim-ready.
    if is_fallback:
        data["paper_claim_ready"] = False
        data["supports_safe_prose"] = False
    # We never auto-promote paper_claim_ready to True; that requires manual review.
    _atomic_write(stub_path, json.dumps(data, indent=2) + "\n")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def emit_leaderboard(
    model_metrics_path: "str | Path",
    baselines_glob: str = "",
    out_dir: "Optional[str | Path]" = None,
    stub_path: "Optional[str | Path]" = None,
) -> dict[str, Path]:
    """Aggregate metrics.json files into the N-T1 comparison leaderboard.

    Args:
        model_metrics_path: Path to ``results/niche-lens-st/metrics.json``
            (or equivalent).  Required.
        baselines_glob: Glob pattern for baseline metrics files
            (e.g. ``"results/baselines/*/metrics.json"``).  Empty string or
            a pattern matching zero files is valid — emits a model-only table.
        out_dir: Output directory.  Defaults to
            ``<repo_root>/results/paper_metrics``.
        stub_path: Optional path to ``n-t1_metrics_stub.json`` to update
            after emit.  When ``None``, uses the canonical location under the
            repo root if it exists; a non-existent path is silently skipped.

    Returns:
        ``{"csv": Path, "md": Path, "json": Path}`` — paths to emitted artifacts.
    """
    model_metrics_path = Path(model_metrics_path).resolve()
    if out_dir is None:
        out_dir = _REPO_ROOT / "results" / "paper_metrics"
    out_dir = Path(out_dir)

    # --- collect rows -------------------------------------------------------
    model_row = _extract_row(model_metrics_path, "niche-lens-st")
    model_dataset_card_id = str(
        _load_json(model_metrics_path).get("dataset_card_id", "unknown")
    )
    # sha256 of the model metrics.json bytes — enables drift detection.
    source_sha256 = hashlib.sha256(model_metrics_path.read_bytes()).hexdigest()

    baseline_rows: list[dict[str, Any]] = []
    if baselines_glob:
        for p in sorted(glob_module.glob(str(baselines_glob))):
            bp = Path(p).resolve()
            method = bp.parent.name  # directory name == baseline name
            baseline_rows.append(_extract_row(bp, method))

    # Stable order: model first, baselines sorted lexicographically by method.
    baseline_rows.sort(key=lambda r: r["method"])
    all_rows = [model_row] + baseline_rows

    # --- fallback detection -------------------------------------------------
    is_fallback, fallback_warning = _detect_fallback(model_metrics_path)

    # --- emit artifacts -----------------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "leaderboard.csv"
    md_path = out_dir / "leaderboard.md"
    json_path = out_dir / "leaderboard.json"

    _emit_csv(all_rows, csv_path, model_dataset_card_id, fallback_warning)
    _emit_md(all_rows, md_path, model_dataset_card_id, fallback_warning)
    _emit_json(all_rows, json_path, model_dataset_card_id, fallback_warning, source_sha256)

    # --- update stub --------------------------------------------------------
    # Only update when the caller explicitly provides a stub_path; no
    # auto-detection here so that programmatic / test invocations never
    # accidentally mutate the committed stub file.  The CLI passes the
    # canonical path explicitly.
    if stub_path is not None:
        _update_stub(Path(stub_path), is_fallback)

    return {"csv": csv_path, "md": md_path, "json": json_path}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model-metrics",
        default=str(_REPO_ROOT / "results" / "niche-lens-st" / "metrics.json"),
        help="Path to the model run metrics.json (default: results/niche-lens-st/metrics.json).",
    )
    parser.add_argument(
        "--baselines-glob",
        default=str(_REPO_ROOT / "results" / "baselines" / "*" / "metrics.json"),
        help='Glob for baseline metrics files (may match zero files). Default: results/baselines/*/metrics.json.',
    )
    parser.add_argument(
        "--out-dir",
        default=str(_REPO_ROOT / "results" / "paper_metrics"),
        help="Output directory for leaderboard artifacts (default: results/paper_metrics).",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    paths = emit_leaderboard(
        model_metrics_path=args.model_metrics,
        baselines_glob=args.baselines_glob,
        out_dir=args.out_dir,
        stub_path=_REPO_ROOT / _N_T1_STUB_REL,
    )
    print(f"leaderboard.csv  -> {paths['csv']}")
    print(f"leaderboard.md   -> {paths['md']}")
    print(f"leaderboard.json -> {paths['json']}")
    print()
    print(paths["md"].read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
