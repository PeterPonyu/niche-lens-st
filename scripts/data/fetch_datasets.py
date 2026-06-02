#!/usr/bin/env python3
"""Unified dataset fetch + ingestion framework for NicheLens-ST.

This CLI is the single entry point for every spatial-transcriptomics dataset the
project intends to ingest (see ``scripts/data/registry.py`` for the canonical,
machine-readable registry and ``docs/DATASETS.md`` for the human guide). It maps
each dataset onto the MVP input contract
(``src/nichelens_st/schemas.py::validate_inputs``) via a platform-native squidpy
reader and dispatches a per-dataset fetch.

Design constraints (intentionally a framework, not full runnable loaders):
  * **Never fabricate URLs or write placeholder bytes.** Real downloads run ONLY
    for sources with a verified, anonymous URL (``verified_direct``) or a real
    squidpy builtin (``squidpy_builtin``). ``verified_page`` / ``gated`` /
    ``unverified`` sources resolve to a guarded stub that points at the tracking
    issue instead of guessing a hotlink.
  * **Offline-safe surface.** ``--list``, ``--help``, ``--card`` and the default
    ``--dry-run`` never touch the network and never import squidpy. The heavy
    readers import lazily, only when an actual ``--download`` is requested for a
    fetchable source.

Examples (all offline)::

    python scripts/data/fetch_datasets.py --list
    python scripts/data/fetch_datasets.py --dataset xenium_breast_janesick   # dry-run plan
    python scripts/data/fetch_datasets.py --card cosmx_nsclc_nanostring
    python scripts/data/fetch_datasets.py --emit-cards                       # write data/cards/*.yaml
    python scripts/data/fetch_datasets.py --ligrec --help                    # derived-data step (#57)

A real pull is opt-in and never runs in CI/smoke::

    python scripts/data/fetch_datasets.py --dataset gse208253_oscc_visium --download

The actual download / CCC (ligrec) paths import squidpy lazily and require the
optional data dependencies. Install them with the ``[data]`` extra (or the ``dl``
conda env) before using ``--download`` or ``run_ligrec``::

    pip install "nichelens-st[data]"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running both as a module and as a script (``python scripts/data/fetch_datasets.py``).
sys.path.insert(0, str(Path(__file__).resolve().parent))

from registry import (  # noqa: E402
    DATASETS,
    FIRST_PULL_ORDER,
    RAW_COUNT_POLICY,
    Dataset,
    all_issue_numbers,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"
CARDS_DIR = REPO_ROOT / "data" / "cards"


class FetchError(RuntimeError):
    """Raised when a dataset cannot be fetched (guarded stub / unverified URL)."""


# Gated-dependency guidance for the optional ``[data]`` extra. Mirrors the
# ``_NO_TORCH_MSG`` / ``_require_torch`` pattern in
# ``src/nichelens_st/encoder.py`` so the squidpy/scanpy fetch + CCC paths fail
# with a clear, actionable hint instead of a raw ``ModuleNotFoundError``.
_NO_DATA_MSG = (
    "This path requires the optional data dependencies (squidpy/scanpy/anndata). "
    'Install the data extra: `pip install "nichelens-st[data]"` '
    "(or use the `dl` conda env), then re-run."
)


def _require_data(component: str = "this dataset path"):
    """Import squidpy or raise a guided ``FetchError`` naming the ``[data]`` extra.

    Returns the imported ``squidpy`` module so callers can do
    ``sq = _require_data("ligrec")``. Kept out of the offline surface
    (``--list``/``--card``/dry-run) because squidpy is imported lazily here.
    """
    try:
        import squidpy as sq  # lazy: not a base dependency
    except ImportError as exc:  # pragma: no cover - optional heavy dep
        raise FetchError(f"{component}: {_NO_DATA_MSG}") from exc
    return sq


# --------------------------------------------------------------------------- #
# Listing / cards (offline)
# --------------------------------------------------------------------------- #

def _status_badge(status: str) -> str:
    return {
        "verified_direct": "verified (direct)",
        "verified_page": "verified page (hotlink UNVERIFIED)",
        "squidpy_builtin": "squidpy builtin",
        "gated": "GATED (registration)",
        "unverified": "UNVERIFIED URL",
        "derived": "derived (no download)",
    }.get(status, status)


def list_datasets() -> None:
    print(f"NicheLens-ST dataset registry — {len(DATASETS)} datasets")
    print(f"Raw-count policy: {RAW_COUNT_POLICY}")
    print(f"Issues covered: {', '.join('#' + str(n) for n in all_issue_numbers())}")
    print(f"First-pull order: {' -> '.join(FIRST_PULL_ORDER)}")
    print()
    header = f"{'id':<34} {'platform':<28} {'issues':<14} status"
    print(header)
    print("-" * len(header))
    for ds in DATASETS.values():
        issues = ",".join("#" + str(n) for n in ds.issues)
        print(f"{ds.id:<34} {ds.platform:<28} {issues:<14} {_status_badge(ds.url_status)}")


def _yaml_scalar(value: object) -> str:
    """Minimal, dependency-free YAML scalar emitter (quotes when needed)."""
    if value is None:
        return "null"
    text = str(value)
    if text == "":
        return '""'
    if any(c in text for c in ':#"\n') or text[0] in "[]{}>|*&!%@`" or text.strip() != text:
        return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'
    return text


def card_text(ds: Dataset) -> str:
    """Render a dataset's tracked YAML card (machine + human readable)."""
    lines: list[str] = []
    lines.append(f"id: {_yaml_scalar(ds.id)}")
    lines.append(f"name: {_yaml_scalar(ds.name)}")
    lines.append(f"platform: {_yaml_scalar(ds.platform)}")
    lines.append(f"tissue: {_yaml_scalar(ds.tissue)}")
    lines.append("issues: [" + ", ".join(str(n) for n in ds.issues) + "]")
    lines.append(f"url_status: {_yaml_scalar(ds.url_status)}")
    lines.append(f"page_url: {_yaml_scalar(ds.page_url)}")
    lines.append(f"direct_url: {_yaml_scalar(ds.direct_url)}")
    lines.append(f"reader: {_yaml_scalar(ds.reader)}")
    lines.append(f"citation_key: {_yaml_scalar(ds.citation_key)}")
    lines.append(f"size: {_yaml_scalar(ds.size)}")
    lines.append(f"raw_count_artifact: {_yaml_scalar(ds.raw_count_artifact)}")
    lines.append(f"raw_count_policy: {_yaml_scalar(ds.raw_count_policy)}")
    if ds.aliases:
        lines.append("aliases: [" + ", ".join(_yaml_scalar(a) for a in ds.aliases) + "]")
    lines.append("contract:")
    for key, val in ds.contract.items():
        lines.append(f"  {key}: {_yaml_scalar(val)}")
    lines.append(f"notes: {_yaml_scalar(ds.notes)}")
    return "\n".join(lines) + "\n"


def print_card(dataset_id: str) -> None:
    ds = _require(dataset_id)
    print(f"# data/cards/{ds.id}.yaml")
    print(card_text(ds), end="")


def emit_cards() -> None:
    """Write/refresh every ``data/cards/<id>.yaml`` from the registry (offline)."""
    CARDS_DIR.mkdir(parents=True, exist_ok=True)
    for ds in DATASETS.values():
        path = CARDS_DIR / f"{ds.id}.yaml"
        path.write_text(card_text(ds), encoding="utf-8")
        print(f"wrote {path.relative_to(REPO_ROOT)}")


def _require(dataset_id: str) -> Dataset:
    if dataset_id not in DATASETS:
        known = ", ".join(sorted(DATASETS))
        raise FetchError(f"unknown dataset {dataset_id!r}; known ids: {known}")
    return DATASETS[dataset_id]


# --------------------------------------------------------------------------- #
# Fetch dispatch
# --------------------------------------------------------------------------- #

def _issue_ref(ds: Dataset) -> str:
    return ", ".join("#" + str(n) for n in ds.issues)


def fetch(dataset_id: str, *, download: bool = False) -> Path | None:
    """Dispatch a per-dataset fetch.

    Returns the local raw directory when a real download path runs, else prints
    the dry-run / guarded-stub plan and returns ``None``. Never fabricates URLs
    and never writes placeholder bytes.
    """
    ds = _require(dataset_id)
    dest = RAW_DIR / ds.id
    print(f"[{ds.id}] {ds.name}")
    print(f"  platform : {ds.platform}")
    print(f"  reader   : {ds.reader}")
    print(f"  issues   : {_issue_ref(ds)}")
    print(f"  status   : {_status_badge(ds.url_status)}")
    print(f"  policy   : {ds.raw_count_policy}")

    if ds.url_status == "derived":
        print("  -> derived-data step: no raw download. See --ligrec / docs/DATASETS.md.")
        return None

    if not download:
        target = ds.direct_url or ds.page_url or "(resolve from issue)"
        print(f"  -> dry-run (default). Source: {target}")
        print(f"     Re-run with --download once the source is verified for {ds.id}.")
        return None

    # --download requested: only proceed for genuinely fetchable sources.
    if ds.url_status == "verified_direct" and ds.direct_url:
        return _download_url(ds, ds.direct_url, dest)
    if ds.url_status == "squidpy_builtin":
        return _load_squidpy_builtin(ds, dest)

    # Everything else is a guarded stub — never invent a URL.
    raise FetchError(_guard_message(ds))


def _guard_message(ds: Dataset) -> str:
    if ds.url_status == "unverified":
        return (
            f"{ds.id}: URL UNVERIFIED — see issue {_issue_ref(ds)}. "
            f"Resolve the official bundle from {ds.page_url or 'the data-availability statement'} "
            "and set direct_url in the registry before downloading. No URL will be fabricated."
        )
    if ds.url_status == "gated":
        return (
            f"{ds.id}: source is form/registration GATED — see issue {_issue_ref(ds)}. "
            f"Register and download manually from {ds.page_url}, then point the reader "
            "at the extracted directory. No anonymous URL exists."
        )
    if ds.url_status == "verified_page":
        return (
            f"{ds.id}: dataset page is verified but the per-file hotlink is UNVERIFIED — "
            f"see issue {_issue_ref(ds)}. Resolve the official output bundle from "
            f"{ds.page_url}, record it as direct_url in the registry, then re-run --download."
        )
    return f"{ds.id}: not fetchable automatically — see issue {_issue_ref(ds)}."


def _download_url(ds: Dataset, url: str, dest: Path) -> Path:
    """Download a verified, anonymous artifact (opt-in; never runs in CI)."""
    import urllib.request

    dest.mkdir(parents=True, exist_ok=True)
    fname = url.split("?")[0].rstrip("/").split("/")[-1] or f"{ds.id}.tar"
    out = dest / fname
    print(f"  -> downloading verified source to {out} ...")
    urllib.request.urlretrieve(url, out)  # noqa: S310 — verified anonymous source only
    print(f"  -> done: {out}")
    print("  NOTE: this is a raw bundle; map to the MVP contract via the reader above.")
    return dest


def _load_squidpy_builtin(ds: Dataset, dest: Path) -> Path:
    """Load a squidpy builtin dataset (figshare mirror) and cache it locally."""
    sq = _require_data(ds.id)
    loader_name = ds.reader.rsplit(".", 1)[-1]  # e.g. 'merfish'
    loader = getattr(sq.datasets, loader_name)
    dest.mkdir(parents=True, exist_ok=True)
    print(f"  -> loading squidpy.datasets.{loader_name}() (figshare mirror) ...")
    adata = loader()
    out = dest / f"{ds.id}.h5ad"
    adata.write_h5ad(out)
    print(f"  -> cached AnnData ({adata.shape[0]} cells x {adata.shape[1]} genes) -> {out}")
    return dest


# --------------------------------------------------------------------------- #
# Contract helper (squidpy-backed, per PR #44 design)
# --------------------------------------------------------------------------- #

def build_contract(adata, section_key: str, k: int = 8):  # noqa: ANN001 - AnnData optional
    """Map a loaded AnnData onto the MVP input contract ``(X, coords, section_id, edges)``.

    Builds a per-section kNN graph that never bridges a section/FOV boundary, by
    delegating to ``nichelens_st.graph.build_graph`` (the repo's existing entry
    point — no new abstraction). Imports numpy lazily so the module's offline
    surface (``--list``/``--help``) stays dependency-light.

    Returns ``(X, coords, section_id, edges)`` suitable for
    ``nichelens_st.schemas.validate_inputs``.
    """
    import numpy as np

    sys.path.insert(0, str(REPO_ROOT / "src"))
    from nichelens_st.graph import build_graph  # noqa: E402

    X = np.asarray(adata.X.todense() if hasattr(adata.X, "todense") else adata.X, dtype=np.float32)
    coords = np.asarray(adata.obsm["spatial"], dtype=np.float32)
    codes = adata.obs[section_key].astype("category").cat.codes.to_numpy()
    section_id = codes.astype(np.int64)
    edges = build_graph(coords, section_id, k=k, method="knn")
    # Invariant guaranteed by build_graph; assert to fail loudly if violated.
    if edges.size:
        assert (section_id[edges[0]] == section_id[edges[1]]).all(), "cross-section edge!"
    return X, coords, section_id, edges


# --------------------------------------------------------------------------- #
# Derived-data step: cell-cell communication (#57)
# --------------------------------------------------------------------------- #

def run_ligrec(adata, cluster_key: str, *, n_perms: int = 100, threshold: float = 0.01, seed: int = 0):  # noqa: ANN001
    """Derived-data step (issue #57): ligand-receptor inference via OmniPath.

    No raw download — OmniPath resolves and caches its reference at call time.
    Returns the squidpy ``ligrec`` result dict (``means`` / ``pvalues``); pair it
    with per-niche ``prototype_id`` from the encoder to build ``interaction_summary``
    (see docs/MVP_DESIGN.md).
    """
    sq = _require_data("ligrec")
    return sq.gr.ligrec(
        adata, cluster_key=cluster_key, n_perms=n_perms, threshold=threshold, copy=True, seed=seed
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Unified dataset fetch + ingestion framework for NicheLens-ST "
        "(framework + registry; never fabricates URLs; --list/--card/dry-run are offline).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--list", action="store_true", help="list every registered dataset (offline)")
    p.add_argument("--dataset", metavar="ID", help="dataset id to fetch / plan")
    p.add_argument("--card", metavar="ID", help="print the YAML card for a dataset (offline)")
    p.add_argument("--emit-cards", action="store_true", help="write data/cards/*.yaml from the registry")
    p.add_argument(
        "--download",
        action="store_true",
        help="actually download (opt-in; verified/builtin sources only; never in CI)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="print the fetch plan without downloading (default behaviour)",
    )
    p.add_argument(
        "--ligrec",
        action="store_true",
        help="describe the cell-cell-communication derived-data step (issue #57)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list:
        list_datasets()
        return 0
    if args.emit_cards:
        emit_cards()
        return 0
    if args.card:
        print_card(args.card)
        return 0
    if args.ligrec:
        ds = DATASETS["ligrec_omnipath"]
        print(f"[{ds.id}] {ds.name} (issue {_issue_ref(ds)})")
        print(f"  {ds.notes}")
        print("  usage: fetch_datasets.run_ligrec(adata, cluster_key='<cell-type col>')")
        print("  no raw download — OmniPath reference is resolved/cached at call time.")
        return 0
    if args.dataset:
        fetch(args.dataset, download=args.download)
        return 0

    # No action: show the registry summary (offline) and usage hint.
    list_datasets()
    print()
    print("Use --dataset <id> to plan a fetch, --card <id> for a card, or --help.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FetchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
