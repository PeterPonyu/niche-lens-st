#!/usr/bin/env python
"""Fetch verified, Python-native datasets for NicheLens-ST.

Project rule: no raw sequencing reads (no FASTQ). Only processed, analysis-ready
matrices (A1 MERFISH, A2 seqFISH) and raw single-cell counts (CosMx NSCLC, for
deep-learning training). Full provenance and the input-contract mapping live in
``docs/DATASETS.md`` (Tier B — Python-native quick-load).

Design notes:
- Squidpy/scanpy are imported lazily so ``--list`` and ``--dry-run`` work even
  before the heavy stack is loaded.
- No URL is fabricated. The only external direct download (CosMx) uses the exact
  verified S3 object and is gated behind ``--download``; otherwise the script
  only prints the plan and never writes placeholder bytes.

Examples:
    python scripts/data/fetch_datasets.py --list
    python scripts/data/fetch_datasets.py --dataset merfish_hypothalamus_shared
    python scripts/data/fetch_datasets.py --dataset cosmx_nsclc_nanostring --dry-run
    python scripts/data/fetch_datasets.py --dataset cosmx_nsclc_nanostring --download
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = REPO_ROOT / "data" / "raw"


@dataclass(frozen=True)
class Dataset:
    """One verified dataset entry."""

    dataset_id: str
    description: str
    access: str          # human-readable loader / access call
    platform: str
    raw_counts: bool
    section_key: str     # obs column used to split sections (no cross-section edges)
    url: str | None = None  # set only for external direct download (CosMx)


# All entries verified to exist in the conda env `dl` (squidpy 1.6.5, scanpy 1.10.4)
# on 2026-05-28. See docs/DATASETS.md (Tier B). URLs are real; do not invent alternates.
DATASETS: dict[str, Dataset] = {
    "merfish_hypothalamus_shared": Dataset(
        dataset_id="merfish_hypothalamus_shared",
        description="MERFISH mouse hypothalamus (Moffitt 2018), 73,655 cells x 161 genes",
        access="squidpy.datasets.merfish()",
        platform="MERFISH (imaging, single-cell resolution)",
        raw_counts=False,
        section_key="Bregma",
    ),
    "seqfish_embryo_shared": Dataset(
        dataset_id="seqfish_embryo_shared",
        description="seqFISH mouse embryo (Lohoff 2022), 19,416 cells x 351 genes",
        access="squidpy.datasets.seqfish()",
        platform="seqFISH+ (imaging, single-cell resolution)",
        raw_counts=False,
        section_key="embryo",
    ),
    "cosmx_nsclc_nanostring": Dataset(
        dataset_id="cosmx_nsclc_nanostring",
        description="CosMx NSCLC lung (He 2022, NanoString), ~800k cells, raw counts",
        access="squidpy.read.nanostring(path=..., counts_file=..., meta_file=..., fov_file=...)",
        platform="CosMx SMI (NanoString, single-cell resolution)",
        raw_counts=True,
        section_key="fov",
        url=(
            "https://nanostring-public-share.s3.us-west-2.amazonaws.com/"
            "SMI-Compressed/Lung9_Rep1/Lung9_Rep1+SMI+Flat+data.tar.gz"
        ),
    ),
}


def list_datasets() -> None:
    """Print the dataset registry."""
    print(f"{'dataset_id':<28} {'raw':<4} {'platform'}")
    print("-" * 78)
    for ds in DATASETS.values():
        print(f"{ds.dataset_id:<28} {'yes' if ds.raw_counts else 'no':<4} {ds.platform}")
        print(f"{'':<28} access: {ds.access}")
        if ds.url:
            print(f"{'':<28} url:    {ds.url}")
    print("\nLigand-receptor (CCC) path: squidpy.gr.ligrec() with the OmniPath")
    print("backend; no download required. See docs/DATASETS.md and ligrec_example().")


def load_squidpy_dataset(dataset_id: str):
    """Load A1/A2 via squidpy's Python-native one-line loaders (AnnData out)."""
    import squidpy as sq

    if dataset_id == "merfish_hypothalamus_shared":
        return sq.datasets.merfish()
    if dataset_id == "seqfish_embryo_shared":
        return sq.datasets.seqfish()
    raise ValueError(
        f"{dataset_id!r} is not a squidpy one-line dataset; "
        "use --download for cosmx_nsclc_nanostring."
    )


def download_cosmx(dest_dir: Path, *, dry_run: bool) -> Path:
    """Download the verified CosMx NSCLC tarball (raw counts) to ``dest_dir``.

    Never fabricates a URL and never writes placeholder bytes: with ``dry_run``
    it only prints the plan. The real archive is multi-GB; download only for
    training runs, never in CI/smoke.
    """
    ds = DATASETS["cosmx_nsclc_nanostring"]
    assert ds.url is not None
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / ds.url.rsplit("/", 1)[-1]

    if dry_run:
        print(f"[dry-run] would download: {ds.url}")
        print(f"[dry-run] -> {target}")
        print("[dry-run] then: tar -xzf <archive>; load with squidpy.read.nanostring(...)")
        return target

    import urllib.request

    print(f"Downloading {ds.url}\n  -> {target} (multi-GB; this is slow)")
    urllib.request.urlretrieve(ds.url, target)  # noqa: S310 (trusted public S3 host)
    print(f"Done: {target}. Extract, then load with squidpy.read.nanostring(...).")
    return target


def build_contract(adata, section_key: str, *, n_neighs: int = 8, use_3d: bool = False):
    """Return (X, coords, section_id, edges) matching docs/MVP_DESIGN.md.

    Builds a per-section kNN graph with squidpy's ``library_key`` so no edge
    crosses a section boundary, then emits a symmetric COO edge list.
    """
    import numpy as np
    import squidpy as sq
    from scipy.sparse import triu

    section_id = adata.obs[section_key].astype("category").cat.codes.to_numpy()
    adata.obs["_section_id"] = section_id.astype(str)

    sq.gr.spatial_neighbors(
        adata,
        coord_type="generic",
        n_neighs=n_neighs,
        library_key="_section_id",  # graph computed independently per section
    )

    conn = triu(adata.obsp["spatial_connectivities"].tocoo())
    edges = np.vstack([conn.row, conn.col]).astype(np.int64)

    coords = adata.obsm["spatial3d"] if use_3d else adata.obsm["spatial"]
    # invariant: every edge stays within one section
    assert bool((section_id[edges[0]] == section_id[edges[1]]).all())
    return adata.X, np.asarray(coords, dtype=np.float32), section_id.astype(np.int64), edges


def ligrec_example(adata, cluster_key: str, *, n_perms: int = 100, seed: int = 0):
    """Run the OmniPath ligand-receptor (CCC) summary on any labeled AnnData."""
    import squidpy as sq

    return sq.gr.ligrec(
        adata,
        cluster_key=cluster_key,
        n_perms=n_perms,
        threshold=0.01,
        copy=True,
        seed=seed,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--list", action="store_true", help="list the dataset registry and exit")
    p.add_argument("--dataset", choices=sorted(DATASETS), help="dataset_id to fetch/load")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="print the plan only; never download or write bytes (default for external downloads)",
    )
    p.add_argument(
        "--download",
        action="store_true",
        help="actually perform the external CosMx download (multi-GB)",
    )
    p.add_argument("--out", type=Path, default=DEFAULT_RAW_DIR, help="output dir for raw downloads")
    p.add_argument("--n-neighs", type=int, default=8, help="kNN neighbors per section for the graph helper")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.list or not args.dataset:
        list_datasets()
        return 0

    ds = DATASETS[args.dataset]

    if ds.url is not None:
        # external direct download (CosMx, raw counts)
        dest = args.out / ds.dataset_id
        download_cosmx(dest, dry_run=not args.download)
        if not args.download:
            print("\nPass --download to fetch the real archive.")
        return 0

    # squidpy one-line loader (A1/A2)
    if args.dry_run:
        print(f"[dry-run] would call: {ds.access}")
        print(f"[dry-run] section_key for per-section graph: {ds.section_key!r}")
        return 0

    print(f"Loading {ds.dataset_id} via {ds.access} ...")
    adata = load_squidpy_dataset(args.dataset)
    print(adata)
    print(f"section_key for per-section graph: {ds.section_key!r}; "
          "build edges with build_contract(adata, section_key, n_neighs=...).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
