"""Canonical, machine-readable dataset registry for NicheLens-ST.

This module is the single source of truth for every spatial-transcriptomics
dataset NicheLens-ST intends to ingest. It is intentionally dependency-light
(stdlib only) so it imports offline without numpy/scipy/squidpy and can drive
``scripts/data/fetch_datasets.py`` (the unified fetch CLI) as well as emit the
per-dataset YAML cards under ``data/cards/``.

Why a framework + registry (not full loaders): NicheLens-ST needs
single-cell-resolution imaging ST with **cell-segmentation centroids** (→
``coords``) and **transcript locations**. This registry pins, per dataset, how
its raw artifacts map onto the MVP input contract
(``src/nichelens_st/schemas.py::validate_inputs``) and which squidpy reader
produces the AnnData. Loaders are dispatched per-dataset; actual downloads run
ONLY for sources with a verified, anonymous URL. UNVERIFIED / form-gated /
registration sources resolve to a guarded stub that points at the tracking
issue instead of fabricating a URL or writing placeholder bytes.

Link-verification flags (kept verbatim from the audited registry / issues):
- ``verified_direct``   : anonymous, resolvable artifact URL (download allowed).
- ``verified_page``     : public dataset page is verified, but the per-file
                          direct hotlink is UNVERIFIED — resolve the official
                          output bundle from the page first.
- ``squidpy_builtin``   : ships via ``squidpy.datasets.<name>()`` (figshare
                          mirror) — no URL to invent; loader is real.
- ``gated``             : form/registration gated; no anonymous URL exists.
- ``unverified``        : only a guessed hotlink is known — DO NOT download.
- ``derived``           : no raw download (computed from a loaded AnnData).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Dataset:
    """One registry entry mapping a dataset onto the MVP input contract."""

    id: str
    name: str
    platform: str
    tissue: str
    issues: tuple[int, ...]
    url_status: str
    page_url: str
    reader: str  # squidpy / scanpy entry point used to build the AnnData
    citation_key: str
    raw_count_artifact: str
    raw_count_policy: str
    contract: dict[str, str]
    size: str
    direct_url: str | None = None  # only set when url_status == "verified_direct"
    notes: str = ""
    aliases: tuple[str, ...] = field(default_factory=tuple)


# Contract-mapping shorthand reused across imaging platforms.
_XENIUM_CONTRACT = {
    "X": "cell_feature_matrix.h5 / .zarr.zip — raw integer cell x gene counts (float32 on load)",
    "coords": "cells.parquet / cell_boundaries.parquet segmentation centroids (float32)",
    "section_id": "per-region / per-sample integer codes (multi-region -> distinct sections)",
    "edges": "build_graph(coords, section_id, k, 'knn') int64; no cross-section edges",
    "labels": "derived cluster / annotation (optional, for prototype evaluation)",
    "transcripts": "transcripts.parquet transcript coordinates (QC / sub-cellular)",
}

_COSMX_CONTRACT = {
    "X": "*_exprMat_file.csv — raw integer cell x gene counts (float32 on load)",
    "coords": "*_metadata_file.csv CenterX/Y_global_px centroids (float32)",
    "section_id": "obs['fov'] per-FOV integer codes; graph must never bridge FOVs",
    "edges": "build_graph(coords, section_id, k, 'knn') int64; no cross-FOV edges",
    "labels": "derived annotation (optional)",
    "transcripts": "*_tx_file.csv transcript locations + per-cell polygon files",
}

_MERFISH_CONTRACT = {
    "X": "cell_by_gene.csv — raw integer cell x gene counts (float32 on load)",
    "coords": "cell_metadata.csv centroids (center_x/center_y, optional z) (float32)",
    "section_id": "per-slice / per-section integer codes (e.g. Bregma / brain_section_label)",
    "edges": "build_graph(coords, section_id, k, 'knn') int64; no cross-section edges",
    "labels": "obs cell-class labels where shipped (optional)",
    "transcripts": "detected_transcripts.csv (where shipped)",
}

_VISIUM_CONTRACT = {
    "X": "filtered_feature_bc_matrix.h5 — raw integer SPOT x gene counts (float32 on load)",
    "coords": "tissue_positions_list.csv spot centers (float32) -- NOT cell boundaries",
    "section_id": "per-slide integer codes",
    "edges": "build_graph(coords, section_id, k, 'knn') int64; per-slide only",
    "labels": "n/a (spot resolution)",
    "transcripts": "n/a -- spot resolution carries no transcript coordinates",
}

RAW_COUNT_POLICY = (
    "raw integer count matrix + spatial metadata only; no FASTQ / WSI / "
    "normalized-only objects"
)


DATASETS: dict[str, Dataset] = {
}


# First-pull order from the audited registry (2 Xenium -> 1 CosMx -> MERFISH).
FIRST_PULL_ORDER: tuple[str, ...] = (
    "xenium_breast_janesick",
    "xenium_skin_melanoma",
    "cosmx_nsclc_nanostring",
    "merfish_mouse_brain_receptor_map",
)


def all_issue_numbers() -> list[int]:
    """Every GitHub issue number covered by the registry, sorted."""
    nums: set[int] = set()
    for ds in DATASETS.values():
        nums.update(ds.issues)
    return sorted(nums)
