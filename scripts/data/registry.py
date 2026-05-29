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
    # ---- Xenium ---------------------------------------------------------
    "xenium_skin_melanoma": Dataset(
        id="xenium_skin_melanoma",
        name="Xenium Human Skin / Dermal Melanoma (Multi-Tissue & Cancer panel)",
        platform="Xenium",
        tissue="skin / dermal melanoma",
        issues=(37,),
        url_status="verified_page",
        page_url="https://www.10xgenomics.com/datasets/human-skin-data-xenium-human-multi-tissue-and-cancer-panel",
        reader="squidpy.read.xenium",
        citation_key="tenx_xenium_skin_2023",
        raw_count_artifact="cell_feature_matrix.h5 (+ MEX), 377-plex, ~68k+90k cells",
        raw_count_policy=RAW_COUNT_POLICY,
        contract=_XENIUM_CONTRACT,
        size="377 genes; 150k-250k cells; 0.5-3 GB",
        notes="Direct .h5 hotlink UNVERIFIED -- resolve Output Bundle from the page. "
        "Superseded at higher plex by Xenium Prime 5K (#48).",
    ),
    "xenium_breast_janesick": Dataset(
        id="xenium_breast_janesick",
        name="Xenium Human Breast Cancer (Janesick)",
        platform="Xenium",
        tissue="breast cancer",
        issues=(38,),
        url_status="verified_page",
        page_url="https://www.10xgenomics.com/products/xenium-in-situ/preview-dataset-human-breast",
        reader="squidpy.read.xenium",
        citation_key="janesick2023xenium",
        raw_count_artifact="cell_feature_matrix.h5, 313-plex (280+33), ~110-170k cells",
        raw_count_policy=RAW_COUNT_POLICY,
        contract=_XENIUM_CONTRACT,
        size="313 genes; ~110-170k cells; 0.4-1.5 GB (full outs 8-9 GB)",
        notes="Canonical Xenium demo; ideal first single-cell niche-graph integration test.",
    ),
    "gse293199_tnbc_xenium": Dataset(
        id="gse293199_tnbc_xenium",
        name="GSE293199 TNBC Xenium (OmiCLIP source)",
        platform="Xenium",
        tissue="triple-negative breast cancer",
        issues=(43,),
        url_status="verified_direct",
        page_url="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE293199",
        reader="squidpy.read.xenium",
        citation_key="omiclip2025",
        raw_count_artifact="Xenium output bundle raw counts, 280-plex (RAW.tar 13.6 GB)",
        raw_count_policy=RAW_COUNT_POLICY,
        contract=_XENIUM_CONTRACT,
        size="280-panel; ~160k cells; 13.6 GB full RAW.tar (subset ~3 GB)",
        notes="Verified GEO RAW.tar bundle. Start from the ~3 GB subset. "
        "direct_url is the GEO supplementary RAW.tar resolved from the GSE page.",
        direct_url="https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE293199&format=file",
    ),
    "xenium_lymph_node": Dataset(
        id="xenium_lymph_node",
        name="Xenium Human Lymph Node (Multi-Tissue & Cancer panel preview)",
        platform="Xenium",
        tissue="lymph node (immune / lymphoid)",
        issues=(45,),
        url_status="verified_page",
        page_url="https://www.10xgenomics.com/datasets/human-lymph-node-preview-data-xenium-human-multi-tissue-and-cancer-panel-1-standard",
        reader="squidpy.read.xenium",
        citation_key="tenx_xenium_lymph_node_2023",
        raw_count_artifact="cell_feature_matrix.h5 raw counts, 377-plex, 377,985 cells",
        raw_count_policy=RAW_COUNT_POLICY,
        contract=_XENIUM_CONTRACT,
        size="377-plex; 377,985 cells",
        notes="Healthy lymphoid organ -> germinal-centre/T-zone/B-follicle niches "
        "(immune axis). CC BY 4.0. Direct .h5 hotlink UNVERIFIED -- resolve from page.",
    ),
    "xenium_prime_5k_cancer": Dataset(
        id="xenium_prime_5k_cancer",
        name="Xenium Prime 5K human cancer cohort (Pan Tissue & Pathways panel)",
        platform="Xenium Prime",
        tissue="skin melanoma / breast / ovarian / cervical cancer",
        issues=(48,),
        url_status="verified_page",
        page_url="https://www.10xgenomics.com/datasets/xenium-prime-ffpe-human-skin",
        reader="squidpy.read.xenium",
        citation_key="tenx_xenium_prime_5k_2024",
        raw_count_artifact="cell_feature_matrix.zarr.zip / .h5 raw counts, ~5,006-plex",
        raw_count_policy=RAW_COUNT_POLICY,
        contract=_XENIUM_CONTRACT,
        size="~5,006-plex (skin) / ~5,100-plex (breast); 112,551 (skin) / 699,110 (breast) cells",
        notes="Upgrades #37/#38: ~13-16x plex jump (XOA v3.0.0), CC BY 4.0. Cohort members "
        "(extra section_id samples): skin melanoma, breast, ovarian (FF), cervical (FFPE). "
        "Per-file hotlinks UNVERIFIED -- pull the Xenium Output Bundle from each page.",
        aliases=("xenium-prime-ffpe-human-breast-cancer", "xenium-prime-fresh-frozen-human-ovary"),
    ),
    # ---- CosMx SMI ------------------------------------------------------
    "cosmx_nsclc_nanostring": Dataset(
        id="cosmx_nsclc_nanostring",
        name="CosMx Human NSCLC (Lung9 public)",
        platform="CosMx SMI",
        tissue="NSCLC (8 FFPE)",
        issues=(39,),
        url_status="verified_page",
        page_url="https://nanostring.com/products/cosmx-spatial-molecular-imager/ffpe-dataset/nsclc-ffpe-dataset/",
        reader="squidpy.read.nanostring",
        citation_key="he2022cosmx",
        raw_count_artifact="*_exprMat_file.csv raw cell-by-gene, 960-plex, >800k cells",
        raw_count_policy=RAW_COUNT_POLICY,
        contract=_COSMX_CONTRACT,
        size="960-plex; 50k-100k cells per rep; 2-5 GB",
        notes="Lung9_Rep1 distributed via NanoString/Bruker public S3. Per-FOV section_id. "
        "Resolve the Lung9_Rep1 flat-file bundle URL from the dataset page before download.",
    ),
    "cosmx_brain_frontal_cortex": Dataset(
        id="cosmx_brain_frontal_cortex",
        name="CosMx Human Brain (Frontal Cortex, WTx)",
        platform="CosMx SMI",
        tissue="frontal cortex FFPE",
        issues=(40,),
        url_status="gated",
        page_url="https://www.brukerspatialbiology.com/",
        reader="squidpy.read.nanostring",
        citation_key="bruker_cosmx_brain_2024",
        raw_count_artifact="exprMat raw counts, 6,078-plex (WTx), ~194k cells",
        raw_count_policy=RAW_COUNT_POLICY,
        contract=_COSMX_CONTRACT,
        size="6,078-plex; ~194k cells; ~3 GB",
        notes="Form-gated (name/email) at brukerspatialbiology.com; no anonymous URL. "
        "Download manually, then point the reader at the extracted directory.",
    ),
    "cosmx_wtx_colon": Dataset(
        id="cosmx_wtx_colon",
        name="CosMx Human Whole Transcriptome (WTx) Colon (sigmoid adenocarcinoma)",
        platform="CosMx SMI",
        tissue="sigmoid colon adenocarcinoma (Stage IVA)",
        issues=(49,),
        url_status="gated",
        page_url="https://www.brukerspatialbiology.com/products/cosmx-spatial-molecular-imager/ffpe-dataset/cosmx-human-whole-transcriptome-colon-dataset/",
        reader="squidpy.read.nanostring",
        citation_key="cosmx_wtx_18933plex_2024",
        raw_count_artifact="exprMat_file.csv raw counts, ~18,933-plex (WTx) + polygons + tx",
        raw_count_policy=RAW_COUNT_POLICY,
        contract=_COSMX_CONTRACT,
        size="~18,933-plex; multi-GB (size not printed on gated page)",
        notes="Upgrades #39/#40: ~3x plex over CosMx Brain, ~20x over NSCLC. Form-gated "
        "(registration) at brukerspatialbiology.com -- same access pattern as #40.",
    ),
    # ---- MERFISH / Vizgen ----------------------------------------------
    "merfish_mouse_brain_receptor_map": Dataset(
        id="merfish_mouse_brain_receptor_map",
        name="MERFISH Mouse Brain Receptor Map (9-slice)",
        platform="MERFISH",
        tissue="mouse brain",
        issues=(41,),
        url_status="unverified",
        page_url="https://info.vizgen.com/mouse-brain-data",
        reader="squidpy.read.vizgen",
        citation_key="vizgen_mouse_brain_receptor_2021",
        raw_count_artifact="..._cell_by_gene_S#R#.csv raw counts, 483-plex, 734,696 cells",
        raw_count_policy=RAW_COUNT_POLICY,
        contract=_MERFISH_CONTRACT,
        size="483 genes; 734,696 cells; 9 slices (3 coronal x 3 rep); 3-7 GB",
        notes="Guessed hubfs hotlink UNVERIFIED -- obtain the bundle link from "
        "info.vizgen.com/mouse-brain-data. 9 aligned slices -> strong multi-section signal.",
    ),
    "abc_atlas_zhuang_merfish": Dataset(
        id="abc_atlas_zhuang_merfish",
        name="Allen ABC Atlas - Zhuang MERFISH whole mouse brain",
        platform="MERFISH",
        tissue="whole mouse brain (atlas-scale, many sections)",
        issues=(46,),
        url_status="verified_page",
        page_url="https://alleninstitute.github.io/abc_atlas_access/descriptions/Zhuang-ABCA-1.html",
        reader="anndata.read_h5ad (ABC Atlas AWS S3 public bucket)",
        citation_key="zhang2023abcatlas",
        raw_count_artifact=".h5ad raw cell-by-gene counts + metadata (x,y centroids, z, brain_section_label, CCF)",
        raw_count_policy=RAW_COUNT_POLICY,
        contract={
            **_MERFISH_CONTRACT,
            "section_id": "obs['brain_section_label'] -> integer codes (100s of serial sections)",
            "coords": "metadata x,y centroids (+ z) and CCF coordinates (float32)",
        },
        size="e.g. Zhuang-ABCA-1: 1122-plex, 4.2M cells, 147 sections",
        notes="Open AWS S3 Public Dataset (CC BY 4.0, no gating) -- the only fully-open "
        "pull of the multi-section set. Resolve S3 object paths via abc_atlas_access docs.",
    ),
    "vizgen_ffpe_io_merfish": Dataset(
        id="vizgen_ffpe_io_merfish",
        name="Vizgen MERFISH FFPE Human Immuno-Oncology (16 samples x 8 tumor types)",
        platform="MERFISH",
        tissue="human multi-cancer (colon/liver/melanoma/ovarian/prostate/lung/breast/uterine)",
        issues=(47,),
        url_status="gated",
        page_url="https://vizgen.com/human-ffpe-immunooncology-release-roadmap/",
        reader="squidpy.read.vizgen",
        citation_key="vizgen_ffpe_io_2022",
        raw_count_artifact="cell_by_gene.csv raw counts, 500-plex, ~9M cells + cell_metadata.csv centroids",
        raw_count_policy=RAW_COUNT_POLICY,
        contract=_MERFISH_CONTRACT,
        size="500-plex; ~9M cells; 16 samples x 8 tumor types",
        notes="Vizgen Data Release Program (registration), like #41. No anonymous hotlink "
        "-- resolve per-sample URLs from the portal.",
    ),
    "merfish_hypothalamus_moffitt": Dataset(
        id="merfish_hypothalamus_moffitt",
        name="MERFISH Mouse Hypothalamus Preoptic Region (Moffitt 2018)",
        platform="MERFISH",
        tissue="mouse hypothalamus preoptic region",
        issues=(55,),
        url_status="squidpy_builtin",
        page_url="https://doi.org/10.5061/dryad.8t8s248",
        reader="squidpy.datasets.merfish",
        citation_key="moffitt2018merfish",
        raw_count_artifact="squidpy anchor normalized; raw counts via Dryad CC0 (doi:10.5061/dryad.8t8s248)",
        raw_count_policy="squidpy mirror is normalized; use Dryad raw counts (CC0) for DL training",
        contract={
            **_MERFISH_CONTRACT,
            "section_id": "factorize(obs['Bregma']) -> 8 anterior-posterior section codes",
            "coords": "obsm['spatial'] (2D) or obsm['spatial3d'] (3D) centroids",
            "labels": "obs['Cell_class'] -- niche prototype evaluation",
        },
        size="73,655 cells x 161 genes; 8 Bregma levels",
        notes="Loads via squidpy.datasets.merfish() (figshare 28169379 mirror); resolves "
        "anonymously. Primary single-cell niche-graph anchor.",
    ),
    "seqfish_mouse_embryo_lohoff": Dataset(
        id="seqfish_mouse_embryo_lohoff",
        name="seqFISH Mouse Organogenesis Embryo (Lohoff 2022)",
        platform="seqFISH",
        tissue="mouse organogenesis embryo (3 FOVs)",
        issues=(56,),
        url_status="squidpy_builtin",
        page_url="https://doi.org/10.1038/s41587-021-01006-2",
        reader="squidpy.datasets.seqfish",
        citation_key="lohoff2022seqfish",
        raw_count_artifact="squidpy anchor normalized subset; raw counts via Lohoff 2022 source",
        raw_count_policy="squidpy mirror is normalized; use source raw counts for DL training",
        contract={
            **_MERFISH_CONTRACT,
            "section_id": "factorize(obs[embryo/FOV column]) -> 3 section codes",
            "coords": "obsm['spatial'] (2D centroids)",
            "labels": "obs['celltype_mapped_refined'] -- niche prototype evaluation",
            "transcripts": "n/a in squidpy mirror",
        },
        size="19,416 cells x 351 genes; 3 embryo FOVs",
        notes="Loads via squidpy.datasets.seqfish() (figshare 26098403 mirror). "
        "Per-FOV section isolation required.",
    ),
    "wang2025_matched_tma": Dataset(
        id="wang2025_matched_tma",
        name="Wang 2025 iST FFPE matched multi-platform TMA (Xenium + MERSCOPE + CosMx)",
        platform="Xenium + MERSCOPE + CosMx (matched)",
        tissue="matched TMAs: 17 tumor + 16 normal types",
        issues=(133,),
        url_status="unverified",
        page_url="",
        reader="per-platform: squidpy.read.xenium / read.vizgen / read.nanostring",
        citation_key="wang2025ist",
        raw_count_artifact="per-platform raw count matrices + segmentation centroids (matched sections)",
        raw_count_policy=RAW_COUNT_POLICY,
        contract={
            **_XENIUM_CONTRACT,
            "section_id": "matched TMA core / platform code; loaders must PRESERVE cross-"
            "technology section matching so subgraphs stay platform-comparable",
        },
        size="matched TMAs: 17 tumor + 16 normal types across 3 high-plex platforms + scRNA",
        notes="Registry #24. Same physical sections on Xenium + MERSCOPE + CosMx -> gold "
        "for platform-invariant vs technology-specific niches. URL UNVERIFIED -- resolve "
        "the per-platform bundles from the Wang 2025 data-availability statement; never "
        "fabricate a hotlink.",
    ),
    # ---- Visium (spot-resolution robustness only) ----------------------
    "gse208253_oscc_visium": Dataset(
        id="gse208253_oscc_visium",
        name="GSE208253 OSCC Visium",
        platform="Visium v1",
        tissue="HPV-neg OSCC",
        issues=(42,),
        url_status="verified_direct",
        page_url="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE208253",
        reader="scanpy.read_visium / scanpy.read_10x_mtx",
        citation_key="gse208253_oscc_visium",
        raw_count_artifact="12x filtered_feature_bc_matrix.h5 raw SPOT counts (RAW.tar 153 MB)",
        raw_count_policy=RAW_COUNT_POLICY + " (SPOT resolution -- no cell boundaries)",
        contract=_VISIUM_CONTRACT,
        size="~18k genes; ~2.5k spots; 12 slides; 153.4 MB",
        notes="Spot resolution, NOT single-cell -- no cell boundaries. Use ONLY as a "
        "12-section cross-platform robustness check; each node is a spot, never a cell. "
        "direct_url is the GEO supplementary RAW.tar.",
        direct_url="https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE208253&format=file",
    ),
    # ---- Derived data (no raw download) --------------------------------
    "ligrec_omnipath": Dataset(
        id="ligrec_omnipath",
        name="Cell-cell communication via squidpy.gr.ligrec (OmniPath)",
        platform="derived (any loaded AnnData with a cell-type label)",
        tissue="n/a (runs on any ingested dataset)",
        issues=(57,),
        url_status="derived",
        page_url="https://omnipathdb.org/",
        reader="squidpy.gr.ligrec",
        citation_key="turei2021omnipath",
        raw_count_artifact="n/a -- derived from an already-loaded AnnData",
        raw_count_policy="no raw download; OmniPath ligand-receptor reference resolved/cached at call time",
        contract={
            "input": "any AnnData in the pipeline (post-graph-build, post-encoder) with a cell-type obs column",
            "output": "res['means'] + res['pvalues'] -> interaction_summary (per-niche L-R enrichment)",
            "section_id": "n/a",
            "edges": "n/a",
        },
        size="n/a (no download)",
        notes="Derived-data step: pair res with prototype_id from the encoder to build the "
        "interaction_summary output (docs/MVP_DESIGN.md). No data/cards download path.",
    ),
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
