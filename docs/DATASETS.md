# NicheLens-ST dataset integration guide

Status: dataset-integration plan. This document maps the **audited & corrected**
ST dataset registry onto NicheLens-ST's actual ingestion path. It is the
real-data companion to [`docs/DATA.md`](DATA.md) (which covers the
squidpy-native smoke anchors A1/A2/B7) and tracks the larger imaging-ST corpus
we intend to wire in next.

**Canonical source of truth:**
`~/Desktop/ST_research/datasets/DATASET_REGISTRY.md` (audited 2026-05-28).
All figures, accessions, and link-verification flags below are copied from that
registry — do **not** reuse older uncorrected numbers from earlier audit drafts.
Policy (project-wide): raw integer count matrices + spatial metadata only — no
WSIs, FASTQs, BAMs, or normalized-only objects.

Legend: ✅ verified link · ⚠️ UNVERIFIED (canonical page given; the guessed
direct hotlink is **not** confirmed — resolve the bundle from the page before use).

---

## Method recap (why these datasets)

NicheLens-ST learns interpretable cellular **niche / microenvironment**
representations by building a per-section spatial neighborhood graph, extracting
a cell-centered subgraph as each node's "niche token", and training a contrastive
model that separates **conserved niche prototypes** from sample-specific
variants. Because every graph node must be a *real cell* with a known local
neighborhood, the method needs **single-cell-resolution imaging ST** that ships
**cell-segmentation boundaries** (→ per-cell centroids for `coords`) and
**transcript locations** (→ segmentation/QC and sub-cellular features). Xenium,
CosMx, and MERFISH are therefore first-class; spot-resolution Visium is included
only as a multi-section, cross-platform robustness check (with the caveat that a
spot is not a cell and carries no boundaries).

---

## How a dataset plugs into the niche-graph pipeline

Every dataset below is ingested through the same contract, using the repo's
**existing** entry points (no new abstractions required):

1. **Fetch** — register the dataset in `scripts/data/fetch_datasets.py`
   (`DATASETS` dict) and pull it with `--dataset <id> [--download]`. The script
   never fabricates URLs and never writes placeholder bytes; large pulls are
   opt-in (`--download`), defaulting to `--dry-run`.
2. **Read → AnnData** via the platform-native squidpy reader:
   - Xenium → `squidpy.read.xenium(path)` (cell_feature_matrix + `cells.parquet`
     centroids; `cell_boundaries.parquet` + `transcripts.parquet` carry the
     boundaries/transcript coords),
   - CosMx → `squidpy.read.nanostring(path, counts_file=…, meta_file=…, fov_file=…)`,
   - MERFISH/Vizgen → `squidpy.read.vizgen(path, counts_file=…, meta_file=…)`
     (`cell_by_gene.csv` + `cell_metadata.csv`),
   - Visium (GEO) → `scanpy.read_10x_mtx` / `read_visium`, then attach
     `obsm['spatial']` from the tissue-positions file.
3. **Map to the MVP input contract** (`src/nichelens_st/schemas.py::validate_inputs`):
   `X` (n_cells × n_genes, float32 raw counts) · `coords` (n_cells × 2/3 cell
   **centroids from segmentation boundaries**, float32) · `section_id`
   (per-FOV / per-slice int codes) · `edges` (2 × n_edges int64).
4. **Build the per-section graph** with `src/nichelens_st/graph.py::build_graph(
   coords, section_id, k, method="knn")` — or the squidpy-backed
   `build_contract(adata, section_key, …)` helper in `scripts/data/fetch_datasets.py`,
   which passes `library_key` so **no edge crosses a section / FOV boundary**.
5. **Extract niche tokens** per cell with
   `src/nichelens_st/graph.py::extract_subgraph(edges, center, k_hop)` and feed
   them to the contrastive encoder.

Per-dataset specifics live in `data/cards/<dataset_id>.yaml`
(see `data/cards/cosmx_nsclc_nanostring.yaml` for the schema-mapping template).

---

## Recommended datasets

| # | Dataset | Accession / page | Platform | Tissue / disease | Size | Link | Niche-graph fit (cell boundaries / transcript coords) |
|---|---------|------------------|----------|------------------|------|------|--------------------------------------------------------|
| 9  | Xenium Human Skin (multi-tissue + cancer panel) | 10x dataset page `human-skin-data-xenium-human-multi-tissue-and-cancer-panel` | Xenium | skin | 377 genes · 150k–250k cells · 0.5–3 GB | ⚠️ page `10xgenomics.com/datasets/human-skin-data-xenium-human-multi-tissue-and-cancer-panel` (guessed `.h5` hotlink unconfirmed) | Segmented cells + `cell_boundaries.parquet` + `transcripts.parquet` → centroid `coords`, transcript-level QC; multi-region → `section_id`. |
| 10 | Xenium Dermal Melanoma + Prostate | 10x dataset pages | Xenium | melanoma / prostate ca | panel · 150k+ cells · 1–3 GB | ✅ cited in STORM data-availability; 10x dataset pages | Same Xenium boundary+transcript outputs; adds tumor-niche diversity for conserved-vs-variant prototype separation. |
| 11 | Xenium Breast Cancer (Janesick) | 10x `Xenium_V1_human_Breast` demo | Xenium | breast cancer | 313 genes (280+33) · ~110k–170k cells · 0.4–1.5 GB (full outs 8–9 GB) | ✅ 10x Xenium human breast demo | Canonical Xenium demo with full boundary + transcript bundle; ideal first single-cell niche-graph integration test. |
| 12 | CosMx Human NSCLC | NanoString / Bruker (Lung9 public S3) | CosMx SMI | NSCLC (8 FFPE) | 960-plex · 50k–100k cells · 2–5 GB | ✅ `nanostring.com` / `brukerspatialbiology.com` FFPE NSCLC — **direct S3 already wired** (Lung9_Rep1) | Per-FOV segmented cells + metadata; `obs['fov']` → `section_id` (graph must not bridge FOVs). Raw counts → DL training. |
| 13 | CosMx Human Brain (Frontal Cortex) | Bruker (form-gated) | CosMx SMI | frontal cortex FFPE | 6,078-plex · ~194k cells · ~3 GB | ⚠️ form-gated (name/email) at `brukerspatialbiology.com` | High-plex single cells → rich niche markers; per-FOV `section_id`; manual download (form-gated). |
| 14 | MERFISH Mouse Brain Receptor Map | Vizgen | MERFISH | mouse brain | 483 genes · 734,696 cells · 9 slices (3 coronal × 3 rep) · 3–7 GB | ⚠️ page `info.vizgen.com/mouse-brain-data` (guessed `hubfs` hotlink unconfirmed) | `cell_by_gene` + `cell_metadata` (centroids/volumes) via `read.vizgen`; 9 aligned slices → strong multi-section conserved-niche signal. |
| 15 | GSE208253 (OSCC Visium) | GEO `GSE208253` | Visium v1 | HPV-neg OSCC | ~18k genes · ~2.5k spots · 12 slides · 153.4 MB | ✅ `ncbi.nlm.nih.gov/geo` (`GSE208253_RAW.tar`) | ⚠️ **spot resolution, not single-cell — no cell boundaries.** Use as a 12-section cross-platform robustness check; each node is a spot, not a cell. |
| 18 | GSE293199 (TNBC Xenium) | GEO `GSE293199` | Xenium | TNBC | 280-panel · ~160k cells · 13.6 GB full `RAW.tar` (subset ~3 GB) | ✅ `ncbi.nlm.nih.gov/geo` (OmiCLIP source) | Single-cell Xenium with boundaries/transcripts; TNBC tumor-immune niches; start from the ~3 GB subset. |

> **Also tagged for NicheLens-ST in the registry (multi-platform, shared with
> factorgraph-st):** Cervilla 2026 Xenium & CosMx (Zenodo **17986017**, ✅) and
> XeniumMT & 5K (Zenodo **18000256**, ✅). Not in the first-pull set below; pull
> on demand if a matched multi-platform niche cohort is needed.

**First-pull order (registry):** 2 Xenium → 1 CosMx → MERFISH. Network ≈ 10–18 GB.
Concretely: Xenium Breast (#11) + Xenium Skin/Melanoma (#9/#10) → CosMx NSCLC
(#12, Lung9_Rep1 already wired) → MERFISH (#14).

---

## Local resources

- **Verified papers (8 PDFs + index):** `~/Desktop/ST_research/references/`
  (incl. the OmiCLIP / STORM papers cited above for #10 and #18).
- **Corrected provenance + `curl` commands:** `st_dataset_provenance_and_policy.md`
  (research brain dir) — concrete download commands; 3 URLs flagged ⚠️ UNVERIFIED
  (Xenium Skin, MERFISH, Visium HD CRC).
- **Audit trail:** `~/Desktop/ST_research/audits/` (`findings_*.md` + AUDIT_SUMMARY).
  The registry **supersedes** the older coverage matrix / audit report where they
  differ (e.g. GSE223561 was mislabeled as HCC; not used by this repo).
- **Repo-local cache (gitignored):** `data/raw/<dataset_id>/` (raw) and
  `data/processed/<dataset_id>/` — written by `scripts/data/fetch_datasets.py`,
  excluded via `.gitignore` (only `data/cards/` is tracked).
- **Shared research cache:** `~/Desktop/ST_research/data_cache/raw/<dataset_slug>/`
  (raw counts + spatial only) for caches reused across the sibling repos.

---

## ⚠️ UNVERIFIED-URL caveats (resolve the page bundle before downloading)

These three are *not* confirmed direct downloads — open the canonical page and
take the official output bundle, then record the resolved URL in the dataset card:

1. **Xenium Human Skin (#9)** — the `cf.10xgenomics.com/.../xenium_human_skin/…`
   hotlink is guessed. Use the output bundle linked from the skin dataset page
   `10xgenomics.com/datasets/human-skin-data-xenium-human-multi-tissue-and-cancer-panel`.
2. **MERFISH Mouse Brain (#14)** — the `info.vizgen.com/hubfs/v1/…cell_by_gene.csv.gz`
   hotlink is guessed. Obtain the link from `info.vizgen.com/mouse-brain-data`.
3. **CosMx Human Brain Frontal Cortex (#13)** — **form-gated** (name/email) at
   `brukerspatialbiology.com`; there is no anonymous direct URL. Download
   manually, then point the loader at the extracted directory.

(For contrast, CosMx NSCLC #12 *is* a verified anonymous S3 object and is already
wired into `scripts/data/fetch_datasets.py`; GSE208253 #15 and GSE293199 #18 are
verified GEO `RAW.tar` bundles.)

---

## Provenance / policy notes

- **No FASTQ / no WSI / no normalized-only objects.** Raw integer counts +
  spatial metadata only. Imaging platforms (#9–#14, #18) give true single-cell
  counts with boundaries; GSE208253 (#15) is spot-level and flagged accordingly.
- Figures here are the **corrected** registry values; older audit drafts carry
  known errors (notably GSE223561 liver-regeneration mislabel) and are not used.
- Each integrated dataset must land a `data/cards/<dataset_id>.yaml` card and a
  row in this table; large downloads stay opt-in and never run in CI/smoke.

---

## Extended validation datasets

This section records (a) the **raw-count verification verdict** for every dataset
already tracked as a `data` issue, and (b) **focused expansion** datasets that
close niche-validation gaps. All verdicts were re-checked against the canonical
registry and the distributing source's own file listing (2026-05-28). Policy is
unchanged: **raw integer count matrices + spatial metadata only**.

Legend: ✅ raw counts + single-cell boundaries/transcripts confirmed ·
⚠️ raw counts confirmed but caveated (access-gated, or spot-resolution = no cell
boundaries) · ❌ not a raw count matrix (none found — no replacements required).

### Verification verdicts (existing issues)

| Issue | Dataset | Reg # | Platform | Raw-count artifact (verified) | Boundaries / transcripts | Verdict |
|-------|---------|-------|----------|-------------------------------|--------------------------|---------|
| #37 | Xenium Human Skin / Dermal Melanoma | 9 / 10 | Xenium | `cell_feature_matrix.h5` (+ MEX) raw counts, 377-plex, ~68k+90k cells | `cell_boundaries.parquet` + `nucleus_boundaries.parquet` + `transcripts.parquet` + `cells.parquet` | ✅ (direct `.h5` hotlink ⚠️ — resolve from page) |
| #38 | Xenium Human Breast (Janesick) | 11 | Xenium | `cell_feature_matrix.h5` raw counts, 313-plex (280+33), ~110–170k cells | full boundary + transcript bundle | ✅ |
| #39 | CosMx Human NSCLC | 12 | CosMx SMI | `*_exprMat_file.csv` raw cell-by-gene, 960-plex, >800k cells | `*_metadata_file.csv` centroids + `fov`, `*_tx_file.csv`, cell polygons | ✅ (Lung9_Rep1 S3 wired) |
| #40 | CosMx Human Brain (Frontal Cortex) | 13 | CosMx SMI | `exprMat` raw counts, 6,078-plex (WTx), ~194k cells | metadata centroids/`fov` + polygons + tx | ⚠️ form-gated (name/email; no anonymous URL) |
| #41 | MERFISH Mouse Brain Receptor Map | 14 | MERFISH | `..._cell_by_gene_S#R#.csv` raw counts, 483-plex, 734,696 cells, 9 slices | `..._cell_metadata` centroids + boundary polygons + detected transcripts | ✅ (direct `hubfs` hotlink ⚠️ — resolve from page) |
| #42 | GSE208253 OSCC Visium | 15 | Visium v1 | 12× `filtered_feature_bc_matrix.h5` raw **spot** counts (RAW.tar 153 MB) | `tissue_positions_list.csv` only — **no cell boundaries** | ⚠️ spot-resolution → secondary/robustness only |
| #43 | GSE293199 TNBC Xenium | 18 | Xenium | Xenium output bundle raw counts, 280-plex (`RAW.tar` 13.6 GB) | `cell_boundaries`/`nucleus_boundaries` + `transcripts` | ✅ (start from ~3 GB subset) |

> **Outcome:** all seven distribute genuine **raw integer** count matrices — **0 ❌**, so no
> replacements were filed. #40 (form-gated) and #42 (spot-resolution, no boundaries)
> remain usable with the caveats above; #42 stays a cross-platform robustness check
> only, never a primary single-cell niche source. Per-issue evidence and exact file
> names are recorded as comments on issues #37–#43.

### New expansion datasets (focused, raw-count verified)

| Issue | Dataset | Platform | Tissue / why it fits | Raw-count artifact + geometry | Access |
|-------|---------|----------|----------------------|-------------------------------|--------|
| #45 | Xenium Human **Lymph Node** (Multi-Tissue & Cancer panel) | Xenium | Healthy lymphoid organ → **immune/germinal-centre niches** (missing axis); clean conserved-niche positive control | `cell_feature_matrix.h5` raw counts (377-plex, **377,985 cells**) + `cell_boundaries`/`nucleus_boundaries` + `transcripts` | ✅ open 10x page, CC BY 4.0 (direct hotlink ⚠️ — resolve from page) |
| #46 | Allen **ABC Atlas — Zhuang MERFISH** whole mouse brain (Zhang 2023) | MERFISH | **Atlas-scale, 100s of serial sections** (e.g. Zhuang-ABCA-1: 1122-plex, 4.2 M cells, 147 sections) → premier conserved-niche-across-sections test | `.h5ad` raw cell-by-gene counts + metadata (`x,y` centroids, `z`, `brain_section_label`) + CCF coords | ✅ open **AWS S3 Public Dataset**, CC BY 4.0 (no gating) |
| #47 | Vizgen **MERFISH FFPE Human Immuno-Oncology** | MERFISH | **16 samples × 8 human tumor types** under one 500-gene IO panel → conserved vs. indication-specific tumor-immune niches | `cell_by_gene.csv` raw counts (500-plex, ~9 M cells) + `cell_metadata.csv` centroids + boundary polygons + detected transcripts | ⚠️ Vizgen Data Release Program (registration); resolve per-sample URLs from portal |

> **Rationale:** these three add the gaps the current corpus misses — a healthy
> **lymphoid/immune** organ (#45), an **atlas-scale multi-section** brain for
> cross-section conserved-niche stability (#46, the only fully-open AWS pull of the
> three), and a **human multi-cancer immuno-oncology** panel (#47). Each was
> verified to distribute a raw integer count matrix with single-cell centroids
> (and, except the MERFISH centroid-only cases, segmentation polygons + transcript
> coordinates) before filing. Ingestion follows the same contract above; no new
> abstractions. Large/registration-gated pulls stay opt-in and never run in CI.

---

## Dataset fetch framework (registry + cards)

The roadmap above is now backed by a concrete, **dependency-light** framework
(framework + registry depth — *not* full runnable loaders, *not* docs-only):

- **`scripts/data/registry.py`** — the canonical, machine-readable registry.
  One `Dataset` entry per dataset records `id`, `platform`, `issues`, the
  link-verification `url_status` (`verified_direct` / `verified_page` /
  `squidpy_builtin` / `gated` / `unverified` / `derived`), `page_url`,
  `direct_url` (only when an anonymous artifact URL is confirmed), the squidpy
  `reader`, a `citation_key`, the raw-count artifact + policy, and the full
  **contract mapping** (`X` / `coords` / `section_id` / `edges` / `transcripts`).
- **`scripts/data/fetch_datasets.py`** — the unified CLI that dispatches a
  per-dataset fetch from the registry. It **never fabricates URLs and never
  writes placeholder bytes**:
  - `verified_direct` → real `urllib` download (opt-in `--download`),
  - `squidpy_builtin` → real `squidpy.datasets.<name>()` load (#55, #56),
  - `verified_page` / `gated` / `unverified` → a **guarded stub** that prints
    `URL UNVERIFIED — see issue #N` (or the gated/manual-download instruction)
    and exits non-zero, pointing at the canonical page to resolve the bundle.
  - `--list`, `--card <id>`, `--emit-cards`, `--ligrec`, and the default
    `--dry-run` are **fully offline** (no network, no squidpy import).
- **`data/cards/<id>.yaml`** — a tracked card per dataset, generated from the
  registry via `python scripts/data/fetch_datasets.py --emit-cards`. The raw/
  processed caches (`data/raw/`, `data/processed/`) stay gitignored; only
  `data/cards/` is tracked.
- **`build_contract(adata, section_key, k)`** (in `fetch_datasets.py`) maps a
  loaded AnnData onto `(X, coords, section_id, edges)` by delegating to the
  repo's existing `nichelens_st.graph.build_graph` — no new abstraction — and
  asserts no edge bridges a section/FOV boundary.

Offline smoke (exit 0, no network):

```bash
python scripts/data/fetch_datasets.py --list
python scripts/data/fetch_datasets.py --card cosmx_nsclc_nanostring
python scripts/data/fetch_datasets.py --dataset xenium_breast_janesick   # dry-run plan
```

---

## Consolidated ingestion roadmap (per issue)

Every dataset is registered in `scripts/data/registry.py` with a `data/cards/<id>.yaml`
card and a row above. Loaders/downloads stay opt-in (`--download`) and never run
in CI/smoke; ⚠️ flags are preserved until the official bundle URL is resolved.

| Issue | Registry id | Status |
|-------|-------------|--------|
| #37 | `xenium_skin_melanoma` | ✅ page verified · hotlink ⚠️ (resolve bundle) |
| #38 | `xenium_breast_janesick` | ✅ page verified · hotlink ⚠️ (canonical first integration test) |
| #39 | `cosmx_nsclc_nanostring` | ✅ Lung9_Rep1 public; resolve flat-file bundle from page |
| #40 | `cosmx_brain_frontal_cortex` | ⚠️ form-gated (manual) |
| #41 | `merfish_mouse_brain_receptor_map` | ⚠️ UNVERIFIED hotlink (resolve from vizgen page) |
| #42 | `gse208253_oscc_visium` | ✅ GEO RAW.tar (spot-resolution robustness only) |
| #43 | `gse293199_tnbc_xenium` | ✅ GEO RAW.tar (start from ~3 GB subset) |
| #45 | `xenium_lymph_node` | ✅ page verified · hotlink ⚠️ (immune/lymphoid axis) |
| #46 | `abc_atlas_zhuang_merfish` | ✅ open AWS S3 (atlas-scale multi-section) |

Source papers for every dataset are tracked in [`LITERATURE_LINKS.md`](../LITERATURE_LINKS.md).
