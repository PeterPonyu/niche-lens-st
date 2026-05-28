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
