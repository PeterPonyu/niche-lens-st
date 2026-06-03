# NicheLens-ST dataset literature links

Source paper / data-release reference for every dataset tracked in
[`docs/DATASETS.md`](docs/DATASETS.md) and `scripts/data/registry.py`. The
`citation_key` column matches the registry's `citation_key` field.

Legend: ✅ DOI/URL confirmed from the issue or a stable identifier ·
⚠️ canonical page known but the exact paper DOI is **UNVERIFIED** — confirm
against the dataset's data-availability statement before citing in the
manuscript. URLs are never fabricated; ⚠️ entries link the dataset page, not a
guessed DOI.

| Issue | citation_key | Source paper / data release | Link | Flag |
|-------|--------------|-----------------------------|------|------|
| #37 | `tenx_xenium_skin_2023` | 10x Genomics — Xenium Human Multi-Tissue & Cancer panel, Skin/Melanoma public dataset (CC BY 4.0). Cited in the STORM data-availability statement. | https://www.10xgenomics.com/datasets/human-skin-data-xenium-human-multi-tissue-and-cancer-panel | ⚠️ data release (no single paper) |
| #38 | `janesick2023xenium` | Janesick et al., "High resolution mapping of the tumor microenvironment using integrated single-cell, spatial and in situ analysis", *Nature Communications* (2023). | https://doi.org/10.1038/s41467-023-43458-x | ⚠️ confirm DOI vs. data-availability |
| #39 | `he2022cosmx` | He et al., "High-plex imaging of RNA and proteins at subcellular resolution in fixed tissue by spatial molecular imaging", *Nature Biotechnology* (2022) — CosMx SMI / Lung9. | https://doi.org/10.1038/s41587-022-01483-z | ⚠️ confirm DOI |
| #40 | `bruker_cosmx_brain_2024` | Bruker Spatial Biology — CosMx Human Brain (Frontal Cortex, WTx 6,078-plex) FFPE public dataset (form-gated). | https://www.brukerspatialbiology.com/ | ⚠️ data release (gated) |
| #41 | `vizgen_mouse_brain_receptor_2021` | Vizgen MERFISH Mouse Brain Receptor Map data release (483-plex, 9 slices). | https://info.vizgen.com/mouse-brain-data | ⚠️ data release (no single paper) |
| #42 | `gse208253_oscc_visium` | GEO Series GSE208253 — HPV-negative OSCC Visium (12 slides). | https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE208253 | ⚠️ confirm associated paper |
| #43 | `omiclip2025` | GEO Series GSE293199 — TNBC Xenium (OmiCLIP source). | https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE293199 | ⚠️ confirm OmiCLIP paper DOI |
| #45 | `tenx_xenium_lymph_node_2023` | 10x Genomics — Human Lymph Node preview, Xenium Human Multi-Tissue & Cancer panel (CC BY 4.0). | https://www.10xgenomics.com/datasets/human-lymph-node-preview-data-xenium-human-multi-tissue-and-cancer-panel-1-standard | ⚠️ data release (no single paper) |
| #46 | `zhang2023abcatlas` | Zhang et al. (Zhuang lab), "Molecularly defined and spatially resolved cell atlas of the whole mouse brain", *Nature* (2023). | https://doi.org/10.1038/s41586-023-06808-9 | ✅ DOI from issue #46 |
| #47 | `vizgen_ffpe_io_2022` | Vizgen MERFISH FFPE Human Immuno-Oncology Data Release (16 samples × 8 tumor types, 500-plex). | https://vizgen.com/human-ffpe-immunooncology-release-roadmap/ | ⚠️ data release (registration) |
| #48 | `tenx_xenium_prime_5k_2024` | 10x Genomics — Xenium Prime 5K Human Pan Tissue & Pathways panel public datasets (2024, CC BY 4.0). | https://www.10xgenomics.com/datasets/xenium-prime-ffpe-human-skin | ⚠️ data release (no single paper) |
| #49 | `cosmx_wtx_18933plex_2024` | "Sub-cellular Imaging of the Entire Protein-Coding Human Transcriptome (18933-plex) … using Spatial Molecular Imaging", *bioRxiv* 2024.11.27.625536. | https://doi.org/10.1101/2024.11.27.625536 | ✅ preprint ID from issue #49 |
| #55 | `moffitt2018merfish` | Moffitt et al., "Molecular, spatial, and functional single-cell profiling of the hypothalamic preoptic region", *Science* (2018). Raw counts: Dryad `doi:10.5061/dryad.8t8s248` (CC0). | https://doi.org/10.1126/science.aau5324 · https://doi.org/10.5061/dryad.8t8s248 | ⚠️ confirm Science DOI; ✅ Dryad DOI from issue #55 |
| #56 | `lohoff2022seqfish` | Lohoff et al., "Integration of spatial and single-cell transcriptomic data elucidates mouse organogenesis", *Nature Biotechnology* (2022). | https://doi.org/10.1038/s41587-021-01006-2 | ⚠️ confirm DOI |
| #57 | `turei2021omnipath` | Türei et al., "Integrated intra- and intercellular signaling knowledge for multicellular omics analysis", *Molecular Systems Biology* (2021) — OmniPath backend for `squidpy.gr.ligrec`. | https://doi.org/10.15252/msb.20209923 | ⚠️ confirm DOI |
| #133 | `wang2025ist` | Wang 2025 — matched multi-platform iST FFPE TMA (Xenium + MERSCOPE + CosMx + scRNA). | (resolve from the data-availability statement) | ⚠️ UNVERIFIED — do not cite until confirmed |

## Notes

- **⚠️ flags are intentional.** Where only a dataset page (not a paper DOI) is
  confirmed, the page is the citable source; resolve the exact paper DOI from the
  dataset's data-availability statement before manuscript use. No DOI is guessed.
- **Baselines** are tracked separately in
  [`BASELINE_REFERENCES.md`](BASELINE_REFERENCES.md), not here.
- **Raw-count policy** (project-wide): raw integer count matrices + spatial
  metadata only — no FASTQ / WSI / normalized-only objects.
