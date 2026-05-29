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

## Notes

- **⚠️ flags are intentional.** Where only a dataset page (not a paper DOI) is
  confirmed, the page is the citable source; resolve the exact paper DOI from the
  dataset's data-availability statement before manuscript use. No DOI is guessed.
- **Baselines** (CellNiche, scComm) are tracked separately in
  [`BASELINE_REFERENCES.md`](BASELINE_REFERENCES.md), not here.
- **Raw-count policy** (project-wide): raw integer count matrices + spatial
  metadata only — no FASTQ / WSI / normalized-only objects.
