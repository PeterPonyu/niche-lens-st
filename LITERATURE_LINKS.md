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

## Notes

- **⚠️ flags are intentional.** Where only a dataset page (not a paper DOI) is
  confirmed, the page is the citable source; resolve the exact paper DOI from the
  dataset's data-availability statement before manuscript use. No DOI is guessed.
- **Baselines** (CellNiche, scComm) are tracked separately in
  [`BASELINE_REFERENCES.md`](BASELINE_REFERENCES.md), not here.
- **Raw-count policy** (project-wide): raw integer count matrices + spatial
  metadata only — no FASTQ / WSI / normalized-only objects.
