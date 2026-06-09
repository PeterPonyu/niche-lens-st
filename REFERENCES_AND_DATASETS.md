# NicheLens-ST — references (with code) & datasets

Consolidated reference + dataset index. Paper DOIs verified via Crossref and code
repositories via the GitHub API on 2026-06-09. See `BASELINE_REFERENCES.md`,
`docs/DATASETS.md`, and `LITERATURE_LINKS.md` for full provenance.

## Reference papers & method baselines (with public code)

| Role | Method | Venue / year | DOI | Code |
|------|--------|--------------|-----|------|
| Primary | CellNiche — contrastive cellular microenvironments at atlas scale | Nature Communications 2026 | `10.1038/s41467-026-71759-4` | https://github.com/Super-LzzZ/CellNiche (Zenodo 19143524) |
| Secondary | scComm — contrastive learning for cell–cell communication | Genome Biology 2026 | `10.1186/s13059-026-04043-9` | https://github.com/ZijieJin/scComm (Zenodo 18946992) |

## Datasets (audited registry — `docs/DATASETS.md`)

- Xenium: Breast (Janesick `Xenium_V1_human_Breast`), Skin/Melanoma, Lymph Node, Prime 5K; TNBC GEO **GSE293199**
- CosMx: NSCLC (Lung9_Rep1, public S3), Brain frontal cortex, WTx Colon
- MERFISH: Mouse-brain receptor map (Vizgen), Allen/Zhuang ABC Atlas whole brain (AWS S3), FFPE IO; OSCC Visium GEO **GSE208253**
- Tier B (squidpy): MERFISH hypothalamus (Moffitt 2018), seqFISH embryo (Lohoff 2022); ligand-receptor via OmniPath

> Verification: CellNiche + scComm DOIs confirmed in Crossref; both code repos live via
> GitHub API. GEO GSE293199 / GSE208253 confirmed accessible (2026-06-09).
