# NicheLens-ST baseline references

Verification date: 2026-05-27

## Baseline decision summary

| Role | Baseline | Decision |
|---|---|---|
| Primary | CellNiche | Use as the primary public-code niche/microenvironment contrastive-learning baseline. |
| Secondary | scComm | Use as a public-code contrastive cell-cell communication reference; not the core niche baseline. |

## Primary baseline: CellNiche

- Paper title: CellNiche represents cellular microenvironments in atlas-scale spatial omics data with contrastive learning
- Venue/date: Nature Communications, published 2026-04-22
- DOI: 10.1038/s41467-026-71759-4
- Article URL: https://www.nature.com/articles/s41467-026-71759-4
- Code URL: https://github.com/Super-LzzZ/CellNiche
- Zenodo: https://doi.org/10.5281/zenodo.19143524
- Verification date: 2026-05-27
- Default branch: main
- Observed HEAD SHA: af58974ded7cf57299a9f8952d4cc6dffee39c6f
- Archive status: not archived at GitHub verification time
- GitHub licenseInfo: MIT
- License note: Article code availability states the software package is deposited at GitHub under the MIT license; GitHub licenseInfo also reports MIT. Re-check before any code reuse.
- Local use: Primary comparison/reference for contrastive cell-centric spatial-proximity subgraph embeddings and atlas-scale niche discovery.
- Fallback: If public code becomes unavailable, run GitHub/code search for a 2026 public-code niche or microenvironment contrastive-learning method. If none exists, select the newest suitable 2025 public-code niche representation reference and label the downgrade explicitly.
- Verification command/evidence:
  - `git ls-remote --heads https://github.com/Super-LzzZ/CellNiche.git`
  - `gh repo view Super-LzzZ/CellNiche --json licenseInfo,isArchived,defaultBranchRef`

## Secondary baseline: scComm

- Paper title: scComm: a contrastive learning framework for deciphering cell-cell communications at single-cell resolution
- Venue/date: Genome Biology, published 2026-03-24; version of record 2026-05-01
- DOI: 10.1186/s13059-026-04043-9
- Article URL: https://link.springer.com/article/10.1186/s13059-026-04043-9
- Code URL: https://github.com/ZijieJin/scComm
- Zenodo: https://doi.org/10.5281/zenodo.18946992
- Verification date: 2026-05-27
- Default branch: main
- Observed HEAD SHA: ed0f372fdc122333afa834150c566948bef68a29
- Archive status: not archived at GitHub verification time
- GitHub licenseInfo: MIT
- License note: GitHub licenseInfo reports MIT. The article itself is CC BY-NC-ND 4.0; do not conflate article license with repository software license.
- Local use: Secondary contrastive-learning reference for communication-aware niche interpretation and ligand-receptor-oriented evaluation ideas.
- Fallback: If public code becomes unavailable, mark as `deferred-unverified` or replace with another public-code 2026 contrastive CCC method; do not list as verified public-code baseline.
- Verification command/evidence:
  - `git ls-remote --heads https://github.com/ZijieJin/scComm.git`
  - `gh repo view ZijieJin/scComm --json licenseInfo,isArchived,defaultBranchRef`
