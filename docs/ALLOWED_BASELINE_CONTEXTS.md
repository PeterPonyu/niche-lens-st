# Allowed baseline contexts

This document is the canonical allowlist of every file or path permitted to
mention CellNiche (`Super-LzzZ/CellNiche`) or scComm (`ZijieJin/scComm`) by
name. Any new file that legitimately needs to cite a baseline must be added
here via PR review before the mention is merged.

`scripts/check_independence.sh` guards `src/`, `tests/`, and `scripts/` against
identifier leakage and references this document in its error output. All paths
listed below are **outside** those guarded directories and therefore implicitly
excluded from the CI scan.

## Permitted files and rationale

| File / path pattern | Justification |
|---|---|
| `BASELINE_REFERENCES.md` | Primary provenance manifest — baseline names are the subject of this file |
| `baseline_repos/` (entire directory) | Checkout instructions and audit metadata for external baseline repos |
| `docs/LICENSE_NOTES.md` | Records baseline license terms; names are required context |
| `docs/PROVENANCE_CHECKLIST.md` | Tracks provenance steps per baseline; names are required context |
| `docs/NAME_AND_BRAND_REVIEW.md` | Brand independence review document; baseline names appear as review subjects |
| `docs/SYNTHETIC_BENCHMARK.md` | Documents baseline-deferral statements; comparative framing requires names |
| `docs/MVP_DESIGN.md` | Line 3 states "CellNiche and scComm are cited only as comparison references" — this is an explicit design-doc context statement, not implementation use |
| `docs/INDEPENDENCE_AND_LEAKAGE_AUDIT.md` | Audit document; names appear in findings and the named-baselines guard table |
| `docs/ALLOWED_BASELINE_CONTEXTS.md` | This file — names appear as allowlist subjects |
| `README.md` (baseline-references section) | Lines 11 and 15 list CellNiche and scComm as prior-art baselines; this is standard academic attribution |
| `.github/ISSUE_TEMPLATE/*.yml` | Issue templates may reference baseline names in structured fields |
| `.github/pull_request_template.md` | PR template may reference baseline names in checklist items |
| `manuscript/DRAFT.md` | Baseline-comparison sections in the manuscript draft require names |

## Policy

- This list is reviewed on every PR that touches `docs/`, `README.md`,
  `BASELINE_REFERENCES.md`, or `.github/`.
- Source-code paths (`src/`, `tests/`, `scripts/`) are **never** on this list.
  If a baseline name is needed there (e.g., an adapter interface), open a
  separate issue to discuss and document it explicitly before merging.
- The CI guard (`scripts/check_independence.sh`) is the enforcement mechanism;
  this document is the human-readable policy record.
