# Contributing to NicheLens-ST

Thanks for contributing! This document consolidates the contribution culture that
the PR template, `CLAIM_LEDGER.md`, and `docs/PROVENANCE_CHECKLIST.md` already
assume, so you have a single entry point. Please read it before opening a PR.

## Development setup

The base package is intentionally lightweight (numpy/scipy only). Optional extras
gate the heavier paths:

```bash
pip install -e ".[test]"     # base + pytest
pip install -e ".[model]"    # adds torch for the contrastive niche encoder
pip install -e ".[data]"     # adds squidpy/scanpy/anndata for dataset fetch + CCC
```

The model encoder and the dataset-fetch / cell-cell-communication paths import
their heavy dependencies lazily and raise a clear, actionable error naming the
relevant extra when it is absent — so the base install stays small.

## Running tests and lint

```bash
python -m pytest -q                 # full test suite
bash scripts/check_independence.sh  # MUST exit 0 (see below)
```

CI also runs `ruff` lint and a Python-version matrix; keep both green. Tests that
exercise the encoder require the `[model]` extra; without it those tests skip
rather than fail.

## Independence guard (must pass)

This repository is an **independent** implementation and must not vendor or
reference third-party baseline source. Before every PR:

- `bash scripts/check_independence.sh` must exit 0.
- Do not add third-party (baseline) source files or git submodules that point at
  a baseline repository.
- Keep baseline provenance in `BASELINE_REFERENCES.md`; baseline checkout
  commands live in `baseline_repos/README.md` (third-party code is **not**
  vendored).
- Any new baseline-method mention (e.g. CellNiche / scComm) must sit in a file
  listed in [`docs/ALLOWED_BASELINE_CONTEXTS.md`](docs/ALLOWED_BASELINE_CONTEXTS.md);
  if it does not, update that allowlist as part of your change.

## Claim-ledger / provenance discipline

- **No performance or biology claim without local-test evidence.** Record claims
  and their supporting evidence in `CLAIM_LEDGER.md`.
- Follow `docs/PROVENANCE_CHECKLIST.md` when adding datasets, references, or
  results. Datasets are described in `docs/DATASETS.md`; downloads stay opt-in.
- If you touch `BASELINE_REFERENCES.md`, re-check the HEAD / license / provenance
  fields.

## Opening a pull request

- Use the [PR template](.github/pull_request_template.md) and complete the
  **Independence and provenance checklist**.
- Link issues with GitHub closing keywords: `Closes #N` to resolve and close on
  merge, or `Refs #N` / `Part of #N` for tracking/roadmap issues that should stay
  open.
- File new work through the issue templates under
  [`.github/ISSUE_TEMPLATE/`](.github/ISSUE_TEMPLATE/) (bug, feature, docs,
  benchmark).
- Keep PRs focused and ensure tests + lint + the independence guard are green
  before requesting review.
