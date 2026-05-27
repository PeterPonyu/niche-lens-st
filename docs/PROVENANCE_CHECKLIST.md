# NicheLens-ST pre-implementation checklist

Run this before any baseline execution or third-party-aware change.

## Baseline HEAD / license / provenance

For each baseline in `BASELINE_REFERENCES.md` (CellNiche, scComm):

- [ ] Re-run `git ls-remote --heads <baseline-url>` and record the HEAD SHA in `BASELINE_REFERENCES.md`.
- [ ] Re-run `gh repo view <baseline> --json licenseInfo,isArchived,defaultBranchRef`.
- [ ] Confirm GitHub `licenseInfo` matches the article-stated license; flag any mismatch in `BASELINE_REFERENCES.md`.
- [ ] Confirm the baseline is not archived and not removed.
- [ ] If the baseline becomes unavailable, apply the fallback rule recorded for that baseline.

## Vendoring policy

- [ ] No baseline source is added to this repository's tree.
- [ ] No git submodule references a baseline repository.
- [ ] `baseline_repos/` contains clone-at-SHA instructions only, never code.

## Brand / leakage scan

Run from repo root:

    bash scripts/check_independence.sh

The script must exit 0. Cross-brand terms are listed inside the script.

- [ ] `scripts/check_independence.sh` exits 0.
- [ ] No new prose claims that NicheLens-ST outperforms or validates against any baseline.
- [ ] No copy-paste from a baseline README, paper, or source file into this repository.
