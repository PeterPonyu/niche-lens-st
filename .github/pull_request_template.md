<!--
NicheLens-ST pull request template.
New here? See the contributor guide: ../CONTRIBUTING.md
-->

## Summary

<one-paragraph description of the change>

## Related issues

<!--
Use GitHub closing keywords to link issues:
- Closes #N  — resolves and closes the issue when this PR is merged
- Refs #N / Part of #N  — references without closing (use for roadmap/tracking issues)
-->

Closes #
<!-- Refs # -->

## Independence and provenance checklist

- [ ] No third-party (baseline) source is added to this repository.
- [ ] No git submodule references a baseline repository.
- [ ] `bash scripts/check_independence.sh` exits 0.
- [ ] No cross-brand references.
- [ ] Any new CellNiche / scComm mention sits in a file listed in
      [`docs/ALLOWED_BASELINE_CONTEXTS.md`](../docs/ALLOWED_BASELINE_CONTEXTS.md);
      if it does not, this PR updates that allowlist as part of the change.
- [ ] No performance or biology claim is asserted without local-test evidence (see `CLAIM_LEDGER.md`).
- [ ] If `BASELINE_REFERENCES.md` is touched, HEAD / license / provenance fields are re-checked.
