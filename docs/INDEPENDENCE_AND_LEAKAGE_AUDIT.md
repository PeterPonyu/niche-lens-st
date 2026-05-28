# Independence and leakage audit

Date: 2026-05-27

## Scope

Audit target: repository scaffold and baseline manifests.

## Findings

- No third-party source code is vendored.
- Baseline repositories are referenced through metadata and clone-at-SHA commands only.
- README and draft text avoid relying on other local project names for identity.
- Claims are `planned` only; no performance superiority is asserted.
- Baseline papers are described as prior art and comparison targets, not templates.

## Required future checks

Before implementation:

1. Re-run baseline license checks.
2. Avoid copying upstream function/class/file names unless required by an adapter interface and documented.
3. Keep figures and manuscript structure independently designed.
4. **Automated source-code leakage checks are now active from the first implementation commit** (not deferred to post-manuscript). `scripts/check_independence.sh` enforces this on every push and PR via the `ci.yml` workflow — see companion issue #21.

### Named baseline brands to guard against

The following upstream project identifiers must never appear as Python class/function/variable names, module imports, inline strings, or file names inside `src/nichelens_st/`, `tests/`, or `scripts/`:

| Upstream project | GitHub slug | Guarded identifier patterns |
|---|---|---|
| CellNiche | `Super-LzzZ/CellNiche` | `cellniche`, `cell_niche`, `cell-niche` |
| scComm | `ZijieJin/scComm` | `sccomm`, `sc_comm`, `sc-comm`, `scomm` |

These patterns are enforced by `scripts/check_independence.sh` (added 2026-05-27, issue #21).

## 2026-05-27 audit-cycle update

**Finding:** `src/`, `tests/`, and `scripts/` are **clean** — no CellNiche or scComm identifiers found in any tracked Python, shell, or TOML file.

**Scan command:**
```bash
grep -rniE 'cellniche|cell_niche|cell-niche|sc.?comm|scomm' src/ tests/ scripts/ \
    --include='*.py' --include='*.sh' --include='*.toml'
# Result: (no output — currently clean)
```

**Legitimate mention sites (named exceptions):**

| File | Content | Status |
|---|---|---|
| `docs/MVP_DESIGN.md` line 3 | "CellNiche and scComm are cited only as comparison references" | ✅ Design-doc context statement — permitted |
| `README.md` lines 11, 15 | Baseline references section listing CellNiche and scComm as prior-art repos | ✅ Baseline attribution — permitted |

These files are listed in the canonical allowlist at `docs/ALLOWED_BASELINE_CONTEXTS.md`.
