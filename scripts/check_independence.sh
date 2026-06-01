#!/usr/bin/env bash
# Fail if any cross-brand reference appears in tracked files.
# Run from repo root.
set -euo pipefail

PATTERN='lumina-?st|aether-?3d|FactorGraph|factorgraph'
# ``src/nichelens_st/results_contract.py`` is the VENDORED canonical
# cross-project results contract (byte-identical to the parent orchestration
# repo's ``scripts/contract/results_contract.py``; enforced by
# ``tests/test_contract_schema.py``). It legitimately enumerates the four
# sibling projects (lumina-st, aether-3d, factorgraph-st, niche-lens-st) and
# MUST NOT be edited to remove them, so it is excluded from the cross-brand
# scan exactly like this guard script excludes itself.
HITS=$(grep -rilE "$PATTERN" . \
    --exclude-dir=.git \
    --exclude-dir=.omc \
    --exclude-dir=baseline_repos \
    --exclude-dir=.venv \
    --exclude-dir=__pycache__ \
    --exclude='check_independence.sh' \
    --exclude='results_contract.py' 2>/dev/null || true)

if [ -n "$HITS" ]; then
    echo "FAIL: cross-brand references found in:" >&2
    echo "$HITS" >&2
    exit 1
fi

echo "PASS: no cross-brand references."

# ---------------------------------------------------------------------------
# Issue #21: Guard against CellNiche/scComm upstream identifier leakage.
# Scans src/, tests/, scripts/ for Python, shell, and TOML source files.
# Allowed-context files (docs/, README.md baseline section, baseline_repos/,
# manuscript/, .github/ templates) live outside these paths and are therefore
# implicitly excluded. The script itself is excluded via --exclude to avoid
# self-matching on the pattern string.
# Permitted mention sites are documented in docs/ALLOWED_BASELINE_CONTEXTS.md.
# ---------------------------------------------------------------------------
UPSTREAM_PATTERN='cellniche|cell_niche|cell-niche|sc.?comm|scomm'
UPSTREAM_HITS=$(grep -rilE "$UPSTREAM_PATTERN" src/ tests/ scripts/ \
    --include='*.py' --include='*.sh' --include='*.toml' \
    --exclude='check_independence.sh' 2>/dev/null || true)

if [ -n "$UPSTREAM_HITS" ]; then
    echo "FAIL: CellNiche/scComm identifier leakage found in guarded source paths:" >&2
    echo "$UPSTREAM_HITS" >&2
    echo "See docs/ALLOWED_BASELINE_CONTEXTS.md for the canonical allowlist." >&2
    exit 1
fi

echo "PASS: no CellNiche/scComm identifiers in guarded source paths (src/, tests/, scripts/)."

# ---------------------------------------------------------------------------
# Issue #23: Brand-metadata consistency check.
# Verify pyproject.toml description and CITATION.cff title use NicheLens-ST.
# ---------------------------------------------------------------------------
if ! grep -qiE 'NicheLens-ST' pyproject.toml 2>/dev/null; then
    echo "FAIL: pyproject.toml does not mention NicheLens-ST in description." >&2
    exit 1
fi

if ! grep -qE '^title:' CITATION.cff 2>/dev/null || ! grep -qiE 'NicheLens-ST' CITATION.cff 2>/dev/null; then
    echo "FAIL: CITATION.cff does not carry NicheLens-ST title." >&2
    exit 1
fi

if ! grep -qE '^keywords:' CITATION.cff 2>/dev/null; then
    echo "FAIL: CITATION.cff is missing a keywords field." >&2
    exit 1
fi

echo "PASS: brand-metadata fields present in pyproject.toml and CITATION.cff."

# Issue #23 (schema-level): validate CITATION.cff structure if cffconvert is
# available. The grep checks above cover content; cffconvert checks the file
# parses as a valid CFF document. Skipped (not failed) when cffconvert is not
# installed so the script remains usable in minimal local environments.
if command -v cffconvert >/dev/null 2>&1; then
    if ! cffconvert --validate >/dev/null 2>&1; then
        echo "FAIL: cffconvert --validate did not accept CITATION.cff." >&2
        cffconvert --validate >&2 || true
        exit 1
    fi
    echo "PASS: CITATION.cff parses as a valid CFF document (cffconvert)."
else
    echo "INFO: cffconvert not installed; skipping schema validation locally." >&2
fi
