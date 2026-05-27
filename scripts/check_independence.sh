#!/usr/bin/env bash
# Fail if any cross-brand reference appears in tracked files.
# Run from repo root.
set -euo pipefail

PATTERN='lumina-?st|aether-?3d|FactorGraph|factorgraph'
# .github/ is excluded so templates can legitimately name the forbidden
# sibling brands inside their guardrail checkboxes without self-flagging.
HITS=$(grep -rilE "$PATTERN" . \
    --exclude-dir=.git \
    --exclude-dir=.omc \
    --exclude-dir=baseline_repos \
    --exclude-dir=.venv \
    --exclude-dir=.github \
    --exclude='check_independence.sh' 2>/dev/null || true)

if [ -n "$HITS" ]; then
    echo "FAIL: cross-brand references found in:" >&2
    echo "$HITS" >&2
    exit 1
fi

echo "PASS: no cross-brand references."
