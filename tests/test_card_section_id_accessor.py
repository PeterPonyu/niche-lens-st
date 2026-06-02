"""Regression test for #265: data-card ``section_id`` accessors must be well-formed.

The seqFISH card (``data/cards/seqfish_mouse_embryo_lohoff.yaml``) shipped a
malformed mapping::

    section_id: factorize(obs[embryo/FOV column]) -> 3 section codes

``obs[embryo/FOV column]`` is not a valid AnnData accessor — it is an
unquoted placeholder, unlike the correct sibling pattern
``factorize(obs['Bregma'])`` in ``merfish_hypothalamus_moffitt.yaml``. A user
copying it into ``build_contract`` / ``fetch_datasets.py`` would hit a
``KeyError``/``SyntaxError`` instead of selecting a real column.

This test asserts that **every** card whose ``contract.section_id`` uses
``factorize(...)`` references a quoted ``obs['<column>']`` accessor. It fails on
the malformed card and passes once the accessor is corrected. The cards are the
emitted artifact of ``scripts/data/registry.py`` (``--emit-cards``), so this
also guards the registry source of truth against re-introducing the bug.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

CARDS_DIR = Path(__file__).resolve().parents[1] / "data" / "cards"

_HAS_FACTORIZE = re.compile(r"factorize\(")
# A well-formed factorize accessor: obs['col'] or obs["col"] with a
# non-empty, quoted column key.
_GOOD_ACCESSOR = re.compile(r"factorize\(\s*obs\[\s*['\"][^'\"]+['\"]\s*\]")


def _cards() -> list[Path]:
    return sorted(CARDS_DIR.glob("*.yaml"))


def test_cards_dir_is_present() -> None:
    assert _cards(), f"no data cards found under {CARDS_DIR}"


@pytest.mark.parametrize("card", _cards(), ids=lambda p: p.name)
def test_section_id_factorize_uses_quoted_obs_accessor(card: Path) -> None:
    for lineno, line in enumerate(card.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("section_id:") and _HAS_FACTORIZE.search(stripped):
            assert _GOOD_ACCESSOR.search(stripped), (
                f"{card.name}:{lineno} — section_id uses factorize() without a "
                f"quoted obs['<column>'] accessor (issue #265): {stripped!r}"
            )
