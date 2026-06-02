"""Data-card hygiene guards (issue #265).

Every tracked ``data/cards/*.yaml`` must parse as YAML, and any ``section_id``
mapping that references an ``obs[...]`` accessor must quote the column key
(``obs['col']``) so it is valid Python/AnnData syntax — not the malformed bare
``obs[embryo/FOV column]`` form that #265 flagged.

The quoted key must also be a *real* AnnData column name — a single token with
no whitespace or ``/`` — so the placeholder ``obs['embryo/FOV column']`` (still a
quote-wrapped non-column) is rejected just like the bare form. The seqFISH card
now names the real column ``obs['embryo']``.
"""

import pathlib
import re

import pytest

yaml = pytest.importorskip("yaml")

CARDS_DIR = pathlib.Path(__file__).resolve().parents[1] / "data" / "cards"
CARD_PATHS = sorted(CARDS_DIR.glob("*.yaml"))


def test_cards_present():
    assert CARD_PATHS, "no data/cards/*.yaml found"


@pytest.mark.parametrize("path", CARD_PATHS, ids=lambda p: p.name)
def test_card_parses_and_section_id_accessor_is_quoted(path):
    card = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(card, dict), f"{path.name} did not parse to a mapping"
    section_id = str(card.get("contract", {}).get("section_id", ""))
    # If the mapping references an obs[...] accessor, the key must be quoted.
    for match in re.findall(r"obs\[([^\]]*)\]", section_id):
        is_quoted = (match.startswith("'") and match.endswith("'")) or (
            match.startswith('"') and match.endswith('"')
        )
        assert is_quoted, (
            f"{path.name}: unquoted obs accessor key in section_id: obs[{match}]"
        )
        # The quoted key must be a real AnnData column: a single token with no
        # whitespace or '/'. This rejects the #265 placeholder
        # obs['embryo/FOV column'] (quoted yet not a real column) — only a
        # concrete name like obs['embryo'] passes.
        key = match[1:-1]
        assert key and not re.search(r"[\s/]", key), (
            f"{path.name}: placeholder obs accessor key in section_id: "
            f"obs[{match}] — name the real column (e.g. obs['embryo'])"
        )
