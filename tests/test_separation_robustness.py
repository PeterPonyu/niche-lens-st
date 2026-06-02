"""Robustness of the conserved/sample_specific separation threshold (issue #105).

`_separation_head` is pure numpy (no torch), so these run without the optional
[model] extra. The strict default still requires presence in every section; a
``min_section_coverage`` < 1.0 tolerates a conserved prototype missing from a
small/shallow section by sampling chance.
"""

from __future__ import annotations

import numpy as np
import pytest

from nichelens_st.model import _separation_head


def test_strict_default_requires_every_section():
    # proto 0 in sections {0,1,2}; proto 1 in {0,1} (missing section 2).
    prototype_id = np.array([0, 0, 0, 1, 1])
    section_id = np.array([0, 1, 2, 0, 1])
    kinds = _separation_head(prototype_id, section_id, 2)  # default 1.0
    assert kinds[0] == "conserved"
    assert kinds[1] == "sample_specific"


def test_threshold_tolerates_one_missing_section():
    # 2 of 3 sections covered; ceil(0.6*3)=2 -> conserved under the relaxed rule.
    prototype_id = np.array([0, 0, 0, 1, 1])
    section_id = np.array([0, 1, 2, 0, 1])
    kinds = _separation_head(prototype_id, section_id, 2, min_section_coverage=0.6)
    assert kinds[0] == "conserved"
    assert kinds[1] == "conserved"


def test_threshold_still_flags_truly_specific():
    # proto 1 in only 1 of 3 sections; ceil(0.6*3)=2 -> still sample_specific.
    prototype_id = np.array([0, 0, 0, 1])
    section_id = np.array([0, 1, 2, 0])
    kinds = _separation_head(prototype_id, section_id, 2, min_section_coverage=0.6)
    assert kinds[1] == "sample_specific"


def test_single_section_unknown():
    prototype_id = np.array([0, 0, 1])
    section_id = np.array([0, 0, 0])
    assert _separation_head(prototype_id, section_id, 2) == ["unknown", "unknown"]


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.5])
def test_invalid_coverage_rejected(bad):
    with pytest.raises(ValueError):
        _separation_head(np.array([0, 0]), np.array([0, 1]), 1, min_section_coverage=bad)
