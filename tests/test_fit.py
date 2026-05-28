"""Regression tests for fit_niche_model input contract (issue #62).

`fit_niche_model` previously skipped `validate_inputs`, so out-of-range or
otherwise schema-violating `edges` fell straight through to torch and surfaced
as an opaque `IndexError` from `index_add_`. The fix is to call
`validate_inputs` at the top of `fit_niche_model` so callers get the actionable
`SchemaError` the package already defines.
"""

from __future__ import annotations

import numpy as np
import pytest

from nichelens_st.model import TORCH_AVAILABLE, NicheModelConfig, fit_niche_model
from nichelens_st.schemas import SchemaError

pytestmark = pytest.mark.skipif(
    not TORCH_AVAILABLE, reason="requires the optional [model] extra (torch)"
)


def test_invalid_edges_raises_clear_error():
    """Out-of-range edge index → SchemaError, not opaque torch IndexError."""
    X = np.zeros((5, 4), dtype=np.float32)
    coords = np.zeros((5, 2), dtype=np.float32)
    section_id = np.zeros(5, dtype=np.int64)
    bad_edges = np.array([[0, 99], [1, 1]], dtype=np.int64)  # 99 ∉ [0, 5)
    cfg = NicheModelConfig(epochs=1, embed_dim=4, n_prototypes=2)
    with pytest.raises(SchemaError, match="outside"):
        fit_niche_model(X, coords, section_id, bad_edges, cfg)
