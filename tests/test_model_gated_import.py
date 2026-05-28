"""The model module must import without torch installed (dependency-light base).

torch is an optional extra (``pip install nichelens-st[model]``). Importing
``nichelens_st.model`` must always succeed; the clear error is deferred until a
torch-backed code path is actually exercised.
"""

from __future__ import annotations

import importlib


def test_model_module_imports_without_torch():
    # Importing the module must never require torch at import time.
    module = importlib.import_module("nichelens_st.model")
    assert hasattr(module, "fit_niche_model")
    assert hasattr(module, "NicheModelConfig")
    assert hasattr(module, "NicheModelResult")


def test_torch_availability_flag_exists():
    module = importlib.import_module("nichelens_st.model")
    # A boolean flag advertises whether the optional extra is installed.
    assert isinstance(module.TORCH_AVAILABLE, bool)
