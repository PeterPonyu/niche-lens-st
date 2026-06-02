"""Regression test for #266: optional ``[data]`` extra must fail gracefully.

The squidpy/scanpy-backed paths in ``scripts/data/fetch_datasets.py``
(``_load_squidpy_builtin`` for ``--download``, ``run_ligrec`` for ``--ligrec``)
import dependencies that ship only in the optional ``[data]`` extra. When that
extra is absent (the normal case after a base ``pip install ".[model,test]"``),
a bare ``import squidpy`` raised a raw ``ModuleNotFoundError`` with no hint
about how to fix it.

The fix adds ``_require_data(module)`` — mirroring the gated ``_NO_TORCH_MSG``
pattern in ``src/nichelens_st/encoder.py`` — which raises a ``FetchError`` naming
``pip install "nichelens-st[data]"`` when the import fails. This test pins that
contract using an injected, guaranteed-absent module name, so it runs regardless
of whether squidpy is installed in the test environment.

Fails on base (``_require_data`` does not exist → ``AttributeError`` escapes the
``pytest.raises(FetchError)``); passes once the guard is added.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

# Import the script module by path (scripts/data is not on sys.path). The module
# itself prepends its own dir so its ``from registry import ...`` resolves.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO_ROOT / "scripts" / "data" / "fetch_datasets.py"

_spec = importlib.util.spec_from_file_location("fetch_datasets", _SCRIPT)
assert _spec is not None and _spec.loader is not None
fetch_datasets = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("fetch_datasets", fetch_datasets)
_spec.loader.exec_module(fetch_datasets)


def test_require_data_missing_module_raises_actionable_fetch_error():
    """A missing optional dep surfaces as a FetchError naming the [data] extra."""
    with pytest.raises(fetch_datasets.FetchError) as excinfo:
        fetch_datasets._require_data("nichelens_definitely_absent_data_dep")

    msg = str(excinfo.value)
    assert "nichelens-st[data]" in msg, f"message must name the [data] extra: {msg!r}"
    assert "pip install" in msg, f"message must give the install command: {msg!r}"
    # The offender module name is surfaced so the user knows what was missing.
    assert "nichelens_definitely_absent_data_dep" in msg


def test_require_data_returns_module_when_present():
    """When the dependency exists, the imported module is returned unchanged."""
    import math as _math

    assert fetch_datasets._require_data("math") is _math


def test_offline_surface_imports_without_data_extra():
    """Importing the module must not require squidpy/scanpy (offline surface)."""
    assert callable(fetch_datasets.list_datasets)
    assert callable(fetch_datasets._require_data)
