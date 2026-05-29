from __future__ import annotations

import pathlib
import tomllib


def test_numpy_is_runtime_dependency_for_numpy_importing_modules():
    root = pathlib.Path(__file__).resolve().parents[1]
    pyproject = tomllib.loads((root / "pyproject.toml").read_text())
    deps = pyproject["project"].get("dependencies", [])
    assert any(dep.lower().startswith("numpy") for dep in deps)


def test_test_extra_does_not_hide_runtime_numpy_requirement():
    root = pathlib.Path(__file__).resolve().parents[1]
    pyproject = tomllib.loads((root / "pyproject.toml").read_text())
    test_deps = pyproject["project"].get("optional-dependencies", {}).get("test", [])
    assert all(not dep.lower().startswith("numpy") for dep in test_deps)


def test_data_extra_declares_squidpy_and_scanpy():
    """Issue #71: the data-ingestion path imports squidpy/scanpy, so a ``data``
    extra must exist to make that path installable (``pip install .[data]``)."""
    root = pathlib.Path(__file__).resolve().parents[1]
    pyproject = tomllib.loads((root / "pyproject.toml").read_text())
    data = pyproject["project"].get("optional-dependencies", {}).get("data", [])
    names = [dep.lower() for dep in data]
    assert any(n.startswith("squidpy") for n in names), data
    assert any(n.startswith("scanpy") for n in names), data
