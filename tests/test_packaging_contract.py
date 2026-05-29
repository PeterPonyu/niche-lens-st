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


def test_model_extra_declares_torch():
    """Lock the torch-gating contract (#65): the encoder/model path is opt-in via
    the ``model`` extra, so that extra must actually declare a torch requirement."""
    root = pathlib.Path(__file__).resolve().parents[1]
    pyproject = tomllib.loads((root / "pyproject.toml").read_text())
    model = pyproject["project"].get("optional-dependencies", {}).get("model", [])
    assert any(dep.lower().startswith("torch") for dep in model), model
