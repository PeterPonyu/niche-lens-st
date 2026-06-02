"""Repo-hygiene guards (issues #63, #266).

Env-independent: imports the fetch CLI without touching the network or the
optional ``[data]`` extra, and checks the contributor entry point exists.
"""

import importlib.util
import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_contributing_exists():
    """#63: a single contributor entry point lives at the repo root."""
    assert (REPO_ROOT / "CONTRIBUTING.md").is_file()


def _load_fetch_module():
    path = REPO_ROOT / "scripts" / "data" / "fetch_datasets.py"
    spec = importlib.util.spec_from_file_location("fetch_datasets", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_data_extra_guard_message():
    """#266: the data-path guard names the optional ``[data]`` extra."""
    fetch = _load_fetch_module()
    assert 'nichelens-st[data]' in fetch._NO_DATA_MSG
    assert callable(fetch._require_data)
