"""Regression test for #79: pytest must collect on a fresh checkout.

The project uses a src-layout (``src/nichelens_st/``). Without
``pythonpath`` set in ``[tool.pytest.ini_options]`` (or a root
``conftest.py``), running ``python -m pytest`` on a fresh clone fails
to import ``nichelens_st`` and aborts collection with
``ModuleNotFoundError``. Issue #79 reproduced 12 collection errors in
that state.

Two complementary assertions:

1. The pytest config in ``pyproject.toml`` declares ``pythonpath`` with
   the src-layout directory. This is the static contract: a contributor
   editing the config cannot drop the setting without the test failing.

2. Spawn ``python -m pytest --collect-only -q`` as a subprocess with an
   environment that does **not** set ``PYTHONPATH`` and assert it exits
   0. This is the runtime contract: the fresh-checkout flow described
   in the issue actually works.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys

try:
    import tomllib  # Python >= 3.11
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_pyproject_declares_src_pythonpath_for_pytest():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    pytest_cfg = pyproject.get("tool", {}).get("pytest", {}).get("ini_options", {})
    pythonpath = pytest_cfg.get("pythonpath")
    assert pythonpath is not None, (
        "[tool.pytest.ini_options] is missing `pythonpath`; "
        "without it `python -m pytest` cannot import nichelens_st "
        "on a fresh src-layout checkout (issue #79)."
    )
    # Accept either ["src"] or a list containing "src" — the contract
    # is that the src dir is on sys.path during collection.
    if isinstance(pythonpath, str):
        pythonpath = [pythonpath]
    assert "src" in pythonpath, (
        f"`pythonpath` must include 'src' for the src-layout to import; "
        f"got {pythonpath!r}."
    )


def test_pytest_collect_only_succeeds_without_pythonpath_env():
    """Spawn pytest collect-only in a clean env (no PYTHONPATH override)."""
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, (
        "`python -m pytest --collect-only -q` failed on a fresh checkout "
        f"(rc={result.returncode}).\n"
        f"stdout (tail):\n{result.stdout[-2000:]}\n"
        f"stderr (tail):\n{result.stderr[-2000:]}"
    )
