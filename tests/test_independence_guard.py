"""Regression test for #58: brand/leakage guard must stay green.

`scripts/check_independence.sh` is the brand/leakage guard the CI
``independence`` job runs on every push / PR
(``.github/workflows/ci.yml``), and the PR template asserts it exits 0
(``.github/pull_request_template.md``). Issue #58 showed the policy
could be silently violated by a docs change adding a sibling-brand
string anywhere outside the allowlisted directories.

Running the script under pytest gives us a fast local signal that
matches CI's ``independence`` job, so a contributor cannot land a
cross-brand reference without seeing this test fail locally first.

NOTE: this file intentionally avoids hard-coding the forbidden brand
strings inline so it does not trigger the very guard it asserts. The
regex source of truth lives in ``scripts/check_independence.sh``.
"""

from __future__ import annotations

import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_check_independence_script_exits_zero():
    result = subprocess.run(
        ["bash", "scripts/check_independence.sh"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        "scripts/check_independence.sh exited non-zero "
        f"(rc={result.returncode}).\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
