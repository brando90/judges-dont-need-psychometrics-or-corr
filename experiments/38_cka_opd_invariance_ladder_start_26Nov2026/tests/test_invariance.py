"""Pytest wrapper around the deterministic numeric checks."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_numeric_script_passes_twice_identically():
    script = ROOT / "proofs_numeric.py"
    runs = []
    for _ in range(2):
        result = subprocess.run(
            [sys.executable, str(script)],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        runs.append(result.stdout)
    assert runs[0] == runs[1]
    assert "ALL PASS" in runs[0]
    assert "FAIL" not in runs[0]
