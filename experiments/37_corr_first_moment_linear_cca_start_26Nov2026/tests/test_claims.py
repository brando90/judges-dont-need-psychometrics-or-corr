"""Pytest wrapper around the deterministic numerical checks. No network, no human ratings."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_numeric_claims_twice_identical():
    script = ROOT / "proofs_numeric.py"
    runs = []
    for _ in range(2):
        proc = subprocess.run([sys.executable, str(script)], cwd=ROOT, capture_output=True, text=True, check=False)
        runs.append((proc.returncode, proc.stdout))
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert "FAIL " not in proc.stdout
        assert proc.stdout.count("PASS ") == 7
    assert runs[0] == runs[1]


def test_toy_figures_exist():
    subprocess.run([sys.executable, str(ROOT / "make_toy_figures.py")], cwd=ROOT, check=True)
    for name in (
        "toy_nonlinear_vs_linear.png",
        "toy_onehot_cca_vs_proper.png",
        "toy_collapse_inflates_corr.png",
    ):
        path = ROOT / "figures" / name
        assert path.is_file() and path.stat().st_size > 1000
