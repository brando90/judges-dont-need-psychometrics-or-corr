"""Seeded exact checks of the simulated-population identities. Run twice; output must match."""

from __future__ import annotations

import sys
from fractions import Fraction as F
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark import (  # noqa: E402
    LAW_A,
    LAW_B,
    M1,
    M2,
    expected_rps_rows,
    leakage_report,
    moment_panels,
    pearson,
    tvd_rows,
)


def check(name: str, ok: bool) -> None:
    print(f"{'PASS' if ok else 'FAIL'} {name}")
    if not ok:
        raise SystemExit(1)


def main() -> None:
    y = np.arange(6)
    check("M1 and M2 share mean 5/2", abs(M1 @ y - 2.5) < 1e-12 and abs(M2 @ y - 2.5) < 1e-12)
    check(
        "M1 and M2 share variance 4",
        abs(np.sum(M1 * (y - 2.5) ** 2) - 4) < 1e-12 and abs(np.sum(M2 * (y - 2.5) ** 2) - 4) < 1e-12,
    )
    gap = float(expected_rps_rows(M2[None, :], M1[None, :])[0] - expected_rps_rows(M1[None, :], M1[None, :])[0])
    check("RPS gap of M2 on M1 is 9/512", abs(gap - float(F(9, 512))) < 1e-12)
    check("TVD of M1 and M2 is 9/16", abs(float(tvd_rows(M2[None, :], M1[None, :])[0]) - float(F(9, 16))) < 1e-12)
    panels = moment_panels()
    check("panel Eρ² values match and equal 1/5", abs(panels["coeffs_U"]["erho2"] - F(1, 5)) < 1e-12 and abs(panels["coeffs_V"]["erho2"] - F(1, 5)) < 1e-12)
    panel_gap = panels["rps_V_on_U"] - panels["rps_U"]
    check("three-item RPS gap is 167/23040", abs(panel_gap - float(F(167, 23040))) < 1e-12)
    check("three-item TVD is positive", panels["tvd"] > 0.25)
    for law in (LAW_A, LAW_B):
        check(
            "endpoint law mean 3/2 and variance 2",
            abs(float(law @ y) - 1.5) < 1e-12 and abs(float(law @ (y - 1.5) ** 2) - 2) < 1e-12,
        )
    means = np.array([1.6, 2.0, 2.4, 2.8, 3.2, 3.6])
    check("constant shift preserves Pearson", abs(pearson(means, means - 0.8) - 1) < 1e-12)
    # A false claim must fail: M1 and M2 are not the same distribution.
    check("M1 is not M2", not np.allclose(M1, M2))
    leak = leakage_report()
    check("fit ignores a overwritten test array", leak["fit_ignores_test_array"])
    check("rater ids are disjoint across splits", leak["rater_ids_disjoint_across_splits"])
    check("item ids are disjoint across splits", leak["item_ids_disjoint_across_splits"])
    print("ALL PASS")


if __name__ == "__main__":
    main()
