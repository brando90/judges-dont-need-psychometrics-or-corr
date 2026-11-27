#!/usr/bin/env python3
"""Exact-fraction check of the Study B population absolute-error formulas.

This is a numerics/algebra check, not a Lean proof. It confirms the closed
forms used in the report:

  v_Delta(8 x 2) = 23/80 + (3/8) t
  v_Delta(4 x 4) = 31/80
  v_Delta(2 x 8) = 53/80 - (3/8) t

and the crossovers t = 4/15 and t = 11/15.
"""
from __future__ import annotations

import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import gtheory_lib as L  # noqa: E402

C1 = {
    "u": Fraction(1),
    "a": Fraction(3, 5),
    "b": Fraction(1, 10),
    "ua": Fraction(3, 5),
    "ub": Fraction(1, 10),
    "ab": Fraction(1, 10),
    "e": Fraction(1, 2),
}
C2 = {
    "u": Fraction(1),
    "a": Fraction(1, 10),
    "b": Fraction(3, 5),
    "ua": Fraction(1, 10),
    "ub": Fraction(3, 5),
    "ab": Fraction(1, 10),
    "e": Fraction(1, 2),
}

CLOSED = {
    (8, 2): lambda t: Fraction(23, 80) + Fraction(3, 8) * t,
    (4, 4): lambda t: Fraction(31, 80),
    (8, 2): lambda t: Fraction(23, 80) + Fraction(3, 8) * t,
    (2, 8): lambda t: Fraction(53, 80) - Fraction(3, 8) * t,
}
# fix dict (duplicate key above is harmless but keep one copy)
CLOSED = {
    (8, 2): lambda t: Fraction(23, 80) + Fraction(3, 8) * t,
    (4, 4): lambda t: Fraction(31, 80),
    (2, 8): lambda t: Fraction(53, 80) - Fraction(3, 8) * t,
}


def mix(t: Fraction):
    return {c: (1 - t) * C1[c] + t * C2[c] for c in L.COMPS}


def check(name, ok, detail=""):
    print(("PASS" if ok else "FAIL"), name, detail)
    return ok


def main() -> int:
    n_fail = 0
    t_grid = [Fraction(0), Fraction(3, 20), Fraction(3, 10), Fraction(2, 5),
              Fraction(9, 20), Fraction(1, 2), Fraction(11, 20), Fraction(3, 5),
              Fraction(7, 10), Fraction(17, 20), Fraction(1)]
    # include exact crossovers
    t_grid = t_grid + [Fraction(4, 15), Fraction(11, 15)]
    for t in t_grid:
        v = mix(t)
        for design, formula in CLOSED.items():
            got = L.d_study(v, *design)["v_Delta"]
            want = formula(t)
            ok = (got == want)
            if not ok:
                n_fail += 1
            if t in (Fraction(0), Fraction(1, 2), Fraction(1), Fraction(4, 15), Fraction(11, 15)) or not ok:
                check(f"v_Delta({design[0]}x{design[1]}) at t={t}", ok, f"got={got} want={want}")
    v_half = {d: CLOSED[d](Fraction(1, 2)) for d in CLOSED}
    n_fail += not check(
        "t=1/2 uniquely prefers 4x4",
        v_half[(4, 4)] < v_half[(8, 2)] and v_half[(4, 4)] < v_half[(2, 8)],
        {k: str(v) for k, v in v_half.items()},
    )
    n_fail += not check(
        "crossover 8x2 vs 4x4 at t=4/15",
        CLOSED[(8, 2)](Fraction(4, 15)) == CLOSED[(4, 4)](Fraction(4, 15)),
        CLOSED[(8, 2)](Fraction(4, 15)),
    )
    n_fail += not check(
        "crossover 4x4 vs 2x8 at t=11/15",
        CLOSED[(4, 4)](Fraction(11, 15)) == CLOSED[(2, 8)](Fraction(11, 15)),
        CLOSED[(4, 4)](Fraction(11, 15)),
    )
    n_fail += not check(
        "8x2 best at t=0",
        CLOSED[(8, 2)](0) < CLOSED[(4, 4)](0) < CLOSED[(2, 8)](0),
        "",
    )
    n_fail += not check(
        "2x8 best at t=1",
        CLOSED[(2, 8)](1) < CLOSED[(4, 4)](1) < CLOSED[(8, 2)](1),
        "",
    )
    # second run is deterministic; no RNG
    print("n_fail", n_fail)
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
