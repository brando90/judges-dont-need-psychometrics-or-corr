# 38. Linear CKA, normalized OPD, and the invariance-group ladder

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/38_cka_opd_invariance_ladder_start_26Nov2026/README.md>

**TLDR:** Linear centered kernel alignment (CKA) on one column is squared Pearson correlation. Linear CKA and normalized orthogonal Procrustes distance (OPD) are unchanged by `X -> a X Q + b` with `a ≠ 0` and `Q` orthogonal, so a sign flip, a rubric permutation, or a rotation leaves them fixed while cross-entropy, Brier, and ranked probability score (RPS) move. A shear, which is invertible but not orthogonal, moves CKA and OPD and leaves canonical correlation analysis (CCA) fixed. Collapsing a within-prompt residual that is orthogonal to the human ratings raises linear CKA and lowers OPD, and the point mass at the conditional mean makes cross-entropy infinite.

The Lean file is `CkaOpd.lean` (Lean 4.36.0, no Mathlib, no `sorry`). The compiler log is `lean_check.txt`. Numeric checks are `proofs_numeric.py` and `tests/test_invariance.py`. Held-out numbers are in `results.json`, `tables/`, and `figures/`. How to read them is `HOW_TO_READ_RESULTS.md`. The one-paragraph remark and the group table are in `REPORT.md`.

Run from this folder:

```bash
lake build
python3 proofs_numeric.py
python3 -m pytest tests/test_invariance.py -q
python3 run_real.py
python3 make_figures.py
```

`lake exe cache get` does not apply: this development does not depend on Mathlib, so Mathlib was not compiled.
