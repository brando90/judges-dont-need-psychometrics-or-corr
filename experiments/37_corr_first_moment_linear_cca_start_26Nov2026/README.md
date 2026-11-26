# Experiment 37: correlation, first moments, and canonical correlation

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/37_corr_first_moment_linear_cca_start_26Nov2026/README.md>

**TLDR:** Across prompts, Pearson correlation is a standardized linear second-moment statistic, not a first-moment functional. Conditionally on the prompt, an independent judge's covariance equals the covariance of the conditional means, and within-prompt variance only enlarges the denominator. Canonical correlation analysis (CCA) is invariant to invertible affine mixes. On held-out SummEval expert items, reversing the class axis leaves the canonical correlations unchanged (maximum absolute difference 1.4e-16) and moves test cross-entropy from 1.244 [1.206, 1.280] to 2.720 [2.685, 2.758]. No VeriBench `R_v1` file was found. The experiment 09 export has 262 scores on 106 items. Read `HOW_TO_READ_RESULTS.md` before quoting a number.

## Question

Did the claim "correlation only captures first moments" overstate the fact, and does canonical correlation analysis repair it?

## Run

```bash
cd experiments/37_corr_first_moment_linear_cca_start_26Nov2026
python3 proofs_numeric.py
python3 -m pytest tests/test_claims.py -q
lake build
python3 run_real.py
python3 make_real_figures.py
```

`VERIBENCH_FSC_DIR` may point at the experiment 09 `ratings/` directory. The default path used for the committed numbers is the sparse clone recorded in `tables/veribench_status.md`.

## Read

- `REPORT.md` — the three answers, the reply, and the tables
- `results.md` — short ledger
- `tables/` — the grids the report quotes
- `lean_check.txt` — compiler and axiom list
