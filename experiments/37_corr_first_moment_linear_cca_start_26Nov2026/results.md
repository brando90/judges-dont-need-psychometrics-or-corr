# Results — correlation, first moments, and CCA

**Status:** COMPLETE

**Last updated:** 2026-11-26

**Bottom line.** Sanmi was right that "correlation only captures first moments" overstates the fact. Across prompts, Pearson correlation is a linear second-moment statistic (shift-invariant, scale-invariant in the square, and zero for `Y = X^2` on `{-1, 0, 1}`). Conditionally on the prompt, for an independent judge, the covariance equals the covariance of the conditional means and within-prompt variance only enlarges the denominator. Canonical correlation analysis (CCA) inherits the same invariance: on every held-out set, sign flips, label permutations, and orthogonal mixes change the canonical correlations by at most 4.2e-14, while reversal of SummEval expert probabilities moves test cross-entropy from 1.2442 [1.2055, 1.2801] to 2.7195 [2.6849, 2.7583]. There is no VeriBench `R_v1` file. The experiment 09 export has 262 scores on 106 items; on its 24 test items the same reversal moves cross-entropy from 1.1245 [0.7249, 1.5686] to 4.3581 [4.1534, 4.5388].

| Phase | State |
| --- | --- |
| Lean 4.36.0 `lake build`, axioms printed | DONE |
| Pytest of the exact and random checks, twice | DONE |
| SummEval, ChaosNLI, DICES-990, VeriBench FSC export | DONE |
| `R_v1` 343-item table | NOT FOUND, not imputed |

Read `REPORT.md` before quoting a number. Intervals are percentile 95%, item resample, B = 1000, seed 37.

**TLDR-end:** [judges-corr-vs-loss: pearson moments and CCA] Pearson correlation is a linear second moment across prompts and, conditionally, a covariance of means; CCA stays invariant to the maps that break cross-entropy, Brier, and ranked probability score, and the VeriBench export is 262 scores on 106 items rather than an `R_v1` file.
