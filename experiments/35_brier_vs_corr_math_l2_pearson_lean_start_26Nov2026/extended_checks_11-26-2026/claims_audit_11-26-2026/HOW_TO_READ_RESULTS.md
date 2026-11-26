# How to read the results of the claims audit, 11-26-2026

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/35_brier_vs_corr_math_l2_pearson_lean_start_26Nov2026/extended_checks_11-26-2026/claims_audit_11-26-2026/HOW_TO_READ_RESULTS.md>

**Data.** No human ratings and no trained judges: every number is exact rational arithmetic or floating-point arithmetic on small synthetic laws written in `check_claims_audit.py`. "Random joint law" means a finite joint pmf of $(X,Y)$ on 3 to 7 distinct pairs drawn from the grid $\lbrace0,\dots,5\rbrace^2$, with integer weights from 1 to 9 normalized to sum to one (every weight an exact fraction); 300 of them are drawn with Python's `random.Random(26112026)`. Other instances are named in each check: random rating pmfs on six ratings (integer weights 0 to 6), random positive pmfs for the log-loss check (2,000 pairs, $K$ from 2 to 6), random decision problems for the Bayes-rule check (200 problems, 50 rival rules each), and two Gaussian densities for the quadrature. There is no train, validation or test split and no prior exposure.

**Metrics.**

- "max gap": the largest absolute difference, over all instances in a check, between a quantity computed from its definition and the closed form claimed on the page. Exact checks (fractions) have gap 0 by equality; floating-point checks print the gap and must be below $10^{-12}$ (they print values around $10^{-15}$).
- "differs on $m$ of $n$ laws": for a line that the audit finds false, the number of random laws on which the written expression differs from the correct one by more than $10^{-9}$. It is a count, not a rate estimate.
- Squared difference $E[(X-Y)^2]$, correlation $\rho$, covariance, variance, Brier distance $\sum_r(p(r)-q(r))^2$: as defined on the pages and in [`../squared_difference_vs_distribution_11-26-2026/HOW_TO_READ_RESULTS.md`](../squared_difference_vs_distribution_11-26-2026/HOW_TO_READ_RESULTS.md). Lower is better for the squared difference and the Brier distance; zero means equal variables or equal laws.
- "Free directions": the dimension of the set of signed measures on $K$ ratings with total mass zero and the first $j$ raw moments zero; an exact rank computed with fractions and, in Lean, a `finrank` of a kernel.

**What is subtracted.** Nothing is compared across methods in this folder. Where an entry subtracts, both terms are the same quantity on the same law (for example the page-2 line as written minus $2(1-\rho)$ on the same joint law).

**Uncertainty.** None to report: no check is a Monte Carlo estimate. The random laws only supply many instances of an identity; each check prints how many it ran.

**How instances were chosen.** Laws and problems are random from the fixed seed above, plus hand-written examples where a specific counterexample is the point (the two coins on $\lbrace0,5\rbrace$ and $\lbrace2,3\rbrace$, the truth $p=(\tfrac35,\tfrac25)$ with the report $(1,0)$, the three-point law with weights $\tfrac14,\tfrac12,\tfrac14$). Nothing is selected for a favourable outcome: the script reports PASS or FAIL for every check and exits with a nonzero status if one fails.

**Lean.** A theorem compiled with `lake env lean` and printing only the standard axioms is a proof for all finite laws satisfying the stated hypotheses, not a test on examples. The hypotheses (for example $0<\mathrm{Var}[X]$) are the ones the corresponding page line needs; `instance_hypotheses` shows one explicit law on which all of them hold, so the theorems are not vacuous. The file contains no `sorry`.

**Two corrections to the checks themselves.** The first version of C11b normalized a histogram over the plotted range only and reported 0.333 instead of 0.2887; the first version of C8b described the square-root score as proper without testing it. Both are fixed in the committed script (see the ledger, "Errors in my own earlier work").
