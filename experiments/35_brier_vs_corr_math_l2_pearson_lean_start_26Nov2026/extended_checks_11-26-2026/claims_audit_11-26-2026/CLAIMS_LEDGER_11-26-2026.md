# Claims audit of the handwritten pages: what is right and what needs a fix (11-26-2026)

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/35_brier_vs_corr_math_l2_pearson_lean_start_26Nov2026/extended_checks_11-26-2026/claims_audit_11-26-2026/CLAIMS_LEDGER_11-26-2026.md>

**TLDR:** Brando asked on 11-26-2026 whether the proofs are checked with Lean and quick numerics, because some of what he sent might be wrong. This ledger checks every mathematical line on four of his pages (the 12:03 and 13:01 derivation pages and the annotated 6.437 page) that way. Of 24 items, 12 are right as written, 3 are right with a note, 4 are wrong or cannot be shown (a missing factor two on page 2, a scribbled variance line he had already crossed out, "six degrees of freedom" that should be four, and the goal "show Brier is local", which is false for three or more ratings), 2 are imprecise, 1 has a valid conclusion but an invalid reason ("log is one-to-one"), and 2 are questions. 45 Lean theorems compile with no `sorry`; 36 of 36 numeric checks pass. Synthetic data and exact arithmetic only; no human ratings.

**Status:** complete (11-26-2026). Companion to the round-2 transcriptions in [`../../assets/`](../../assets/) and to the notes in [`../`](../README.md).

## How to read this

1. **Verdicts.** CORRECT: the line holds as written. CORRECT, WITH A NOTE: it holds, but a symbol or a hypothesis has to be stated. WRONG: the line is false as written, or the goal cannot be shown. IMPRECISE: true under a condition the page does not state. REASON INVALID: the conclusion is true but the stated reason does not prove it. NOT A CLAIM: a question or a remark.
2. **Notation.** As written on the pages, $\tilde X$ is the standardized variable. Your PS on page 1 changes this: $\tilde X$ becomes the centered variable $X-\mu_X$ and the standardized variable becomes $Z_X=(X-\mu_X)/\sigma_X$. The "Written" lines below copy the page, tildes included; my own statements use $Z$. $\rho=\operatorname{corr}(X,Y)$.
3. **Where each check lives.** Names in backticks are Lean theorems in [`AuditClaims.lean`](AuditClaims.lean), unless a file is named. `C1a` and so on are numeric checks in [`check_claims_audit.py`](check_claims_audit.py), captured in [`results/check_claims_audit.out`](results/check_claims_audit.out). "Numerics only" means the claim is a calculus fact that I did not formalize.
4. **What the checks do and do not show.** They show that the stated identity or counterexample is true or false for the finite laws and ratings used (Lean: for all finite laws satisfying the stated hypotheses; numerics: for 100 to 2,000 random instances, exact fractions wherever the quantity is rational, tolerance $10^{-12}$ otherwise). They do not show anything about human raters, and they do not check how well I read the handwriting.

## Honest coverage: what was and was not machine-checked

1. **Before this audit.** The Brier, squared-error and correlation claims were checked in Lean 4 and by seeded numerics: the Mathlib-free file [`../../BrierCorr.lean`](../../BrierCorr.lean), 18 main theorems over Mathlib in `../lean/ProperScoring/` (`Basic.lean` and `Counterexamples.lean`), the Brier note in [`../brier_proper_6437_style_11-26-2026/Brier6437.lean`](../brier_proper_6437_style_11-26-2026/Brier6437.lean), and the scripts `check_math.py`, `discrete_means_check.py`, `check_brier_proper_6437_style.py` and `check_squared_difference_vs_distribution.py`.
2. **The gap.** For the page-3 note, the general statements (conditional-mean decomposition, lower bound over couplings, independent-draws formula, degrees-of-freedom count) had paper proofs and numerics; Lean covered only concrete instances (`../squared_difference_vs_distribution_11-26-2026/MeanMatching.lean` says so in its header). The three-term identity and the standardized identity $E[(Z_X-Z_Y)^2]=2(1-\rho)$ were already general Lean theorems (`mse_corr_form` and `standardized_mse` in `ProperScoring/Basic.lean`). The other line-by-line identities of pages 1 and 2, Gibbs' inequality for log-loss and the Bayes rule of the 6.437 page had no Lean statement of their own.
3. **Now.** `AuditClaims.lean` states each of those in general form over finite laws, with one concrete three-point law showing the hypotheses are satisfiable (`instance_hypotheses`). It compiles with no `sorry`, and every theorem uses only the three standard axioms (`propext`, `Classical.choice`, `Quot.sound`); see [`lean_check.txt`](lean_check.txt).
4. **Still not Lean-checked.** (a) How a pdf changes under rescaling, the Gaussian-noise reading of squared error and the integral form of the Brier divergence: calculus, numerics only (C10, C11). (b) Theorem 1 printed on the 6.437 page (log-loss is the only smooth, local, proper cost for three or more outcomes): cited, not re-proved. (c) The training simulations in `discrete_means_check.py`: they are experiments, not proofs; two of its 24 checks FAIL by design and are explained in its explainer. (d) My reading of the handwriting.

## The ledger

### Page 1: the ruled notebook sheet (12:03 photograph and the 13:01 extension)

**P1.1. Top line `2(1 - xy) = 2 - 2xy = 1 + 1 - xy - xy`. CORRECT.** Algebra, with `xy` the correlation: it restates the target of the derivation below it, $2(1-\rho)=1+1-\rho-\rho$. Checked by `ring` and by C2c.

**P1.2. "Q: why is Brier proper while l2 loss (iid eps) is 'gaussian noise'?" NOT A CLAIM (a question).** The short answer the checks give: squared error is the negative log-likelihood of Gaussian noise up to constants (C10a), and as a rule for forecasting it is proper only for the mean, since the conditional mean is its unique minimizer (`condmean_unique_minimizer`, C5b). Brier is proper for the whole distribution: the excess expected Brier score of reporting $q$ when the truth is $p$ is $\lVert p-q\rVert^2$ (`brier_proper` in `Brier6437.lean`). They are rules for different objects, so there is no tension.

**P1.3. `Cov`, `Corr` and `Corr = E[Z_X Z_Y]` as defined on the page. CORRECT.** `rho` and `E_ZZ` (the expectation of $Z_XZ_Y$ equals $\rho$); C2c.

**P1.4. Right-margin identities: $\operatorname{Cov}=E[XY]-E[X]E[Y]$; $\operatorname{Var}[X]=E[X^2]-\mu_X^2=E[X^2]-E[X]^2$; $E[X^2]=\sigma_X^2+\mu_X^2$; $E[XY]=\operatorname{cov}+\mu_X\mu_Y$. CORRECT.** `cov_eq_raw`, `var_eq_raw`, `E_sq`, `E_mul`; C1a to C1c, exact fractions on 300 random joint laws.

**P1.5. $E[(X-Y)^2]=E[X^2+Y^2-2XY]=E[X^2]+E[Y^2]-2E[XY]$. CORRECT.** `sqdiff_expand`; C1d.

**P1.6. Boxed: $E[(X-Y)^2]=\sigma_X^2+\sigma_Y^2-2\operatorname{cov}+\mu_X^2+\mu_Y^2-2\mu_X\mu_Y$. CORRECT.** `boxed_raw`; C1e, and the regrouping $(\mu_X-\mu_Y)^2+(\sigma_X^2+\sigma_Y^2-2\operatorname{cov})$ in C1f. Not on the page but useful for page 3: the same quantity equals $(\mu_X-\mu_Y)^2+(\sigma_X-\sigma_Y)^2+2\sigma_X\sigma_Y(1-\rho)$, a location gap plus a scale gap plus a coupling term (`three_term`, C2e).

**P1.7. Standardized derivation, boxed $E[(\tilde X-\tilde Y)^2]=2(1-\operatorname{corr})$. CORRECT, WITH A NOTE.** It holds when the tilde is the standardized variable and both standard deviations are positive (`std_sqdiff`, C2c). Under your PS (tilde becomes the centered variable) the boxed line is false: the centered version is $\sigma_X^2+\sigma_Y^2-2\operatorname{cov}$, which equals $2(1-\rho)$ only when both variances are one (`centered_sqdiff`; C3c: different on 300 of 300 random laws). So after the symbol change the three lines down to the box need $Z$ in place of the tilde, and the top-of-page definition `Z_X = X~` becomes $Z_X=\tilde X/\sigma_X$. The typeset [`../../DERIVATION.md`](../../DERIVATION.md) already uses the new convention; the frozen photographs and their transcription keep the old one.

### Page 2: the loose-leaf page (same writing in the 12:03 and 13:01 photographs)

**P2.1. $(a+b)^2=a^2+2ab+b^2=1^2+2ab+1^2$ and $(a-b)^2=a^2-2ab+b^2=1-2ab+1^2$. CORRECT.** Valid when $a$ and $b$ have second moment one, as the standardized variables do: $E[(a\pm b)^2]=2\pm2E[ab]$ (C2c, C2d).

**P2.2. First expansion line $E[(\tilde X-\tilde Y)^2]=E[\tilde X^2-\tilde X\tilde Y+\tilde Y^2]$. WRONG: the cross term is missing a factor two.** It should read $-2\tilde X\tilde Y$. As written, the line equals $2-\rho$ rather than $2(1-\rho)$, and the two agree only when $\rho=0$ (`slip_missing_two`, `slip_agrees_iff`). On the three-point law of `instance_slip_differs` ($\rho=\sqrt{2/3}\approx0.816$) the written line gives about 1.184 and the correct value is about 0.367. C3a: different on 300 of 300 random laws. The next line on the page already has the 2, and page 1's new derivation has it throughout, so this is a one-character slip.

**P2.3. $E[\tilde X]=E[X-\mu]/\sqrt{E[(X-\mu)^2]}=0$. CORRECT.** `E_Z`; C2a.

**P2.4. Scribbled-out line $\operatorname{Var}[\tilde X]=\operatorname{Var}[X-\mu]/\sqrt{\operatorname{Var}[\tilde X]}$. WRONG as written, and crossing it out was right.** The right-hand side contains the quantity being defined, and dividing a variable by $\sigma$ divides its variance by $\sigma^2=\operatorname{Var}[X]$, not by a square root. With the true $\operatorname{Var}[\tilde X]=1$ the right-hand side is $\operatorname{Var}[X]$, so the line is false unless $\operatorname{Var}[X]=1$ (`scribbled_line_false`, `var_centered`, and `instance_scribbled_false` with $\operatorname{Var}[X]=1/2$; C3b: false on 300 of 300 random laws). The correct chain, three lines lower, is P2.6.

**P2.5. $\operatorname{Var}[X]=E[(X-\mu)^2]=E[X^2]-E[X]^2$, with the margin notes $x^2-2\mu x+\mu^2$ and $x^2-\mu^2$. CORRECT.** `var_eq_raw`; C1b.

**P2.6. $\operatorname{Var}[\tilde X]=E[\tilde X^2]=E[(X-\mu)^2]/\sigma^2=\operatorname{Var}[X]/\operatorname{Var}[X]=1$. CORRECT.** `var_Z`, `E_Z_sq`; C2b.

**P2.7. "centering doesn't really chg pmf/pdf of a 1-1 invertible op RV". IMPRECISE.** For a pmf, an injective map carries every mass to a new support point and changes no mass (`masses_preserved_under_injective`, C11a; the function changes only because its support moves). For a pdf, translation keeps the shape, but dividing by $\sigma$, or any nonlinear invertible map, changes the height by the Jacobian: $p_Z(z)=\sigma\,p_X(\mu+\sigma z)$. The standardized Uniform$[0,1]$ has density $\sigma=0.2887$ on $(-1.732,1.732)$, not 1 (C11b, C11c; numerics only). A wording that holds: centering moves a pmf or pdf; standardizing also rescales a pdf's height; no map of this kind changes a pmf's masses.

**P2.8. $D_{\text{Brier}}(p^*\Vert\hat q)=\sum_x(p^*(x)-\hat q(x))^2$, or $\int(p^*(x)-\hat q(x))^2dx$. CORRECT, WITH A NOTE.** For pmfs it is exactly the excess expected Brier score (`expBrier_excess` and `brier_proper` in `Brier6437.lean`). For densities the integral equals the expected quadratic-score gap (C10b: quadrature on two Gaussians agrees with the closed form to ten digits), provided the densities are square-integrable. It is a divergence because the Brier score is proper, not by definition.

**P2.9. Closing note: "Brier vs $E[(X-Y)^2]$ is obviously very different. One is a diff at the distribution level the other is at the individual values of the RVs." CORRECT in substance (plain $X$, $Y$: no tildes).** What the checks add is that the squared difference of two random variables depends on how they are paired. With a fixed pairing it identifies only the conditional mean (`fixed_pairing_decomposition`, `condmean_unique_minimizer`; C5a, C5b). With a free pairing its minimum is a squared Wasserstein distance, zero only for equal laws (`sqdiff_lower_bound`, `zero_sqdiff_forces_equal_laws`; C6). Against independent draws it is improper (`indep_formula`, `truthful_not_optimal`; C7). Brier compares the two pmfs directly.

### Page 3: "the next step"

**P3.1. "if $E[(X-Y)^2]=0$ ... if $X=Y$ then the squared difference of $X,Y$ is zero too". CORRECT, and the converse holds.** Zero squared difference forces $X=Y$ almost surely and therefore equal laws (`sqdiff_self`, `zero_sqdiff_forces_equal_laws`; C6c).

**P3.2. $E[(X-Y)^2]=\mu_X^2+\mu_Y^2-2\mu_X\mu_Y+\sigma_X^2+\sigma_Y^2-2\operatorname{cov}$. CORRECT.** Same as P1.6 (`boxed_raw`, C1e).

**P3.3. "6 degrees of freedom, but only 1 equation, so it has an infinite set of pdfs". WRONG count; the conclusion stands.** Six masses on the ratings $0,\dots,5$ must sum to one and have the given mean: two equations, so four free directions, not six (and not five). With the variance matched too, three remain; five moments pin a six-point law; with two ratings the mean already fixes the law (`finrank_ker_momMap` in general, with $K-1-j$ free directions after normalization and $j$ moments, `four_free_directions`, `three_after_variance`, `none_after_five_moments`, `two_ratings_mean_is_distribution`). C4a: exact rank for $K=2,\dots,10$ and every $j$; C4b: $K=6$ gives 4, 3, 2, 1, 0 for $j=1,\dots,5$; C4c: four different pmfs on six ratings that share the mean 5/2.

**P3.4. "(B) is ignoring all higher order moments, so (B) can never do distribution matching". IMPRECISE: true for a fixed pairing, not in general.** The first half is right: $E[(X-Y)^2]$ depends on the pair only through $\mu_X,\mu_Y,\sigma_X,\sigma_Y,\rho$ (`three_term`). The second half holds for the mean difference, for correlation (blind to positive affine maps) and for the squared difference when the data fix the pairing, since then only $E[X\mid Z]$ is identified: two coins with the same conditional mean 5/2 and irreducible errors 25/4 and 1/4 have different laws and the same minimizer (`same_conditional_mean_same_minimizer`, `coins_same_conditional_mean`, `coins_irreducible_error`; C5c). It fails when the pairing is free: the minimum over couplings is a squared Wasserstein distance, zero exactly when the laws are equal (`sqdiff_lower_bound`; C6a to C6d; example C6d: halves on 0 and 5 against halves on 2 and 3 have minimum 4). Wording that holds: "can never do distribution matching when the pairing is fixed".

### The printed 6.437 notes page (12:25 photograph)

**S1. "TODO! Show Brier is local in this style of argument." WRONG as a goal: it cannot be shown for three or more outcomes.** Brier is local only for two outcomes. Exact counterexample with three: for $q_A=(\tfrac12,\tfrac12,0)$ and $q_B=(\tfrac12,\tfrac14,\tfrac14)$ the outcome 0 has $q(0)=\tfrac12$ in both and Brier cost $\tfrac12$ against $\tfrac38$, so the cost is not a function of $q(x)$ alone (`brier_qA`, `brier_qB`, `brier_three_not_local`; with two outcomes it is local, `brier_two_local`; all in `../brier_proper_6437_style_11-26-2026/Brier6437.lean`). This is consistent with the printed Theorem 1 (cited, not re-proved here).

**S2. "it's obvious log loss is proper ... log is a 1-1 function with its input!" REASON INVALID; the conclusion is right.** Log-loss is strictly proper by Gibbs' inequality (`gibbs`; C8a: no violation on 2,000 random pairs). Being one-to-one in the probability assigned to the outcome does not give properness. With truth $p=(\tfrac35,\tfrac25)$ the score $q(x)$ itself is one-to-one, yet reporting the truth earns $\tfrac{13}{25}$ while the point mass $(1,0)$ earns $\tfrac{15}{25}$ (`linear_score_improper`); the score $q(x)^2$ fails the same way (`square_score_improper`); the best report under $\sqrt{q(x)}$ is $\tfrac9{13}=0.692$, not $0.6$ (C8b). What makes log special is that it is both proper and local; the printed Theorem 1 says it is the only smooth rule with both for three or more outcomes.

**S3. Top-right margin: $\hat A(\cdot)=\arg\min_{\hat A}E_{H\mid y}[C(\hat A(y),H)]$ (the Bayes decision rule). CORRECT.** Choosing, for each observation $y$, the action with the smallest posterior expected cost minimizes the expected cost over all decision rules (`bayes_rule_optimal`; C9: 200 random problems against 50 rival rules each).

**S4. The questions on the page (point estimates and the partition function; the meaning of "peaky"; why define proper and local). NOT CLAIMS.** Nothing to check.

## Tally

| Verdict | Items | Count |
| --- | --- | --- |
| CORRECT | P1.1, P1.3, P1.4, P1.5, P1.6, P2.1, P2.3, P2.5, P2.6, P3.1, P3.2, S3 | 12 |
| CORRECT, WITH A NOTE | P1.7, P2.8, P2.9 | 3 |
| WRONG or cannot be shown | P2.2, P2.4, P3.3, S1 | 4 |
| IMPRECISE | P2.7, P3.4 | 2 |
| REASON INVALID | S2 | 1 |
| NOT A CLAIM | P1.2, S4 | 2 |
| Total | | 24 |

## Errors in my own earlier work that this audit found or confirmed

1. The 12:03 transcription read the closing note on page 2 as having tildes on $X$ and $Y$; it has none. This was found before this audit and recorded as an erratum in the round-2 transcription (P2.9 uses the corrected reading).
2. The general statements of the page-3 note had paper proofs and numerics but Lean only on concrete instances (see "Honest coverage"); this folder closes that gap.
3. Two of my own checks in this folder were wrong when first written. C8b described the square-root score as "proper-looking" without testing it; it is improper (best report 0.692), and the check now asserts that. The first C11b reported a density of 0.333 instead of 0.2887 because numpy's `density=True` normalizes over the plotted range only, which cuts off part of the support; it now divides by all samples. Both were fixed before the results were saved.

## Not audited here

1. The text of the Cursor cloud agent's report and of any other model's output you relayed. Say which lines you want checked and they get the same treatment.
2. The prose of the plan for Sanmi (not mathematics).
3. Anything about human raters: no human ratings were used.

## Files

- [`AuditClaims.lean`](AuditClaims.lean) and [`lean_check.txt`](lean_check.txt): the Lean file and its compile and axiom log.
- [`check_claims_audit.py`](check_claims_audit.py) and [`results/check_claims_audit.out`](results/check_claims_audit.out): the numeric checks and their captured output (36 PASS, 0 FAIL).
- [`HOW_TO_READ_RESULTS.md`](HOW_TO_READ_RESULTS.md): data, metrics and tolerances behind the numbers.
