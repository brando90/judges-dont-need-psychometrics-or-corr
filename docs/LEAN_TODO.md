# Lean TODO: claims not yet Lean-checked

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/docs/LEAN_TODO.md>

**TLDR:** Every theorem, proposition, lemma or proof in the paper, notes or docs that is not yet machine-checked in Lean 4 is listed here with its status label (rule: `AGENTS.md`, section "Every proof in Lean 4, with a quick numerical check first"). Remove a row once the Lean file is cited next to the claim.

Status labels: "numerically checked, Lean pending", "paper proof, not yet Lean-checked", "Lean-checked special case" (case stated).

| Claim | Location | Status | What is missing |
|---|---|---|---|
| Point 2 note, Props. `prop:identity`, `prop:murphy`, `prop:expected` beyond finite laws (general distributions with finite second moment) | `docs/notes/point2_encodings_vs_first_moments_27Nov2026.tex` | Lean-checked special case (finite and empirical laws; `docs/notes/point2_support_27Nov2026/lean/`) | measure-theoretic statement |
| Point 2 note, closing sentence of Prop. `prop:sampled`: the collapsed judge is strictly worse than the truth on population-average Brier, RPS and CE | same | per-item Brier and RPS strict propriety Lean-checked (`rps_strictly_proper`, `brier_strictly_proper`); averaging over items: paper proof, not yet Lean-checked; CE: cited (Gibbs' inequality) | Lean lemma summing per-item gaps with positive weights; a CE (log score) propriety lemma |
| Point 2 note, Remark "CRPS as the continuous limit" (CRPS equals unnormalized RPS on the integer grid) | same | paper proof, not yet Lean-checked; CRPS propriety cited | Lean statement on step CDFs |
| Paper section 4 and supplement S8: the per-sample risk (mean over an item's ratings, then over items) is unbiased for the expected score, exactly so with one rating per item | `paper_latex_and_notes/AAAI_2027_EvalAIScientists_JudgesKL/4-proposal.tex`, `12-supplementary-pool-strata-power.tex` | paper proof (linearity of expectation), not yet Lean-checked; simulated in Experiment 52 (E1, numerator row) | Lean statement of linearity for a finite joint law |
| Same places: the skill ratio is consistent but not unbiased (bias of order 1/n when the base rate comes from the same ratings) | same | numerically checked, Lean pending (Experiment 52, exact-fraction pytest case) | Lean ratio-bias counterexample and the order of the bias |
| Supplement S8: at a fixed total number of ratings the variance of a paired mean difference V(N,k) is nondecreasing in k | `12-supplementary-pool-strata-power.tex` | numerically checked, Lean pending (Experiment 44, H-M2: 50 of 50 pairs); the algebra is one line | Lean inequality for V = (sigma_b^2 + sigma_w^2/k)/N with N = B/k |
| Supplement S8: binned Jensen-Shannon divergence is unchanged when forecasts are reassigned among the items of one bin | same | numerically checked, Lean pending (Experiment 44 Arm J: largest change 2.4e-16 over 200 shuffles in 25 cells); true by construction | Lean statement for finite bins |
| Accuracy against the majority label is proper but not strictly proper; kappa statements | `11-supplementary-related-work.tex`; README open item (4) | paper proof, not yet Lean-checked (the mode lemma `Point1.agrees_iff_mode` is the first step) | Lean statement for the expected-accuracy score |

## Done 11-27-2026

- Point 1 (target versus ruler): agreement with a label depends on a strict-mode forecast only through its mode (any number of categories), plus four closed three-category instances (same mode ties agreement but not Brier excess; the pool changes the Brier excess; sharpening is rewarded on a point-mass item and penalized on a contested one): core Lean, no Mathlib, `docs/notes/point1_support_27Nov2026/Point1Checks.lean` (Lean 4.36.0, exit 0, zero `sorry`, axioms `propext` and `Quot.sound` only; output `Point1Checks_lean_output_27Nov2026.txt`). Numerics first: `point1_claims_numerics_27Nov2026.py` (seed 20261127, 10 of 10 PASS, identical twice, pytest wrapper).
- Point 2 note: every proposition and the corollary are Lean-checked for finite laws in `docs/notes/point2_support_27Nov2026/lean/` (43 theorems, zero `sorry`; claim map in that folder's `README.md`). Numerical checks run first: `docs/notes/point2_support_27Nov2026/point2_claims_numerics_27Nov2026.py` (seed 20261127, 15/15 PASS).

## Reconciled 11-27-2026

- Square-root form of anti-properness (`r_collapsed = sqrt(rho) > rho = r_truthful`, sampled readout, finite joint law): Lean-checked as `PlanAudit.Model.anti_proper` in `experiments/35_brier_vs_corr_math_l2_pearson_lean_start_26Nov2026/extended_checks_11-26-2026/gemini_plan_claims_audit_11-26-2026/PlanClaims.lean`. Rebuilt 11-27-2026 with Lean 4.36.0 and Mathlib `fc4f59e577feb1a4956671bc04ac18aaaf8cd842`: exit 0, 21 theorems, each depending only on `propext`, `Classical.choice`, `Quot.sound` (output: `docs/notes/point2_support_27Nov2026/lean/planclaims_rebuild_27Nov2026.txt`). The earlier AGENTS.md wording "square-root form is Monte Carlo only" was out of date.
