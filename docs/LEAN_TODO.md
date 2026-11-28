# Lean TODO: claims not yet Lean-checked

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/docs/LEAN_TODO.md>

**TLDR:** Every theorem, proposition, lemma or proof in the paper, notes or docs that is not yet machine-checked in Lean 4 is listed here with its status label (rule: `AGENTS.md`, section "Every proof in Lean 4, with a quick numerical check first"). Remove a row once the Lean file is cited next to the claim.

Status labels: "numerically checked, Lean pending", "paper proof, not yet Lean-checked", "Lean-checked special case" (case stated).

| Claim | Location | Status | What is missing |
|---|---|---|---|
| Point 2 note, Props. identity, Murphy, expected readout, encoding rank, RPS propriety; Cor. same-mean tie | `docs/notes/point2_encodings_vs_first_moments_27Nov2026.tex` | numerically checked, Lean pending (in progress 11-27-2026) | Lean theorems in `docs/notes/point2_support_27Nov2026/lean/` |

## Reconciled 11-27-2026

- Square-root form of anti-properness (`r_collapsed = sqrt(rho) > rho = r_truthful`, sampled readout, finite joint law): Lean-checked as `PlanAudit.Model.anti_proper` in `experiments/35_brier_vs_corr_math_l2_pearson_lean_start_26Nov2026/extended_checks_11-26-2026/gemini_plan_claims_audit_11-26-2026/PlanClaims.lean`. Rebuilt 11-27-2026 with Lean 4.36.0 and Mathlib `fc4f59e577feb1a4956671bc04ac18aaaf8cd842`: exit 0, 21 theorems, each depending only on `propext`, `Classical.choice`, `Quot.sound` (output: `docs/notes/point2_support_27Nov2026/lean/planclaims_rebuild_27Nov2026.txt`). The earlier AGENTS.md wording "square-root form is Monte Carlo only" was out of date.
