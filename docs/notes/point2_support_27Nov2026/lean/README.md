# Lean 4 proofs of the Point 2 note (27 Nov 2026)

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/docs/notes/point2_support_27Nov2026/lean/README.md>

**TLDR:** Every proposition and corollary of `docs/notes/point2_encodings_vs_first_moments_27Nov2026.tex` is proved here in Lean 4 with Mathlib, for finite (including empirical) laws. Zero `sorry`, zero `admit`, no new axioms; 43 theorems print only `propext`, `Classical.choice`, `Quot.sound`. The seeded numerical checks were run first (`../point2_claims_numerics_27Nov2026.py`, 15/15 PASS).

## Build

- Toolchain: `leanprover/lean4:v4.36.0` (`lean-toolchain`), Lake 5.0.0.
- Mathlib: `v4.36.0` tag, revision `fc4f59e577feb1a4956671bc04ac18aaaf8cd842` (`lake-manifest.json`).
- Commands, from this folder: `lake exe cache get` (prebuilt Mathlib, never rebuild it), then `lake build`, then `lake env lean Axioms.lean`.
- Saved outputs: `build_output_27Nov2026.txt` (exit 0) and `axioms_output_27Nov2026.txt` (43 lines, all standard axioms).

## Claim to theorem map

All names are in namespace `Point2`. `FinLaw ι` is a weight vector on a finite type (an empirical evaluation set is the uniform case).

| Note claim | Lean theorem(s) | File | Scope |
|---|---|---|---|
| Prop. `prop:affine`, Pearson affine invariance with sign | `FinLaw.pearson_affine` (factor `a/\|a\|`), `pearson_affine_pos`, `pearson_affine_neg` | `Point2/Moments.lean` | any real `a ≠ 0`, finite law |
| Prop. `prop:identity`, $r^2 = 1 - \min_{a,b} E(Y-a-bS)^2/\mathrm{Var}(Y)$ | `FinLaw.r_sq_identity` (`IsLeast` form), `r_sq_eq_one_sub_min`, `mse_affine_ge`, `mse_affine_opt` (attained at $a^\ast, b^\ast$) | `Point2/Moments.lean` | finite law, `Var S > 0`, `Var Y > 0` |
| Remark on rank correlations | `FinLaw.spearman_strictMono`, `kendall_strictMono`, `midrank_strictMono`; sign flip `spearman_strictAnti`, `kendall_strictAnti` | `Point2/Ranks.lean` | population mid-rank Spearman, Kendall $\tau_a$ |
| Prop. `prop:murphy`, uncertainty − resolution + reliability | `FinLaw.murphy` (any score that is a function of the conditioning key), `murphy_scalar`, `reliability_zero_iff`, `r_sq_resolution` (second display), `condE_affine` | `Point2/Murphy.lean` | finite law, so finite forecast support |
| Murphy decomposition, multiclass Brier | `FinLaw.brier_murphy`, `brier_reliability_zero_iff` (reliability 0 iff calibrated on the support) | `Point2/Murphy.lean` | conditions on the forecast vector |
| Prop. `prop:expected`, expected-score readout | `FinLaw.expected_readout` (identity and bound), `expected_readout_attained` (if), `expected_readout_equality` (only if: $\mu = a^\ast + b^\ast s$ on the support, $b^\ast > 0$), `cov_sq_le` | `Point2/Murphy.lean` | finite law; also assumes `Var Y > 0` |
| Cor. `cor:tie`, same mean ⇒ tie | `FinLaw.same_mean_tie` (Pearson, Spearman, Kendall, MSE of the readout), `scalar_same_mean_tie` | `Point2/Ranks.lean`, `Point2/Encodings.lean` | joint law over (item, rating) |
| Prop. `prop:sampled`, square-root form $r_D = \sqrt\rho > \rho = r_T$ | `Sampled.Model.anti_proper` (port of `PlanAudit.Model.anti_proper`), instance `Sampled.noteModel_anti_proper` ($1/4$ vs $1/2$) | `Point2/SampledReadout.lean` | finite joint law, sampled readout |
| Prop. `prop:onehot`, Brier strictly proper | `brier_expected`, `brier_strictly_proper` | `Point2/Encodings.lean` | any finite $K$ |
| Prop. `prop:encoding`, encoding rank | `expScore_eq`, `enc_proper`, `strictlyProper_iff` (iff injective on $D$), `expScore_congr`; bullets: `brier_strictly_proper`, `cumEnc_injOnD`, `scalar_not_strictly_proper` ($K \ge 3$), `scalar_two_strictly_proper` ($K = 2$) | `Point2/Encodings.lean` | any finite $K$, any finite row set |
| Prop. `prop:rps`, RPS strictly proper | `rps_eq_encScore`, `rps_expected`, `rps_strictly_proper` | `Point2/Encodings.lean` | $K = n + 1$ ordered ratings, thresholds $0..K-2$ |

The worked-example numbers (Brier and RPS tables, distance sensitivity, the population table, the shift-and-stretch value) are checked by evaluation in `../Point2WorkedExamples.lean` (no Mathlib; `lean Point2WorkedExamples.lean`; output `../lean_check.txt`).

Not formalized here (see `docs/LEAN_TODO.md`): measure-theoretic versions beyond finite laws, the CRPS remark, cross-entropy statements (cited: Gibbs' inequality), and the step from per-item strict propriety to population averages in the closing sentence of Prop. `prop:sampled`.
