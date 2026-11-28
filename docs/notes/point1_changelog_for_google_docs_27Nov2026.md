# Point 1 changelog (Friday 11-27-2026): target versus ruler, in the paper and the experiments

Paste-ready summary of what changed and why. Main note: `docs/notes/point1_target_vs_ruler_27Nov2026.md`.

## The answer in four sentences

1. Pavlick and Kwiatkowski (2019) and Plank (2022) established the target: a human label is a distribution. We adopt that and do not claim it.
2. Our contribution is the ruler: how a judge should be scored once the target is a distribution, and whether current practice scores it that way.
3. It has four parts: a measured practice gap (43 of 49 judge-validation papers report no item-level proper score; 29 of 30 score-based papers validate by correlation), an identification result (machine-checked in Lean for finite laws), a protocol (a strictly proper score matched to the label encoding, with a base-rate skill score and agreement strata), and evidence.
4. Real-judge rank reversal is not yet established; the Experiment 46 results are pending, so the paper does not claim it.

## What we conceded and how

1. **Rater pool and rubric are the construct.** The target is now written p*(r | x, P, rho): a rating of input x from a stated rater pool P under a stated rubric rho. Changing either changes p*, so the choice is fixed before any score is computed. Added to the introduction ("Target versus ruler") and to the problem section ("The target is set by a rater pool and a rubric").
2. **Consensus items.** On high-agreement items p* is peaked and a peaked judge is correct. The protocol therefore reports scores by agreement stratum (agreement measured on one half of each item's ratings, score computed on the other half).
3. **Softened impossibility wording.** "Cannot validate" became "cannot, in general, validate" in the abstract. The problem section says none of the statistics identifies the full conditional distribution in general, that this does not make correlation always misleading, and that with two categories the mean fixes the distribution.

## Corrections to the pasted draft

1. "Neither paper says anything about it" is too absolute: nearby work (Nie et al. 2020; Uma et al. 2021; Elangovan et al. 2025; Lyu, Do and Pergola 2026; Li et al. 2026) already evaluates against distributions. The identification statement in terms of encodings, with a machine check, is ours.
2. Agreement ties forecasts with the same mode, not the same mean. Correlation and squared error tie forecasts with the same mean. (Lean, core: `Point1.agreement_depends_only_on_mode`.)
3. A strictly proper score ranks the true distribution ahead of every other forecast. Two different wrong forecasts can still tie.
4. Lean does not cover accuracy, kappa or cross-entropy propriety (cited, Gibbs); those are labeled analytical or cited.
5. The per-sample risk is unbiased with one rating per item. The skill ratio is consistent, with a bias of order 1/n when the base rate comes from the same ratings (Experiment 52, synthetic).
6. "The field didn't act" is scoped to LLM-judge validation practice in a 49-paper convenience sample.

## New results from this session (Experiment 44, pre-registered, saved forecasts, CPU, $0)

1. **Rater pool (H-P1, H-P2): not supported.** Demographic sub-pools of 11 to 139 raters; pool distributions differ by mean total variation distance 0.08 to 0.18, as do random rater groups. 0 of 6 contrasts and 0 of 30 judge-by-contrast cells are Holm-significant; Kendall tau-b between judge orders under the two pools is 0.8 or 1.0; the best and second-best judge keep their places in all six contrasts. This is a power statement, not evidence that the pool never matters.
2. **Consensus and contested items (H-S1): supported, with a nuance.** Sharpening helps relatively more on consensus items in 25 of 25 cells, but improved the consensus stratum in only 5 of 25; on contested items the softest temperature was best in 23 of 25.
3. **Binned divergence (H-J1, H-J2): supported.** Binned JSD is unchanged (to 2.4e-16) when forecasts are reassigned inside a bin, while the proper-score risk worsens in all 25 cells (+0.0086 to +0.1075).
4. **Power (H-M1, H-M2): supported.** Minimum detectable paired difference at 80% power with 400 items and one rating each: 0.014 to 0.065; 0 to 6 of 10 judge pairs resolvable per set; at a fixed rating budget the variance never decreases with ratings per item.
5. Related, from the parallel experiments on main: Experiment 50 (frontier skill falls from 0.16 to 0.91 in the highest-agreement fifth to minus 0.38 to 0.11 in the lowest; the raw training-objective gap is larger on high-agreement items, so we do not claim correlation-trained judges fail mainly on ambiguous items).

## Files changed

1. Paper (`paper_latex_and_notes/AAAI_2027_EvalAIScientists_JudgesKL/`): `1-abstract.tex`, `2-introduction.tex`, `3-problem.tex`, `4-proposal.tex`, `7-discussion.tex`, `11-supplementary-related-work.tex`, new `12-supplementary-pool-strata-power.tex`, `00_main_aaai_2027.tex`, and both rebuilt PDFs (`00_main_aaai_2027.pdf`, 12 pages; `00_main_arxiv.pdf`, 23 pages).
2. Experiments: `experiments/44_pool_as_construct_consensus_sharpness_binned_jsd_power_start_27Nov2026/` (protocol, amendment, code, tests, results, figures, README, RESULTS, HOW_TO_READ_RESULTS); `experiments/README.md`; `experiments/41_.../BLOG_FACT_CHECK_11-27-2026.md` (a Gemini share link redacted).
3. Docs: root `README.md`, `docs/LEAN_TODO.md`, `docs/notes/point1_target_vs_ruler_27Nov2026.md`, `docs/notes/point1_support_27Nov2026/` (core-Lean file, exact-fraction numerics, outputs, pytest), `DEFENSE_ONE_SLIDE_EVAL_NOTE.md`.

## Checks run

1. Lean 4.36.0, no Mathlib: `Point1Checks.lean` compiles, exit 0, zero `sorry`, axioms `propext` and `Quot.sound` only. Mathlib-based theorems are cited from the saved build of the Point 2 Lean project (I did not rebuild Mathlib here).
2. Exact-fraction numerics, seed 20261127: ten claims, all PASS, identical output twice; pytest passes. Claims that must fail are tested as failing cases.
3. Paper builds: zero LaTeX errors, zero undefined references, zero `??`.
4. Hygiene greps: zero hits for the withdrawn arXiv and OpenReview ids, the Gemini share link, "ICLR 2026" next to Li et al., and the burden-shifting sentence in active and paper files.

## Open for Brando

1. I changed one phrase of the approved abstract ("cannot, in general, validate a judge"). Revert if you prefer the original.
2. The main body still ends on page 5, one page past the four-page target in Rule 89; compressing the Section 6 paragraph on Experiments 26, 28 and 29 (details are already in the supplement) would fix it. I did not do it without your go-ahead.
3. Real-judge rank reversal (Experiment 46) is pending; the paper says so.
4. Several internal notes in this repository name the advisor (for example `STATUS_11-20-2026.md`, `paper_latex_and_notes/notes/`). I did not scrub them; tell me if you want that.

**TL;DR:** They established the target; we supply the ruler, the identification result, the protocol and the audit, with rank reversal still open. The rater-pool definition and the consensus-item point are conceded and now in the paper and the experiments.
