# Point 1 changelog (Friday 11-27-2026): target versus ruler, in the paper and the experiments

Paste-ready summary of what changed and why. Main note: `docs/notes/point1_target_vs_ruler_27Nov2026.md`.

## The answer in four sentences

1. Pavlick and Kwiatkowski (2019) and Plank (2022) established the target: a human label is a distribution. We adopt that and do not claim it.
2. Our contribution is the ruler: how a judge should be scored once the target is a distribution, and whether current practice scores it that way.
3. It has four parts: a measured practice gap (43 of 49 judge-validation papers report no item-level proper score, 44 of 49 no item-level strictly proper one under the stricter code of Experiment 54; 29 of 30 score-based papers report correlation), an identification result (machine-checked in Lean for finite laws), a protocol (a strictly proper score matched to the label encoding, with a base-rate skill score and agreement strata), and evidence.
4. Real-judge rank reversal is established only narrowly (Experiment 46, interim): none under the pre-registered Pearson test, top-1 agreement or kappa; two Spearman reversals for one judge pair on two sets, removed by calibration. A B = 20,000 sensitivity run (a labelled deviation) adds reversals only among small local judges, one also on Pearson. The paper says Spearman.

## What we conceded and how

1. **Rater pool and rubric are the construct.** The target is now written p*(r | x, P, rho): a rating of input x from a stated rater pool P under a stated rubric rho. Changing either can change p*, so the choice is fixed before any score is computed. Added to the introduction ("Target versus ruler") and to the problem section ("The target is set by a rater pool and a rubric").
2. **Consensus items.** On high-agreement items p* is peaked and a peaked judge is correct. The protocol therefore reports scores by agreement stratum (agreement measured on one half of each item's ratings, score computed on the other half).
3. **Softened impossibility wording.** "Cannot validate" became "cannot, in general, validate" in the abstract; the abstract was then rewritten to the advisor's notes S1 to S9 and no longer has that sentence, and its remaining claims are scoped. The problem section says none of the statistics identifies the full conditional distribution in general, that this does not make correlation always misleading, and that with two categories the mean fixes the distribution.

## Corrections to the pasted draft

1. "Neither paper says anything about it" is too absolute: nearby work (Nie et al. 2020; Uma et al. 2021; Elangovan et al. 2025; Lyu, Do and Pergola 2026; Li et al. 2026) already evaluates against distributions. The identification statement in terms of encodings, with a machine check, is ours.
2. Agreement ties forecasts with the same mode, not the same mean. Correlation and squared error tie forecasts with the same mean. (Lean, core: `Point1.agreement_depends_only_on_mode`.)
3. A strictly proper score ranks the true distribution ahead of every other forecast. Two different wrong forecasts can still tie.
4. Lean does not cover accuracy, kappa or cross-entropy propriety (cited, Gibbs); those are labeled analytical or cited.
5. The per-sample risk is unbiased with one rating per item. The skill ratio is consistent, with a bias of order 1/n when the base rate comes from the same ratings (Experiment 52, synthetic).
6. "The field didn't act" is scoped to LLM-judge validation practice in a 49-paper convenience sample. An independent pre-declared census of 40 sources (Experiment 45) agrees on the split: correlation is the headline for graded-score judges and leaderboard validation (16 of 40 overall), agreement, accuracy, F1 or kappa for the rest, and 1 of 40 scores the rating distribution with a proper rule.

## New results from this session (Experiment 44, pre-registered, saved forecasts, CPU, $0)

1. **Rater pool (H-P1, H-P2): not supported.** Demographic sub-pools of 11 to 139 raters; pool distributions differ by mean total variation distance 0.08 to 0.18, as do random rater groups. 0 of 6 contrasts and 0 of 30 judge-by-contrast cells are Holm-significant; Kendall tau-b between judge orders under the two pools is 0.8 or 1.0; the best and second-best judge keep their places in all six contrasts. This is a power statement, not evidence that the pool never matters.
2. **Consensus and contested items (H-S1): supported, with a nuance.** Sharpening helps relatively more on consensus items in 25 of 25 cells, but improved the consensus stratum in only 5 of 25; on contested items the softest temperature was best in 23 of 25.
3. **Binned divergence (H-J1, H-J2): supported.** Binned JSD is unchanged (to 2.4e-16) when forecasts are reassigned inside a bin, while the proper-score risk worsens in all 25 cells (+0.0086 to +0.1075).
4. **Power (H-M1, H-M2): supported.** Minimum detectable paired difference at 80% power with 400 items and one rating each: 0.014 to 0.065; 0 to 6 of 10 judge pairs resolvable per set; at a fixed rating budget the variance never decreases with ratings per item.
5. Related, from the parallel experiments on main: Experiment 50 (frontier skill falls from 0.16 to 0.91 in the highest-agreement fifth to minus 0.38 to 0.11 in the lowest; the raw training-objective gap is larger on high-agreement items, so we do not claim correlation-trained judges fail mainly on ambiguous items).

## Files changed

1. Paper (`paper_latex_and_notes/AAAI_2027_EvalAIScientists_JudgesKL/`): `1-abstract.tex`, `2-introduction.tex`, `3-problem.tex`, `4-proposal.tex`, `7-discussion.tex` (including one calibration-and-drift sentence), `8-appendix-rps.tex` (the Lean sentence now directly follows the identities it covers), `11-supplementary-related-work.tex`, new `12-supplementary-pool-strata-power.tex`, `00_main_aaai_2027.tex`, and both rebuilt PDFs (`00_main_aaai_2027.pdf`, 13 pages; `00_main_arxiv.pdf`, 25 pages).
2. Experiments: `experiments/44_pool_as_construct_consensus_sharpness_binned_jsd_power_start_27Nov2026/` (protocol, amendment, code, tests, results, figures, README, RESULTS, HOW_TO_READ_RESULTS); `experiments/README.md`; `experiments/41_.../BLOG_FACT_CHECK_11-27-2026.md` (a Gemini share link redacted).
3. Docs: root `README.md`, `docs/LEAN_TODO.md`, `docs/notes/point1_target_vs_ruler_27Nov2026.md`, `docs/notes/point1_support_27Nov2026/` (core-Lean file, exact-fraction numerics, outputs, pytest), `DEFENSE_ONE_SLIDE_EVAL_NOTE.md`.

## Checks run

1. Lean 4.36.0, no Mathlib: `Point1Checks.lean` compiles, exit 0, zero `sorry`, axioms `propext` and `Quot.sound` only. Mathlib-based theorems are cited from the saved build of the Point 2 Lean project (I did not rebuild Mathlib here).
2. Exact-fraction numerics, seed 20261127: ten claims, all PASS, identical output twice; pytest passes. Claims that must fail are tested as failing cases.
3. Paper builds: zero LaTeX errors, zero undefined references, zero `??`.
4. Hygiene greps: zero hits for the withdrawn arXiv and OpenReview ids, the Gemini share link, "ICLR 2026" next to Li et al., and the burden-shifting sentence in active and paper files.

## Open for Brando

1. The abstract was rewritten to your advisor's notes S1 to S9 after my first pass, so the three abstract edits I had flagged are gone (the Lean-scope sentence is no longer in the abstract; the supplement states it, for finite laws, in both the squared and the square-root form). On the rewritten abstract I made two precision edits: "Correlation is a function of squared error after the best affine refit" (squared correlation is one minus that error over the variance, not the error itself), and the rating written as r like the body. Revert if you prefer the original.
2. The venue edition's main body is past the four-page target in Rule 89 (and the workshop's 2 to 4 main-body pages). Measured on the current build: the Conclusion starts at the bottom of page 5 and runs onto the top of page 6, where the references begin, about 2.1 columns past page 4. Before the Point 1 work it ran about 0.6 of a column onto page 5, and after it about 1.6 columns; the rest of the growth is the abstract and introduction rewrite. The arXiv edition has no limit. Compressing the Section 6 paragraph on Experiments 26, 28 and 29 (details are already in the supplement) and the dated "earlier text; corrected" asides in Section 6 would recover most of it. I did not do it without your go-ahead.
3. Experiment 46's README says "H1 is supported", but the frozen H1 is about Pearson, which shows no significant reversal; the two significant reversals are Spearman. The paper, README and notes say Spearman and state the Pearson null. The Experiment 46 README is another session's active file, so I did not edit it.
4. Other sessions' files that a second independent review flagged and I did not edit: (a) the Experiment 46 README also credits Experiment 47's "100 to 200 labels" to the frontier judges, although Experiment 47 tested weak linear heads and the frontier judges were calibrated with all labels in five folds; (b) `experiments/README.md` has index rows for Experiments 40 to 45, 53 and 54 but none for 46 to 52; (c) `paper_latex_and_notes/AAAI_2027_EvalAIScientists_JudgesKL/checks/HalfBrierRps.lean` contains a theorem named `one_rating_unbiased` whose statement is `x = x` (proved by `rfl`), so it checks nothing; the paper correctly says the unbiasedness derivation is not machine-checked, so do not cite that theorem as a check.
5. Several internal notes in this repository name the advisor (for example `STATUS_11-20-2026.md`, `paper_latex_and_notes/notes/`). I did not scrub them; tell me if you want that.

**TL;DR:** They established the target; we supply the ruler, the identification result, the protocol and the audit, with real-judge rank reversal narrow (Spearman only, one judge pair, interim). The rater-pool definition and the consensus-item point are conceded and now in the paper and the experiments.
