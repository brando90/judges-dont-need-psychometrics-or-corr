# Point 1: the target is a distribution; what we add is the ruler

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/docs/notes/point1_target_vs_ruler_27Nov2026.md>

**TLDR:** Pavlick and Kwiatkowski (2019) and Plank (2022) established the target: a human label is a distribution. We adopt that and do not claim it. Our contribution is the ruler: how a judge should be scored once the target is a distribution, and whether current practice scores it that way. It has four parts: a measured practice gap, an identification result, a protocol, and evidence. Real-judge rank reversal is not yet established, so we do not claim it. We concede the two points of the advisor's Point 1 (define the rater pool and rubric up front; on consensus items a peaked judge is correct) and softened the appendix wording to "cannot identify the distribution in general".

**As of:** 11-27-2026 (Friday, Pacific time). CPU only, $0. Companion files: [`point1_support_27Nov2026/`](point1_support_27Nov2026/) (core-Lean file, exact-fraction numerics, compiler output) and [`point1_changelog_for_google_docs_27Nov2026.md`](point1_changelog_for_google_docs_27Nov2026.md) (paste-ready).

## The short answer

Pavlick and Kwiatkowski, and Plank, argue that a human label is a distribution and that models should be judged against that distribution. That is a claim about the target, and it is not ours. Our claim is about the ruler. Once you accept the target, how should a judge be scored, and does the field score it that way?

Four things are new on top of those papers, and one of them is still open.

First, a measured practice gap. In the forty-nine-paper audit (Experiment 43, a purposive sample coded by one person, not a systematic review), forty-three papers report no item-level proper score, twenty-nine of the thirty papers that score outputs validate by correlation, and all thirteen pairwise-judge papers report agreement, accuracy or kappa. The careful sentence is that, in this sample, LLM-judge validation practice did not act on the point that the target is a distribution.

Second, an identification result. Correlation is the squared error of the best affine refit, so forecasts that share a conditional mean tie on Pearson, Spearman, Kendall and scalar squared error whatever the human distribution is, while every strictly proper score ranks the true distribution strictly ahead of any forecast that shares its mean. With two categories the mean fixes the distribution, so the tie needs three or more. This is machine-checked in Lean for finite laws, together with the sampled-score result that a collapsed judge can out-correlate a truthful judge that samples from the true distribution. None of it says correlation is always wrong; it says correlation cannot identify the distribution in general.

Third, a protocol. Score with a strictly proper score matched to the label encoding: one-hot Brier or cross-entropy for categorical labels, cumulative RPS for ordinal ones, CRPS or log-density for continuous ones. Compute it per sample on held-out individual ratings, which is unbiased with one rating per item. Report a base-rate skill score and stratify by human agreement. The skill ratio is consistent, not exactly unbiased.

Fourth, evidence that the choice of ruler changes what you see. On seven real rating sets a cross-entropy-trained linear head beats a Pearson-trained one by up to 0.1282 normalized RPS, and at most 0.0024 remains once each head has its own validation-fitted calibration map (Experiment 41). That closure is in-domain: with 25 single held-out ratings the calibrated Pearson-trained head is still worse than the raw cross-entropy-trained head by 0.0097 to 0.0225 normalized RPS on all 16 views (Experiment 47), and a map fitted on another dataset or rubric reopens the gap in 53 of 82 frontier-judge cases (Experiment 48), so each new target needs its own held-out ratings scored with a proper rule. Stratified by human agreement (Experiment 50), the skill of frontier judges over the base rate falls from 0.16 to 0.91 in the highest-agreement fifth of items to between minus 0.38 and 0.11 in the lowest, while their Pearson correlation on the contested NLI items stays between 0.38 and 0.86. What is open: whether correlation and proper-score rankings of independently built real LLM judges actually reverse. The Experiment 46 scripts exist and the results are pending, so the paper does not claim a reversal.

The pasted draft ("neither paper says anything about it") is too absolute, and four of its sentences need correcting; see the next section.

## Corrections to the pasted draft

These are the places where the draft says more than we can support. Each is fixed in the paper text, or the paper never made the claim.

1. **"Neither paper says anything about it."** Pavlick and Kwiatkowski and Plank do not state an identification result about correlation or squared error. But nearby work evaluates against distributions: Nie et al. (EMNLP 2020, ChaosNLI), Uma et al. (JAIR 2021), Elangovan et al. (ICLR 2025, "Beyond Correlation"), Lyu, Do and Pergola (arXiv 2610.37868) and Li et al. (Findings of EMNLP 2026). The claim we can defend is that the identification statement, in terms of encodings and with a machine check, is ours; the observation that distributions matter is not.
2. **"Correlation, MSE and agreement provably can't tell apart judges with the same conditional mean."** Agreement is wrong in that list. Agreement with the majority label depends on a forecast only through its mode, so it ties forecasts with the same mode, not the same mean. Same-mean forecasts can have different modes. Both statements are checked: the mode statement in core Lean for any number of categories (`Point1.agrees_iff_mode`, `Point1.agreement_depends_only_on_mode`) and by exact fractions (claims C2 and C4 in the numerics output).
3. **"Every strictly proper score can separate them."** Strict propriety means the true distribution is the unique minimizer of the expected score. It does not mean two different wrong forecasts always differ: two forecasts at equal distance from the truth can tie on Brier. The paper now says the true distribution scores strictly better than any forecast that shares its mean (`2-introduction.tex`, `3-problem.tex`, `11-supplementary-related-work.tex`).
4. **"Machine-checked in Lean" for the whole set.** Lean covers the correlation and scalar-MSE ties, strict propriety of Brier and RPS, the scalar two-category versus three-category contrast, the identity for squared correlation, and both forms of the sampled-score result for finite joint laws. Cross-entropy strict propriety is cited (Gibbs), not formalized. Accuracy against the majority label being proper but not strictly proper, and anything about kappa, is argued by hand.
5. **"A skill score that's unbiased with one rating per item."** The per-sample risk estimate is unbiased. The skill score is a ratio of two risk estimates, so it is consistent, and when the base rate is estimated from the same ratings it has a bias of order one over n (Experiment 52, synthetic, normalized RPS, judge J1: about minus 0.034 at 25 items, minus 0.0017 at 500 and minus 0.0006 at 1,000, with one rating per item; essentially none with a known base rate).
6. **"The field didn't act on their point."** Scope it: LLM-judge validation practice in a convenience sample of forty-nine papers. We did not survey the human label variation literature itself, which does evaluate against distributions.
7. **"Real-judge rank reversal is running now."** Not established. Experiment 46 has scripts and frozen subset ids; no results are committed. Until they are, the paper says it is not yet established and points to Guerdan et al. (NeurIPS 2025) as the nearest evidence on selection effects.

## What we conceded, and where it now lives

**The rater pool and the rubric are part of the construct.** The target is written p*(r | x, P, rho), the distribution of a rating of input x from a stated rater pool P under a stated rubric rho. Changing the pool or the rubric changes p*, so that choice is the construct, fixed before any score is computed. It is stated in the paper's introduction (paragraph "Target versus ruler") and in the problem section (paragraph "The target is set by a rater pool and a rubric"), and in the experiments in the Experiment 44 README and the root README.

What the pool experiment found (Experiment 44, Arm P, DICES, demographic sub-pools of 11 to 139 raters): the item-level rating distributions of two pools differ by a mean total variation distance of 0.08 to 0.18, but so do two random rater groups. No contrast is distinguishable from a rater-label permutation null (smallest Holm-adjusted p 0.59 over six contrasts), no judge-by-contrast score difference is Holm-significant (0 of 30), and the best and second-best judge keep their places in all six contrasts. The hypotheses that the pool changes the verdict (H-P1, H-P2) were **not supported**. That is a statement about power and about asking judges to forecast a whole crowd, not evidence that the pool never matters, so the construct statement stays a requirement of the method, not a finding.

**On consensus items p* is peaked, and a good judge should be too.** That is why the protocol stratifies by human agreement, with agreement measured on one half of each item's ratings and the score computed on the other half. What the data say is narrower than "judges should be peaked on consensus items":

- Experiment 44, Arm S: sharpening a forecast (temperature 0.5) helps relatively more, or hurts less, on consensus items than on contested ones in 25 of 25 judge-by-set cells (interval excludes 0 in all 25), but it improved the consensus stratum in only 5 of 25 cells; on contested items the softest temperature (2) was best in 23 of 25. These judges are overconfident where raters disagree.
- Experiment 50: the raw training-objective gap (Pearson-trained versus cross-entropy-trained head) is larger on high-agreement items, the opposite of our frozen prediction R1 (held on 2 of 12 views). We therefore do not claim correlation-trained judges fail mainly on ambiguous items.

**Softened wording.** "Correlation cannot validate a judge" became "cannot, in general, validate a judge" in the abstract. The problem section now says that none of the statistics identifies the full conditional distribution in general, that this is not a claim that correlation is always misleading, and that with two categories the mean fixes the distribution. No sentence in the paper says correlation is always wrong or that an impossibility holds on every item.

## Where each change landed

Paper (`paper_latex_and_notes/AAAI_2027_EvalAIScientists_JudgesKL/`):

1. `1-abstract.tex`: "cannot, in general, validate a judge", and the same-mean sentence made exact (a different forecast that shares the true mean ties with it on Pearson, Spearman, Kendall and scalar mean squared error, while every strictly proper score ranks the true distribution strictly ahead of it); two phrase edits to the approved abstract, flagged for Brando.
2. `2-introduction.tex`: paragraph "Target versus ruler"; contributions rewritten as practice gap, identification result, protocol, evidence.
3. `3-problem.tex`: rater pool and rubric paragraph; "In general, none of these statistics identifies..." wording.
4. `4-proposal.tex`: unbiased risk with one rating per item; skill score and agreement strata paragraph.
5. `7-discussion.tex`: limitations (power of the pool analysis, strata on contested items, the gap is not concentrated on contested items) and one sentence on calibration (needs about 100 to 200 held-out ratings, often does not transfer) with rating-budget split under drift marked as future work.
6. `11-supplementary-related-work.tex`: identification sentence made exact.
7. `12-supplementary-pool-strata-power.tex` (new, arXiv edition): pool, consensus and contested items, binned divergence, power, calibration needs in-domain held-out ratings (Experiments 47 and 48), machine-checked claims.
8. `00_main_aaai_2027.pdf` and `00_main_arxiv.pdf` rebuilt.

Experiments: Experiment 44 (protocol, amendment, code, nine result tables, figures, `RESULTS.md`, `HOW_TO_READ_RESULTS.md`); root `README.md`; `experiments/README.md`; this note.

## Machine-check status

Labels follow `AGENTS.md`: "Lean-checked", "Lean-checked special case", "cited", "analytical".

1. Same conditional mean implies equal Pearson, Spearman, Kendall and scalar MSE (finite laws): **Lean-checked**, `Point2.FinLaw.same_mean_tie`, `Point2.scalar_same_mean_tie` (Lean 4.36.0, Mathlib `fc4f59e`, saved outputs in `docs/notes/point2_support_27Nov2026/lean/`). I did not rebuild Mathlib in this session; the status rests on that saved build.
2. Brier and RPS strictly proper (finite laws): **Lean-checked**, `Point2.brier_strictly_proper`, `Point2.rps_strictly_proper`.
3. Scalar encoding strictly proper for two categories, not for three or more: **Lean-checked**, `Point2.scalar_two_strictly_proper`, `Point2.scalar_not_strictly_proper`.
4. Squared correlation equals one minus the mean squared error of the best affine fit over the variance of the rating: **Lean-checked** for finite laws, `Point2.FinLaw.r_sq_identity`.
5. Sampled-score anti-properness, squared form and square-root form (finite joint law): **Lean-checked**, `Point2.Sampled.Model.anti_proper` and `PlanAudit.Model.anti_proper` in experiment 35.
6. Agreement with a label depends on a strict-mode forecast only through its mode, any number of categories: **Lean-checked (core Lean, no Mathlib)**, `Point1.agrees_iff_mode`, `Point1.agreement_depends_only_on_mode` in `point1_support_27Nov2026/Point1Checks.lean`. Axioms: `propext`, `Quot.sound` only; zero `sorry`; Lean 4.36.0; compiler output saved.
7. Same mode but different Brier excess; the pool changes the Brier excess of a fixed forecast; sharpening rewarded on a point-mass item and penalized on a contested one: **Lean-checked special case** (three categories, probabilities scaled by ten), the four `*_instance` theorems in the same file.
8. Cross-entropy strict propriety: **cited** (Gibbs' inequality). Accuracy proper but not strictly proper, and kappa statements: **analytical**.
9. Skill ratio consistent but not unbiased; unbiasedness of the per-sample risk estimate; variance nondecreasing in k at a fixed rating budget; binned JSD unchanged by within-bin reassignment: **numerically checked, Lean pending** (Experiments 52 and 44; listed in `docs/LEAN_TODO.md`).

The numerics were run before the Lean file: `point1_claims_numerics_27Nov2026.py` (seed 20261127, exact fractions, ten claims, all PASS, identical output on two runs, pytest wrapper passes). It tests the claims that must fail as failing cases: same mode does not imply same Brier excess (C3), same mean does not imply same agreement (C4), sharpening toward the mode does not always lower the excess (C9).

## Not run, and open

1. Real-judge rank reversal of correlation versus held-out proper score with paired pre-registered tests: Experiment 46 results pending; not claimed.
2. The budget split (more items versus more ratings per item) as an empirical study, and drift under model and task change: Future Work in the paper. Only a synthetic estimator check (Experiment 52) and the variance formula are in.
3. No API or GPU job was run for this work; all analyses use saved forecasts.

## Reproduce

From the repository root (Python 3; Lean 4.36.0 via elan):

```bash
python3 docs/notes/point1_support_27Nov2026/point1_claims_numerics_27Nov2026.py
python3 -m pytest -q docs/notes/point1_support_27Nov2026/test_point1_claims.py
cd docs/notes/point1_support_27Nov2026 && lean Point1Checks.lean
cd experiments/44_pool_as_construct_consensus_sharpness_binned_jsd_power_start_27Nov2026 && python3 -m pytest -q tests
```

**TLDR-end:** [judge objectives: Point 1, target versus ruler] The target being a distribution is Pavlick and Kwiatkowski's and Plank's; our contribution is the ruler: a measured practice gap (43 of 49), an identification result, a proper-score protocol with skill and agreement strata, and evidence, with real-judge rank reversal still open. The rater pool and rubric are now part of the stated construct, and the "impossibility" wording is softened to "cannot identify in general".
