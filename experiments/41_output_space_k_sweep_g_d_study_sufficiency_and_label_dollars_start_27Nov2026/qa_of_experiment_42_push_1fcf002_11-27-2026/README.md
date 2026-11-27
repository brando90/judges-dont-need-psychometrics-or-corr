# QA of Cursor's Experiment 42 push (commit 1fcf002)

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/41_output_space_k_sweep_g_d_study_sufficiency_and_label_dollars_start_27Nov2026/qa_of_experiment_42_push_1fcf002_11-27-2026/README.md>

**TLDR:** An independent reviewer (Claude Opus at maximum effort, about twelve minutes, CPU only) re-ran Cursor's Experiment 42 (`experiments/42_multimodal_simulated_population_ce_rps_vs_corr_gtheory_start_27Nov2026/`, a simulated population on known laws) from a fresh clone and recomputed its exact numbers with its own code. The arithmetic is right and everything reproduces, but the headline attribution is wrong. The 0.0595 ranked probability score (RPS) gap between the "Pearson binomial" and persona cross-entropy compares a one-parameter binomial family with a six-class histogram; no arm in the experiment is trained or selected by correlation, and under Experiment 41's matched calibration design the gap would be 0 or about 0.005, not 0.06. The "400 of 400" generalizability-coefficient selection is true by construction, because the pool holds one point mass and every point mass has coefficient one. Twelve findings are below; the exact risks on the constructed laws, the equal-moment identities and the pooled-versus-persona crossover survive.

**As of:** 11-27-2026 (Friday, Pacific time). Local CPU, $0, no model calls. I did not edit Experiment 42.

## What is in this folder

```text
REVIEW_by_independent_opus_subagent_11-27-2026.md   the full review: verdict, twelve findings (severity, location, evidence, smallest fix), what survives, what was not checked
scripts/                                            the reviewer's three scripts (own recomputation, own Monte Carlo, a number-to-summary.json trace)
outputs/reviewer_*.txt                              the reviewer's saved outputs
outputs/my_*.txt                                    my own first-pass runs on a scratch copy of the pushed folder
```

The reviewer's scripts contain hard-coded paths of the review machine (`/tmp/qa42/...`). They are evidence of what was run, not a turnkey reproduction.

## The findings in one line each (numbering follows the review)

1. **Blocking for any "correlation versus proper loss" reading: the headline gap measures model family and conditioning, not the training objective.** The "Pearson binomial" is the mean-matched binomial, which is also what cross-entropy training of a binomial gives. Nothing in the code optimizes or selects by Pearson, and the set of forecasts with Pearson correlation 1 contains the oracle itself. A Pearson-scored scalar followed by a cross-entropy calibration map reproduces pooled or persona cross-entropy, so at 256 ratings per item the matched gap is about 0.0048 (pooled) or 0 (persona), against the reported 0.059517. Experiment 42 does not cite Experiment 41.
2. **Important: "selected by that coefficient in 400 of 400 replications" is true by construction.** The selection pool holds exactly one point mass, and any point mass has Eρ² = 1 while the other forecasts are always below 1. It is 400 of 400 at all four budgets, not only the two reported. The Spearman point mass, which also has Eρ² = 1 and a better RPS, was removed from the pool. The README, the results index and the technical report present the count as a replicated result.
3. **Important: sentences that are wrong.** "Every component has the bin mean" (strict is minus 0.8, lenient plus 0.8; the mixture mean matches only because they cancel). The README lists DICES-990 as a base measure of the four subpopulations; it is not used in any of them. `results.md:112` prints two total variation distances (0.147154 and 0.569058) that no script produces; `summary.json` has 0.147187 and 0.569134. The technical report calls 2.742210 "smoothed" cross-entropy; it is the unsmoothed value. Add-one is described as the minimizer of cross-entropy plus a Dirichlet(1) penalty; it is the posterior mean under Dirichlet(1). "23 of 23 passed, including leakage" rests on a vacuous leakage test.
4. **Important: the Lean statements claim more than the theorems prove.** No theorem covers the three-item gap 167/23040 or the panel total variation distance although `results.md:20` says Lean checks them. `erho_depends_only_on_moments` is plain function congruence. `panel_erho` uses hard-coded vectors not linked to the laws. The step from the RPS decomposition to the gap is a definition, not a proof, and `hypotheses_can_hold` is a conjunction of computed facts.
5. **Important: 11 of the 23 self-checks hold by construction, and the leakage check is vacuous.** It zeroes a copy of the test array that is never passed to anything, and the fit function has no test input.
6. **Important: a simulation is reported with the machinery of empirical inference.** Student-t intervals and p-values below 1e-300 are given for quantities that can be computed exactly, so they measure Monte Carlo error only; the eight items of a bin share one law, so the item split and the item-clustered bootstrap test nothing. It should be labeled a verification on known laws.
7. **Minor to important: preregistration is unverifiable, and one threshold was set after the fact.** Protocol and results are in one commit. The generalizability-coefficient Φ cut exists only at 256 ratings; at 64 the peaked share is 86 percent, below the 95 percent bar. The moment threshold of 0.007 sits just below the computed 0.007248.
8. **Minor: figures.** The y-axis titles of figures 1 and 2 are cut off, so "(lower is better)" is not visible; RPS and TVD are not spelled out on an axis; figure 2 uses `ax.grid(True, axis="y")`.
9. **Minor: hygiene.** `results.md` line 9 commits an email address (the owner's own Gmail address, on a Google Doc's sharing line); the index row says "Active" while the README says "complete".
10. **Minor: the quick numerics are not independent or randomized** (they import the experiment's own functions and test fixed instances).
11. **Minor: the validation rule sees twice as many ratings per bin as the persona learner** ("quota-sized" is ambiguous). No conclusion changes.
12. **Minor: disclosure.** The low pole of the polarized law is a reflected histogram (the README says so only for the strict persona); the pole locations and the 0.8 shift are free choices with no sensitivity analysis.

## What survives (independently confirmed by the reviewer)

1. The exact risks on the constructed laws: oracle RPS 0.221429; Pearson binomial RPS 0.281693 and total variation distance 0.614594; train mixture 0.226916; the strict-target and shifted-mixture values.
2. The finite-sample expectations, within Monte Carlo error: persona cross-entropy 0.244564, 0.231699, 0.224612 and 0.222275 at 4, 16, 64 and 256 ratings; pooled minus persona is negative at 4 and 16 ratings and positive from 64.
3. The equal-moment panels, exactly in fractions: center gap 9/512, center total variation distance 9/16, panel gap 167/23040, panel total variation distance 43/144, and equal coefficients for the two panels (1/5 with divisor N, 3/11 with divisor n minus 1; the equality is the point).
4. The honest main point: forecasts whose Pearson correlation with the bin means is exactly 1 range in expected RPS from 0.221429 (the oracle) to 0.281693 (the mean-matched binomial) on this law, so correlation cannot tell them apart. That is the same non-identifiability the Lean witnesses in Experiment 41 state, now on a constructed bimodal law.

## What I re-checked myself

1. **Reproduction (outputs `my_01` to `my_04`).** Quick numerics 14 of 14 PASS, pytest 10 passed, `lake build` succeeds on Lean 4.36.0 with only `propext`, `Classical.choice` and `Quot.sound`, and a rerun of `run_experiment.py` ("checks 23/23 passed") reproduces all 4,086 numbers of `results/summary.json`; only the numpy version string and the wall-clock seconds differ.
2. **Four of the reviewer's claims, read straight from the code:** the leakage check zeroes `test_copy` and never passes it to `fit_from_train(train)` (`benchmark.py` lines 565 to 568); no script writes `results.md`; `results.md:112` prints 0.147154 and 0.569058 where `summary.json` has 0.147187 and 0.569134; the maximum-Eρ² pool is `ce_polarized`, `rps_polarized`, `ce_pooled`, `pearson_binomial` and `lenient_point`, so there is one point mass, and its tie-break runs toward the earlier proper-score name, which works against the headline rather than for it.
3. **One qualification of finding 1.** "Reproduces pooled or persona cross-entropy exactly" holds for a lookup-table calibration map over the six distinct scores. Experiment 41's map is a parametric multinomial logit in the standardized score, which can only approximate six arbitrary histograms, so under that map the matched gap would be small but not exactly 0 or 0.0048. The reviewer's lookup-table values (+0.004823 [0.004774, 0.004873] pooled, 0 persona at 256 ratings) are the right order of magnitude; the exact value for Experiment 41's map was not computed. The conclusion that family and conditioning, not the objective, drive the 0.0595 stands.
4. **Privacy-scanner hits.** The counts-only scanner's one email hit is the owner's own Gmail address on `results.md` line 9 (checked by domain and prefix only); the committed copy of the reviewer's output masks it. The other `@` in that file is the VeriBench git pin. The ten `part0` hits in this review folder are all the product name in "Google Doc" and in the docs.google.com link (viewed as masked context lines, no candidate printed), so they are benign here. I also viewed the hits in Experiment 42's own `data/empirical_shapes.json` and in the technical-report pages as masked context lines: they are the same product name (a `google-research-datasets` source string and Google Doc links), so benign. The one real item is the owner's address in `results.md`, which Cursor should remove.

## What I did not do

1. I did not edit Experiment 42, the technical report or the index rows.
2. Neither the reviewer nor I re-implemented the Monte Carlo selection loop, the shift and panel Monte Carlo rows, the bootstrap interval or the external data hashes (the VeriBench `R_v1` pin still returns HTTP 404 here).
3. I did not rebuild the technical report's PDF; the reviewer confirmed that it contains the Experiment 42 text.

## Smallest fixes for Cursor (full wording in the review)

1. Rename the arm "mean-matched binomial"; add "Pearson scalar plus cross-entropy calibration map" (pooled and persona-conditional) and "cross-entropy-trained binomial"; rewrite the headline as a non-identifiability statement and cite Experiment 41.
2. Say that every point-mass forecast has Eρ² = 1 by definition; drop the counts, intervals and p-values for that rule; define the within-item variance term.
3. Correct the six wrong sentences and the two printed distances; have a script write `results.md`.
4. Restate what Lean checks; add theorems for the panel gap, the panel total variation distance and the reversed laws, and link the hard-coded vectors to the laws.
5. Report the number of substantive checks; make the leakage check pass a test array to the fit.
6. Label the study a verification on known laws; drop the p-values.
7. State that protocol and results share a commit; state the 0.007 threshold and its origin.
8. Fix the figure axes and add `ax.grid(True)` to every axis.
