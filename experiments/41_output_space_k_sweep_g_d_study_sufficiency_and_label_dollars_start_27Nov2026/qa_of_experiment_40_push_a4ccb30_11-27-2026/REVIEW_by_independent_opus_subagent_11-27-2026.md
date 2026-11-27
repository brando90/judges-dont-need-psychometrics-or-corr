# Independent QA review of Experiment 40 (Cursor commit a4ccb30)

Folder reviewed: `experiments/40_g_d_theory_dollar_pareto_crps_start_27Nov2026/` at commit `a4ccb30585fa4f69f363cd1f6f19b06caf028f44` (read-only checkout `/tmp/e40_cursor`), plus its row in `experiments/README.md`. All runs were done in copies under `/tmp/opus_review_exp40c/`. Scripts are in `scripts/`, outputs in `outputs/`. No file under the repository or `/tmp/e40_cursor` was modified. Reviewed 11-27-2026, CPU only, no model calls.

## (a) Verdict

The code runs and reproduces itself exactly (every table and REPORT.md byte-identical, results.json to 3e-7), the six bridged files are byte-identical to the sister repository at efdc8c3, no rater name or identifier leaks, and the four general scoring-rule theorems (Brier and RPS decomposition and strict properness) are real, general Lean proofs with standard axioms. The headline claims do not survive scrutiny. The one "detectable misallocation" (VeriBench `native_main`, B = 64) loses significance under a task-clustered bootstrap, under any multiplicity correction, and under retraining, and its test prompts are not sealed but already used in sister-repo Experiments 27 and 28. The "agreement once the budget is large" message is contradicted by the folder's own results.json (21 cells whose intervals exclude 0 with the D-study worse, 13 after Holm correction, 6 at B of 256 or more), rests on rows where the budget is not actually equal across k, and uses intervals covering 0 with no equivalence margin; the executive paragraph that states it is a hard-coded string, not computed. The dollar frontier is vacuous: a constant forecast that ignores the text meets the 0.05-nat tolerance on 6 of 7 sets, at the same or a smaller budget than the head on 5 of them. The Lean "G-theory blindness" witness is an instance whose G-coefficients are constants unconnected to the two laws, and one cell of its 2 x 2 table is impossible on the witness's own 0-3 scale (I proved this in Lean); the real-data K-sweep table is mostly infinite, and the synthetic K = 2 crossover is an artifact of a competitor bug.

## (b) Findings, most severe first

### F1. Blocker: the VeriBench B = 64 headline is not a robust finding, and "sealed" is wrong
- **Where:** `REPORT.md:15` and `:27`, `README.md:5`, `experiments/README.md:61`, generated from the hard-coded string at `run_all.py:905`.
- **What is wrong:**
  1. The interval resamples artifacts, but the split unit is the task and the 40 test artifacts come from 10 tasks (4 each). With the task as the cluster the 95% interval is [-0.0284, 0.1695] and covers 0 (`outputs/05b_allocation_stats_summary.txt`).
  2. Multiplicity: the folder computes 47 D-study-minus-oracle intervals (41 compare different cells). The two-sided bootstrap p-value of this cell is 0.0394 (10,000 draws). Holm over the 41 cells gives 0.83; Holm over only the 5 non-identical largest-budget cells gives 0.197. It is not the only, nor the strongest, signal (F2).
  3. The interval is for a fixed head. Across 200 alternative design seeds (which train artifacts and which ratings enter), the difference has mean 0.057 and standard deviation 0.056, a variance component the fixed-head interval omits and larger than its own standard error (about 0.033). A task-cluster bootstrap that retrains both heads in every replicate gives [-1.035, 0.322], median 0.010, positive in only 55% of replicates (`outputs/11_veribench_b64_robustness.txt`).
  4. The budget is not equal. The 40 train artifacts have 2 ratings (22) or 3 ratings (18), 98 ratings in all. The "oracle k = 3" cell is necessarily the same 18 artifacts that happen to have a third rating and uses 54 ratings; the D-study cell uses 64; k = 1 uses 40.
  5. Prior exposure: the sister repo's `PREREG.md` (efdc8c3) says the exp27 test tasks were scored in Experiment 27, and Experiment 28 then evaluated judges on exactly this held-out test (`df7f6af`, `efdc8c3`). This repository's Experiments 36 and 38 also used these native_main ratings, and Experiment 37 used a 106-item export of the same collection. HOW_TO_READ item 1 mentions experiment-36 exposure for the public sets only. Calling the split "sealed" in REPORT and in the index row is inaccurate.
  6. What does hold: refitting both cells with 300 iterations changes nothing (0.0813), the validation-chosen k is also 3, and the selection bias of the test oracle is negligible here (0.0001).
- **Proposed fix:** report this cell as "suggestive, not significant after task clustering and multiplicity correction", disclose the equal-budget violation and the prior exposure, remove "sealed", and add the task-clustered and retraining intervals next to the item interval.

### F2. Blocker: the executive "agreement" message contradicts the folder's own results and is hard-coded
- **Where:** `REPORT.md:15` (string literal `run_all.py:905`), the README TLDR (`README.md:5`) and `experiments/README.md:61`. Also `REPORT.md:5`: "Numbers below are written by run_all.py from results.json".
- **What is wrong:**
  1. "Once the nominal budget reaches a few hundred ratings the held-out CE is within a few thousandths of a nat across k, and the interval covers 0": at B = 256 the spread of test CE across k is 0.010 to 0.087, and at B = 512 it is 0.006 to 0.046. Six D-study-minus-oracle intervals exclude 0 at B of 256 or more (SummEval crowd B = 512 and 1024, POPQUORN offensiveness B = 1024, DICES B = 512, ChaosNLI B = 256 and 1024; `outputs/12_claims_vs_results.txt`).
  2. "The detectable misallocation is VeriBench": 21 of 47 intervals exclude 0, all with the D-study worse, and 13 survive Holm over 41 tests (SummEval expert B = 32/64/128, SummEval crowd B = 32/64/128/512, POPQUORN politeness B = 32/64, POPQUORN offensiveness B = 64/128, ChaosNLI B = 32/64). The same holds against the deployable validation-chosen allocation (`outputs/05b_allocation_stats_summary.txt`).
  3. The "agreement" rows at B = 4096 are not equal-budget comparisons. `run_all.py:199` caps N at the number of eligible train prompts, so in 8 budget rows some k use far less than B. At B = 4096: on DICES, k = 1 to 5 use 594 to 2,970 ratings; on POPQUORN offensiveness, the k = 10 cell that is both the D-study and the oracle pick uses 1,590 ratings (39% of B). PROTOCOL.md:7 defines the comparison "at a fixed rating budget B = N x k" (`outputs/04_audit_committed_results.txt`).
  4. "Agree" is inferred from intervals that cover 0, with no equivalence margin declared in PROTOCOL.md. In 2 of the 7 headline rows the two allocations are the same cell, so [0.0000, 0.0000] is automatic.
  5. "At small budgets that k is often infeasible": 3 of 50 budget rows.
  6. The whole executive paragraph (17, 10, Phi(10) = 0.7875, 0.081 [0.013, 0.141], "B = 32 or 64 except SummEval expert at 128") is a string literal at `run_all.py:905` and `:907`. It matches the current results.json, but it would not change if the data did. The WMT sentence (`run_all.py:982`) is also a literal; no script in the folder fetches WMT.
- **Proposed fix:** generate the executive text from results.json. Restate the finding as "the D-study k is significantly worse than a validation-chosen k at small budgets on five of six public sets; at the largest budgets the CE differences are under 0.002 nats, but those rows do not hold the budget fixed across k". Either cap B at N_train times the smallest k or report ratings actually used, and pre-declare an equivalence margin before calling anything "agreement".

### F3. Important (blocker for any paper use): the dollar frontier measures the cost of a base rate, not of a judge
- **Where:** `REPORT.md:17` and `:29-41`, `tables/dollar_thresholds.md`, `figures/pareto_*.png`, `dollar_hits()` at `run_all.py:285`.
- **What is wrong:**
  1. The full-train head beats a constant train-marginal forecast by only 0.0113 (SummEval expert), 0.0040 (POPQUORN offensiveness), 0.0316 (POPQUORN politeness), 0.0059 (DICES) and 0.0040 nats (ChaosNLI). It is 0.0008 nats worse than the constant on SummEval crowd. Only on VeriBench does it gain more, 0.059 (`outputs/05b_allocation_stats_summary.txt`). The 0.05-nat tolerance is larger than the whole learnable signal on all six public sets.
  2. A constant forecast estimated from 32 train ratings meets the tolerance at B = 32 on SummEval expert, SummEval crowd, POPQUORN offensiveness, DICES and ChaosNLI, and at B = 64 on POPQUORN politeness. On SummEval expert (head: 128) and POPQUORN offensiveness (head: 64) it gets there with fewer ratings than the head. At B = 32 the head's frontier is worse than that constant on SummEval expert (1.6005 versus 1.2410), SummEval crowd, POPQUORN offensiveness and ChaosNLI (`outputs/16_constant_forecast_frontier.txt`).
  3. 12 of the 14 dollar hits equal the smallest grid budget (32), so they are left-censored: the true threshold is "at most 32".
  4. The frontier is the minimum over k of test CE, so the test set chooses k. The figures' y-axes do not say this.
- **Proposed fix:** report the head's gain over the constant forecast first. Set the tolerance as a fraction of that gain (the user's study uses 10% as a sensitivity check). Extend the grid below 32 or report "at most 32". Use a validation-chosen k for the frontier.

### F4. Important: the Lean "G-theory blindness" witness does not formalize what README and REPORT say
- **Where:** `theory/GTheory.lean:495-523` (`cellMean`, `sigmaRes2 := 5/4`), `:605-619` (`g_theory_blindness_witness`), `:556-568`, `:451-458`, `:627-646`; `README.md:5` ("therefore the same generalizability coefficients").
- **What is wrong:**
  1. The G-components are closed constants, computed from a hand-written 2 x 2 table and a defined residual of 5/4. They do not take `pStar` or `qBad` as arguments, so "both judges have the same G coefficients" holds by construction, not by derivation. The theorem is a conjunction of separate numeric facts. No G-coefficient is defined as a function of a judge's rating laws, and the general statement (equal cell means and variances imply equal coefficients) is not stated.
  2. The table is not realizable. Cell (1,1) has mean 1/4 and within-cell variance 5/4, and no rating law on {0,1,2,3} can have both: with mean 1/4, E[Y^2] is at most 3/4, so the variance is at most 11/16 (Bhatia-Davis). I proved this in Lean as `cell11_infeasible` and `cell11_second_moment_bound` (standard axioms only; `scripts/ReviewChecks.lean`, `outputs/08_review_checks_lean.txt`). The two witness laws have mean 3/2, so they can sit only in cell (0,0).
  3. The cross-entropy separation in Lean is only the rational inequality prod(q) < (1/4)^4. The log step is numerics only. The Lean docstring says so; README and REPORT list cross-entropy without that label.
  4. `dstudy_depends_only_on_k` proves two numeric values (Phi(1) = 3/31, Phi(4) = 3/10). Independence from N holds only because `phiAt` has no N argument.
  5. Refinement: `rps_term_nonneg`'s docstring says "refinement of the cut set is monotone", but it states only that one term is nonnegative. `refinement_adds_a_nonnegative_term` is `a <= a + extra`. `rps_refines_middle_cut` is a K = 4, middle-cut instance about a separately written `rps4Terms`. I checked that `rps4Terms` equals `rps 4` (`rps4Terms_eq_rps`, compiles), so the instance does concern the same RPS. No general subset-of-cuts theorem is proved.
- **Proposed fix:** define G-coefficients as a function of a table of cell laws. Pick cell means inside (1/2, 5/2), for example by shifting the table by +1/2, which leaves every component unchanged. Give two different laws per cell with equal mean and variance, and prove both "G equal" and "Brier/RPS differ" for those laws. Label the CE step "numerics only". Rename `dstudy_depends_only_on_k`. Either prove general refinement or call it a K = 4 instance.

### F5. Important: the real-data rebin table is mostly infinite and prints invalid intervals
- **Where:** `tables/ksweep.md` (Real rebin section), `rebin_sweep()` at `run_all.py:354-396`, `percentile_interval` in experiment 36 `metrics.py:140-149` (drops non-finite replicates).
- **What is wrong:** the "MSE-trained" head is the two-point distribution on adjacent grid values (`two_point_on_grid`), so held-out CE is infinite whenever a rating falls outside the two points. In 52 of 69 rows the MSE-head CE is infinite. In 8 rows a finite "95% interval" is still printed, computed from the 11 to 390 bootstrap replicates (out of 1000) that happened not to draw an infinite item, for example `inf [0.0030, 0.0217]`. The 4 DICES "native" rows duplicate the K3 rows. REPORT never discusses this table (`outputs/14_rebin_inf_and_native_consistency.txt`).
- **Proposed fix:** give the mean-based competitor full support (a discretized Gaussian or beta-binomial around the predicted mean, or the calibrated map the user's study uses). Make intervals NaN whenever any replicate is non-finite. Drop the duplicate rows.

### F6. Important: the synthetic K-sweep crossover at K = 2 is an artifact of a competitor bug
- **Where:** `run_all.py:469-482` (`sigma` from the residuals of the k-rating MEAN), `REPORT.md:51-58`, `tables/ksweep.md`, `figures/synthetic_kstar.png`.
- **What is wrong:** the Gaussian competitor's predictive SD comes from residuals of the mean of k ratings. It shrinks from 26.3 at k = 1 to 5.1 at k = 32, so the competitor grows more overconfident as k rises. With the SD taken from individual ratings (about 27 at every k), the "CE head better" verdict flips at K = 2 for every k from 2 to 32, so no k shows a CE advantage at K = 2 (5 verdicts change). For K of 3 or more the verdict stays "better at k = 1", but the gaps for k of 2 or more shrink by 1.3 to 4.5 times (K = 3, k = 32: 0.296 to 0.065; K = 101, k = 32: 6.54 to 1.95) (`outputs/09_synthetic_sigma_check.txt`). k* describes this pair of estimators, not a rating design. "Finite-grid CRPS" is exact only at K = 101, where unit spacing matches integer support. For K below 101 the code scores RPS on bin indices, so magnitudes are not comparable across K; within-K comparisons are fine. Test draws also differ across cells, because the RNG is consumed sequentially (`run_all.py:465`). The sweep is labeled synthetic and is not used for a real-data claim.
- **Proposed fix:** use the individual-rating SD, and re-state "K = 2: no crossover; K of 3 or more: CE better from k = 1". Plot the CI-based k*, or the gap itself, rather than a flat line of point estimates.

### F7. Important: the GPU-bridge note is inaccurate
- **Where:** `REPORT.md:62`, `tables/paper_master_results.md`, `run_all.py:614-618`.
- **What is wrong:** "They were not recomputed: the JSON stores probabilities, not the human ratings". The two bridged CSVs hold the ratings of all 175 items, the same 175 ids as the logit JSONs. From them I recomputed normalized RPS exactly for both single-pass rows: 0.1482 (Qwen3-4B) and 0.2145 (Qwen3-8B), with item-equal mean human score 3.742 (`outputs/13_recompute_gpu_bridge_rows.txt`). The quoted CE column (3.589, 5.375) does not equal the raw mean of -log q (11.97, 57.60), so some floor or smoothing applies that is not documented here.
- **Proposed fix:** recompute what can be recomputed and state the CE convention. Rename "GPU bridge" to "copied sister-repo outputs".

### F8. Important: the fixed-head and item-clustered intervals under-cover for the stated claims
- **Where:** `HOW_TO_READ_RESULTS.md:6`, `REPORT.md:72`, `summarize_losses` and `paired_diff` at `run_all.py:93-119`.
- **What is wrong:** the intervals cover test-prompt sampling only. On SummEval (20 test articles) and VeriBench (10 test tasks) the split unit is the group, and group-clustered intervals are much wider. For example, SummEval crowd B = 512 goes from [0.0065, 0.0229] by item to [-0.0022, 0.0315] by article. The training draw, meaning which N prompts and which k ratings, is a single seed; see F1 for its size on VeriBench. This complies with the repository's "item-clustered" wording, but the claims are about allocation rules, which need the training variability.
- **Proposed fix:** report group-clustered intervals wherever the split is by group, and add multiple training-draw seeds or a retraining bootstrap for the allocation comparison.

### F9. Minor: figures and captions
1. `figures/allocation_k_vs_test_ce.png` rebuilds options with `(B // k) == N` (`run_all.py:847`), which silently drops capped cells. On POPQUORN offensiveness it drops k = 1, 2, 3, 4 and the D-study/oracle k = 10, the "same k" of the headline. On DICES it drops k = 1 to 5, and on SummEval expert k = 1. No oracle marker is drawn.
2. In the Pareto figures the y-axis is the test-chosen minimum over k, which neither the label nor any caption says. There are no reference lines (full-train head, tolerance, constant forecast).
3. `figures/synthetic_kstar.png` plots the point-estimate k*, which is 1 everywhere (a flat line), not the CI-based rule, and its y-label is cut off.
4. No figure is captioned or referenced in REPORT or README, so the repository's rule that captions say how the head and allocation were chosen is not met.
5. Gridlines are present on every axis, with minor gridlines on the log axes. Axis labels name the metric, the held-out split and the direction.

### F10. Minor: provenance and rating-file details
1. REPORT.md:47 says "the three scores were sorted", but 30 of the 75 old rows have 2 scores. `run_all.py:536` hard-codes `ratings_per_item = 3` in results.json (`outputs/02_privacy_counts.txt`).
2. The folder has no sha256 receipt for the six bridged files. I verified them byte for byte (`outputs/01_provenance_bytes.txt`); a receipt file would make that reproducible.

### F11. Minor: process and reporting
1. PROTOCOL.md:3 says "Written before run_all.py printed a held-out number", but the protocol, code and results arrived in one commit, so this cannot be checked.
2. The "$15,000-$75,000 researcher-plus-GPU anchor" (PROTOCOL.md:42, REPORT.md:17 and :41) is labeled "not measured" but has no source. The user's own study records that this range came from an LLM suggestion and was not verified; say so or drop it.
3. In `tables/gstudy.md`, the column headed "smallest k with Phi >= 0.80" shows the fallback k = 10 ("not hit") for POPQUORN offensiveness.
4. Eight L-BFGS-B fits stop at the 35-iteration cap with `success = False` (1 SummEval expert, 3 SummEval crowd, 1 POPQUORN offensiveness, 3 VeriBench). Refitting the VeriBench cells at 300 iterations changes nothing to 4 decimals. HOW_TO_READ says "convex" but not that some fits stopped at the cap.
5. Tests:
   1. `tests/test_gstudy.py:29` is a vacuous assertion (`... is False or True`).
   2. The numeric twin tests no RPS strict-properness case and does not test whether the G table is feasible.
   3. There is no PASS/FAIL claims script in the format CLAUDE.md asks for.
6. Nits:
   1. `lean_check.txt` omits the "Build completed successfully" line and the toolchain line.
   2. Lean emits a deprecation warning (`if_true`).
   3. The commit title says "CRPS sweep" for a sweep that is CRPS only at K = 101.

## (c) Verified OK

1. **`run_all.py` reproduces.** It runs in 139 s single-threaded. results.json matches on 12,749 numeric leaves, max absolute difference 3.1e-7. REPORT.md, HOW_TO_READ_RESULTS.md, results.md and all six tables are byte-identical. The figures differ only in font rendering; I compared the content visually (`outputs/06_run_repro.txt`, `outputs/10_compare_repro.txt`).
2. **`pytest`:** 11 passed.
3. **`lake build` (Lean v4.36.0, no Mathlib) succeeds.** All nine `#print axioms` lines list only `propext`, `Classical.choice` and `Quot.sound`. There is no `sorry`, `admit`, `native_decide`, `decide`, `axiom`, `unsafe` or `implemented_by` in `GTheory.lean` (`outputs/07_lake_build.txt`).
4. **The four scoring-rule theorems are general.** `brier_risk_decomposition`, `brier_strictly_proper`, `rps_risk_decomposition` and `rps_strictly_proper` hold for any K over the rationals. K = 2 has one cut and is covered; K = 1 is trivial. Strictness is in the form "equal expected score if and only if q = p", and with the nonnegative decomposition that is strict properness. RPS strictness needs sum(q) = 1, which is assumed. The order is the index order of `Fin K`, so any fixed order works.
5. **Witness arithmetic is correct.** Both means are 3/2 and both variances 5/4. The Brier gap is 1/20, the RPS gap 3/200, and the CE gap 0.12178 nats (numerics). The G numbers 9/64, 1/16, 9/89, 3/31 and 3/10 are right for the stated table.
6. **`rps4Terms` equals `rps 4`.** My Lean theorem compiles with standard axioms.
7. **The six bridged files are byte-identical** to the sister repo's objects at `efdc8c392c04...`. Row counts are 100 (new) and 75 (old) data rows. The native_main ratings loaded from `460466d` (100 items, 250 ratings) agree with the efdc8c3 CSV in all 25 tasks. `exp27_split` is task-pure: 40/20/40 native artifacts, with 0 tasks in two splits.
8. **Privacy.**
   1. No exact-case rater name, rater id or e-mail appears in any committed experiment-40 file. Patterns were built at run time from the sister repo's manifest and roster: 34 ids, 36 names, 12 e-mails.
   2. All 46 case-insensitive first-name hits are ordinary words.
   3. The rating CSVs have no per-rater column, and the JSONs have no identity keys.
   4. The bridged sister files are already publicly readable (an unauthenticated raw fetch returned HTTP 200), so copying them into this public repository discloses nothing new.
9. **The hard-coded executive numbers currently match results.json:** k = 17, k = 10, Phi(10) = 0.78752, 0.0813 [0.0128, 0.1406], and B_ce = 128/32/64/32/32/32/32. The experiments/README.md row matches REPORT.
10. **The test oracle adds negligible selection bias.** Bootstrap estimates are at most 0.0022 nats, and validation-chosen comparisons give the same picture. No budget has an exact tie among options, so the Experiment 39 tie-break problem does not arise.
11. **The two sister-repo checks hold.** The R_v1 loader returns HTTP 404 here too. The GPU JSON integrity checks reproduce (n = 175, probability-sum error 2e-6, expected-score error 5e-6 to 6e-6).
12. **The WMT claim is right at metadata level.** Head commit is `301de385`, train.csv is 638,191,673 bytes, and the header is `lp,src,mt,ref,score,raw,annotators,domain,year` (a mean plus an annotator count) (`outputs/15_wmt_metadata.txt`).
13. **Prices are labeled as assumptions.** The $0.05 rate is close to POPQUORN's published $12 per hour times 13 minutes per 50 annotations, about $0.052 per rating. No price is called published.
14. **The G-study estimator works.** Henderson III matches the balanced ANOVA and is unbiased in an unbalanced simulation (pytest). D-study choice follows PROTOCOL: train split, target 0.80, smallest feasible k, largest feasible k as the fallback. k = 1 is in the grid, and cells with N below 4 are skipped.

## (d) Could not check

1. **R_v1 at `veribench@d1c47355`:** HTTP 404, private repository.
2. **Sister-repo visibility through the GitHub API:** access is not enabled in this session. The unauthenticated raw fetch shows it is publicly readable.
3. **Whether PROTOCOL.md predates the results:** there is only one commit.
4. **WMT has no individual rater rows:** checked from the header only. I did not download the 638 MB file.
5. **The quoted CE column of the copied GPU rows:** the smoothing or floor convention is not in the bridged files.
6. **Byte-level figure reproduction:** fonts and the matplotlib version differ here. I compared one figure visually; the others were checked only through their data.
7. **Exact match of the bootstrap p-values to the folder's own intervals:** the folder publishes no p-values.

## (e) Comparison with your study (now folder 41, README only)

1. **ICC and Phi(1).** For the nested designs Phi(1) equals ICC(1) by construction, and Cursor's all-splits values match yours: SummEval expert 0.7187 versus 0.719, SummEval crowd 0.4920 versus 0.492, ChaosNLI SNLI+MNLI 0.4851 versus 0.485. For the crossed sets Cursor uses Henderson III with a rater component, which explains the small gaps: POPQUORN offensiveness 0.2945 versus 0.287, POPQUORN politeness 0.4446 versus 0.439, DICES-990 0.1935 versus 0.201. Cursor's train-split values are 0.7200, 0.4806, 0.2704, 0.4425, 0.1962 and 0.4835.
2. **k\*.** The two studies agree on every set within your intervals:
   1. SummEval expert 2 and 2.
   2. SummEval crowd 5 and 5.
   3. ChaosNLI 5 and 5.
   4. POPQUORN offensiveness 10 and 10: Cursor's train split misses 0.80 (Phi(10) = 0.7875), and the all-splits row hits it.
   5. POPQUORN politeness: Cursor 6 on train and 5 on all splits, yours 6 [5, 6].
   6. DICES: Cursor 17, yours 16 [15, 18]. The difference comes from the crossed versus one-way estimator.
3. **One rating per item at equal budget.** Your P7 has one rating per item better or tied in 19 of 20 cells. Using Cursor's own cells at budgets where k = 1 is binding, point estimates favor k = 1 in 23 of 35 and the D-study k in 12. The D-study wins come at B of 64 or more on SummEval expert, POPQUORN and DICES, by 0.0007 to 0.048 nats (`outputs/12_claims_vs_results.txt`). Direction agrees with yours at small budgets and on SummEval crowd and ChaosNLI, and Cursor's significant cells (F2) support your reading. Cursor's headline framing ("agree once the budget is large") comes from the non-binding largest-budget rows. The remaining gap is explained by design: one training draw versus your three, CE versus normalized RPS, a test-oracle grid versus k = 1 against k\*, and different budget grids.
4. **Dollars.**
   - Yours: $6.40 to $25.60 at $0.05, and $640 to $2,560 at $5 (128 to 512 items, saturation margin 0.001 normalized RPS).
   - Cursor's: $1.60 to $6.40 and $160 to $640 (B = 32 to 128 ratings, +0.05 nats of a test-chosen frontier).

   Different criteria explain the gap. Both datasets show the heads learn almost nothing beyond the constant forecast. Your README says so; Cursor's does not, and F3 shows its thresholds are met by the constant forecast itself.
5. **The $15k-$75k range.** You flag it as unverified (LLM-proposed). Cursor calls it a "stated anchor, not measured" with no source.
6. **Lean.** Both studies prove instance witnesses that the moments are blind to shape: your Brier divergence is 7/40, Cursor's is 1/20 (RPS 3/200). Cursor's added G-table is incoherent (F4). Your K = 2 "mean identifies the law" result explains F6: once its SD bug is fixed, Cursor's synthetic sweep finds no CE advantage at K = 2.
7. **VeriBench.** Your README says the VeriBench views were not run because the pinned file sits in a repository this machine cannot read. That holds for R_v1 only. Cursor loaded `native_main` from the public sister repository at `460466d`, and that path works here, so your study could add native_main through the same loader. Note the prior exposure (F1).
8. **No contradiction needs a numeric reconciliation.** Every difference traces to the estimator (crossed versus one-way), the split used (train versus all items), or the design of the equal-budget and sufficiency criteria.

## Files

- Scripts: `scripts/01_provenance_bytes.sh`, `02_privacy_counts.py`, `03_privacy_hit_contexts.py`, `04_audit_committed_results.py`, `05_allocation_stats.py`, `06_run_repro.sh`, `09_synthetic_sigma_check.py`, `10_compare_repro.py`, `11_veribench_b64_robustness.py`, `12_claims_vs_results.py`, `13_recompute_gpu_bridge_rows.py`, `14_rebin_inf_and_native_consistency.py`, `15_wmt_metadata.sh`, `16_constant_forecast_frontier.py`, `ReviewChecks.lean` (built against a copy of `GTheory.lean` in `lean_build/`).
- Outputs: the matching files in `outputs/`.
- Environment: `SANMI36_DATA=/tmp/opus_review_exp40c/data_cache` (copied, sha-verified by the loaders), `TTJ_ROOT=/tmp/ttj_exp28_efdc8c3` (read with `git show` at pinned commits), and OMP/BLAS threads set to 1.
