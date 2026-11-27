# Independent adversarial review: trustworthy-theorem-judges Experiments 29 and 30 (commit f76d502)

Reviewer: independent Opus subagent, 11-27-2026, 19:43 to about 20:10 UTC. Read-only on the source repository. All work was in a scratch copy at `/tmp/qa2930/repo`. Independent code is in `/tmp/qa2930/indep/` (`recompute29.py`, `extra29.py`, `f8diff.py`, `power30.py`, `rank30.py`). None of it imports the experiments' code, except one marked call that measures their own optimizer's weight norm.

## 1. Verdict

**Experiment 29.** The numbers are right, but the headline overstates what is new. The pre-registered primary on the 20 opened test tasks fails the written rule: its cross-entropy is worse than the set-aware constant's. That failure is the confirmatory result. The second claim is that the H3-free 50-task nested head beats the set-aware constant: ranked probability score zero point zero six eight four against zero point one one eight six, a paired difference of minus zero point zero five zero two, interval from minus zero point zero seven two six to minus zero point zero two nine zero. I reproduced every one of those digits with my own code. The result is robust: it survives dropping any single task, any single fold, other bootstrap seeds and a cluster t interval, and thirty-nine of the fifty tasks favor the head.

That 50-task claim is exploratory, not confirmatory:

- **Not new.** Experiment 28 was already on main and had already shown the same thing on the same fifty tasks, the same folds, the same ratings and the same constant. Its difference was minus zero point zero five zero one.
- **Pre-registration cannot be dated.** The pre-registration was committed in the same commit as the results.
- **The script does not run as documented.** The README's command sequence crashes at the eval stage, because the committed selection file was edited outside the committed code.
- **The CE-versus-RPS comparison is confounded.** The same L2 penalty weighs about ten to twenty times more against the ranked probability score loss than against cross-entropy, because the RPS loss is about that much smaller.

Safe to quote: "on these fifty already-analyzed tasks, a nested cumulative-logit head beats the set-aware constant, consistent with Experiment 28." Not safe to quote: that this is a new or pre-registered confirmation, or anything about cross-entropy versus RPS training.

**Experiment 30.** The ranking numbers are correct, and I reproduced them independently: twenty-one fives out of sixty-one ratings in the top twenty-five, and sixty-six out of seventy in the bottom twenty-five. But the committed script cannot run. It is missing an import, so the receipt came from a script that is not in the repository. After a one-line patch, everything reproduces.

The power claims are weaker than written:

- **The forward simulation cannot fail.** It draws "new" ratings from the current histograms, so it can only return the current interval. It shows nothing about real extra ratings. Its general reason, that a task bootstrap does not shrink when you add ratings, is false in general.
- **The practical conclusion is plausible, for a different reason.** On these data, rating noise is only about seven percent of the task-level variance.
- **Twenty-three tasks is a coin-flip number.** It is roughly the fifty-percent-power point. Under reasonable alternatives, the planning range is about twenty-six to sixty-two tasks, and more if the true effect is smaller.
- **It is framed around a head Experiment 29 had already superseded.** The whole question is about the F8 head, while Experiment 29's other heads already exclude zero on the same twenty tasks.

## 2. Checks run (exit codes in brackets)

- **A, Experiment 29 reproduction:**
  - `build_snapshot.py` [0]. Output is byte-identical to the committed `artifacts_11-27-2026.csv` and `join_audit_11-27-2026.json`.
  - `unittest` [0]. Three tests, OK.
  - `--stage select` [0]. The grid CSV is byte-identical. The selection JSON matches on every shared key but lacks `best_converged_primary` and `amendment_1`.
  - `--stage eval` [1]. Crashes with `KeyError: 'best_converged_primary'` at `run_experiment.py:236`. With the committed selection file restored (a deviation), eval runs [0] and `heldout_eval_11-27-2026.json` is byte-identical.
  - `--stage nested` [0]. `nested_oof_11-27-2026.json` is byte-identical.
- **A, Experiment 30 reproduction:**
  - `run_analysis.py` [1]. Fails with `NameError: name 'defaultdict' is not defined` at `scripts/run_analysis.py:191`.
  - After adding `from collections import defaultdict` in a scratch copy: [0], 81 seconds. Both priority CSVs are byte-identical. `analysis_11-27-2026.json` matches to about 1e-6 relative (NumPy 2.5.3 here, 1.26.4 in the receipt). Every printed number is unchanged at its printed precision.
- **B, independent recomputation (own code, task bootstrap, 10,000 draws, seed 20261127 plus seeds 7 and 99):**
  - RPS definition: REPRODUCED and the definition is not disputed. The code's histogram formula equals the mean over individual ratings of (1/5) Σ_{t=0..4} (F_Q(t) − 1[r ≤ t])², averaged within artifact and then equally over artifacts (max abs difference 1.1e-16 on 175 artifacts). That is what AGENTS.md:31 specifies.
  - Set-aware constant, 20-task test: 0.1088 REPRODUCED.
  - Fold-wise constant, 50-task: 0.1186 REPRODUCED.
  - F8 head: 0.0737, paired −0.0351 [−0.0764, +0.0020] REPRODUCED. The upper bound is +0.0029 with seed 7.
  - Best converged head: 0.0653, −0.0435 [−0.0782, −0.0139] REPRODUCED.
  - Primary head: NOT REPRODUCED EXACTLY, because it depends on the optimizer. My longer fit gives 0.0689, −0.0399 [−0.0725, −0.0097] and CE 1.244, against their 0.0698, −0.0390 and 1.249. Verdict unchanged.
  - Nested 50-task head: 0.0684 against 0.1186, paired −0.0502 [−0.0726, −0.0290] REPRODUCED exactly from the receipt's per-fold configurations refit with my own cumulative-logit code. CE 1.003 against 1.308 and TVD 0.345 against 0.492 also REPRODUCED.
- **C, pre-registration history:** both PREREG.md files have exactly one commit, the results commit (5438370 for Experiment 29, f76d502 for Experiment 30), on every ref. Totals scored:
  - **20-task validation grid:** 2,760 rows (552 configurations × 5 temperatures; 960 rows primary-eligible).
  - **Test:** 6 named heads and 3 constants, under 3 rating targets.
  - **Nested pass:** 48 configurations × 2 temperatures × 4 inner folds × 5 outer folds.
- **D, leakage and baseline:**
  - Outer folds are task-level: 5 old and 5 new tasks per fold, no task in two folds or splits.
  - The nested selection is genuinely nested. Inner 4-fold on the 40 outer-training tasks; standardization and constants come from training rows only.
  - H3 is correctly excluded from the 50-task pass, and H4 is zero-demonstration.
  - Raters and solver agents are not separated across splits.
  - The set-aware constant is stronger than the literature-standard pooled train marginal (pooled constant RPS 0.1643 on the 50-task out-of-fold data), so the baseline is fair.
- **E, intervals:** percentile task bootstrap, 10,000 draws. The same seed is reused for every comparison, so all intervals share the same task draws. For the nested result:
  - **Tasks:** 39 of 50 favor the head.
  - **Leave one task out:** the point stays in [−0.0519, −0.0459] and the worst upper bound is −0.0265.
  - **Leave one fold out:** every interval is below 0 (upper bounds −0.0264 to −0.0286).
  - **Most influential task dropped:** −0.0459 [−0.0673, −0.0266].
  - **Cluster-t interval:** [−0.0727, −0.0277].
  - **Cohorts:** old −0.0794 [−0.1201, −0.0412], new −0.0283 [−0.0500, −0.0095].
  - **20-task heads:** F8 is knife-edge; 4 of 20 single-task deletions push its upper bound below 0, and the upper bound ranges from −0.0062 to +0.0088. The primary and best converged heads stay below 0 under all 20 deletions.
- **F, CE versus RPS fairness:** the grids and the optimizer start are the same for both objectives, but the penalty is not matched in effect (finding 5). This was checked on the validation grid and by matched test refits.
- **G, Experiment 30:**
  - The rank score reproduces independently, including item045's member means [3, 5, 3.3, 3.154, 4.345, 4.07], split counts 13 train, 3 validation and 9 test, and Spearman −0.617.
  - I read the simulation code. I ran my own bootstrap-shrinkage and variance-decomposition experiment and my own power and extrapolation sensitivity analysis.
  - The derivation states no theorem, and nothing in it is machine-checked; there is no `.lean` file.
- **H, narrative against data:** mismatches and untraceable numbers are listed in findings 7, 11 and 13. The unit tests are reviewed in finding 17.
- **I, reporting standard:** gaps are listed in findings 10, 14 and 15. Neither experiment has figures, so the gridline rule does not apply.
- **J, paper side:** the Experiment 29 and 30 numbers in `latex/results/29_*.tex`, `30_*.tex`, `mega_results.tex:10-11` and `mega_meta.tex` match the JSON. `pdftotext` finds all the headline numbers in `mega_tech_report.pdf` (CreationDate 19:27:08 UTC, commit 19:28:06 UTC). There are no absolute home paths, tokens or cache directories, and no rater names. The owner's personal email is committed (finding 18).

## 3. Findings, most severe first

1. **Blocking (reproducibility). Experiment 30's only documented command fails.**
   - `scripts/run_analysis.py:191` calls `defaultdict`, but the imports (lines 9–15) never import it, so `python3 scripts/run_analysis.py` fails on a fresh clone (README.md:22) with exit 1.
   - The receipt records NumPy 1.26.4 and SciPy 1.13.1 (RESULTS.md:90). The committed results were therefore produced by a script that is not the committed one.
   - With the one-line import added, every output reproduces (section 2).
   - Smallest fix: add `from collections import defaultdict`, rerun on a fresh clone, and commit.

2. **Blocking (reproducibility and provenance). Experiment 29's README sequence crashes, and its selection file was edited by hand.**
   - `select()` writes a payload without `best_converged_primary` or `amendment_1` (`scripts/run_experiment.py:146-159`). `evaluate()` requires `selection["best_converged_primary"]` (`run_experiment.py:236`).
   - Running the README commands in order (README.md:22-28) overwrites the committed selection file and then fails at `--stage eval` with a KeyError, exit 1.
   - The committed `results/validation_selection_11-27-2026.json` carries the two keys, so it was edited outside the committed pipeline.
   - The claim "Selection, written before test scores" (results.md:10) therefore cannot be checked against code.
   - Smallest fix: compute `best_converged_primary` in `select()` (best primary-eligible row with `optimizer_success`), write the amendment note from code, and rerun the five commands from a clean clone.

3. **Important (pre-registration unverifiable). "Pre-registered" and "written before" cannot be checked for either experiment.**
   - Both PREREG.md files first appear in the same commit as all results: 5438370 for Experiment 29 and f76d502 for Experiment 30. No earlier commit exists on any ref.
   - This affects "Written before `--stage select` and before any test metric" (Experiment 29 PREREG.md:7), "the sigmoid-clip amendment written before any test score" (README.md:9) and "Predictions, committed before the run" (Experiment 30 PREREG.md:38).
   - It also affects "The five predictions in PREREG.md all landed" (Experiment 30 RESULTS.md:82). Two of those five predictions only replay numbers that were already published (PREREG.md:40-41).
   - Experiment 28 shows the repository can do this properly: PREREG in 0414a74, frozen winners in 94c5cd8, results in df7f6af.
   - Smallest fix: relabel each PREREG as "written in the same commit as the results; timing not verifiable from history". Next time, commit the PREREG first.

4. **Important (overstated; unlabeled prior exposure). The 50-task "works" verdict repeats Experiment 28 and is secondary by the PREREG's own rule.**
   - Experiment 28 (RESULTS_11-26-2026.md:18-19, 64-69, 91) already scored the same 50 tasks, the same `cv5_fold` folds, the same 445 ratings and the same fold-wise constant (0.1186):
     - H3-free nested head: −0.0483 [−0.0724, −0.0258].
     - Post-hoc all-frontier head: −0.0501 [−0.0753, −0.0266].
     - Nested CE C7 head: RPS 0.0683.
   - Experiment 29's −0.0502 [−0.0726, −0.0290] is a third look at the same data with a similar head. Exp 29's results.md:28 cites Exp 28's nested RPS values but not that they had already cleared the bar.
   - The PREREG's decision rule is stated "for the one primary head" on "the pooled 20 test tasks" (PREREG.md:57-61). The nested pass only gets "its own verdict … labeled as the 50-task estimate" (PREREG.md:63). So the confirmatory verdict is "not found".
   - Yet README.md:5, results.md:5 ("also written down before the test scores") and results.md:24 ("it is the one that clears the bar") put the nested result on equal or higher footing, as does mega_results.tex:10.
   - "Before the test scores" has no force for the nested pass, since every one of its 175 artifacts was scored out-of-fold a day earlier.
   - Smallest fix: the TLDR should say "Primary (confirmatory): not found. Secondary 50-task nested pass: −0.0502 [−0.0726, −0.0290], which reproduces Experiment 28's earlier result on the same tasks and folds (−0.0501). Not independent evidence."

5. **Important (CE versus RPS comparison confounded). The shared penalty grid is not matched in effect.**
   - The L2 term is added to the mean loss: `score.mean() + 0.5*alpha*(w@w)` (`lib29.py:169`). RPS values are about 0.05–0.1 and CE about 0.8–1.3, so each α regularizes an RPS fit roughly 10–20 times harder.
   - Validation grid, matched ordinal pairs with the same features, mixture and temperature:
     - At α = 0, the RPS-fit head wins 55 of 75 pairs (median −0.0063).
     - At α = 0.1, 1 and 10, it loses 75 of 75 pairs at each value.
   - Matched test refits (test RPS of the RPS-fit head minus test RPS of the CE-fit head; positive means the CE fit is better):
     - Best-converged configuration: +0.0079 [+0.0031, +0.0135].
     - F8 features at α = 1: +0.0342 [+0.0200, +0.0531].
     - F8 features at α = 0: −0.0035 [−0.0098, +0.0025].
   - At α = 0 the RPS objective has no finite minimizer. With their code the weight norm is 163.7 on train and 219.6 on train+validation, never converging; my longer fit ran to 848.
   - So the grid offers RPS only a degenerate option or an over-penalized one. Two statements reflect that parameterization, not the objectives: "RPS training did not produce a head that wins the proper-score panel" (results.md:95) and the nested pass picking CE in 4 of 5 folds (results.md:22).
   - Smallest fix: state that the comparison is confounded. Next time, scale α by each objective's loss magnitude, or give each objective its own α grid, and report a matched CE-versus-RPS test table.

6. **Important (selector artifact). "Temperature 2" is a near-tie of a saturated fit, and the primary's numbers depend on the optimizer budget.**
   - The top five primary-eligible validation rows are the same unconverged fit (h2_h3, RPS objective, α 0, mixture 0.25) at T = 2, 0.5, 0.75, 1.5 and 1. Their validation RPS spans only 0.04930 to 0.04939, so T is effectively unidentified.
   - results.md:95 nonetheless reads it as a failed prediction ("the argmin used temperature 2").
   - The primary's test numbers depend on `maxiter=250` (`lib29.py:171`): my longer fit gives RPS 0.0689 and paired −0.0399, and primary minus F8 is −0.0047 rather than the printed −0.0039 (results.md:62).
   - The verdict does not change.
   - Smallest fix: say T is not identified for this fit, and label the primary's digits as depending on the optimizer budget.

7. **Important (wrong reason in paper-side text). The LaTeX blames non-convergence; the rule fails on cross-entropy.**
   - `latex/results/29_refit_cumulative_logit_heads_ce_versus_rps.tex:46` says: "The primary argmin did not converge. The written rule therefore says not found."
   - The rule has no convergence clause (PREREG.md:59-61; `run_experiment.py:271-275`). It says "not found" because CE 1.249 is worse than 1.159.
   - Line 9 of the same file says CE is worse "because the unpenalized RPS fit did not converge". That is unsupported: my longer fit terminates at an even more saturated solution and still has CE 1.244, above 1.159. The cause is that an unpenalized RPS fit has no finite optimum.
   - Smallest fix: "not found because its CE (1.249) is worse than the constant's (1.159)." Then rebuild the PDF.

8. **Important (overgeneral claim; the simulation cannot fail). Experiment 30's forward simulation is uninformative, and its general reason is false.**
   - The forward simulation draws extra ratings from each test artifact's current empirical histogram (`run_analysis.py`, `forward_curves`: `rng.choice(6, p=probs[i])`) with the forecasts fixed.
   - The paired RPS per artifact is linear in the empirical CDF, so the interval must return to the current one. The "finding" that it stays near +0.002 (README.md:17; RESULTS.md:62) holds by construction.
   - README.md:15 promises "the reason a task bootstrap does not shrink when more ratings are added to the same tasks", and the latex fragment at line 37 repeats it ("More ratings drawn from those histograms do not shrink it"). As a general claim this is false: more real ratings remove within-artifact rating noise from the observed between-task spread.
   - My synthetic check (20 tasks, 3 artifacts each, task SD 0.03, artifact SD 0.02, per-rating noise SD 0.12; illustrative parameters) gives median 95% width 0.0495 at 2 ratings per artifact, 0.0395 at 4, 0.0293 at 16 and 0.028 at 64.
   - On the real F8 data, the per-rating linearization attributes 7.2% of the task-level variance to rating noise. Removing all of it moves the normal-approximation upper bound only from +0.0037 to +0.0023.
   - So the practical conclusion (more ratings on these 20 tasks are unlikely to clear F8) is plausible, for a reason the experiment did not measure.
   - The derivation's sentence (`.tex:85`; `.md:33`), "except insofar as that noise has already changed d_i", understates this mechanism.
   - Smallest fix: replace the general sentence with the variance decomposition, and label the forward curve "uninformative by construction".

9. **Important (framing). Experiment 30 asks about F8 only and ignores heads Experiment 29 had already shown to exclude zero.**
   - README.md:5 says "Extra ratings on the same 20 held-out tasks do not push the RPS interval below 0", and RESULTS.md:17 says "The binding limit is the 20 tasks". Both are about the F8 head only.
   - Experiment 29, pushed 80 minutes earlier on the same tasks and snapshot, has two heads whose intervals already exclude 0: the primary at −0.0390 [−0.0721, −0.0087] and the best converged head at −0.0435 [−0.0782, −0.0139]. Both survive all 20 leave-one-task-out deletions.
   - F8's +0.0020 is fragile: 4 of 20 deletions flip it, and seed 7 gives +0.0029.
   - Smallest fix: scope every sentence to "the F8 head" and cite Experiment 29.

10. **Important (planning figure fragile). "23 tasks" is the fifty-percent-power point.**
    - The arithmetic is right under the stated assumption: U(22) = +0.00028 and U(23) = −0.00050, both reproduced.
    - But 23 is where the expected upper bound touches 0, which is roughly 50% power if the observed effect were the truth. Under alternative assumptions (my `power30.py`):

      | Assumption | Tasks needed |
      | --- | --- |
      | Their rule (normal quantile) | 23 |
      | t quantile with an m/(m−1) correction | 26 |
      | 80% power | 46 |
      | 90% power | 62 |
      | Task SD at the chi-square 95% bounds | 13 to 48 |
      | True effect −0.025 | 45 |
      | True effect −0.020 | 69 |
      | True effect −0.0176 (half the estimate) | 90 |
      | Seed-7 upper bound | 24 |

    - The point estimate comes from an opened split that has been reused across experiments, so a winner's-curse shrinkage is likely.
    - "Three more tasks … about 26 ratings" (RESULTS.md:66; `.tex` "26.4 ratings") is the line most likely to be misquoted.
    - The "planning" label does travel with the figure (README.md:5,17; RESULTS.md:5,17,66,92; `latex 30 .tex:18-19`; mega_results.tex:11). "Not a certification" is missing from the README TLDR and the mega summary row.
    - Smallest fix: report a range (about 26–62 tasks for 50–90% power at the observed effect, more if the effect is smaller) and drop the ratings translation.

11. **Minor-important (narrative contradicts the receipt). The task-subsample sentence draws the opposite conclusion from its own number.**
    - RESULTS.md:64 says "The fraction of subsamples with an upper bound below 0 stays near 0.20 … so shrinking the test set does not manufacture a significant result."
    - The receipt shows 0.18–0.22 of the 6- to 18-task subsamples have an upper bound below 0. A smaller test set would show a spurious pass about one time in five.
    - Smallest fix: reverse the sentence.

12. **Minor (near-vacuous test). The active-versus-uniform five-rate comparison can barely differ.**
    - Prediction 5 and RESULTS.md:60 compare active 0.454 with uniform 0.482 among the "next 100 revealed" ratings.
    - Only 106 extra ratings exist, so both policies reveal 100 of the same 106. They can differ by at most 6 ratings. No Monte Carlo spread is given.
    - Smallest fix: state this, or compare the first 20–40 reveals with an interval.

13. **Minor (untraceable numbers). Several Experiment 29 numbers are printed by no committed script and appear in no JSON.**
    - Best converged minus F8: −0.0083 [−0.0219, +0.0039], p 0.0945 (results.md:21, 62). I reproduce it: −0.0083 [−0.0219, +0.0039], 0.0944.
    - Primary minus F8: −0.0039 [−0.0152, +0.0069] (results.md:62). This depends on the optimizer.
    - "Weight norm about 165" (results.md:19). Their code gives 163.7 on train.
    - "Weight norms between 2.9 and 5.8" and "All five refits converged" (results.md:22).
    - Smallest fix: write these into the JSON from `run_experiment.py`.

14. **Minor (mislabel). A bootstrap fraction is called a p-value.**
    - "p-val=0.0327" (Experiment 30 RESULTS.md:15, :80) is the plus-one bootstrap fraction of draws at or above 0, not a null-hypothesis p-value.
    - Smallest fix: rename it "bootstrap fraction ≥ 0".

15. **Minor (reporting standard). The Experiment 29 README TLDR misses several required elements.**
    - README.md:5 gives no subtraction order in words ("head minus fold-wise set-aware constant, negative favors the head").
    - It gives no interval type, unit or number of draws, no "lower is better", and no definition of RPS or of the fold-wise constant.
    - It does not say the 50 tasks were already scored out-of-fold in Experiment 28. results.md:40 covers the bootstrap details only.

16. **Minor (unstated limitation). Splits are by task only.**
    - Raters and solver agents cross the train/test split and the folds. AGENTS.md:32 asks for task-family, solver and rater separation, and AGENTS.md:9 notes that strict separation needs a prospective amendment.
    - Neither Experiment 29 document says so. Task-family straddling, for example the several "unsafe*_safe" security tasks in the old cohort, was not checked.

17. **Minor (code hygiene and weak tests).**
    - `run_experiment.py:218-221` holds a dead `verdict()` containing `< None or True`, which would raise a TypeError if called.
    - `build_snapshot.py:237` hard-codes `"new_percent_fives": 185 / 250` instead of computing it. The value is correct: 185 fives verified.
    - `build_snapshot.py:75-77` relabels any attestation other than yes, partial or blank as `other:<x>`. The schema filter at :120-121 would therefore silently drop genuine `no` and `unsure` rows. None exist on this snapshot.
    - `build_snapshot.py:63-64` silently skips rater A1, the author, who has 0 numeric scores, so there is no effect. It is undocumented.
    - `tests/test_heads_and_join.py:56-58` assert literals the script writes. Nothing tests the RPS definition, the bootstrap, select-to-eval compatibility (a test there would have caught finding 2) or nesting.

18. **Minor (hygiene).**
    - Experiment 30 RESULTS.md:9 and results.md:9 commit the owner's personal Gmail address to a public repository.
    - RESULTS.md and results.md are byte-identical duplicates.
    - `run_analysis.py` writes an uncommitted `results/work/` scratch folder. Under the no-`.gitignore` policy, `git add -A` would commit it.
    - Clean: no absolute home paths, tokens or caches, and no rater names. The single roster-name match is the owner's own name on the PREREG "Owner:" line.

19. **Minor (derivation precision).**
    - `.tex:61` says the (n+1)(n+2) denominator is "the marginal shrinkage of a variance that scales as 1/n". It is the shrinkage of 1/(n+1); 1/n would give n(n+1).
    - `.tex:71` says RPS is proper because "a constant shift of the mean changes it", which conflates propriety with sensitivity to location.
    - No theorem is stated, so there is no Lean obligation. Every formula is either a definition, a limit under an assumption or a heuristic. I checked the RPS identity and the extrapolation arithmetic numerically.

## 4. What survives

- **Experiment 29 data.** The snapshot is rebuilt byte-identically:
  - 445 ratings (195 old, 250 new; 185 of the 250 new are fives), 175 artifacts, 50 tasks.
  - Attestations: 118 yes, 100 partial, 32 blank.
  - The item020 row is harmless.
  - No task sits in two splits or two folds.
- **RPS definition.** The RPS is the stated per-rating normalized score, averaged within artifact and then equally over artifacts.
- **20-task numbers reproduced independently:** constant 0.1088, F8 −0.0351 [−0.0764, +0.0020], best converged −0.0435 [−0.0782, −0.0139]. The primary's "not found" holds, because its CE is worse in any fit.
- **The 50-task nested head beats the fold-wise set-aware constant:** 0.0684 against 0.1186, −0.0502 [−0.0726, −0.0290]. CE and TVD agree, both cohorts agree, and every robustness check in section 2 holds.
- **The nesting is genuine.** "Correlation was not a training objective" is true.
- **Experiment 30 ranking.** The rank score, item045 as the top item, 21 of 61 against 66 of 70, 185 of 250 and Spearman −0.617 are all reproduced independently. The Experiment 27 replay alignment reproduces. The extrapolation arithmetic is correct under its assumption.

## 5. Could not check

- Whether either PREREG was written before its results. History cannot show this.
- H2 prompt contents and harness differences between the cohorts. I relied on the Experiment 27 and 28 PREREG statements for H3 (training-task demonstrations) and H4 (zero demonstrations).
- Task-family overlap across folds.
- The Google Doc and Drive links, which I did not open.
- I did not re-implement the inner-fold selection. I reproduced it by rerunning their code (byte-identical) and checked the outer results with my own code.
- I did not read Experiment 28's code beyond its results file. The prior audit was skimmed only for its data facts, which match mine.
