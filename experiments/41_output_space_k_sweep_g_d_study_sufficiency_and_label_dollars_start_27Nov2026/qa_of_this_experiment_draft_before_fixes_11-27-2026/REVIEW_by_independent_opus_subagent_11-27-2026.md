# Adversarial QA review of Experiment 40 (output-space K-sweep, G/D-study, sufficiency, label dollars)

Reviewer: Claude Opus 5.5 at maximum effort, 11-27-2026. I made no edits to the repository. Every run used copies under `/tmp/opus_review_exp40/` with `python3 -I -B`, and I wrote no bytecode into the repository. Scratch scripts are in `/tmp/opus_review_exp40/scripts/`. In file:line references, "E40/" means `experiments/40_output_space_k_sweep_g_d_study_sufficiency_and_label_dollars_start_27Nov2026/`.

## (a) Verdict

The arithmetic, preregistration timing, Lean compilation, numerics and leakage controls hold up. The headline matched-gap result (at most 0.0024) reproduces and survived one robustness probe. The sufficiency, saturation and dollar conclusions do not hold as written: they are artifacts of a fixed L2 = 0.01 penalty. Several sentences also misstate numbers or the preregistration status, the preregistered mean-squared-error (MSE) head is never reported, and the "calibration, not shape" reading uses a statistic that cannot detect shape.

## (b) Findings, most severe first

### 1. BLOCKER: "saturation at 128 to 512 items" and the dollar figures at saturation are artifacts of the fixed L2 = 0.01 penalty

- **Where:** E40/README.md:67 (train side), :69 (dollars), :5 (TLDR "these heads learn so little"), :121 (TLDR-end "label dollars are small"); E40/HOW_TO_READ_RESULTS.md:28.
- **What is wrong:** The objective is the mean loss plus 0.5·0.01·‖W‖². The penalty does not shrink with N, and the 24 principal-component features have standard deviations of only 0.077 to 0.167. The curve therefore has to flatten at the penalized optimum, whatever the data could support.
- **Evidence:**
  - `scripts/probe_lc_l2.py` uses POPQUORN politeness (K = 5, k = 1, split seed 0) with the same subsample orders and seeds as `fit_eval_subset`.
  - At L2 = 0.01, test normalized RPS at N = 128, 512 and 2230 is 0.1738, 0.1718 and 0.1716. This reproduces the repository's curve exactly.
  - At L2 = 1e-4 the same three points are 0.1854, 0.1680 and 0.1637. The curve still improves by 0.0043 from 512 to 2230 items, which is 4.3 times the 0.001 margin.
  - At L2 = 1e-4 the best head beats the best L2 = 0.01 head by 0.0079. Its gain over the constant forecast (0.1775) is 0.0138 instead of 0.0059.
  - The K-sweep split shows the same thing (`scripts/probe_fit.py`, seed 0). The CE-trained head scores 0.1714 at L2 = 0.01 and 0.1622 at L2 = 1e-4 (K = 5), and 0.2298 versus 0.2215 (K = 2). The largest fitted weight at L2 = 0.01 has absolute value 0.77.
- **Fix:**
  - Choose L2 on validation for each N (or scale it as λ/N), then recompute the saturation points and dollars.
  - Until then, delete "saturates at 128 to 512", the sentence that the answer "is informative for ... POPQUORN politeness and SummEval expert", and all dollar figures priced at saturation. The alternative is to label them "for L2 = 0.01 heads only; with L2 = 1e-4 the POPQUORN politeness curve is still improving at all 2,230 training items".

### 2. IMPORTANT: the 0.001 saturation margin and the power rule are called "preregistered", but they are not

- **Where:** E40/README.md:67 ("the preregistered margin"), :69 ("preregistered-margin label sets"), :91 (Limitation 7: "fixed before the results").
- **Evidence:**
  - `grep -n -i "0\.001\|saturat\|power\|dollar\|learning curve\|margin" PREREGISTERED_PREDICTIONS_11-27-2026.md` matches only the L2 line.
  - The preregistration commit e067462 (reflog, 09:06:28 −0800) contains only the preregistration file, `e39_common.py` and `k_sweep_matched_calibration.py`.
  - `sufficiency_and_dollars.py` (where `SAT_MARGIN = 0.001` is set) and `gd_study_and_allocation.py` were never committed (`git status`: untracked). So the claim that the margin was fixed beforehand cannot be verified.
  - The same applies to the implementation choices behind P6 and P7: budgets of 128, 256 and 512; k* capped at the 20th percentile of ratings per item; the N ≥ 16 rule; split seed 0 only.
- **Fix:** "Fixed in the analysis script before it ran (not in the preregistration file, and the script was not committed beforehand)."

### 3. IMPORTANT: number error used to justify the post-hoc 10% sensitivity rule

- **Where:** E40/README.md:67 ("That margin is as large as everything the head can learn on three of the five views"), E40/README.md:91 ("equals the whole learnable gain on three sets"), E40/HOW_TO_READ_RESULTS.md:28 ("equals or exceeds the whole learnable gain on three sets").
- **Evidence:** Table B of `results/sufficiency_and_dollars_tables.md` gives these learnable gains:
  - 0.0015 on POPQUORN offensiveness
  - 0.0014 on DICES-990
  - 0.0009 on ChaosNLI SNLI+MNLI

  Only ChaosNLI is at or below 0.001.
- **Fix:** "The margin exceeds the whole learnable gain on one set (ChaosNLI SNLI+MNLI, 0.0009) and is about two thirds of it on two (POPQUORN offensiveness 0.0015, DICES-990 0.0014)."

### 4. IMPORTANT: wrong range for the signal over the constant forecast

- **Where:** E40/README.md:44, "0.0003 to 0.0027 on every other view with signal".
- **Evidence:** `constant_minus_ce_raw.seed_mean` in `results/k_sweep_summary.json` is 0.01155 for POPQUORN politeness K = 3 and 0.00598 for K = 5. The smallest value on any other view is 0.00036.
- **Fix:** "0.0004 to 0.0027 on the four other datasets with signal; 0.0116 (K = 3) and 0.0060 (K = 5) on POPQUORN politeness."

### 5. IMPORTANT: the preregistered MSE-trained head is silently unreported, and its results cut against the narrative

- **Where:** PREREGISTERED_PREDICTIONS:17 lists "scalar mean squared error (MSE-trained head)". `grep -i mse README.md HOW_TO_READ_RESULTS.md` returns nothing.
- **Evidence (from `results/k_sweep_summary.json` and the per-seed rows):**
  - On POPQUORN politeness, the MSE-trained head minus the CE-trained head after the same validation-fitted map is −0.0022 (K = 2), −0.0022 (K = 3) and −0.0018 (K = 5). Negative means the MSE-trained head is better.
  - Per-seed intervals lie entirely below 0 in 3, 3 and 4 of the five seeds.
  - At K = 2 the raw MSE-trained head beats the raw CE-trained head on 4 of 6 binary views. On POPQUORN politeness the difference is −0.0089, with all five seed intervals below 0.
  - This is the empirical test of the 11-25 "MSE plus L2" comment. README:75–77 answers that comment with identities only.
- **Fix:** Add MSE-trained rows to the K-sweep table and to the MSE+L2 section. Say that after recalibration the MSE-trained head ties or beats the CE-trained head on these heads.

### 6. IMPORTANT: "one shared calibration map" misdescribes the control

- **Where:** E40/README.md:5, :42; E40/HOW_TO_READ_RESULTS.md:13; the axis label of figure `k_sweep_raw_vs_matched_gap.png`; the header of `results/k_sweep_summary_table.md`.
- **Evidence:** `scripts/k_sweep_matched_calibration.py:112-118` calls `calibrate(mv, mt, ...)` inside the per-objective loop. Each head therefore gets its own validation moments and its own fitted map.
- **Fix:** "The same map family (a multinomial logit on the standardized scalar mean), fitted separately for each head on the validation ratings."

### 7. IMPORTANT: "calibration, not shape learning" uses the wrong statistic, and the residual gap is a discrimination effect of the objective

- **Where:** E40/README.md:45 ("almost all of the raw cross-entropy advantage is calibration, not shape learning"); PREREGISTERED:29 (P1 rationale) and :31 (P3 rationale).
- **The matched gap cannot see shape.** The matched control replaces every head's distribution with a one-dimensional curve of its scalar mean. A matched gap near 0 therefore says only that the two scalar scores rank items about equally well.
- **The right statistic exists but is not cited.** The shape test is CE raw versus CE matched (`ce_raw_minus_ce_calibrated`, already in the JSON). It is at or above 0 on 14 of 16 views: the CE-trained head's own K-class distribution loses to a one-dimensional recalibration of its mean, by up to +0.0069 on POPQUORN politeness (intervals above 0 in 5 of 5 seeds). This result is confounded by L2 shrinkage. At L2 = 1e-4, CE raw and CE calibrated are nearly equal (0.1622 versus 0.1625; `probe_matched_l2.txt`).
- **The residual on POPQUORN politeness at K = 2 is neither calibration nor shape.** The residual is +0.0024, with the interval above 0 in 5 of 5 seeds. A binary forecast has no shape beyond its mean, and each head's map is fitted separately. So the residual is a ranking (discrimination) difference between the two objectives' scores: the objective matters beyond calibration there.
- **P1's rationale is wrong.** "A scalar score plus a calibration map spans the Bernoulli family, so the matched gap is zero up to noise" ignores that two different scores give different forecasts unless one is a monotone transform of the other. P1 passed only because the tolerance is 0.003.
- **Fix:** "After each head's scalar mean is recalibrated on validation, almost all of the raw gap disappears. The CE-trained head's own K-class distribution never beats that recalibration, so these L2-shrunk heads show no usable shape learning. A small, significant ranking (discrimination) advantage for the CE-trained score remains on POPQUORN politeness at every K and on DICES-990."

### 8. IMPORTANT: the raw-gap magnitudes, the P2 exception and the P5 failure depend on L2

- **Where:** E40/README.md:5 ("up to 0.128") and :41 ("which is what an uncalibrated output distribution costs").
- **Evidence:**
  - `probe_matched_l2.txt`, POPQUORN politeness, seed 0, L2 = 0.01 versus L2 = 1e-4:

    | K | raw gap at L2 = 0.01 | raw gap at L2 = 1e-4 | matched gap at L2 = 0.01 | matched gap at L2 = 1e-4 |
    |---|---|---|---|---|
    | 5 | +0.0703 | +0.1710 | +0.0010 | +0.0024 |
    | 3 | +0.0252 | +0.1044 | +0.0016 | +0.0025 |
    | 2 | −0.0025 | +0.0601 | +0.0025 | +0.0028 |

  - The sign flips at K = 2, so the only P2 exception and the P5 failure disappear under weaker regularization. The Pearson-trained head's raw normalized RPS rises from 0.2417 to 0.3332 at K = 5.
  - At L2 = 0.01 the Pearson-trained head stays near W = 0. Its training objective (−Corr + penalty) is −0.003, −0.013 and −0.046 on POPQUORN offensiveness, SummEval expert and POPQUORN politeness at K = 2, seed 0 (`probe_pearson.py`).
  - Its raw normalized RPS is within 0.08 of the uniform forecast's on all 16 views (`probe_uniform.py`).
  - The matched-gap headline survived this probe, but it moved closer to the 0.003 tolerance.
- **Fix:** Label the raw-gap magnitudes and the P2 and P5 exceptions as conditional on L2 = 0.01. Run a validation-chosen L2 sweep before quoting "up to 0.128".

### 9. IMPORTANT: P7 is reported more favorably than P5, and one cell was dropped silently

- **Where:** E40/README.md:63 ("supported in 19 of 20 cells"), :87 (Limitation 3 lists only P5 and P6 as failed); `scripts/score_gd_predictions.py:48`.
- **P7 fails as written.** P7 is a universal statement. It fails on SummEval expert at B = 512: the k* head is better, with the k*-head-minus-k1 difference −0.0006 [−0.0011, −0.0001].
- **A cell was dropped silently.** The DICES B = 128 cell is skipped by `continue` because the k* plan is absent (k* = 16 leaves 8 items, below the 16-item minimum). This breaks the preregistration's own rule that "missing cells are reported, not dropped".
- **POPQUORN offensiveness did not use k\*.** It used k = 8, capped from 10, which README:63 does not mention.
- **Six of the 20 cells are on sets with no signal**, where "k = 1 better" means "closer to the constant forecast". On the 14 cells with signal: 11 better, 2 tied, 1 worse.
- **The P7 implementation was post hoc** (see finding 2).
- **Fix:** "P7 not supported as written (fails on SummEval expert at B = 512). k = 1 was better or tied in 13 of the 14 cells with signal. The DICES B = 128 cell could not be run. POPQUORN offensiveness used k = 8, capped from 10."

### 10. IMPORTANT: a risk-estimation formula is used to explain a training result

- **Where:** E40/README.md:63, "Item-to-rating cost ratio zero ... makes one rating per item cost-optimal for the held-out risk".
- **What is wrong:** k_opt = max(1, sqrt(c·σ_w²/σ_c²)) (`sufficiency_and_dollars.py:24-26`) minimizes the variance of an estimated mean loss. It is a test-side precision argument, not a training argument.
- **Evidence from training:** k > 1 was best in several cells: SummEval expert B = 128 and 512 (k = 2), POPQUORN offensiveness B = 128 (k = 2), DICES B = 128 (k = 4), ChaosNLI abductive NLI B = 256 (k = 2).
- **Fix:** "For estimating a held-out risk when prompts are free, one rating per item minimizes variance (closed form). For training, k = 1 was best or tied in 19 of 20 cells here."

### 11. IMPORTANT: the "$0.05 per rating" crowd rate is not a published price

- **Where:** E40/README.md:5 and :69; the legend of figure `label_dollars_vs_researcher_month.png` ("published, POPQUORN"); the header of `scripts/sufficiency_and_dollars.py` ("PUBLISHED").
- **Evidence:**
  - The [POPQUORN paper (LAW 2023)](https://aclanthology.org/2023.law-1.25.pdf) states "Each annotator is paid $12 per hour as suggested by Prolific".
  - For offensiveness it states "The medium time of finishing 50 annotations is 13 minutes".
  - $12/h × 13 min / 50 ≈ $0.052 is a derivation for offensiveness only. Politeness timing is not reported.
  - The repository's own Experiment 23 protocol (PROTOCOL_11-21-2026.md:26) says an hourly wage alone does not identify a per-rating cost.
  - The claim was inherited from Experiment 36 (SANMI_PITCH.md:17, :585).
- **Fix:** "About $0.05 per rating, derived from the published $12/hour and the 13-minute median for 50 offensiveness ratings; an assumption for every other set and task."

### 12. IMPORTANT: the estimation-rate correction is generalized too far

- **Where:** E40/README.md:73, "Sanmi's high-dimension worry therefore applies to log-score and density estimation, not to the squared-error scores".
- **What is wrong:** Claims C9a–C9e test the plug-in frequency estimate of a single distribution from N draws. The excess risk of a learned K-output judge grows with its parameter count for any proper score. Also, "O(K/N) ... wrong for Brier and RPS": O(K/N) is a valid bound, just a loose one.
- **Fix:** "For the plug-in estimate of one item's rating distribution from N ratings, the expected Brier excess is (1 − ‖p‖²)/N ≤ 1/N and the normalized-RPS excess is at most 1/(4N) for every K, while the add-one KL excess is about (K − 1)/(2N). O(K/N) is loose for Brier and RPS. This does not describe how a learned judge's excess risk scales with K."

### 13. MINOR: some Lean statements are weaker than the prose

- **Ceiling theorems.** `ceiling_sq` and `ceiling_rho` (E40/lean/E40_theorems.lean:289-307) bound Cov(f, μ)²/(Var f · Var Y), not Corr(f, Y)². The step Cov(f(c), Y) = Cov(f(c), μ(c)) appears only in the docstring; no joint law of item and rating is defined. Numerics F1 computes the same Cov(f, μ). The prose "squared correlation ... with a single rating" (lines 18 and 298) is therefore only partly machine-checked. Fix: define the pooled pair law and prove the tower step, or reword.
- **Decision accuracy.** `decision_accuracy_not_strict` (lines 129-140) is a single K = 2 instance that proves non-strictness only. "Proper", claimed in its docstring and at README:33, is checked nowhere, in Lean or in numerics.
- **File header.** Lines 6-7 say every theorem is general unless its docstring says "instance". That is false for `brier_not_local_three` (K = 3), `decision_accuracy_not_strict` and `same_two_moments_different_law` (K = 5 fixed numbers). Only `accuracy_prefers_overconfidence` says "instance".
- **Accuracy optimality (nit).** `sampled_accuracy_eq_max_iff` and `truthful_sampled_accuracy_optimal_iff` take any upper bound m. "Optimal" needs m = max p, which is not a hypothesis. The statements are true, but the names promise more than the types.

### 14. MINOR: some numeric checks are weak or tautological

- **E3** (numerics_theorems.py:202-203) defines `exy` as Σμ²/n, which is the conclusion, so it tests nothing.
- **F3** only checks that sqrt(0.5) > 0.5.
- **D3** repeats part of D1.
- **G3** does not test "yet MSE and Brier change".
- **Unchecked claims:** "decision accuracy is proper" and "chi-squared is O(K/N)" (README:73) are checked nowhere.
- So "30 claims, all PASS" overstates how many independent checks there are.

### 15. MINOR: reporting-standard violations (repository CLAUDE.md)

- No figure axis title and none of the three `results/*_tables.md` spells out "ranked probability score (RPS)". "CE" and "ICC" are also not spelled out in the figures.
- The log x axes of `learning_curves_and_saturation.png` and `label_dollars_vs_researcher_month.png` have no faint minor gridlines.
- The learning-curve and reliability-curve figures show no uncertainty.
- The bar in panel B of the strata figure is the mean of per-seed interval endpoints, which is not an interval of anything.
- Figures 2 to 6 do not state how the judge was chosen (objective, optimizer, budget).
- The "learnable gain" is the constant minus the minimum of the test-set curve, so it is selected on test and optimistic.
- The README G-study table does not say it uses all ratings, unsplit. HOW_TO_READ does say so.
- The dollars figure draws stems from an arbitrary 1e-4 baseline on a log axis.

### 16. NIT: stale or inaccurate text

- E40/README.md:121 still says "[judge objectives: experiment 39]", left over from before the renumbering to 40.
- The compile-output header says "Linter warnings are kept below", but there are none.
- E40/HOW_TO_READ_RESULTS.md:20 says the stop rule is "relative tolerance 1e-9". The fits actually stop on scipy's projected-gradient tolerance (`CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL`) with finite-difference gradients (there is no `jac`). This is harmless: tight refits give the same objective.
- README:77 says MSE "equals the Brier score only at K = 2 (two-class Brier is twice the scalar error)". The right word is "proportional".
- The TLDR-end phrase "one rating per item is almost always the better use of a fixed budget" needs "for training a proper-loss head". For the precision of item means, k* applies (Experiment 39 shows the same thing).

### 17. NIT: the P3 decision rule was made concrete without saying so

- `scripts/analyze_k_sweep.py:130` turns "interval excludes 0" into "at least 3 of 5 seeds' intervals above 0".
- It also drops the clause "while its K = 2 view shows about 0".
- Neither change is in the preregistration, and neither is labeled. Neither changes the verdict.

### 18. NOTE: the folder changed during this review

- At about 10:01 PST someone else added five untracked files: `lean/AllocationAndClosedForms.lean`, its compile output, `scripts/check_allocation_and_closed_forms.py`, and two run logs. README does not reference them.
- On a copy, the Lean file compiles (exit 0). All 17 `#print axioms` lines show standard axioms only, and there is no `sorry`. The check script exits 0 with 19 PASS.
- I did not review what those files state.

## (c) Verified OK, with evidence

1. **Preregistration timing.**
   - The reflog shows commit e067462 at 09:06:28 −0800. The result JSONs are dated 09:12:02 and 09:12:04.
   - The rebased commit b811fd9 is content-identical to e067462 (`git diff` is empty).
   - The renumbering diff changes numbers only. The post-preregistration diff of `k_sweep_matched_calibration.py` is infrastructure only: `sys.path`, the VeriBench try/except moved last, an assertion, and an `unavailable` field.
   - The working copy equals HEAD for both committed scripts.
2. **Lean.** My compile took 24 s and exited 0. All 19 `#print axioms` lines show `[propext, Classical.choice, Quot.sound]`. There is no `sorry`, `admit` or `axiom`. The output is identical to the saved one apart from timing. Every other theorem statement matches its prose: `binary_mean_identifies`, `mean_not_identifying` (general K ≥ 3), `brier_binary_local`, `same_two_moments_different_law` (7/40 checked by hand), `total_variance`, `icc_eq_rho`, `cov_sq_le`, `covF_centered`, `betweenVar_eq_varF`, `ceiling_attained`.
3. **Numerics.** Two runs on a copy both exit 0. The outputs are byte-identical to each other and to `logs/numerics_theorems_output_11-27-2026.txt`, with 30 PASS. A1, A3 and C3 test the false claims as failing cases.
4. **Cheap scripts reproduce.** `analyze_k_sweep.py`, `score_gd_predictions.py` and `sufficiency_and_dollars.py` re-run on a copy reproduce all six results files byte-for-byte, and all six PNGs pixel-for-pixel.
5. **K-sweep numbers in README and HOW_TO_READ**, except findings 3 and 4:
   - constant forecast 0.1156 to 0.2501
   - raw gap positive on 15 of 16 views; 0.1282 and 0.1277
   - Pearson-trained raw 0.2274 to 0.2598
   - largest absolute matched gap 0.0024; largest at K ≥ 3 is 0.0016
   - matched/raw ratios 5.46% and at most 1.25%
   - POPQUORN politeness intervals above 0 in 5 of 5 seeds; DICES in 1 of 5
   - every P1 to P5 value
   - 80 cells, at most 19 and 24 iterations, all marked successful; the 35-iteration and 300-iteration runs have identical rows
6. **Optimizer.** The CE and Pearson heads sit at the optimum of their penalized objectives. Tight-tolerance refits give the same objective to about 1e-9 and the same test normalized RPS to about 1e-6 (seed 0; 2 CE views, 3 Pearson views). `fe.MAXITER` is read at call time.
7. **G-study and D-study.** All seven rows of ICC(1) with its CI and k* with its CI match the JSON. Recomputing k* = ceil(4(1−ICC)/ICC) gives 10, 6, 2, 5, 16, 5 and 3. The ceiling table, the leave-one-out formulas and the finite-n formula ρ/sqrt(ρ+(1−ρ)/m) are correct. P6 is scored as written.
8. **Dollar arithmetic** in README:69: $6.40 to $25.60; $640 to $2,560; 0.021 to 0.085; $51.20, $5,120 and 0.171; 90.1 h and 841.5 h; $4,507 to $42,075; 0.15 to 1.40.
9. **No leakage.**
   - Splits are by prompt group, with `assert_no_prompt_leakage`.
   - The principal-component basis and the constant marginal use training data only.
   - The calibration standardization moments and the map are fitted on validation only, then applied to test.
   - Entropy strata come from half A and are scored on half B.
   - The paired bootstrap resamples test items, using the same indices for both heads.
10. **Formulas.** The re-binning maps and the K assertion, the power formula, the k_opt formula and the saturation function logic are all correct as defined.
11. **Privacy.**
    - I built the patterns at run time from the sister roster sources and from the public datasets' rater ids. Counts only:
      - roster full names 0
      - roster ids 0
      - long public ids 0
      - 34 roster-token hits, all generic words ("rater" and "simulated"; checked with masked contexts)
      - 118 quoted short-id hits, all numeric JSON keys (learning-curve N, budgets, cost ratios)
    - The PNG metadata holds only "Software".
    - Two addressee names (the user and the advisor) appear in README and the preregistration file (3 and 8 mentions), not as raters.
12. **Cited work exists.**
    - The Experiment 36 O(K/N) row exists (RESULTS_SUMMARY.md:561, SANMI_PITCH.md:610) and is inconsistent as described.
    - The cited Lean names exist: Experiment 35 BrierCorr.lean, and Experiment 36 SanmiPitch.lean:527 and :669.

## (d) Could not check

1. Full reruns of the two heavy scripts (excluded by the instructions). My single-view probes reproduced repository values exactly, for example the learning-curve points 0.1738, 0.1718 and 0.1716.
2. The VeriBench views (HTTP 404).
3. An L2 sweep over all views and seeds. I probed only POPQUORN politeness, seed 0.
4. What the files added at 10:01 state (see finding 18).
5. The sister-repository inputs: 4.0 minutes per rating and the 37.3-minute assumption. I saw the pinned aggregates only.
6. The journal citations in "Positioning". Volumes and pages match my knowledge, but I fetched none.
7. Rendered text inside the PNGs. I checked them visually only.

**Process disclosure.** One directory listing of the sister checkout printed file paths that contain rater names into my tool log. Nothing from it went into any file or into this report.

**Experiment 39, one line.** The two experiments converge: in Experiment 39 Study A, the RPS oracle picks the maximum-item design while item-mean precision wants more raters, the same direction as P7. Experiment 39 concludes "proper-score design does not replace D-theory". Experiment 40's answer to the user's question 2 should cite that, and Experiment 40 never mentions Experiment 39.

TL;DR: The K-sweep matched-gap headline, the Lean and the numerics hold. The sufficiency and dollar answers are L2 = 0.01 artifacts, and several sentences misstate numbers, preregistration status, the calibration control, the $0.05 "published" rate, and the unreported MSE-trained head.
