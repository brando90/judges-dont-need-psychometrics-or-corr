# Independent QA review of Experiment 39 (commit 1cd4fc6)

Scope: `experiments/39_proper_score_vs_d_study_budget_allocation_start_26Nov2026/` at commit `1cd4fc6`. All runs were done in a copy at `/tmp/opus_review_exp39/work`; nothing under `/tmp/e39_cursor` or `/home/claude/judges-dont-need-psychometrics-or-corr` was modified. Review scripts are in `/tmp/opus_review_exp39/scripts/`, their outputs in `/tmp/opus_review_exp39/outputs/`, and an independent Lean check in `/tmp/opus_review_exp39/lean/`.

## (a) Verdict

The experiment reproduces exactly (all six input hashes match, the Study A summary is identical in all 42,451 values, the Study B table is byte-identical, 25/25 checks pass) and every headline count matches the committed JSON, so nothing is fabricated; but the main interpretive claims do not hold as written, because the "modal RPS oracle prefers more items" and "U1 agrees with the RPS oracle in only 0.00–0.36" statistics are produced by an undeclared tie-break (75% of comparable replications have an exact tie for best test RPS), the two D-study "selectors" are constant rules (so the item-mean and grand-mean MSE wins hold by construction), and the proper-score selector uses 4.5 to 7.7 times more ratings than the 80 × 16 pilot it is said to use. What survives is narrower: for this 27-transform recalibration of frozen judges, max-items designs give slightly lower held-out RPS than max-raters designs (0.4–2.6% of total RPS; the fixed max-items rule does it in 12/20 cells and the selector in 10/20, which is not the protocol's pre-declared majority), the Study B closed forms are right (now also Lean-checked), and "proper-score design does not replace D-theory" is a framing choice rather than a measured result.

## (b) Findings, most severe first

### 1. Blocker (for the oracle statistics): the RPS oracle is mostly decided by an undeclared "more items" tie-break

- **Where:** `src/study_a.py:265-279` (`pick_min`), `:290` (RPS oracle, ties to more items), `:292` (item-mean oracle, ties to more raters); `PROTOCOL.md:51` (no oracle tie rule declared); `README.md:28`; `REPORT.md:19`, `:55`, table columns "PS / RPS-oracle" and "PS = RPS-oracle" (`REPORT.md:66-87`); the Experiment 39 row of `experiments/README.md` ("the modal ranked probability score (RPS) oracle prefers more items").
- **What is wrong:** The 27-transform head is discrete, so different designs often fit the same transform and produce identical test RPS. In 75.0% of comparable replications at least two designs tie exactly for the best test RPS, and the oracle then takes the design with more items. The item-mean oracle never ties (0.00 in all 20 cells) and its tie rule goes the other way. So "the modal RPS oracle is the max-item design", "U1 agrees with the RPS oracle in only 0.00–0.36", "U2 agrees 117/150", and the "PS = RPS-oracle" column are largely produced by the tie rule, and "the two oracles disagree systematically" is half by construction. The mean-RPS paired comparisons are not affected.
- **Evidence:** `python3 scripts/oracle_ties.py work` → `overall fraction of comparable replications with an exact tie for best test RPS: 0.750`. Columns are: tie rate | U1 agrees with RPS oracle (as reported) | U1 design tied-or-best | max-items tied-or-best | max-items unique best:
  - `dices990 512 | 1.00 | 0.00 | 1.00 | 1.00 | 0.00`
  - `dices350 256 | 0.97 | 0.00 | 0.87 | 0.91 | 0.00`
  - `chaosnli_snli 512 | 0.93 | 0.01 | 0.93 | 0.99 | 0.07`
  - The max-items design is the unique best in 0.00–0.44 of replications (at most 0.07 in 13 of 20 cells).
  - Tie-neutral pairwise comparison (`scripts/tie_neutral.py`), averaged over the 20 cells: the max-items design has strictly lower test RPS than the U1 design in 22% of replications, is exactly equal in 65%, and is strictly worse in 13%.
  - The selector's own tie rule (declared in `PROTOCOL.md:50`) has the same effect at large budgets. Its mean validation RPS ties exactly for best in some replications, and then the "more items" rule decides (`scripts/ps_val_ties.py`): SNLI 512: 39/150; alphaNLI 512: 32/150; DICES-350 256: 19/150 and 512: 47/150; DICES-990 256: 54/150 and 512: 96/150; never on MNLI-m. So "the modal PS design is the max-items design" at B ≥ 256 is also partly set by a rule.
- **Proposed fix:** Declare the oracle tie rule and report P(strictly better) / P(exact tie) / P(strictly worse), or give tied designs fractional oracle credit, or compute oracle statistics on the continuous A1 mixture. Remove "the modal RPS oracle is the max-item design" and the RPS-oracle agreement rates, or recompute them tie-neutrally, in README, REPORT (verdict bullet 3, line 55, table) and the experiments index row.

### 2. Important: both D-study "selectors" are constant rules, so the MSE results hold by construction and the D-theory verdict is not tested

- **Where:** `README.md:5`, `:28`, `:41`; `REPORT.md:15-20`, `:44-45`, `:55-60`, `:162`; `PROTOCOL.md:48-49`; `src/study_a.py:285-287`.
- **What is wrong:** U1 chose the max-raters design in 150/150 replications of all 30 dataset-budget cells. U2 chose the max-items design in 150/150 replications of all 20 comparable cells (the pilot estimate of v_u was positive every time). Neither rule ever used the variance-component values to change a decision. As a result:
  1. "U1 beats U2 on item-mean MSE 20/20" and "U2 beats U1 on grand-mean MSE 18/20" follow from finite-population sampling variance. The analytic expectation gives the same ordering in 20/20 cells; observed item-MSE means are within 5.1% of the analytic values; the two grand-MSE "ties" at B = 64 are power failures, not reversals.
  2. "U1 matches the item-mean oracle in 0.83–0.97" is the same fact plus noise.
  3. At fixed B the U2 criterion is v_u/n_i + constant, so "a grand-mean D-study already points at the same corner" (REPORT:20, :162) is algebra, not evidence.
  4. "Proper-score design does not replace D-theory" cannot be supported by this design. A fixed rule ("always max raters") reproduces U1 exactly. A cross-validated selector scored by item-mean squared error (a proper score for the mean) would, by the same variance formula, also pick max raters (reasoned, not run). The experiment shows that the optimal allocation depends on the target; it does not show that estimating variance components adds information.
- **Evidence:** `verify.py` check `u1_always_max_raters` PASS. `scripts/audit_counts.py` raises no "U2 not always max items" flag in any comparable cell. `scripts/analytic_mse.py` → `analytic ordering (U1<U2 on item MSE and U2<U1 on grand MSE) holds in 20/20; max relative gap observed vs analytic item-MSE mean: 5.1%`, for example `chaosnli_snli B=64: item MSE U1 obs 0.05684 vs analytic 0.05773; U2 obs 0.10878 vs 0.11310`. Lean theorems `item_mean_more_raters` and `grand_mean_more_items` (`lean/VDelta.lean`) compile.
- **Proposed fix:** Label these results as analytic sanity checks and state that U1 and U2 did not depend on the pilot in any replication. Suggested verdict: "The best split of a fixed rating budget depends on the target. For held-out RPS of this recalibrated-judge predictor, more items is better or tied. For per-item mean precision, more raters is better, as v_e/n_r implies. Neither D-study rule changed its decision with the pilot estimates, so this experiment does not test whether variance-component estimation adds value beyond a fixed rule." A real test needs a decision that depends on the estimates, for example crossed rater effects on DICES-350 or a reliability-target k\*.

### 3. Important: the proper-score selector uses far more rating data than the 80 × 16 pilot it is described as using (undisclosed protocol deviation)

- **Where:** `PROTOCOL.md:50` ("From the same pilot only"); `README.md:22`; `REPORT.md:36`, `:46`; `src/study_a.py:229` (`val_h = h_full[val_items]`) and `:236` (training ratings drawn by `sample_counts` from each item's full pool).
- **What is wrong:** The validation targets are the full rating histograms of the held-out pilot items: 100 ratings each on ChaosNLI, 123 on DICES-350, 69–76 on DICES-990. Inner training ratings are also resampled from each item's full pool, not from its 16 pilot ratings. The selector therefore uses about 5,800–9,800 ratings of the 80 pilot items, versus 1,280 in the pilot table. U1 and U2 use only the 80 × 16 table, and ignore its values. Separately, the 1,280-rating pilot exceeds every deployment budget (32–1,024) and is not counted in B, while U1 and U2 need no pilot at all.
- **Evidence:** `scripts/extra_checks.py` → `dices350: pilot table 80x16 = 1280 ratings; PS selector draws from up to 80 x mean pool 123.0 = ~9840 ratings`.
  - I re-ran a protocol-faithful version (`scripts/ps_pilot_only.py`, same seeds; validation on the 16-rating pilot histograms; training subsampled from the 16 pilot ratings). Results are in `outputs/ps_pilot_only_analysis.txt`:
    - `U1 vs PS-pilot-only: {'PSpilot beats U1': 9, 'tie': 11}`
    - `U2 vs PS-pilot-only: {'U2 beats PSpilot': 4, 'tie': 16}`
    - The original selector and the pilot-only selector tie in 20/20 cells.
  - The fixed max-items rule (identical to U2) beats U1 in 12/20 cells (`extra_checks.py`), more often than the selector's 10/20.
- **Proposed fix:** Disclose the deviation and report the pilot-only selector (9/20 wins over U1; U2 better in 4/20), or amend the protocol text. Report the pilot cost next to B, and state that the fixed max-items rule, which needs no pilot, does at least as well as the cross-validated selector.

### 4. Important: the intervals leave out test-item sampling, there is no multiplicity correction, and the repository's item-clustered bootstrap rule is not met

- **Where:** `README.md:22`; `REPORT.md:24`, `:49`, `:91` (flagship means); Figure 1 bands and caption (`REPORT.md:140`). `HOW_TO_READ_RESULTS.md:46` is the only disclosure. The project's `CLAUDE.md` (11-26-2026) requires item-clustered bootstrap 95% intervals.
- **What is wrong:** The Student-t intervals are over 150 replications with a fixed frozen test set (105–180 items) and a fixed dev pool. They describe only pilot and training-draw variability, and README and REPORT do not say so.
  - For paired U1−PS differences, a two-way bootstrap (replications × test items) widens the intervals modestly.
  - For absolute RPS levels (flagship means, Figure 1 bands), the reported intervals are 7–16 times too narrow as statements about new items.
  - Per-item scores are not stored, and the Study A per-replication CSV is not committed. I rebuilt per-item test RPS for all 27 transforms from the data and the stored transform indices; this matches the stored per-replication RPS to 1e-12 in all 20 cells.
- **Evidence:** `scripts/item_bootstrap.py work` (4,000 draws):
  - The two-way interval is wider than the Student-t interval by a median factor of 1.14 (range 1.01–1.73).
  - DICES-350 B = 256 goes from [+0.00008, +0.00159] to [−0.00017, +0.00244], so 9/20 cells stay significant instead of 10/20.
  - SNLI B = 64: [+0.00078, +0.00378] becomes [+0.00078, +0.00389]. DICES-350 B = 128: [+0.00040, +0.00253] becomes [+0.00001, +0.00360].
  - Absolute levels (`extra_checks.py`): SNLI B = 64, U1 mean 0.11765, Student-t ±0.00138 versus two-way ±0.00986 (×7.1). DICES-350 B = 128: ±0.00093 versus ±0.01454 (×15.6).
  - Multiplicity (`audit_counts.py`), for the 10 "PS beats U1" cells:
    - Holm keeps 3 with m = 20 (MNLI-m 256, alphaNLI 256, DICES-990 256), or 4 with m = 19 after excluding the exact-tie cell (adds SNLI 64).
    - Benjamini-Hochberg at q = 0.05 keeps 6 (adds SNLI 512 and DICES-350 128).
    - For "U2 beats PS", Holm keeps 1 of 2 (DICES-350 128, p = 0.00117).
- **Proposed fix:** Use the two-way bootstrap for every interval, or at least add the test-item component. State the resampling unit in README, REPORT and the figure captions. Store per-item test scores (or the per-replication transform index plus per-item per-transform scores) and commit the Study A replication CSV. Report Holm- or BH-adjusted counts next to the raw 10/20.

### 5. Important: the pre-declared decision criterion was not met, and the freeze cannot be verified

- **Where:** `PROTOCOL.md:3`, `:61`; `AMENDMENT_A1.md:3`; `REPORT.md:9`, `:162`; `README.md:7`; git history.
- **What is wrong:**
  1. **Criterion not met.** The protocol's own test requires significantly lower test RPS for the selector in a "pre-declared majority" of cells. 10/20 (or 10/30 counting singleton budgets) is not a majority, and the second condition (the RPS oracle's n_r disagrees with the item oracle's n_r) is compromised by finding 1. The report never evaluates the criterion, yet still says "a proper-score selector recovers that split" (`REPORT.md:162`).
  2. **Freeze not verifiable.** Every file in the folder was first committed on 11-27-2026 between 09:44:09 and 09:45:11 PT, about 17 hours after the run logged 11-26 16:23–16:29 PT. README and REPORT with results (`a58d617`, 09:44:09) were committed before PROTOCOL.md and AMENDMENT_A1.md (`fb652b3`, 09:44:17). No earlier commit mentions the folder or its seed, and no smoke-test output for A1 exists.
  3. **Other departures from the protocol:**
     - The selector's data use (finding 3).
     - Undeclared oracle tie rules (finding 1).
     - An undeclared fold-assignment seed stream `[20261126, 6, k, rep]` (`study_a.py:219`).
     - The hash path is given as `data/raw/SHA256SUMS.txt` (`PROTOCOL.md:19`) but committed as `data/SHA256SUMS.txt`.
     - The t-grid is concentrated at 0.40–0.60 because of the false equal-v_Δ premise, and no grid point lies at the true crossovers. This is disclosed only as a "typo".
     - The mid-point check in `verify.py` was rewritten after the results were seen (disclosed).
- **Evidence:** `git -C /tmp/e39_cursor log --format='%h %ad %s' --date=iso -- experiments/39_.../` gives 8 commits, all 11-27 09:44–09:45. `git log --all -S proper_score_vs_d_study`: the earliest commit is `a58d617`. Run times are in `results/run_ab_11-26-2026.txt`.
- **Proposed fix:** State in the REPORT that the pre-declared majority criterion was not met. Drop "frozen before measurement" or add evidence, for example a PROTOCOL.md hash recorded in a pre-run commit or log. List all departures in one section.

### 6. Important: the documented RPS formula is not the one computed, and the effect sizes are small relative to what can be learned

- **Where:** `PROTOCOL.md:12`, `:55`; `HOW_TO_READ_RESULTS.md:20`; `src/study_a.py:63` (identical to Experiment 28's `lib28.rps_of`).
- **What is wrong:**
  - The code computes mean_k [F² + G(1 − 2F)] = E over y drawn from the item's histogram of (1/(K−1)) Σ_k (F(k) − 1{y ≤ k})². That is the normalized individual-rating RPS averaged over all of a test item's ratings, with items weighted equally.
  - The documents state (1/(K−1)) Σ_k (F(k) − G(k))². That omits the forecast-independent item term (1/(K−1)) Σ_k G(k)(1 − G(k)).
  - So the paired differences are correct, and the absolute levels and the "typical values 0.11–0.23" are those of the individual-rating metric (the repository's primary-metric form), but the formula is labelled wrongly.
  - `HOW_TO_READ_RESULTS.md:20` also gives K = 3 for ChaosNLI (alphaNLI has K = 2) and calls the NLI labels ordinal. They are nominal, so RPS depends on the imposed order. CE and TVD agree in direction in all 10 significant cells, which softens this.
- **Effect size:**
  - The irreducible term averages 0.0961 (SNLI), 0.1387 (MNLI-m), 0.0878 (alphaNLI), 0.1890 (DICES-350) and 0.1806 (DICES-990).
  - The significant U1−PS gaps (0.00047–0.00294) are 0.37–2.6% of U1's total RPS and 2.2–11.7% of U1's reducible RPS.
  - On DICES-350, U1 at B = 64 and 128 (0.2296, 0.2289) scores worse than a constant forecast equal to the dev-pool base rate (0.2275). That constant uses far more ratings than B, so it is a reference only.
  - The verdict's "more items is the better split" is right in direction, but "systematically disagree" and "does not replace" overstate an effect of 0.0005–0.003 RPS.
- **Evidence:** `scripts/item_bootstrap.py`: the reconstruction assert passes; see the `mean irreducible constant=` lines and the `gap/excess` columns.
- **Proposed fix:** Correct the formula and say it is the individual-rating RPS averaged over each item's ratings. Report the irreducible floor and the reducible excess, and state effect sizes relative to the excess in the verdict. Fix K for alphaNLI and note that the NLI labels are nominal.

### 7. Important: a fresh clone cannot reproduce or verify the run as documented

- **Where:** `data/README.md:5`, `:21-25`; `REPORT.md:8`, `:158`; `src/prepare_data.py`; `src/verify.py:77`.
- **What is wrong:**
  - `data/README.md` says `prepare_data.py` "writes `data/raw/`" and downloads from "the commit pinned in `src/prepare_data.py`" and "the raw GitHub URL pinned in `src/prepare_data.py`". The script has no URL, no commit pin and no download code. It only reads six files from `data/raw/` under specific names; the DICES files must be called `diverse_safety_adversarial_dialog_350.csv` and `_990.csv`.
  - `verify.py` reads `results/study_a_replications.csv`, which is not committed (REPORT:158 says per-replication CSVs are committed). So `verify.py` fails on a fresh clone until Study A is re-run.
  - The Experiment 28 forecast file is already in the repository (`experiments/28_.../forecasts/forecasts_11-23-2026.csv`, sha256 `54fbff95…`), contrary to "downloaded, not vendored".
- **Evidence:** `grep -n "http\|huggingface\|urllib\|requests\|download\|wget\|curl" src/prepare_data.py` finds nothing (exit 1). `git ls-tree` of `results/` lists no `study_a_replications.csv`. Running `verify.py` on the committed results gives `FileNotFoundError: ... results/study_a_replications.csv`.
- **Proposed fix:** Either add download code with pins, or change the README to "place these six files here". Commit the 7.8 MB Study A replication CSV, or have `verify.py` skip the row-count check when it is absent. Point to the in-repository Experiment 28 file.

### 8. Minor: Study B wording and the "small pilot hurts" claim

- **Where:** `README.md:24`, `:30`; `REPORT.md:22`, `:130`, `:132`, `:142`; `src/make_figures.py:169`.
- **What is wrong:**
  1. **"Falls to near chance."** At t = 0.30 and 0.70 the third design is picked in only 13/1000 and 9/1000 replications, so the choice is effectively between two designs and chance is 1/2. The observed 0.50–0.64 is about a coin flip, not three-way chance (1/3). The oracle's advantage at those points is only 0.0125 (3.2% of v_Δ(4×4)), so wrong picks are cheap.
  2. **"W-shaped."** Supported at the grid level: both minima fall at t = 0.30 and 0.70 for all five pilots, with a local maximum at 0.45–0.50. The crossovers 4/15 and 11/15 themselves are not on the grid.
  3. **"A small pilot can be significantly worse than equal allocation."**
     - On (4/15, 11/15) equal 4×4 is the oracle, so any wrong pick has positive true regret. The sign is guaranteed; only the size is empirical.
     - The noise-free true regret is already in the JSON (`true_regret`): at t = 0.5 with pilot 16×4 it is 0.0245 (6.3% of 0.3875), versus the noisy observed +0.040.
     - Six cells are nominally worse: t0.45_64x4, t0.50_16x4, t0.50_32x8, t0.55_64x4, t0.60_16x4 and t0.60_64x4. Only two are reported. Three of them use the 64-object pilot 64×4, so the driver is the number of facet levels (4), not pilot size.
     - None survive Holm over the 55 cells (smallest p = 0.00295 > 0.05/55 = 0.00091). BH keeps 4 of the 6, not including 32×8.
  4. **"As the facets become harder to tell apart"** (README:24, Figure 2 title) repeats the false premise. At t = 1/2 the facets are symmetric, yet every t in (4/15, 11/15) has the same unique optimum, 4×4.
- **Evidence:** `scripts/extra_checks.py` and `outputs/audit_misc.txt` (P(match) grid and choice counts).
- **Proposed fix:** Suggested wording: "Near the crossovers (grid points t = 0.30 and 0.70) REML picks the oracle in 50–64% of replications, about a coin flip between the two nearly tied designs; a wrong pick there costs 0.0125 in v_Δ (3%)." Report the true regret, attribute the mid-region loss to 4-level facets, apply Holm or BH, and fix the title.

### 9. Minor: documentation that contradicts the code

1. `REPORT.md:36`, `:140` and `HOW_TO_READ_RESULTS.md:38` say deployment training uses "only the remaining dev pool". The code draws from the full dev pool, pilot items included (`study_a.py:251`; `PROTOCOL.md:44` allows this).
2. `REPORT.md:152` says deployment draws ratings "with replacement". The code samples without replacement (`study_a.py:124`).
3. `HOW_TO_READ_RESULTS.md:5` and `:24` say item-mean and grand-mean MSE are computed "over held-out test items". The code uses the deployment dev items against the mean of their remaining ratings, and the dev-pool mean of item means (`study_a.py:256-257`). `PROTOCOL.md:56-57` is correct.
4. `AMENDMENT_A1.md:7` says "Laplace-smoothed". The code adds 0.5 per category (`study_a.py:90`), which is Jeffreys (Krichevsky-Trofimov) smoothing.
5. The "transform tie" column (`REPORT.md:66`) is the U1-versus-U2 transform agreement (`primary_transform_tie_u1_u2`), printed next to U1−PS differences.
6. `REPORT.md:44` says "verified 150/150 cells". It is 150/150 replications in each of 30 cells.
7. The Figure 1 x-axis label is "Held-out design budget" (`make_figures.py:104`). B is the training (deployment) budget, not a held-out quantity.
8. `REPORT.md:8` says the Experiment 28 forecasts are "downloaded, not vendored". They are in the repository.

### 10. Minor: U2's "max items" prescription assumes no rater main effect

- **Where:** `REPORT.md:45`, `:162`; `PROTOCOL.md:48-49`.
- **What is wrong:** v_u/n_i + v_e/B is the grand-mean error variance when raters are nested in items, which is how this deployment samples. With crossed raters (DICES-350 is fully crossed), absolute error is v_u/n_i + v_r/n_r + v_e/B, which has an interior optimum. Example (hand arithmetic): v_u = 0.3, v_r = 0.05, v_e = 0.4, B = 64 gives 0.0375 for 16×4 and 0.0406 for 32×2. The one-facet ANOVA ignores DICES-350's rater identities.
- **Proposed fix:** Add a caveat that U2 = max items holds only for nested or freshly sampled raters.

### 11. Minor: the repository's proof rule is not met for the formal claims

- **Where:** `qa/CHECKS_11-27-2026.md:13-21`; `checks/check_vdelta_closed_forms.py`.
- **What is wrong:** The project's `CLAUDE.md` (11-26-2026) requires both Lean 4 and Python for formal claims. The closed forms, crossovers and uniqueness are checked only in Python, through the same `gtheory_lib.d_study` used in the run, so the check is not independent. The report correctly labels it "numerics only".
- **Evidence (partial remedy):** I wrote an independent core-Lean check, `lean/VDelta.lean`, with its own definition of v_Δ. It proves:
  - `v_8x2`, `v_4x4`, `v_2x8` for all rational t;
  - `cross_lo`, `cross_hi`, `half_values`;
  - `mid_unique` on the open interval (4/15, 11/15);
  - `low_8x2`, `low_8x2_vs_2x8`, `high_2x8`;
  - `grand_mean_more_items` and `item_mean_more_raters`, plus one concrete instance.

  `~/.elan/bin/lean VDelta.lean` exits 0, and every `#print axioms` lists only `propext`, `Classical.choice`, `Quot.sound` (`outputs/lean_output.txt`).
- **Proposed fix:** Add a Lean file to `checks/`. This one compiles with core Lean 4.36.0 and needs no Mathlib.

### 12. Nits

1. Rounding in the Study B table: t = 0.15, pilot 16×4 shows "−0.061" for −0.06048 (should be −0.060); t = 0.70, pilot 64×8 shows lower bound "−0.004" for −0.00348 (should be −0.003).
2. `results/study_a_summary.json` contains 1,017 `NaN` tokens, so it is not strict JSON.
3. REML did not converge in 7 of 11,112 Study B replications. Those replications were still used, and this is not reported.
4. `study_a.py:199` computes `universe_mean` wrongly and is overwritten at `:201` (dead code).
5. For alphaNLI (K = 2) the quadratic coefficient has no effect, so the family has 9 distinct transforms, not 27 (max |Δq| = 2.2e-16). Harmless with the earliest-index tie rule.
6. `anova_one_facet` (`study_a.py:135-149`) uses the crossed residual with (n_i−1)(n_r−1) degrees of freedom, not the nested ratings-within-items estimator the protocol describes. Immaterial (one pilot: v_e 0.2250 vs 0.2263), because U1 and U2 ignore the values.
7. The pilot label "32x8" means 32×8×4.

## (c) Verified OK

1. **Inputs.** The five dataset files in `/tmp/e40_data_cache` and the in-repository Experiment 28 forecasts match `data/SHA256SUMS.txt` (`sha256sum -c` gives six OKs). The forecasts are at `experiments/28_.../forecasts/forecasts_11-23-2026.csv` (`54fbff95…`). The 11,348 valid judge-item rows equal the manifest sum.
2. **Data preparation.** `prepare_data.py` regenerates `splits.json` and `manifest.json` identical to the committed files.
3. **Study A re-run** (311.7 s). Software here was numpy 2.5.3, scipy 1.18.1 and pandas 3.0.5, versus the reported numpy 2.5.5 and pandas 3.1.1. `study_a_summary.json` is identical in all 42,451 values except runtime and the numpy version string.
4. **Study B re-run** (11.1 s). The summary is identical, and `study_b_replications.csv` is byte-identical (sha256 `bb0ff306e437…`).
5. **`verify.py`.** 25/25 PASS, and `verification.json` is identical to the committed one (this needs the regenerated Study A CSV).
6. **`checks/check_vdelta_closed_forms.py`.** Exit 0 on two runs with identical output, matching the committed output.
7. **Study B mathematics.**
   - The closed forms v_Δ(8×2) = 23/80 + 3t/8, v_Δ(4×4) = 31/80 and v_Δ(2×8) = 53/80 − 3t/8 are correct.
   - The crossovers at 4/15 and 11/15 are correct, and "4×4 is uniquely optimal at t = 1/2 (31/80 vs 19/40)" holds, as does uniqueness on the whole open interval. Checked by hand and in Lean.
   - The D-study absolute error includes the facet main effects (`gtheory_lib.py:151-156`), and evaluation redraws the facet effects (`study_b.py:46-64`).
   - The REML constraint is as stated (`gtheory_lib.py:249`).
8. **Headline counts recomputed from the committed JSON.**
   - PS vs U1: 10 / 0 / 10.
   - U2 vs PS: 0 PS wins, 2 U2 wins, 18 ties.
   - ps_mix vs U1: 12 / 0 / 8.
   - Item-mean MSE: 20/20. Grand-mean MSE: 18/20 (ties are DICES-350 and DICES-990 at B = 64).
   - Within the 10 RPS-significant cells, CE is significant in 9 and TVD in 8, with the exceptions as stated and no significant contradictions.
9. **Study A table and report numbers.** All 20 rows (modes, means, CIs, p-values, agreement rates, tie column) match the JSON, as do the flagship means, the item- and grand-MSE examples, and the U2-vs-PS intervals.
10. **Study B table and reproduction.** The table rows and reproduction numbers match, apart from the two rounding nits. The published Experiment 04 values (256/256; −0.107 [−0.149, −0.065]; −0.072 [−0.108, −0.035]) match. I recomputed Wilson and Student-t intervals by hand:
    - t = 0.5, pilot 16×4: 144/200, [0.6541, 0.7776]; +0.03997 [+0.01379, +0.06616], p = 0.00295.
    - t = 0, pilot 64×8: −0.1194 [−0.1636, −0.0752], p = 2.32e-7.
11. **Paired intervals and selector rebuilt independently.**
    - U1−PS intervals recomputed from per-item, per-transform test RPS and the stored transform indices agree (SNLI B = 64: +0.00228 [+0.00078, +0.00378]).
    - A separate re-implementation of the selector's inner cross-validation (`scripts/ps_val_ties.py`) reproduces the stored choice in 3,000/3,000 replication-cells (150/150 in each of the 20 comparable cells).
12. **Code paths.**
    - No test leakage: the pilot, the inner CV and deployment draw only from `dev_idx` (`study_a.py:214`, `:228`, `:251`). Test histograms are used only to score fitted forecasts and for the declared oracles.
    - The transform family, floor and tie rule are identical to Experiment 28's `lib28.py`, and the RPS function is identical to `lib28.rps_of`.
    - Seeds match PROTOCOL.md.
    - "U1 minus PS, positive means PS better" is implemented correctly (`study_a.py:373-377`).
13. **W shape at grid level.** Both minima fall at t = 0.30 and 0.70 for all five pilots.
14. **Privacy** (counts only; patterns built at run time).
    - 0 matches for any of the 283 DICES rater ids.
    - 0 email addresses, 0 home-directory paths, 0 @handles.
    - 6 person-name token matches, all the repository owner's handle inside the github.com doc-link URLs.
    - Rater ids exist only in the gitignored `data/processed` bundle.

## (d) Could not check

1. That PROTOCOL.md and AMENDMENT_A1.md predate the measurement, and that the A1 smoke test happened: no pre-run artifact exists in git.
2. Where the data were downloaded from (HuggingFace commit, DICES URL): the code has no pins. I used cached copies whose hashes match.
3. Per-item test scores from the original run: they are not stored. My item bootstrap rebuilds them (checked to 1e-12 against the stored replication RPS). I did not bootstrap the A1 mixture or the CE and TVD differences.
4. Whether the conclusions change under an order-free score (Brier) for the nominal NLI labels: not run.
5. Byte identity of the SVG figures: the regenerated SVGs differ in embedded font glyph paths (554 vs 545 `<path>` elements in Figure 1). I inspected the regenerated PNGs visually; they are drawn from identical summaries.
6. Commit `593bf45` (11-27 10:06, Google Doc link) is later than the reviewed commit and was not inspected.

## Comparison with Experiment 40 (secondary)

Experiment 39's U2 result and Experiment 40's result agree, and in fact share the same algebra. At a fixed budget B = N·k, Experiment 40's variance of a clustered mean loss, (σ_c² + σ_w²/k)/N, equals σ_c²/N + σ_w²/B. That is exactly U2's criterion v_u/n_i + v_e/B, so both say that when items are free, the budget should go to items. Experiment 39's mean-RPS results point the same way as Experiment 40's "one rating per item beats or ties k\* in 19/20 cells".

Experiment 39 cannot speak to k = 1, though:

- Its grid has no n_r = 1 and caps n_i at 64, so at B = 256 and 512 its "max-items" designs still use 4 and 8 raters.
- With the discrete 27-transform head, the max-items design is strictly better than the max-raters design in only 22% of replications, with exact ties in 65%.
- Its RPS gains (0.0005–0.003) are the same size as Experiment 40's matched gaps and learnable gains.

Experiment 39 contradicts no number in Experiment 40's README. Its only argument that D-theory is still needed is the per-item-mean target, which Experiment 40 already concedes ("the classical k\* answers a different question, the reliability of each item's mean"). The tension is in framing only: Experiment 39 says "does not replace D-theory", while Experiment 40 asks whether D-theory is needed once a proper loss is used.

The two experiments also have complementary interval limits. Experiment 40's training comparisons resample test items but not training draws (its limitation 8). Experiment 39's resample training and pilot draws but not test items.

## Review artifacts

- Re-run copy: `/tmp/opus_review_exp39/work/`, including the regenerated `results/study_a_replications.csv`.
- Committed outputs saved before the re-run: `/tmp/opus_review_exp39/committed_results/`.
- Scripts: `/tmp/opus_review_exp39/scripts/`:
  - `audit_counts.py`, `audit_table.py`, `audit_misc.py`
  - `analytic_mse.py`, `item_bootstrap.py`, `extra_checks.py`
  - `oracle_ties.py`, `tie_neutral.py`
  - `ps_pilot_only.py`, `analyze_pilot_only.py`, `ps_val_ties.py`
  - `privacy_scan.py`
- Outputs: `/tmp/opus_review_exp39/outputs/`.
- Lean: `/tmp/opus_review_exp39/lean/VDelta.lean` and `lean_output.txt`.

TL;DR: The numbers reproduce exactly, but three headline claims are produced by the design rather than measured (a tie-break that manufactures "the RPS oracle prefers more items", D-study rules that never use the data, and a selector given more data than its pilot). The defensible result is small: more items beats more raters for held-out RPS by 0.4–2.6% of total RPS.
