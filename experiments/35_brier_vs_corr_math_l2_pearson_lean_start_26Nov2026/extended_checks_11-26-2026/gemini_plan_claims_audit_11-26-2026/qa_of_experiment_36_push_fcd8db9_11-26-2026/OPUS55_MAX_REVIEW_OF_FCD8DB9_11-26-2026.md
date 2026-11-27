# Opus 5.5 maximum-effort review of Experiment 36, commit `fcd8db9`, as the agent delivered it | 11-26-2026

**What this file is.** The final report of the background reviewer (an Opus 5.5 agent at maximum effort, run in the same sandbox after the push landed). It could not write its own `REPORT.md` (its sandbox blocks report files), so it handed the text back as a message, and I saved it here: the wording is the reviewer's, the heading layout is normalized, and two rater identifiers are replaced by `[rater ids redacted]`. Its labels "known finding N" refer to the findings of my earlier audit of Cursor's commit `036cafa` (`../qa_of_experiment_36_11-26-2026/`). Its four scratch scripts are in `agent_scripts/` (verbatim, with the `/tmp` paths of its sandbox); it saved no outputs. Where I rely on one of its numbers I recomputed or read it myself (see `README.md` in this folder, section "The Opus reviewer's claims: what I confirmed").

**Status of every claim in it.** The reviewer labels its own claims "Verified" or "Suspected". That labelling is the reviewer's, not mine. Which claims I then confirmed, did not confirm or could not check is in `README.md` of this folder. One point below I do not agree with: its "false" verdict on the "$0.05 ... now measured directly" line, because the pitch text reads as saying that POPQUORN is now scored directly.

---

## Summary (at most 350 words)

I could not write `/tmp/opus_qa_fcd8db9/REPORT.md`: the harness blocks subagents from writing report files, and I did not work around that. The full report follows this summary. I did not modify the repo. My scratch scripts are in `/tmp/opus_qa_fcd8db9/scripts/`.

Verified defects, most severe first:

1. **The headline win is a calibration artifact.** Known finding 1 is still present and now also shows on POPQUORN. On offensiveness, normalized RPS is 0.2523 for the Pearson-trained head and 0.1595 for the CE-trained head. I fitted a scalar-to-distribution map on the validation split, which the experiment never reads. After it, the Pearson-trained head scores 0.1580. Its paired difference against the raw CE-trained head is −0.0014 [−0.0025, −0.0004]; on politeness it is −0.0057 [−0.0076, −0.0037]. The featureless train-marginal forecast also beats the Pearson-trained head in 9 of 9 views. The Pearson objective is the correlation of a sampled judge (`fit_eval.py:128`), which collapses to near point masses.
2. **The sealed FSC test split is burned.** The experiment uses its own seed-0 task split. Pooled `n_train` is 107, but only 105 artifacts lie outside the sealed test split, so at least 2 sealed-test artifacts were trained on. No document discloses the sealed split or prior exposure.
3. **Overclaims.**
   - The note to Sanmi (`SANMI_PITCH.md:552`) and README row 36 omit the CertJudge E39 RPS result (40/56) and the Exp 67 RPS intervals, which all include 0. Stage 1's claim that the note was "scoped the same way" is false.
   - "Correlation often higher for collapsed judges" is contradicted in 6 of 9 views.
   - The power comparison is driven by effect size: the Pearson median difference is 0.0006.
   - "With signal" and "$0.05 measured directly" (line 585) are false.
4. **Statistics.**
   - The bootstrap resamples items although splits are by task or article (`fit_eval.py:252`).
   - There are no paired intervals, and "9 of 9" counts point estimates only (`write_reports.py:305`).
   - `x_val` is never used (`run_all.py:73`).
   - One cell reports +inf [0.620, 0.645] (`metrics.py:140`).
   - ChaosNLI: 160 of 624 test premises also appear in train or validation (`data_io.py:341`).
5. **Protocol and wording.**
   - Amendment 1 is called "pre-registered" but was committed together with the results (`PROTOCOL.md:44`).
   - FSC is expanded as "functional semantic correctness" (`PROTOCOL.md:50` plus about 8 other files).
   - A rater's personal name appears at `PROTOCOL.md:60`, `SANMI_PITCH.md:621` and `write_reports.py:504`.
6. **Figures.** The power-figure titles overlap the y-labels, a grey band has no label, the CertJudge panel shows CE only, and normalized RPS is never plotted.
7. **Reproducibility.** A fresh clone cannot run `run_all.py`: the private sibling repos return 404 and abort the whole run, and the README does not mention the environment variables.

Fine: the Lean files are byte-identical to 036cafa7, the POPQUORN loader is exact, the pinning is sound, every axis has a grid, and no data or secrets are tracked.

**Not checked:** VeriBench and CertJudge reruns, the exact list of sealed tasks that overlap, SummEval and DICES reruns, and who the R_v1 raters [rater ids redacted] are.

---

# Full report: independent QA of commit fcd8db9 (Experiment 36)

Scope: `git diff 45e0968 fcd8db9` (39 paths). E = `experiments/36_sanmi_distributional_pitch_anti_proper_corr_start_26Nov2026`.

## 1. Headline comparison measures calibration, not training objective (high)

**Known finding 1: still present, and it now also shows on POPQUORN.**

Script: `/tmp/opus_qa_fcd8db9/scripts/pearson_head_checks.py`, run on the cached POPQUORN CSVs, which match their pinned SHA-256.

**Reproduction.** The CE-trained head reproduces exactly: offensiveness CE 1.2983, normalized RPS 0.1595. The Pearson-trained head's normalized RPS also matches (0.2523), but its CE drifts: 4.9959 vs 4.9955 committed, and 9.5598 vs 9.5475 on politeness.

**Matched calibration on the unused validation split.** I fitted the same map for every head: a multinomial logit on [scalar mean, 1], fitted by CE on validation ratings. Results are held-out test normalized RPS, with paired item bootstrap 95% intervals (B=2000).

- Offensiveness:
  - Pearson-trained head raw minus CE-trained head raw: +0.0928 [+0.0810, +0.1035].
  - Calibrated Pearson-trained head minus CE-trained head raw: −0.0014 [−0.0025, −0.0004]. The calibrated Pearson head is better.
  - Calibrated Pearson-trained head minus calibrated CE-trained head: −0.0000 [−0.0004, +0.0003].
- Politeness:
  - Calibrated Pearson-trained head minus CE-trained head raw: −0.0057 [−0.0076, −0.0037].
  - Calibrated Pearson-trained head minus calibrated CE-trained head: +0.0010 [+0.0007, +0.0014].

**The constant forecast beats the Pearson-trained head everywhere.** The train-marginal forecast beats the Pearson-trained head on normalized RPS and CE in 9 of 9 ordinal and 3-class views (`results.json` matrices, `marginal`). For example, SummEval expert: 0.1421 vs 0.2490.

**Objective mismatch.**
- Training: `fit_eval.py:128` sets `var_x = E[Var(X|c)] + Var(mu)`. That is the correlation of a sampled judge, which the folder's own theorem says collapses.
- Evaluation: Pearson is computed on the mean score.
- Effect on head entropy: train mean entropy is 0.69 for the Pearson-trained head vs 1.21 for the CE-trained head. With L2 = 0 it falls to 0.001.

**The Pearson-trained head often loses on Pearson itself.** Its held-out Pearson r is lower than the CE-trained head's in 6 of 9 views: R_v1 0.328 vs 0.594, native_main −0.062 vs 0.362. On native_main its optimizer reports `success=False` at the 35-iteration cap.

**Hash features carry almost no signal (known finding 2: still present).** The CE-trained head beats the constant by:
- 0.0042 nats on POPQUORN offensiveness (1.2983 vs 1.3025);
- 0.0040 nats on ChaosNLI 3-class;
- 0.0058 nats on DICES-990.

## 2. Sealed FSC test split trained on (high, new; verified by arithmetic)

The Exp 35 inventory (lines 20–23) documents the sealed split: 10/5/10 tasks per set, so 30/15/30 artifacts for the old set and 40/20/40 for the new set. Non-test artifacts therefore total 105.

Exp 36's own seed-0 60/20/20 task split gives pooled `n_train` = 107 (`results.json`). By pigeonhole, at least 2 sealed-test artifacts are in its training set. For R_v1 alone or native_main alone, avoiding every sealed task would require roughly a 1-in-3.3-million coincidence.

Grepping E's markdown for `exp27`, `sealed` and `prior exposure` gives 0 hits, so there is no disclosure.

**Suspected, confirm from `task_splits_11-24-2026.csv`:** the exact list of overlapping tasks.

## 3. Claims vs evidence (I sampled more than 25 numbers; all match their JSON source)

The numbers match. The framing does not.

- **Note to Sanmi (`SANMI_PITCH.md:552`).** It cites E39 CE 48/56 only. It omits E39 RPS 40/56 (mean 0.1536 vs 0.1547) and the fact that all 3 Exp 67 RPS intervals include 0. Stage 1 (`qa1`, major 2) says the note was "scoped the same way" as the bottom line; it was not.
- **README row 36.**
  - It never gives RPS, the primary metric.
  - It names methods by bare metric ("CE 1.298 vs Pearson 4.996").
  - It gives no split or interval.
  - It says "20 theorems", but `SanmiPitch.lean` has 56 theorem declarations; 20 is the number of `#print axioms` lines.
- **"Correlation is often higher for worse-calibrated or collapsed judges" (`SANMI_PITCH.md:5`).**
  - The matrices contradict it in 6 of 9 views.
  - In E63, the best-calibrated arm also has the highest Spearman (0.746).
  - In E39, CE-fit has the higher mean Spearman: 0.508 vs 0.492 overall, and 0.616 vs 0.568 on test.
- **Power claim.** It compares the power of two different metrics. The CE gap is caused by miscalibration: the POPQUORN offensiveness median difference is 3.45 nats for CE but 0.0006 for Pearson, so a Pearson "power" of 0.06 is about α.
- **`SANMI_PITCH.md:585` (false).**
  - It says "with signal", but SummEval crowd test r for the CE-trained head is −0.051.
  - It says "$0.05 ... published ... now measured directly", but `results.json` contains no cost fields. Exp 35 derives about $0.052 by arithmetic, and for offensiveness only.
- **Binary exception.** It is disclosed for CE only. The Pearson-trained head is also better on Brier (0.4547 vs 0.4596) and normalized RPS (0.2273 vs 0.2298).
- **Cell coverage.** There are 16 views × 5 heads × 8 metrics. "9 of 9" compares point estimates only (`write_reports.py:305–311`), with no paired intervals. On native_main the RPS intervals overlap heavily: [0.056, 0.182] vs [0.078, 0.238], from 20 test items in 5 tasks.

## 4. Statistics (verified from the code)

- **Bootstrap unit.** It resamples items (`fit_eval.py:252`). VeriBench splits by task (R_v1 test: 15 items from 5 tasks) and SummEval by article (64 items per article). **Suspected:** the intervals are too narrow.
- **No validation selection (known finding 3: still present).** `x_val` is computed but never used (`run_all.py:73,76`).
- **Interval without its point.** `percentile_interval` drops non-finite draws (`metrics.py:140–148`). As a result, `popquorn_politeness_binary` MSE CE is reported as +inf [0.620, 0.645], built from 390 of 1000 draws.
- **ChaosNLI leakage (known finding 5: still present).** It is still grouped by uid (`data_io.py:341`). With the pinned files, 160 of 624 test items share an exact premise with train or validation; by SNLI image id, 92 do. The QA's "zero leakage" check passes only because it tests the same grouping.

## 5. Protocol, wording and privacy

- **"Pre-registered" (known finding 4: still present).** It appears at `PROTOCOL.md:44`, `README.md:37`, `run_all.py:3` and `fit_eval.py:11`. `PROTOCOL.md` and `results.json` changed in the same commit, and `qa1` m8 admits this cannot be checked.
- **FSC mis-expanded** as "functional semantic correctness" in:
  - `PROTOCOL.md:50`
  - `SANMI_PITCH.md:183, 199, 215, 284`
  - `RESULTS_SUMMARY.md:19`
  - `data_io.py:8, 512`
  - `write_reports.py:209`
  - `tables/objective_metric_matrix.md:163, 179`
  - `tables/certjudge_real_panels.md`
- **Rater personal name** at `PROTOCOL.md:60`, `SANMI_PITCH.md:621` and `write_reports.py:504`. Stable rater IDs (allowed) appear in `COMMON_11-26-2026.md`, `qa1`–`qa3`, `data_io.py` (the coordinator) and the Exp 67 `per_rater` keys in `results.json`.
- **Lean.** `SanmiPitch.lean` and `lean_check.txt` are byte-identical to 036cafa7 (`git diff` gives 0 lines). Coverage is still overstated (known finding 6): `SANMI_PITCH.md:609, 615` cite `anti_proper_from_parts`, but that theorem only proves 0 < ρ < 1 ⇒ ρ² < ρ over `Rat`. The step from the law of total covariance is argued by hand and backed by numerics only.
- **Math wording (pre-existing).** "O(K/N)" at line 610 is imprecise: the stated identity gives at most 1/N.

## 6. Figures (all 10 opened)

Every axis has a grid (`_style`; `make_figures.py:120–121`).

- **Power figures.** The suptitle overlaps the right-hand y-label in both `eval_power_mdd_*.png` (`eval_power_mdd.py:197`).
- **Unlabeled grey band.** The k = 4–8 band in the crossover figures has no legend entry.
- **CertJudge right panel.** It is titled "CE-fit beats Pearson-fit" but shows CE only, with no intervals and no RPS.
- **Primary metric missing.** No figure plots normalized RPS. The y-axis "Held-out test score" mixes metrics, and the Pearson-trained head's CE (5–14) squashes the other curves.
- **Matrix strips.** They are 6000×800 and 4800×800 with 5.8-pt text, unreadable when embedded. Coloring by rank exaggerates near-ties.

## 7. Reproducibility and hygiene

- **Fresh clone fails.** `load_all()` (`data_io.py:743–774`) needs the three private sibling repos. The unauthenticated raw URL fallback (`data_io.py:182–183`) returns 404 and the whole run aborts. The README's Reproduce section omits `VERIBENCH_ROOT`, `TTJ_ROOT` and `CERT_JUDGE_ROOT`.
- **Pinning is sound.** It uses `git show`, per-artifact, manifest, packet-tree and CertJudge SHAs.
- **POPQUORN loader is exact.** 13,036 / 1,500 and 25,042 / 3,718 ratings and items, 0 dropped rows, 0 duplicate (user, item) pairs, 0 texts shared across items.
- **Nothing sensitive tracked.** No `data_cache`, CSV or secret is in git.

## QA fixes checked

**Confirmed:**
- `git show` pinning and per-artifact SHA checks
- TTJ and CertJudge SHAs
- raw-name collision check
- integer guard
- E39 filter and counts
- MSE finite in 6 of 7 binary views
- README row 36 numbers
- `HOW_TO_READ_RESULTS.md` braces
- rollout bars
- colormap and visible "inf" label
- mathtext label fix

**Not confirmed:**
- the note to Sanmi scoped "the same way" as the bottom line
- "zero overlapping text" in the figures
- "zero leakage"

## Not checked

- VeriBench and CertJudge reruns (private repos)
- the exact list of overlapping sealed tasks
- SummEval and DICES reruns
- whether R_v1 all5 raters [rater ids redacted] are human experts
- what reward the E63 GRPO arm was trained on
