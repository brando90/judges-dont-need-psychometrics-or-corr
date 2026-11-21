# Experiment 18: when does cross-entropy (CE) beat correlation? Notebook page, per-term math, and a sweep over data amount, flexibility, regularization and rater noise (started 11-21-2026)

**Status:** started 11-21-2026 11:23 AM. Step 1 (transcription and images) done and pushed; steps 2-3 (math, sweep) planned, marked below as planned until a script prints the numbers.

**Doc link:** none yet (one dated Google Doc will be added here when the results exist).
**TLDR:** Brando's notebook page (11-21-2026) asks for (a) the equation per loss term of the correlation objective versus CE and a clear, checked proof that Pearson is only a function of the judge's mean (linear), and (b) the experiment that says when CE wins. This folder holds the page and Sanmi Koyejo's 11-16-2026 5:15 PM Discord message as images plus a word-for-word transcription, then the math and the sweep.

## Files
- [TRANSCRIPTION_11-21-2026.md](TRANSCRIPTION_11-21-2026.md): the handwritten page and Sanmi's five points, transcribed (unclear spots marked).
- `images/`: the three originals, unchanged bytes (notebook page IMG_0008.jpg; Sanmi's Discord screenshots IMG_0010.png and IMG_0009.png). Same files are in Brando's Drive folder "judges-dont-need-psychometrics-or-corr".

## Status log
- 11-21-2026 11:23: transcription and images pushed.
- 11-21-2026 11:50: math done and pushed: [MATH_11-21-2026.md](MATH_11-21-2026.md), checks in `math/` (sympy, numpy, Lean 4.22 + Mathlib).
- 11-21-2026 12:20: sweeps A and B pre-registered below, then launched (`sweep/scripts/`).

## Sweep design and predictions (written 11-21-2026 12:20, before any sweep cell ran)
**Question:** when does training or selecting a judge by cross-entropy (CE) beat doing it by correlation, as a function of data amount n, rater noise (ratings per item k), judge flexibility, regularization and target complexity? Primary test metrics, all against the TRUE rating distribution on 500 fresh items: RPS, CE excess (KL), TVD (lower is better). Secondary: Pearson and Spearman of the judge mean with the true mean. Synthetic (labelled simulated; Beta-Binomial targets A-D of experiments 08 and 11); real human ratings are a separate step.

**Sweep A (fitted judges).** Scenarios A-D; n in {8,16,32,64,128,256,512}; k in {1,3,10}; flexibility J in {1,3,6} cosine/sine harmonics (D = 3, 7, 13 features, 6 logits each); 20 seeds; 5,040 cells. Arms: constant (train histogram); CE-trained, RPS-trained and MSE-trained softmax judges, each with its ridge chosen on the training tasks only by 3-fold pooled RPS; Pearson-trained judge (fixed ridge 1, as in the paper); Pearson-selected score (ridge by out-of-fold correlation) followed by a calibrated affine head (the score-then-calibrate baseline from Sanmi's point 2; head ridge by out-of-fold RPS). Code: `sweep/scripts/sweep_core.py`, `run_sweep.py`.

**Sweep B (select one judge from a pool of 24 fixed candidates, like prompt or harness arms).** Candidates are the truth tempered, shifted and noised (different calibration quality); select by Pearson, Spearman, CE or RPS on n training items with k ratings; pools: native probabilities, 3-vote histograms with floor 1e-6 (as the cached scalar judges), 3-vote histograms with floor 5e-2; n in {4,...,256}; 100 pools per cell. Code: `sweep/scripts/run_selection.py`.

**Predictions (my guesses, fixed now):**
1. (A) The Pearson-trained judge loses to the constant on test RPS in at least 80% of the 252 (scenario, n, k, J) cells (about 75%).
2. (A) CE-tuned beats the Pearson-selected score-then-calibrate arm on test RPS (paired mean difference, interval excludes 0) in scenarios A and B once n >= 64 at J = 6, and not at n <= 16 (about 65%). In C and D (constant spread) the two are within 0.003 RPS at J = 1 (about 60%).
3. (A) A smaller k (noisier ratings) moves the crossover n (smallest n from which CE-tuned beats the Pearson arm at every larger n) up by about a factor of 2 or more from k = 10 to k = 1 (about 55%).
4. (A) CE-tuned stays within 0.02 of the Pearson-trained judge on test Spearman in at least 70% of cells, i.e. CE does not give up ranking (about 55%).
5. (B) With native probabilities, CE and RPS selection beat Pearson selection on test RPS for n >= 16 (about 70%). With 3-vote histograms and floor 1e-6, CE selection is no better than Pearson selection or worse (reproduces Sanmi's point 1 mechanism; about 65%); with floor 5e-2 CE selection recovers (about 55%).
6. (B) At n <= 8 no selection rule is distinguishable from picking the constant/random candidate; Pearson selection is not worse than CE selection (about 60%).
**Reading rule:** a cell counts as "CE wins" only if the 95% bootstrap interval (over seeds) of the paired difference excludes 0 on test RPS and CE excess both; "tie" if both include 0. Everything is reported, including cells that go against these guesses.

## Plan (planned, not done)
1. Math (page TODO): per-loss-term equations for Pearson correlation of a judge distribution against a target distribution, versus CE, RPS and TVD; proof by hand, sympy and numeric checks; Lean only if feasible.
2. Sweep (builds on, does not redo, experiments 05, 08, 11, 15, 16, 17): when CE wins as a function of n, judge flexibility, regularization and rater noise; primary test metrics CE, TVD and RPS; Pearson and Spearman secondary.
3. Results doc with the answer to "when does CE win?" in its first lines.

Related: experiments 05, 06, 08, 11, 16, 17 and `STATUS_11-20-2026.md` at the repo root.

## Status log
- 11-21-2026 ~12:05: sweeps A (5,040 cells) and B (25,200 pools) and the real-data n_train sweep finished (skampere1, CPU, $0). Results, predictions scored, limits: `RESULTS_11-21-2026.md`; tables `sweep/results/tables.md`; figure `figures/fig_A_rps_vs_n.png`; analysis `sweep/scripts/analyze.py`. Real-data runs: `real_n_sweep/`.
