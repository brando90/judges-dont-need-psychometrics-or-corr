# Experiment 18: when does cross-entropy (CE) beat correlation? Notebook page, per-term math, and a sweep over data amount, flexibility, regularization and rater noise (started 11-21-2026)

**Status:** started 11-21-2026 11:23 AM. Step 1 (transcription and images) done and pushed; steps 2-3 (math, sweep) planned, marked below as planned until a script prints the numbers.

**Doc link:** none yet (one dated Google Doc will be added here when the results exist).
**TLDR:** Brando's notebook page (11-21-2026) asks for (a) the equation per loss term of the correlation objective versus CE and a clear, checked proof that Pearson is only a function of the judge's mean (linear), and (b) the experiment that says when CE wins. This folder holds the page and Sanmi Koyejo's 11-16-2026 5:15 PM Discord message as images plus a word-for-word transcription, then the math and the sweep.

## Files
- [TRANSCRIPTION_11-21-2026.md](TRANSCRIPTION_11-21-2026.md): the handwritten page and Sanmi's five points, transcribed (unclear spots marked).
- `images/`: the three originals, unchanged bytes (notebook page IMG_0008.jpg; Sanmi's Discord screenshots IMG_0010.png and IMG_0009.png). Same files are in Brando's Drive folder "judges-dont-need-psychometrics-or-corr".

## Plan (planned, not done)
1. Math (page TODO): per-loss-term equations for Pearson correlation of a judge distribution against a target distribution, versus CE, RPS and TVD; proof by hand, sympy and numeric checks; Lean only if feasible.
2. Sweep (builds on, does not redo, experiments 05, 08, 11, 15, 16, 17): when CE wins as a function of n, judge flexibility, regularization and rater noise; primary test metrics CE, TVD and RPS; Pearson and Spearman secondary.
3. Results doc with the answer to "when does CE win?" in its first lines.

Related: experiments 05, 06, 08, 11, 16, 17 and `STATUS_11-20-2026.md` at the repo root.
