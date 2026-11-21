# Experiment 17: does a flexible scalar-to-distribution calibrator remove the "one-dimensional bottleneck" claim? (toy, $0)

**Status:** pre-registered 11-20-2026, written and pushed before `scripts/run_sweep.py` was executed; run finished 11-20-2026 23:12, see [results.md](results.md). Retrospective label: none (prospective on synthetic data).

**Doc link:** none yet (the one-page status file at the repo root summarizes it).
**TLDR:** a score-then-calibrate judge is a one-dimensional bottleneck only if the calibrator after the score is simple. Experiment 11 used a 12-parameter affine-in-score head. This experiment replaces it with a flexible (nonlinear) head and asks whether the gap to end-to-end cross-entropy (CE) training survives.

## Question
Experiment 11 showed a gap at large sample size between "Pearson-trained score + tuned affine head" and end-to-end exact-CE training. The paper's wording attributes that gap to the one scalar bottleneck. A reviewer check noted that this is untested for a nonlinear calibrator: an affine-in-score logit head cannot represent a rating distribution whose shape changes with the score (scenarios with a score-dependent spread). A flexible head can.

## Design (same generator, grid and seeds as experiment 11)
- Scenarios A-D, n in {4, 8, 16, 20, 32, 64, 128, 256, 512, 1024, 2048}, 100 seeds, k = 3 ratings per item, 500 fresh test tasks per cell.
- New Stage-2 heads on a scalar score z: softmax of weights on [1, z, 12 Gaussian bumps over z in [-2.5, 2.5]], exact CE, ridge on all non-intercept weights, ridge picked from {0.01 ... 1e8} by 4-fold cross-fitted pooled RPS on training tasks only (1e8 gives the constant baseline).
- Scores: Pearson-trained out-of-fold score, MSE-trained out-of-fold score (both as in experiment 11), and an ORACLE score (the exact true mean rating of each task, standardized on training tasks) with the flexible head and with the affine head.
- Reference arms recomputed on the same data: constant, end-to-end CE fixed and tuned, Pearson fixed, Pearson and MSE affine tuned heads.
- Primary metric: test RPS (ranked probability score) against the true distribution (RPS excess over the true RPS). Secondary: CE excess, Spearman, Pearson, TVD from constant.

## Predictions (written before the run)
1. With the oracle score, the flexible head is within 0.002 RPS of end-to-end CE at n >= 512 in scenarios A-D, and is better than the affine head on at least one scenario (the one where spread varies with the mean). Probability 70%.
2. With the learned Pearson score, the flexible head does NOT close the gap to end-to-end CE at n >= 512 on the scenario where experiment 11 showed the largest gap (the gap comes from the score being a bad one-dimensional summary, not from the head). Probability 60%.
3. At n <= 32 the flexible head is no better than the affine tuned head (ridge selection drives both to the constant). Probability 75%.
4. Reading rule: the "one-dimensional bottleneck" wording stays only if prediction 2 holds with the oracle-score result as the control; otherwise the paper must say "a simple calibrator after a scalar score".

## Run
`python3 scripts/run_sweep.py 16` (resumable, writes `results/raw.jsonl`, then `results/DONE`). End condition: all 4,400 cells written. Free hardware only (shared CPU host, 1 thread per process).
