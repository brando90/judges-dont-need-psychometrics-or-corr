# Protocol: proper-score design allocation versus D-study allocation (frozen before measurement)

**Date:** 11-26-2026 (America/Los_Angeles). **Status:** frozen before the Monte Carlo was run.
**Machine:** local CPU only. **Cost:** `$0`. No model calls, no Claude or Codex jobs, no GitHub push.

**TLDR:** Two studies. Study A compares a generalizability-theory (G-theory) / decision-study (D-study) rule with a proper-score rule for allocating a fixed rating budget $B = n_{\mathrm{items}}\times n_{\mathrm{raters}}$ on ChaosNLI and DICES, and reports separate oracles for held-out ranked probability score (RPS) and for universe-score / mean-estimation error. Study B repeats Experiment 04 Phase B with closer variance-component configurations and smaller pilots.

## Common rules

- Master seed `20261126`. Every random draw uses `numpy.random.SeedSequence` as specified below. The resampling unit for confidence intervals is the replication.
- Uncertainty: Student-$t$ 95% intervals over replications for means and paired differences. Wilson 95% intervals for proportions. A method **beats** another on a metric only when its mean is better and the 95% interval on the paired difference excludes 0. No multiplicity correction.
- Ranked probability score (RPS) is primary for the distributional target, range 0 to 1, lower better: $(1/(K-1))\sum_{k=0}^{K-2}(F(k)-G(k))^2$. Cross-entropy (CE, nats) and total variation distance (TVD, 0 to 1) are secondary. Universe-score mean squared error (MSE) of an item mean, and MSE of a grand mean, are reported separately and are never mixed with RPS.
- "Beats" is never claimed from a lower point estimate alone.

## Study A — allocation of items versus raters

### Data

Public ChaosNLI and DICES tables downloaded 11-26-2026 (hashes in `data/raw/SHA256SUMS.txt`). Item subsets and label order match Experiment 28 (`prepare.py`, `numpy` default generator with `SeedSequence([20261123, k])`):

| Dataset | $k$ | Items | Ratings / item | $K$ | Category order (score $0,\ldots,K-1$) | Rater ids |
|---|---:|---:|---:|---:|---|---|
| chaosnli_snli | 0 | 600 | 100 | 3 | contradiction, neutral, entailment | no |
| chaosnli_mnli | 1 | 600 | 100 | 3 | same | no |
| chaosnli_alphanli | 2 | 400 | 100 | 2 | hypothesis 1, hypothesis 2 | no |
| dices350 | 3 | 350 (all) | 123 (fully crossed) | 3 | No, Unsure, Yes | yes |
| dices990 | 4 | 400 (Experiment 28 subsample) | 69–76 | 3 | No, Unsure, Yes | yes |

Frozen judge probabilities are the Experiment 28 forecasts (`forecasts_11-23-2026.csv`, five judges). They are used only as **frozen features**. No new judge calls.

### Predictor (pre-declared)

For each item, average the valid raw probability vectors of the five judges (skip `invalid_uniform`); renormalize. If no judge is valid, use the uniform distribution. Then apply the Experiment 28 family of 27 logit transforms (temperature in $\lbrace 0.5,1,2\rbrace$ and linear/quadratic category shifts in $\lbrace -1,0,1\rbrace$). **Cross-entropy fit** selects the unique earliest transform (ties as in Experiment 28) that minimizes categorical CE on the training ratings of the design. The fitted transform is applied to held-out items. Training ratings never include test items.

### Splits (pre-declared, no leakage)

For each dataset, permute items with `SeedSequence([20261126, 1, k])`. The first 30%, rounded, are the frozen **test** set. The rest are the **development** pool. Test item identifiers are never used to estimate variance components, to choose a design, or to fit a transform. Test histograms (all available ratings) are used only for the final score of a already-fitted forecast.

### Candidates and pilot

- Pilot, drawn from the development pool in each replication: 80 items and 16 ratings per item, without replacement (`SeedSequence([20261126, 2, k, rep])` for items, `[20261126, 3, k, rep]` for ratings).
- Candidates: every pair $n_i\in\lbrace 16,32,64\rbrace$, $n_r\in\lbrace 2,4,8,16\rbrace$. Budget $B=n_i n_r\in\lbrace 32,64,128,256,512,1024\rbrace$. At each fixed $B$ the three (or fewer) designs have distinct $n_r$.
- Replications: 150 per dataset. Inner Monte Carlo for the proper-score selector: 5-fold over the 80 pilot items (64 train / 16 validation) and 8 rating-subsamples per fold and design (`SeedSequence([20261126, 4, k, rep, fold, design_id, m])`).
- Deployment, after a design is chosen: a fresh sample of $n_i$ items and $n_r$ ratings from the **full development pool** (the pilot may be reused, as a development sample), `SeedSequence([20261126, 5, k, rep, design_id])`. Every candidate is deployed in every replication so oracles and paired differences are well defined.

### Selectors (pre-declared)

1. **G/D-U1 (primary D-study rule, per-item universe score).** One-facet ANOVA on the $80\times 16$ pilot table (ratings are nested / exchangeable; this is the only identifiable model on ChaosNLI). Negative object-variance estimates are clipped at 0; residual variance is clipped at $10^{-12}$. Choose the design of budget $B$ that minimizes $\widehat{v}_\Delta(n_r)=\widehat{v}_e/n_r$. This criterion does not depend on $n_i$, so it selects the largest $n_r$ (fewest items) at each $B$.
2. **G/D-U2 (secondary D-study rule, grand-mean precision).** From the same components, minimize $\widehat{v}_u/n_i+\widehat{v}_e/(n_i n_r)$. At fixed $B$ the residual term is constant, so this selects the largest $n_i$ (fewest raters) whenever $\widehat{v}_u>0$.
3. **Proper-score.** From the same pilot only, estimate expected validation RPS of the CE-fitted forecast on held-out *pilot* items, for each candidate with $n_i\le 64$. Choose the design of budget $B$ with the lowest mean validation RPS. Ties: more items, then more raters, then the first listed design.
4. **Oracles (not selectors used in the wild).** In each replication, the candidate with the lowest *deployment* test RPS is the RPS oracle; the candidate with the lowest per-item held-out mean MSE is the universe-score oracle; the candidate with the lowest grand-mean MSE is the grand-mean oracle. Oracles may use test outcomes and are reported as references only.

### Metrics on deployment

- **Distributional, test items:** RPS, CE, and TVD of the fitted forecast against each test item's full empirical histogram, averaged equally over test items. (Same convention as Experiment 28.)
- **Per-item universe-score MSE:** on the deployment training items, the squared error of the $n_r$-rating mean versus the mean of that item's *held-out* ratings, then averaged over those items. This uses no judge and no test items.
- **Grand-mean MSE:** squared error of the mean of the $n_i$ sample means versus the mean of all development-item universe means (full-data item means in the development pool). This is a descriptive population-mean metric; it is not a proper score of $p(y\mid x)$.

### What would support replacing D-theory

If, at a pre-declared majority of dataset-by-budget cells, the proper-score design has significantly lower test RPS than G/D-U1, *and* the RPS oracle's $n_r$ disagrees with the universe-score oracle's $n_r$, then the two targets have different optimal allocations and a proper-score rule is the one that tracks the distributional target. That would **not** show that D-theory is wrong for universe-score precision. If G/D-U1 and the proper-score rule pick the same design, or if their test-RPS interval covers 0, they agree for that cell.

## Study B — harder Phase B of Experiment 04

Crossed Gaussian objects $\times a\times b$ model, same generators, REML, ANOVA, and D-study formulae as Experiment 04 (`gtheory_lib.py`, `run_d_study.py`). Absolute error is the mean over 256 fresh objects of $(\text{observed mean}-\text{universe score})^2$, expectation $v_\Delta$.

- Original components: C1 $=(v_u,v_a,v_b,v_{ua},v_{ub},v_{ab},v_e)=(1,0.6,0.1,0.6,0.1,0.1,0.5)$, C2 $=(1,0.1,0.6,0.1,0.6,0.1,0.5)$.
- Interpolation: $v(t)=(1-t)\mathrm{C1}+t\mathrm{C2}$ for $t\in\lbrace 0,0.15,0.30,0.40,0.45,0.50,0.55,0.60,0.70,0.85,1\rbrace$.
- Pilots $(n_u,n_a,n_b)\in\lbrace (16,4,4),(32,4,4),(32,8,4),(64,4,4),(64,8,8)\rbrace$.
- Budget 16 designs: $8\times 2$, $4\times 4$, $2\times 8$. Selector: constrained REML, minimize $\widehat{v}_\Delta$. Baselines: oracle (true components) and equal $4\times 4$.
- Replications: 200 per $(t,\text{pilot})$ cell. Seeds: `SeedSequence([20261126, 20, t_index, pilot_index, rep, stage])` with stage 0 = calibration, 1 = evaluation.
- Reproduction cells: $t=0$ and $t=1$ at pilot $64\times 8\times 8$, 256 replications, same evaluation design as Experiment 04, reported against the published C1/C2 selection rates and paired errors. These two cells are a sanity check, not a new claim.
- No vulnerability battery (V1–V4) is re-run here.

### What would count as a weak-signal failure of REML allocation

The fraction of replications in which REML matches the oracle design, and the paired held-out absolute-error gap versus the oracle and versus $4\times 4$, as $t\to 1/2$ and as the pilot shrinks. Near $t=1/2$ the three designs have nearly equal $v_\Delta$, so mis-selection is expected and the error gap should be small. The scientifically important case is intermediate $t$, where the designs differ by a modest amount and a small pilot may pick the wrong one with a material error gap.
