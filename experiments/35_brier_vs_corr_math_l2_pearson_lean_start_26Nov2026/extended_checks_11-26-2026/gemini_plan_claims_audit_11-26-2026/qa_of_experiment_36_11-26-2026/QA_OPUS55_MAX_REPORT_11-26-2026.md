# QA report: experiment 36, the Gemini plan, and the experiment 35 claims audit (11-26-2026)

Audited commit: `036cafa7ab08c9ad9d6c06516cd691538e644520` (origin/main). I fetched again at 23:19 UTC: there were no new commits, so I found no new experiment 36 work or new experiment folders. Nothing in the repository was modified. All scratch work is in `/tmp/qa55/` (scripts in `code/`, downloads in `data/`, outputs in `out/`).

At 23:23 UTC an untracked folder appeared in the working tree: `experiments/35_.../extended_checks_11-26-2026/gemini_plan_claims_audit_11-26-2026/`. Another process created it, not me, and I did not audit it.

What I ran:
1. The Lean compile, pytest, and a full `run_all.py` rerun in a scratch copy (downloads match the `data_io.py` SHA-256 values; storage.googleapis.com was reachable).
2. Fairness and calibration diagnostics on five real sets.
3. Recomputation of the plan's numbers, and recounts of the DICES, ChaosNLI and POPQUORN files.
4. The experiment 35 numerics (twice) and a Mathlib compile of `AuditClaims.lean` with `#print axioms`.

Labels: CONFIRMED means I reproduced it; PLAUSIBLE means I could not.

## Findings, most severe first

### 1. Conference-blocking: the Pearson-trained head is not a fair competitor, and the headline gaps are artifacts (CONFIRMED)

Where: `fit_eval.py:39-41` (L2=1e-2, MAXITER=35), `fit_eval.py:97-100` (the intercept is unpenalized), `fit_eval.py:122-138` and `:141`, and `RESULTS_SUMMARY.md:17`.

**Is the head trained at all?** Yes. Experiment 36's surrogate is not degenerate at the zero start. It is the correlation of a sampled rating, and `var_x` includes the conditional variance, which is positive under the uniform start. I measured the start gradient norm at 1.0–1.6×10⁻² on all five sets. L-BFGS-B ran 3–17 iterations and stopped on PGTOL.

The zero-start degeneracy you suspected does exist, but for the expected-score Pearson objective used in experiments 24, 25 and 29. That objective has zero gradient at W=0, and L-BFGS-B stops at iteration 0 on every set (`/tmp/qa55/out/fair_*.json`).

**The head is crippled by the penalty.** The same L2=1e-2 is applied to CE (about 1.2 nats, start gradient about 0.4) and to a correlation objective 30–100 times smaller in value and gradient. At L2=1e-2 the training surrogate reaches only 0.014–0.032. At L2≤1e-3 it reaches 0.06–0.26 (0.15–0.26 at L2=0).

The symptom is a correlation-trained head that loses on correlation. Held-out r of the CE head minus r of the Pearson head is:
- SummEval expert: +0.028 [0.011, 0.047]
- DICES-990: +0.017 [0.007, 0.029]

These are paired, item-clustered, B=1000.

**Its proper scores are not identified by its objective.** The surrogate sees per-item spread only through one pooled second moment, so it has flat directions. Rerunning to convergence at the same L2 gives the same objective value but different held-out CE:
- SummEval: 4.73 vs 14.69, and normalized RPS 0.249 vs 0.157
- ChaosNLI SNLI+MNLI: 4.02 vs 11.48
- DICES-990: 1.16 vs 1.99

So "4.73 versus 1.21" reflects where this optimizer happened to stop (numeric gradients, a PGTOL stop at 13–17 iterations), not correlation training as such.

**Matched calibration removes most of the gap.** I applied the same temperature plus K class offsets to both experiment 36 heads, fitted on validation CE (`calib2.py`). The held-out normalized RPS of the Pearson head minus the CE head becomes:
- SummEval expert: +0.0025 [0.0018, 0.0032]
- ChaosNLI SNLI+MNLI: −0.0005 [−0.0012, 0.0002]
- DICES-990: +0.0009 [0.0004, 0.0014]
- SummEval binary: +0.0000 [−0.0003, 0.0003]
- DICES binary: +0.0008 [0.0004, 0.0014]

The reported raw gaps were +0.109, +0.057, +0.060, +0.055 and +0.073. Calibration shrinks them by 40–100× or more, and two sets tie. This matches experiment 28's own finding, which experiment 36 omits: Pearson-then-calibration beat CE-fit on DICES from 64 ratings.

**With fair tuning, correlation is indistinguishable.** I selected L2 on validation from {0, 1e-4, 1e-3, 1e-2} (validation r for the Pearson heads, validation normalized RPS for the CE head), with analytic gradients, 3,000 iterations and 4 starts. The held-out r difference between the CE and Pearson heads is not significant on:
- 4 of 5 sets for the sampled surrogate (DICES binary favours CE)
- 5 of 5 sets for the expected-score objective

**What survives.** Without calibration, every Pearson variant has worse held-out normalized RPS and CE than the CE head, with intervals excluding 0.

Why it matters: a reviewer will read the matrix as a strawman (an over-regularized, uncalibrated head with no validation selection), and CLAUDE.md's fairness rule is not met.

Fix: select L2 and the budget on validation per objective, penalize or calibrate the intercept, add calibrated arms for both heads, report paired intervals, and restate the headline as "correlation leaves calibration unidentified; a small calibration fitted on held-out ratings recovers most of it".

### 2. Conference-blocking: the features carry almost no signal (CONFIRMED)

On SummEval expert, ChaosNLI and DICES-990, the experiment 36 CE head beats the constant train-marginal forecast by only 0.0113, 0.0040 and 0.0059 nats of held-out CE, and by 0.0019 [0.0015, 0.0022], 0.0009 [0.0002, 0.0017] and 0.0022 [0.0009, 0.0035] normalized RPS (`marginal_gain.py`). The shuffle control is within 0.0025 nats of the real head on ChaosNLI.

So the CE-versus-Pearson comparison is close to "calibrated constant versus miscalibrated constant". The raw Pearson head (0.249 normalized RPS on SummEval) is far worse than the constant forecast (about 0.142). These numbers cannot support claims about judges.

Fix: use informative features (frozen LLM embeddings or rating-token logits), and until then present this matrix as a toy.

### 3. High: the validation split is never used, and "pre-registered" cannot be verified (CONFIRMED)

`run_all.py:81` builds `x_val` and nothing reads it. `PROTOCOL.md:15` says "Validation does not choose hyperparameters." This contradicts CLAUDE.md, which makes validation-based selection mandatory with normalized RPS as the common criterion.

`PROTOCOL.md:3` says the protocol was "Written before run_all.py printed a held-out test metric". But the protocol and `results.json` arrived in the same commit, `7b3c7d1`.

Fix: select on validation. Drop the pre-registration language unless an earlier timestamped commit exists.

### 4. High: claims without paired intervals, and one claim contradicted by its own table (CONFIRMED)

`RESULTS_SUMMARY.md:17` says "Its correlation is not higher than the CE head's" for the ordinal and three-class sets. On ChaosNLI SNLI+MNLI the Pearson head is higher on all three correlations: 0.086, 0.118 and 0.078 against 0.078, 0.111 and 0.074. The paired r difference is −0.0085 [−0.0195, 0.0024].

No stated difference in the matrix has a paired interval. The primary metric, normalized RPS, is shown without intervals, while sqrt(RPS) has them.

Fix:
- Report paired item-clustered intervals for every claimed difference.
- Make normalized RPS with its interval the headline column.

### 5. High: POPQUORN was available (CONFIRMED)

`README.md:51` and `data_io.py:103-108` say POPQUORN could not be found. It is public at `github.com/Jiaxin-Pei/Potato-Prolific-Dataset`; its README is titled "POPQUORN: the Potato-Prolific dataset…". I cloned it:
- Offensiveness: 1,500 items, 13,036 ratings, mean 8.69 per item (range 5–10), 262 annotators.
- Politeness: 3,718 items, 25,042 ratings, mean 6.74 per item (range 3–8), 506 annotators.

Both are on a 1–5 scale with a per-rater `user_id`.

Fix: add a loader. It is the plan's ordinal crowd set and its dollar anchor.

### 6. High: Lean coverage is overstated (CONFIRMED: exit 0, no `sorry`, axioms identical to `lean_check.txt`)

1. **General and correct:** `brier_decomposition`, `brier_strictly_proper`, `DB_zero_iff`, `binary_brier_eq_two_l2`, `standardized_mse_corr`, `empirical_l2_rate` and `plugin_DB_bias`. `expN` is exactly the product-measure expectation over n iid draws, so the last two match E‖p̂−p‖²=(1/N)(1−‖p‖²) and the plug-in bias.
2. **Rat-only:** `pearson_affine_invariant` and `mse_decomposition` hold only for data whose variance is a rational square. Not vacuous, but not the real-valued statement.
3. **From parts:** `anti_proper_from_parts` (lines 692-706) proves only that 0<r<1 implies r²<r, for r=a/(a+b), and `collapsed_sqcorr_scales` takes its values as hand-plugged inputs. Neither formalizes the law of total covariance, Corr(X_{p*},Y)=Var μ/Var Y, Corr(X_δ,Y)²=Var μ/Var Y, or the optimality of X_δ.
4. **Arithmetic on typed numbers,** not linked to `brier`, `DB` or any RPS definition: `ordinal_same_mean_different_law`, `fair_coin_couplings`, `binary_realized_prob_drops`, `binary_brier_strictly_worse`, `binary_witness_corr_tie` and `anti_proper_half_example`. RPS is never defined in Lean; its properness rests only on the one-threshold `bernoulli_square_excess`.
5. **A tautology:** `local_score_worsens`, whose hypothesis is its conclusion.
6. **A wrong number:** `binary_brier_strictly_worse` proves 1/25 against 2209/10000, which is scalar (q−y)². The documents call "Mean Brier 0.0800 versus 0.4418" Lean-checked, but those two-class values are twice what Lean proves.

Fix:
- Relabel these theorems as general, from parts or instance.
- State the instances through `DB` or `expBrier`.
- Define RPS and prove its excess identity.
- Or present anti-properness as "numerics plus a paper proof".

### 7. Medium: the chain-of-thought claim is wrong as stated (CONFIRMED by derivation; no LLM run)

`pitch_addenda.py:24` (copied into `SANMI_PITCH.md:218`) says that M chain-of-thought samples estimate "the same marginal" that a direct, no-rationale forward pass reads. They are different conditionals. The direct pass gives q(y | c, answer now). The chain-of-thought marginal is Σ_r q(r|c) q(y|c,r). They are equal only under an unstated assumption that generally fails for language models.

The linear-head rollout figure is also a Jensen tautology: a histogram of M draws from q against q itself. It is not evidence about language models.

### 8. Medium: hard-coded numbers that are not bit-reproducible (CONFIRMED)

The G_THEORY block (`write_reports.py:9-35`), the prose at `write_reports.py:302` and the POWER block in `pitch_addenda.py` are fixed strings, not computed from the run. All currently match `results.json` and `results_power_mdd.json`. My rerun, however, moved Pearson-head values in the fourth decimal (8.6508→8.6505, 4.9262→4.9264) and wrote `+inf` where the commit has `inf`.

Fix: generate every quoted number from the payload.

### 9. Medium: the iso-budget conclusion is not supported (CONFIRMED)

`RESULTS_SUMMARY.md:273` concludes that more prompts with one rating beat fewer prompts with many ratings. The CE-head intervals overlap for (1024,1) 1.054 [1.040, 1.070], (256,4) 1.056 [1.044, 1.069] and (64,16) 1.062 [1.046, 1.077]. The constant marginal is 1.057. Only (32,32), at 1.101, differs. This is one dataset with no paired intervals.

### 10. Medium/low: data caveats

1. **ChaosNLI (CONFIRMED):** split by uid, but 123 of 624 test items share their premise with a train item. Group by premise.
2. **SummEval experts (PLAUSIBLE, from this repo's experiment 23 inventory citing SummEval §4.3):** experts revised scores after seeing others' on large disagreements, so the three ratings are not independent draws.
3. **SummEval crowd (CONFIRMED):** the crowd crossover gives negative held-out r for both heads (−0.051, −0.057).
4. **DICES (CONFIRMED):** the raw file has 69–76 raters per item and 72,103 rows; "60–69" and "64,886" are after the 12-rater filter. `Q_overall` is per-rater ("any Yes, else any Unsure, else No" over that rater's Q2–Q6 answers on 100% of rows), so its use is valid.

### 11. Low

1. `RESULTS_SUMMARY.md:288` gives the frequency error as O(K/N). The identity gives at most 1/N, with no K growth.
2. The witness q*=(.2,.2,.8,.8) with y=(0,0,1,1) is called "calibrated". It is not.
3. The rater check in `data_io.py:387-389` is dead code.
4. The MSE two-point readout has infinite CE by construction, which makes it a strawman readout.
5. `write_reports.py` raises invalid-escape SyntaxWarnings.
6. `SanmiPitch.lean:3` still carries the stale header "BrierCorr.lean".

## B. The Gemini plan (`dcc7eca1-attachment.txt`)

1. **Unbiased per-sample risk: CORRECT.** The Monte Carlo mean was 0.90158 against the true 0.90115 (s.e. 0.00038). This assumes test ratings are iid draws from p*(·|c); see the SummEval caveat above.
2. **Plug-in bias (1/k)(1−‖p‖²): CORRECT.** It is general Lean (`plugin_DB_bias`), and the Monte Carlo agrees.
3. **"Exact ERM rate" E‖q̂_N−p*‖²=(1/N)(1−‖p*‖²): CORRECT** for one context's empirical frequency. **IMPRECISE** if read as a rate for a judge over prompts. The plan's Python asks for N up to 1e5; experiment 36 checks only N∈{10, 50, 200}.
4. **Binary example: CORRECT.** Pearson is 0.6532 for both. Scalar Brier is 43/300=0.1433 against 0.4119, a ratio of 2.874. The bias, scale and alignment decomposition is exact. "Brier" here means scalar (q−Y)².
5. **Corr(X_δ,Y)=√Corr(X_{p*},Y): CORRECT** when X_{p*}⊥Y given C, Var μ>0 and E σ²>0. It concerns sampled judges. When a judge is scored on its expected score, p* attains the maximum, so "not identifying" is accurate where "anti-proper" is not.
6. **Ordinal {0..4} example: CORRECT, and one law gives every number.** Take two equally likely contexts, with p* uniform on {0,2} and on {2,4}, and collapsed forecasts δ₁ and δ₃:
   - Pearson: 0.5 against 0.7071
   - Brier: 1/2 against 2 (4×)
   - RPS: 1/2 against 1 (2×; normalized 1/8 against 1/4)
   - CE: 0.6931 against ∞

   Experiment 36 checks only the correlations on this law.
7. **sqrt(RPS) as the reported metric: CONFLICTS with the project.** The primary metric is normalized RPS. sqrt preserves rankings, but its differences are not risk differences, it is not comparable across K, and the square root of an unbiased mean is biased low.
8. **Datasets:**
   - SummEval: **CORRECT** (1,600 summaries = 100 articles × 16 systems; 3 experts and 5 crowd workers; 4 aspects; 1–5).
   - POPQUORN: counts **CORRECT**. The pay figures are not verified; $12/hr × 13 min per 50 ratings gives $0.052 per rating.
   - DICES "60–112 raters/item": **WRONG.** DICES-350 has 123 per item (104 after the quality filter). DICES-990 has 69–76 (60–69 after the filter).
   - ChaosNLI "3-class (4.6k)": **WRONG.** There are 3,113 three-class items. The 4,645 total includes 1,532 two-class αNLI items. "100 votes per item" is correct.
   - HelpSteer2: the 0–4 scale, 5 attributes and at least 3 annotators are **CORRECT**. "Filtered out >2-point disagreements" is **WRONG** per the data card. Such disagreements triggered extra annotators (up to 5), and the released score is the rounded mean of the 3 most-agreeing annotators. Per-annotator lists are released in `disagreements/`.
   - CertJudge FSC: not verifiable here.
   - Kohli, arXiv:2605.29797: it exists (Guneet Kohli, 28 May 2026). The abstract gives KL saturating by N≈10 (87–95% of the improvement over five seeds) and entropy correlation needing N≈20–50. I read only the abstract, so experiment 36's "91±3%" and "DICES 90% vs 62%" remain unverified.
9. **Asked for by the plan but not done in experiment 36:** POPQUORN (available), CertJudge FSC, real LLM judges, rank reversals, greedy against distributional readout on one model, LoRA (scripts only), a Spearman or RankNet surrogate, ECE, bias and spread diagnostics, TVD (the project's secondary metric), power-law fits in B, a researcher and GPU dollar axis, single-pass against rollouts on a real LLM, Lean RPS properness and probabilistic anti-properness, and validation selection.

## C. Experiment 35 claims audit

1. **Numerics (CONFIRMED).** `python3 -I check_claims_audit.py` gave identical output on two runs, byte-identical to `results/check_claims_audit.out`, with 36 of 36 PASS.
2. **Lean (CONFIRMED).** The SHA-256 of `AuditClaims.lean` matches `lean_check.txt`. All 45 theorems use only propext, Classical.choice and Quot.sound, with no `sorry`. The one compile error came from a bad line I appended myself, not from the committed file.
3. **Verdicts I re-derived:**
   - **P2.2: correct.** The written line evaluates to 2−ρ instead of 2(1−ρ); at ρ=√(2/3) that is 1.184 against 0.367.
   - **P2.4: correct.**
   - **P3.3: correct.** The count is 6−2=4, and `finrank_ker_momMap` is general. One caveat is missing: the "infinite set" conclusion fails when the mean is 0 or 5, where the law is unique.
   - **P3.4: "imprecise" is defensible.** But in the judge setting the judge cannot choose the coupling, so the user's statement holds there. The cited `sqdiff_lower_bound` is a moment bound, (μX−μY)²+(σX−σY)², which is zero for any two laws with equal mean and variance. It does not show "zero exactly when the laws are equal"; that rests on `zero_sqdiff_forces_equal_laws` plus attainment of the minimum, which is not formalized.
   - **S1: correct** (Brier cost 1/2 against 3/8 at the same q(0)).
   - **S2: correct** (13/25 against 15/25; best report 9/13).
4. **An item marked CORRECT that should not be (PLAUSIBLE; it depends on the handwriting).** P2.1, "(a+b)²=a²+2ab+b²=1²+2ab+1²", is false pointwise and holds only in expectation. By the ledger's own rubric it should be "CORRECT, WITH A NOTE".

## Checked and fine

The risk estimator and the CE, Brier, RPS and normalized RPS formulas; the item-clustered bootstrap (B=1000); midrank Spearman and tau-b Kendall with NaN draws dropped; SummEval groups (100 articles, split 60/20/20); DICES and ChaosNLI item groups, with no item in two splits; SVD fit on train only; committed tables equal to a regeneration from `results.json`; power-analysis prose against `results_power_mdd.json`; citations of experiments 28 and 34; pytest (5 passed); figure gridlines.

## Not run or not verified

GPU and LoRA scripts; Kohli's tables (abstract only); POPQUORN pay rates; CertJudge FSC; the SummEval paper text on expert revision; visual inspection of the figures.

TL;DR: The Lean and numerics are sound but over-labelled. Experiment 36's real-data headline is mostly an artifact: an over-regularized, uncalibrated Pearson head on near-signal-free features. After matched calibration the proper-score gaps shrink by 40–100× or vanish, and under fair tuning the held-out correlation differences are not significant.
