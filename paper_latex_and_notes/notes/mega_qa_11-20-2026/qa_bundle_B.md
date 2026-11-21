**Model:** ChatGPT, OpenAI. Exact model ID, reasoning-effort setting, temperature, and sampling parameters are not exposed to me. **Review settings:** supplied text only; no code execution or external verification.

The strongest new human-data results are not independently auditable from this bundle: the primary reports for experiments **09 and 10 are absent**. Several substantive mathematical claims are also wrong.

## 1. MISMATCHES

### Path key

References below expand literally to these paths:

```text
S   = Version 27 one-pager (text of the artifact, 11-17-2026)
I   = experiments/README.md
E00 = experiments/00_synthetic_pearson_vs_kl_squashed_judge_0to5_start_07Nov2026
E02 = experiments/02_exp06_ce_vs_pearson_matched_budget_start_09Nov2026
E04 = experiments/04_g_theory_reliability_vs_distribution_accuracy_start_13Nov2026
E05 = experiments/05_cross_entropy_vs_correlation_information_gap_start_13Nov2026
E06 = experiments/06_toy_rank_and_mixed_correlation_objectives_vs_ce_start_13Nov2026
E07 = experiments/07_toy_optimizer_restarts_and_budget_for_nonconvex_objectives_start_13Nov2026
E08 = experiments/08_toy_target_complexity_and_tuned_regularization_vs_ratings_needed_start_14Nov2026
E11 = experiments/11_toy_mse_and_tuned_head_calibration_vs_ce_start_17Nov2026
```

| Issue | Exact text and both sources | Finding |
|---|---|---|
| Wrong experiment identifier | **E00/README.md:** `# 01_synthetic_pearson_vs_kl_squashed_judge_0to5` versus **S:** `00_synthetic_pearson_vs_kl_squashed_judge_0to5_start_07Nov2026` | README heading says 01; canonical experiment is 00. |
| Impossible finite CE/KL as described | **E05/results.md:** “A1, two-point forecast on {0, 5} with the human mean” reports “1.000; 5.039 nats; 3.635 nats”. **E04/RESULTS_reliability_vs_accuracy.md:** “p_x = (9/10) Binomial(5, θ_x) + (1/10) Uniform{0..5}” | This population assigns positive probability to ratings 1–4; the stated forecast assigns zero. **CE and KL must be infinite.** Finite values require undisclosed smoothing or a different forecast. |
| Scenario A is not “spread only” | **E08/README.md:** “spread only” versus, in the same file, “μ = 0.5 + 0.08x (nearly constant)” | Its mean varies from 2.1 to 2.9. Say **mainly spread**, not exclusively spread. |
| KL mislabeled as total CE | **E11/results.md** and **S:** “③ Val-CE raw (ex., CE nats)” with values including `0.0066`. **E11/README.md:** “Test CE excess (KL(true \|\| judge), nats)” | These are **CE excess/KL**, not raw cross-entropy. The distinction matters when comparing them with human-data CE values such as 1.416. |
| Calibration spike described as eliminated throughout small n | **E11/results.md, TLDR-end:** “eliminates the naive calibration-head spike (`n <= 32`)”. Same file, verdict table: “down to `+0.0145..+0.0387`, and eliminates it by `n = 32` (`-0.0004..+0.0006`)” | It is **reduced**, not eliminated, across n≤32. Even “completely” eliminated at n=32 overstates a residual reaching +0.0006 without a corresponding uncertainty statement. |
| False unqualified historical minimum | **S:** “Lowest historical RPS (0.137)” versus “judge ② has test RPS 0.1419 and judge ④ has 0.1362” | 0.1362 < 0.137. Moreover, different splits cannot support a clean cross-experiment ranking. |
| Exp 04 cannot be a response to later feedback | **S:** “In response to Sanmi's 11-16 feedback” introduces “three follow-ups”, including exp 04. **E04/results.md:** “Phases A, B and C ran on 11-13-2026.” | Exp 04 predates that feedback by three days. Separate Nicole’s earlier study from the later Sanmi follow-ups. |
| Human small-n work simultaneously tested and untested | **S:** “small-sample gains on human ratings remain untested” versus “CE 48-wt heads beat Pearson from 5 training families ( ~52 ratings).” | Retrospective human learning-curve results are claimed. What remains untested is **fresh/prospective** small-n generalization. |
| Exp 02 integration status is stale | **I:** “integrated in AAAI-27 `6-case-study.tex` (`11-17-2026`)”. **E02/README.md:** “The paper has not been changed.” **I** also says “paper integration remains open”. | Documentation disagrees about completion. This is a stale-status problem, not evidence of a bad experiment. |
| Exp 04 integration status is stale | **I:** “main-body cross-reference added in `3-problem.tex` (`11-17-2026`)”. **E04/README.md** and **E04/results.md:** “main-body cross-reference remains optional.” | Update the older status files. |

**Two reporting ambiguities—not established data errors:**

- **E06/results.md:** “2 scenarios × 11 training sizes × 100 seeds × 3 objectives” accompanies “DONE: 2,200 runs”. That is **2,200 scenario/size/seed cells and 6,600 objective fits**. Define “run”; do not imply only 2,200 fits.
- **S:** “Qwen2.5-7B: +0.2088…” versus “Qwen2.5-Coder-7B-Instruct”. Use the full checkpoint name consistently; the shorthand obscures the Coder/Instruct distinction.

### Missing second sources

These are **verification gaps, not demonstrated contradictions**.

| Claim in S | Missing corroboration |
|---|---|
| LoRA effects `+0.1344 [+0.0495, +0.2220]` and `+0.2088 [+0.0957, +0.3037]`; native open-weight results and model/configuration counts | **No second source in this bundle** containing the primary results. `experiments/09_open_weight_real_judges_native_prob_ce_vs_corr_start_16Nov2026/RESULTS_open_weight_ce_vs_corr.md` is linked but not supplied. |
| Native-15 effect `+0.0775 [+0.0437, +0.1135]`; frontier heads, prompt/harness results and seven-model inventory | **No second source in this bundle** containing the primary results. `experiments/10_frontier_ai_judges_native_prob_and_prompt_search_ce_vs_corr_start_16Nov2026/RESULTS_frontier_ce_vs_corr.md` is not supplied. |
| “only 15/75 on-disk reference hashes match annotation metadata”; “ce_iter2_c1 has 7/30 held-out uniform fallbacks” | **No second source in this bundle.** |
| Paper “due 11-20-2026, 06:00 UTC”, page/build assertions, and “0/100 scored as of 11-17-2026” | **No second source in this bundle.** |
| Full archived 11–14 result tables and certificates | Reports absent. Sanmi’s note corroborates some rounded historical effects, not the complete results. |

**Checked: nothing found** in printed interval containment; the differences 0.1426−0.1252=0.0174 and 0.1840−0.1488=0.0352; 4×11×100=4,400 cells; 2×9×30=540 cells; 150+45=195 ratings; exp 04’s 20/23 prediction count and Kendall rank statistic −0.40; the supplied RPS appendix examples.

Also checked: **$17.85 agrees between S and I and is below the $18 cap**. November 15 UTC versus November 14 Pacific in exp 04 is not a date error. Exp 07’s Kendall crossover of 16 versus exp 06’s 8 uses 30 versus 100 seeds; it is not evidence that restarts changed the crossover.

## 2. OVERCLAIMS

### Mathematical errors

- **“1D bottleneck whenever disagreement varies (A, B)” — S.**  
  This explanation is wrong. In both A and B, the true mean is an **injective function of scalar x**. Therefore, an exact mean score contains enough information to recover x and hence κ(x). A sufficiently flexible scalar-to-distribution calibrator could recover the distribution. Your results establish limitations of the **learned score plus affine-logit head**, not an unavoidable one-dimensional information bottleneck.

- **“C and D, a 1-parameter exponential-family mean curve” and “a sufficient scalar summary” — E11/results.md.**  
  Fixed-concentration beta-binomial distributions are one-parameter, but the sample mean is **not generally sufficient**. For example, equal-sum observations `(0,2)` and `(1,1)` have likelihood ratio
  \[
  \frac{p_0p_2}{p_1^2}
  =\frac25\frac{\alpha+1}{\alpha}\frac{\beta+4}{\beta+3},
  \]
  which depends on μ when α=κμ and β=κ(1−μ). You are confusing **a mean that parameterizes a population family** with **a sufficient sample statistic**.

- **“Correlation fixes only the forecast mean” — E05/results.md.**  
  Pearson does not even fix that: positive affine transformations preserve correlation. Say it evaluates mean association without identifying level, scale, or distribution.

### Empirical overreach

- **“three follow-ups resolve the mechanism” / “Switching to native … flips both mixture and single-judge selection” — S.**  
  The mathematical zero-bin penalty is real. But the empirical causal explanation is not isolated: pools, models, prompts, output protocols and possibly splits change. Evidence supports **different outcomes under different protocols**, not that native probabilities alone caused the reversal. Run the matched floor/smoothing ablation.

- **“CE single-judge selection lost (−0.025 in exp 12…)” — S.**  
  Its interval is `[-0.0570, +0.0005]`. Say **lower point-estimate RPS under correlation selection; unresolved at the reported interval level**. The exp 14 single-selection interval does exclude zero.

- **“train-CE … and train-RPS … single judges both beat train-Pearson” — S.**  
  Native-15 CE has a supplied interval excluding zero. For RPS, the bundle supplies the point difference 0.1426−0.1360=**0.0066**, but no corresponding interval. “Both have lower reported point estimates” is supportable; both statistically beat Pearson is not verifiable here.

- **“③ Val-CE wins at every n ≥ 4” — S.**  
  Specify **against the tested calibrated heads**. It is false against all competitors: exp 08 scenario A, n=4 has tuned CE **0.0140** versus raw Pearson **0.0096** excess RPS. Likewise, the “12 ratings in B–D” crossover applies to fixed-ridge comparisons; tuned D crosses at **48 ratings**.

- **“Sanmi's prediction confirmed” — S.**  
  You have heterogeneous comparisons with different search spaces and precision. You have not established an objective-by-modification-axis interaction. Nor does a noisy prompt result establish that correlation is preferable for prompting.

- **Heads win “via lower parameter variance” — S.**  
  Plausible explanation, not measured mechanism. Parameter count, approximation error, regularization and optimization all differ.

- **“Calibrated” means probabilities “match observed rating frequencies” — S.**  
  Fitting a penalized CE head does not establish out-of-sample calibration. Call it a **CE-fitted calibration head**, and measure calibration separately.

### Prospective/preregistration limits

- Exp 08 A/B and exp 11 reuse previously inspected synthetic data/results. Preregistering **new analyses** is not an unseen-data replication.
- **I explicitly says exp 15’s smoke result was read before its README.** It must remain exploratory.
- Exp 04’s late secondary analysis is explicitly labeled post-result; I found no concealed prospective claim there.
- The one-pager generally acknowledges retrospective human exposure. Keep that qualification adjacent to the headline claims, not only in limitations.
- Exp 10’s split-Monte-Carlo intervals are correctly labeled, but they quantify uncertainty over splits of **this fixed dataset**, not population uncertainty over new task families.

## 3. DONE VS NOT DONE

Feedback source: `paper_latex_and_notes/notes/2026-11-16_sanmi_feedback_corr_vs_ce.md`.

| Concern | Status | Evidence and what remains |
|---|---|---|
| RPS-selected arm | **Partly done** | S reports native RPS singles/mixtures and a historical RPS mixture. The specifically requested historical **single-selection** repair is not documented by supplied results. I lists exp 16, but its results are absent. |
| Toy baselines and calibration | **Done as requested; interpretation needs correction** | **E11/README.md and E11/results.md** supply direct MSE, tuned/cross-fitted heads, and raw RPS against truth and constant. Remaining: correct the sufficiency/bottleneck claims and residual small-n wording. |
| Small-n overfitting | **Partly done** | **E07/results.md, E08/results.md, E11/results.md:** budgets/restarts, objective-specific tuning and improved calibration. Still missing: a clean converged/simple correlation comparator and fresh human validation. Most long-budget Pearson fits still fail to converge. |
| Item signal versus rater severity | **Partly done, not answered** | **E04/RESULTS_reliability_vs_accuracy.md** measures agreement; it does not establish why humans disagree. I lists exploratory exp 15, but no results are supplied. Common raters permit estimating relative offsets; two or three raters do not justify declaring all decomposition impossible. |
| References | **Done at the attribution level** | S names Lin, McGraw–Wong and Linacre; E04/README.md identifies the G-theory source. Actual bibliography/source-page verification is unavailable here. |
| Drop “don’t need psychometrics” | **Partly done** | S adopts complementary framing. Repository name remains, and **E04/README.md** still says “not needed to choose or train the judge”. Insufficiency of reliability coefficients does not establish uselessness of psychometrics. |
| Prompts versus weights | **Partly done / not independently verified** | S reports exp 09/10 prompt, harness, head and LoRA comparisons; primary reports are missing. No supplied evidence establishes the claimed interaction or frontier transformer-weight results. |

## 4. COVERAGE

“Frontier” below follows the bundle’s terminology, not an independently checked model classification.

| Experiments | Data/model coverage | Named models available | Missing |
|---|---|---|---|
| 00, 01, 05, 06–08, toy 11, archived 13 | **Synthetic/analytic only; no LLM judge** | None. Sonnet 5.5 in exp 05 is a coding worker, not an evaluated judge. | Human generalization; broader target families; fair flexible calibration control. |
| 04 A/B | **Synthetic/analytic only** | None | Difficult allocation conditions and deployment frequencies; current positive control is easy by construction. |
| 04 C; 02; archived 11/12 | **Cached frontier outputs + real ratings** | `openai/gpt-6-sol-high`; `anthropic/claude-opus-5-5`, max effort. Five prompt/model configurations, not five distinct models. | Fresh ratings, matched probability-estimation controls; archived primary reports missing. |
| Archived 14 | **Frontier + open-weight**, real ratings; 5+39=44 configurations | Frontier pool above; open-weight names not supplied | Open-weight inventory, primary results, matched scalar/native comparison. |
| 09 | **Open-weight**, real ratings; reported ten-model pool | LoRA: `Qwen3-4B-Instruct-2507`, `Qwen2.5-Coder-7B-Instruct`; meta-search: `Qwen3-30B` | Full pool inventory, primary results, fresh families/ratings, reproducible compute accounting. |
| 10 | **Frontier APIs**, real ratings; reported 15 configurations/seven models | Prompt search names `gpt-6.1-sol`, `gpt-5.4-mini`; Mistral models unnamed | Remaining model names, primary results, prospective test; no frontier transformer fine-tuning evidence. |
| 15 / 16 | **Existing real ratings/cached predictions; CPU analyses** | 15 unspecified; 16 reuses archived 12/14 pools | Results absent. Index entries do not establish completed findings. |

**No experiment in the supplied results establishes zero-exposure prospective human performance.**

## 5. TOP FIVE NEXT STEPS

Costs below are planning estimates, not reported expenditures.

| Rank | Action | Rough cost | New human ratings? |
|---|---|---|---|
| **1** | **Resolve provenance and make the headline results auditable.** Recover the exact human-delivered reference bytes; explain the 60 hash discrepancies; supply exp 09/10/15/16 reports, inventories and paired results. Correct the mathematical and reporting errors above. | $0 compute; roughly 1–3 researcher-days if records exist | **No**, unless packet identity cannot be recovered and rerating is necessary |
| **2** | **One genuinely prospective, preregistered comparison.** Freeze packets, prompts, pools, calibration and one primary contrast. Use new families and a balanced crossed-rater design. Compare CE/RPS with both raw and calibrated correlation methods. | Illustratively, 300 ratings at assumed $5–10 each: **$1,500–3,000**, plus ~$20–100 inference | **Yes** |
| **3** | **Isolate the claimed empty-bin mechanism.** First inspect exp 16; then compare CE/Pearson/RPS singles and mixtures on identical scalar pools across smoothing/floor settings. Match native/scalar models and prompts where possible. | **$0** cached CPU analysis; perhaps **$20–100** additional matched calls | **No** |
| **4** | **Complete the rater-severity analysis.** Obtain exp 15 results; report common-rater offsets, sensitivity of judge rankings to correction, and uncertainty. Do not equate residual disagreement with meaningful item ambiguity. | **$0** CPU; roughly 1 day | **No** for exploratory analysis; stronger identification should be built into step 2 |
| **5** | **Replace the false bottleneck argument with a decisive control.** Give scalar scores a flexible distributional calibrator; include a correctly specified beta-binomial baseline and properly nested tuning. Compare against simple/converged correlation and MSE models. | **$0 marginal CPU cost** on existing hardware; roughly 1–3 days | **No** |

One additional implementation warning for step 5: exp 11 describes Stage-2 CV on a precomputed Stage-1 OOF score vector. That is **not automatically nested end-to-end CV**: Stage-2 training features may depend on Stage-2 validation labels through Stage 1. Fresh outer test tasks remain untouched, but the tuning procedure needs an explicit leakage audit.