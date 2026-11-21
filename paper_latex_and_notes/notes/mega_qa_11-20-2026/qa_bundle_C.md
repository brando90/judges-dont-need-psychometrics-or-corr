**Model/settings:** OpenAI ChatGPT; the exact model identifier, temperature, and reasoning-budget settings are not exposed to me. This is a text-only audit: no code execution or external verification.

**Scope:** 11 supplied files. The primary results for experiments **09, 10, and toy 11 are absent**. Your strongest new empirical claims therefore cannot be independently verified from this bundle.

## 1. MISMATCHES

Path abbreviations below expand to these exact supplied paths:

- **V** — `Version 27 one-pager (text of the artifact, 11-17-2026)`
- **I** — `experiments/README.md`
- **A** — `experiments/archive/README.md`
- **12/** — `experiments/archive/12_formulation2_vs_4_corr_vs_ce_80_20_splits_start_10Nov2026_end_10Nov2026/`
- **14/** — `experiments/archive/14_open_weight_judge_pool_corr_vs_ce_80_20_start_10Nov2026_end_10Nov2026/`
- **15/** — `experiments/15_rater_severity_vs_item_signal_start_20Nov2026/`
- **16/** — `experiments/16_rps_selected_arm_and_probability_floor_on_scalar_pools_start_20Nov2026/`

Quotes are exact excerpts; column labels identify isolated table values.

| Issue | Claim and path | Conflicting text and path | Correction |
|---|---|---|---|
| **Nine selections unaccounted for** | **12/results.md:** “④ H2 Opus 601, H0 Sol 313, H2 Sol 77.” | **12/results.md:** “Repeated splits (1000)” | \(601+313+77=991\). Report the remaining nine CE selections. This is an incomplete count, not evidence that only 991 splits ran. |
| **Frontier arm denied despite cached frontier judges** | **12/README.md:** “no frontier-model arm.” | **12/results.md:** “GPT-6 Sol high and Claude Opus 5.5 max” | There **is a cached frontier-model arm**. There are no new frontier scoring calls. |
| **Floored CE called unbounded** | **12/results.md:** “with q floored at 1e-6.” | **12/results.md:** “Proper but unbounded” | The implemented floored loss is bounded at approximately **13.816 nats**. Unclipped CE is unbounded. Flooring also requires qualifying strict-propriety claims about the original forecasts. |
| **Wrong cached-Pearson range** | **14/results.md:** “versus 0.62 to 0.78” | **14/results.md:** `H0_opus`, Pearson column: “0.609” | The printed cached range rounds to **0.61–0.78**, not 0.62–0.78. |
| **Not every chooser worsens** | **14/results.md:** “adding it makes every chooser worse than on the old five” | **14/results.md**, both old and combined tables: “② best single by train Pearson” has RPS “0.1797” | Pearson-single is unchanged across every printed metric; the constant is also unchanged. |
| **Ranking changes denied** | **15/README.md:** “correcting for them changes no judge ranking” | **15/README.md:** “Kendall tau 0.996 (Spearman), 0.998 (Pearson), 1.000 (RPS)” | Some correlation-based rankings change. **The winner stays unchanged; the RPS ranking stays unchanged.** |
| **“Best arm” loses to constant** | **16/results.md:** “Mixing judges by train RPS is the best arm on every pool” | **16/results.md**, open pool: Constant “0.1836”; RPS mixture “0.2601” | Best **nonconstant fitted arm** in that table, not best arm overall. The constant wins on the open pool. |
| **Single selectors are not all equally bad** | **16/results.md:** “it is as bad as picking by CE or by correlation” | **16/results.md:** “CE single minus RPS single is -0.0251 [-0.047, -0.002], so RPS-picking is clearly worse there” | On the open pool, RPS-single is significantly worse than CE-single under the reported interval. Elsewhere, nonsignificance does not establish equality. |
| **Synthetic experiment classified as real** | **V:** “02, 11–14 (real ratings)” | **V:** “13 (synthetic)” | Exclude archived **13** from the real-rating row; distinguish archived 11 from toy 11. |
| **LoRA model name changes** | **V:** “Qwen2.5-7B: +0.2088 [+0.0957, +0.3037]” | **V**, Figure 1b: “Qwen2.5-Coder-7B-Instruct” | Preserve **Coder**. The bundle separately lists Qwen2.5-7B-Instruct and Qwen2.5-Coder-7B-Instruct as different models. |
| **False “lowest historical” label** | **V:** “Lowest historical RPS (0.137)” | **12/results.md:** “④ mixture minimizing train CE” has RPS “0.1362” | 0.137 is not the lowest printed historical result. Moreover, different split protocols should not be ranked without qualification. |
| **Optimizer robustness summary drops changed crossovers** | **I:** “held-out crossovers unchanged under these settings” | **V:** “On held-out test CE, two scenario-A crossovers move by one grid step.” | Say **RPS crossovers unchanged**, not all held-out crossovers. |
| **CE excess mislabeled as CE** | **V**, toy-11 header: “③ Val-CE raw (ex., CE nats)”; A entry: “0.1876 (0.0007, 0.0066)” | **V** defines CE as “average −ln q(rating)” and separately defines “CE excess” as KL divergence | Label 0.0066 **CE excess/KL**, not ordinary CE. With true RPS 0.1869, raw expected CE cannot be 0.0066: even the elementary entropy lower bound \(H(R)\ge 2\,\mathrm{RPS}(p)\) gives 0.3738 nats. |
| **Stale human small-n status** | **V:** “small-sample gains on human ratings remain untested” | **V:** “CE 48-wt heads beat Pearson from 5 training families ( ~52 ratings)” | Specify what remains untested: prospective gains or correlation’s hypothesized advantage. Retrospective human small-n comparisons are already reported. |
| **Paper-integration status conflicts** | **I**, experiment 02 date table: “integrated in AAAI-27 `6-case-study.tex` (`11-17-2026`)” | **I**, historical comparisons table: “paper integration remains open” | Update the stale status, or specify which integration work remains. |
| **“No folder changed” is literally false** | **I:** “No folder changed after its commit (the moves are byte-identical renames).” | **A:** “a `HOW_TO_READ_RESULTS.md` (added 11-13-2026)” | Say **frozen scientific files were unchanged**. Folder contents and navigation documentation changed. This does not establish corruption of frozen results. |

### Unverifiable or unresolved—not demonstrated contradictions

- **V:** “Native-15 CE 1.416 nats”; “+0.1344 [+0.0495, +0.2220]”; “0.1301 RPS”; toy-11 “4,400 cells”: **no second source in this bundle** containing the underlying result tables.
- **V:** “$17.85” agrees with **I**, but there is no billing ledger. The workshop deadline, PDF page/build claims, and “0/100 scored” likewise have **no second source in this bundle**.
- **V:** “exp 02 uses 998/1,000 valid family refits” versus “2 of about 49,000 bootstrap refits failed and were kept”: these use different denominators. Clarify whether failed fits were retained in raw output, interval calculation, or both; the supplied text cannot settle it.
- Experiments 15–16 date to **November 20**, after V’s **November 17** snapshot. Their absence from V is staleness, not a date contradiction.

**Checked:** printed interval endpoints contain their point estimates; principal paired-difference signs agree with their operands; \(75+75+45=195\); \(15\times3-6=39\); \(39+5=44\); experiment 14 has 16 negative open-pool Pearsons, median 0.062 and maximum 0.407. The RPS appendix’s worked examples and migration-file totals also check out. **Nothing found in those checks.** Differences of 0.0001 compatible with rounding were not counted as errors.

## 2. OVERCLAIMS

1. **Native probabilities have not isolated the mechanism.**  
   **V:** “fixed by native probabilities and RPS” and “three follow-ups resolve the mechanism behind the historical 3-sample tie.”  
   You changed models, prompts, pools, harnesses and sometimes protocols—not merely the probability representation. Experiment 16 supplies a narrower intervention: smoothing substantially changes scores, but historical RPS-single still fails to fix selection. Supported: **the newer pipelines perform better; empty predicted bins explain large CE penalties.** Not supported: format alone caused the selection reversal.

2. **The “1D bottleneck” explanation is wrong as a general explanation of these toys.**  
   **V:** “still hits the 1D bottleneck whenever disagreement varies (A, B).”  
   In **both A and B, the true mean is one-to-one in scalar \(x\)**. An exact mean score therefore retains all information needed to reconstruct \(\kappa(x)\). Your **linear-logit 12-parameter head** may be too restrictive; that is not proof of an information-theoretic scalar bottleneck. Test a nonlinear scalar calibrator.

3. **“Beats” needs an uncertainty qualifier.**  
   **V:** “CE single-judge selection lost ( −0.025 in exp 12 …)” conflicts with the strength warranted by **−0.0254 [−0.0570, +0.0005]**. It is a point-estimate loss, not a resolved difference.  
   Native-15 RPS-single’s printed advantage is \(0.1426-0.1360=0.0066\), but its interval is absent. “Help clearly” for the two-step harnesses likewise rests here on point estimates, not supplied paired intervals.

4. **Different significance results do not establish a weights-versus-prompts interaction.**  
   **V:** “Weights vs prompts/harnesses (Sanmi's prediction confirmed).”  
   Strong weight results plus noisy prompt results do not prove their effects differ. They use different search spaces, budgets and validation sizes. Sanmi’s note requests the comparison; it does not supply a preregistered quantitative prediction confirmed by these results.

5. **Variance reduction is an explanation, not a measured mechanism.**  
   **V:** calibration heads edge out CE “via lower parameter variance.”  
   Supported: the smaller heads have lower reported test loss in C/D at \(n=2,048\). You have not supplied a bias–variance decomposition establishing why.

6. **Constant rater offsets do not settle the source of disagreement.**  
   **15/README.md:** “item signal dominates” and “severity is not what the ratings are measuring.”  
   Supported: additive constant offsets explain an estimated **2.1%** of within-artifact variance, with interval **−1.3% to 20.8%**, for these three raters. Rater-specific thresholds, scales, item interactions and transient error remain unresolved. Large between-artifact variation does not demonstrate that **within-artifact disagreement** is meaningful item ambiguity.

7. **Preregistration is partial and conditional on substantial prior knowledge.**  
   **16/README.md:** “Pre-registered rerun” needs its exception attached: old5/1e-6 RPS results were already seen.  
   **V:** “Hypotheses and the decision rule were committed before measurement” for experiment 08 must distinguish newly measured controls/tuning from the already-known A/B fixed-ridge results. New calculations on reused data are not fresh confirmatory evidence. Experiment 15 correctly labels itself exploratory.

8. **Proper scoring rules guarantee an estimand, not finite-sample dominance.**  
   **V:** “fit, select, or calibrate with a strictly proper scoring rule” is defensible as a distributional-target recommendation—not as a theorem that CE/RPS training wins with limited data, restricted models or imperfect optimization.  
   Also, “only proper losses use” disagreement is false: variance-based objectives and improper distribution distances can use it too.

9. **The extremely narrow head intervals are not human-population uncertainty.**  
   **V:** “+0.0250 [+0.0238, +0.0262]” is explicitly conditional split-Monte-Carlo uncertainty. Keep that distinction prominent when claiming an advantage “from ~52 ratings.” Repeatedly splitting the same 19 families does not create additional independent human evidence.

## 3. DONE VS NOT DONE

Feedback source: `paper_latex_and_notes/notes/2026-11-16_sanmi_feedback_corr_vs_ce.md`.

| Concern | Status | Evidence and what remains |
|---|---|---|
| **RPS-selected arm** | **Done** | **16/README.md; 16/results.md** directly add singles and mixtures to the historical pools. The answer is mixed—not a universal rescue. Native results are reported in V but lack primary backing here. |
| **Toy baselines and calibration** | **Partly done** | V’s toy-11 section reports MSE, cross-fitting, tuned slope regularization, and raw/excess RPS. The referenced toy-11 results file is absent. A flexible nonlinear scalar calibrator is still needed before claiming a 1D information bottleneck. |
| **Small-n overfitting** | **Partly done** | V’s experiments **07, 08, 11**, plus reported 09/10 learning curves, address restarts, ridge tuning and calibration. Persistent Pearson nonconvergence, estimator-capacity effects and prospective human small-n behavior remain unresolved. |
| **Item signal versus rater severity** | **Partly done** | **15/README.md** establishes that constant offsets are estimable and small in point estimate. No repeated ratings; no separation of interaction from occasion noise; no general rater-population inference. |
| **References** | **Done in the text** | V explicitly adds Lin, McGraw–Wong, Linacre and complementary G-theory framing. Bibliographic accuracy is not independently checked: no bibliography or cited papers supplied. |
| **Drop “don’t need psychometrics”** | **Partly done** | V’s substantive argument is narrowed appropriately. The displayed repository branding still says `judges-dont-need-psychometrics-or-corr`. |
| **Prompts versus weights** | **Partly done** | V reports experiments **09/10** across prompts, harnesses, heads and LoRA. Their primary results are absent; a matched comparison establishing an axis-dependent effect is not supplied. “Measured” is not “prediction confirmed.” |

## 4. COVERAGE

“Used” includes reused cached outputs—not necessarily new model calls.

| Experiments | Frontier models | Open-weight models | Synthetic only? | Missing |
|---|---|---|---|---|
| **02; archived 11–12; historical 06** | Cached **GPT-6 Sol high; Claude Opus 5.5 max** | None documented | No | Fresh data; primary 02/archived-11 reports absent here. |
| **Archived 14; 16; 15’s cached-judge analysis** | Same cached frontier pair | Fifteen attempted models listed below; 39 usable model×prompt judges | No | Stronger scalar judges, repeated scoring runs and clean human holdout. Experiment 16 itself makes no model calls. |
| **09** | None documented | Reported ten-model pool. Named: **Qwen3-4B-Instruct-2507**, **Qwen2.5-Coder-7B-Instruct**; **Qwen3-30B** named as meta-prompter | No | Primary results, complete ten-model roster, configuration-level provenance and fresh ratings. |
| **10** | Reported seven models across OpenAI/Mistral. Named: **gpt-6.1-sol**, **gpt-5.4-mini** | No local open-weight arm documented | No | Complete seven-model roster, primary tables and billing ledger; fresh ratings. |
| **04** | Historical **H0_opus** explicitly named | None documented | **Mixed:** synthetic counterexamples/design simulations plus historical judges | Primary results; real-data validation of the D-study allocation benefit. |
| **00, 01, 06–08, toy 11, archived 13** | None | None | **Yes**—analytic forecasts or small synthetic softmax judges, not LLM weight training | External validity; stronger scalar-calibration controls. Most primary reports absent. |
| **05** | No run documented | No run documented | Theory/numerical work | Planned VeriBench experiment explicitly not run. |

**Archived-14 attempted open models:** Qwen2.5-7B-Instruct; Qwen2.5-Coder-7B-Instruct; Qwen2.5-Coder-14B-Instruct; Qwen3-4B-Instruct-2507; Qwen3-8B; Qwen3-14B; Qwen3-30B-A3B-Instruct-2507 AWQ; Llama-3.1-8B-Instruct; Mistral-7B-Instruct-v0.3; Granite-3.3-8B-Instruct; Hermes-3-Llama-3.1-8B; GLM-4-9B-0414; Olmo-3-7B-Instruct; Seed-Coder-8B-Instruct; Phi-4-mini-instruct. Phi’s three configurations were dropped.

**Do not count** Claude 3.7 Sonnet artifact generation or Sonnet 5.5 orchestration as additional judge arms.

## 5. TOP FIVE NEXT STEPS

Costs below are estimates, not reported experiment costs. Human-cost examples assume **10 minutes/rating and $50–100/hour**.

| Rank | Step | Rough cost | New human ratings? |
|---|---|---|---|
| **1** | **Freeze and run one genuinely prospective comparison.** New task families; fixed prompts, models, parsing, calibration and primary paired endpoint. Include constant, correlation selection, proper-loss selection and score-then-calibrate. Freeze exact artifact/reference bytes before annotation. | Example: 100 artifacts × 3 ratings = **50 rater-hours, $2,500–5,000**, plus scoring | **Yes** |
| **2** | **Repair provenance and produce a defensible results bundle.** Resolve the 60/75 reference-hash mismatches; supply 09/10/toy-11 tables and contrast definitions; regenerate summary numbers; separate Monte-Carlo from sampling intervals; rerun uncertified RPS fits. | **1–2 analyst-days**, modest CPU cost | **No**, unless original rated inputs cannot be recovered |
| **3** | **Run the matched representation ablation.** Same model, prompt, artifacts and splits: three scalar votes versus native probabilities, crossed with smoothing and selection objective. This tests the claimed mechanism rather than comparing different pipelines. | Budget **$20–100 API**, or cluster time; roughly 1–2 analyst-days | **No** |
| **4** | **Attack the alleged scalar bottleneck.** Add nonlinear scalar-to-distribution calibration, cross-fitted and independently regularized; retain MSE and constant baselines. Report raw loss, not just excess ratios. | CPU cash roughly **$0–20**; about one analyst-day | **No** |
| **5** | **Collect crossed, repeated rater measurements.** Balance item overlap and include repeat occasions; fit rater thresholds/scales and separate repeat noise from stable rater×item effects. Integrate this into step 1 if possible. | Example: 20 items × 6 raters × 2 occasions = **40 rater-hours, $2,000–4,000** | **Yes** |

**Defense boundary:** the nonidentification argument is established by the stated construction. Universal superiority of proper-loss training, the native-probability causal mechanism, and an intrinsic one-dimensional calibration bottleneck are not.