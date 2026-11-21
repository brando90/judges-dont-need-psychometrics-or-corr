**Model: ChatGPT. Exact model/version, temperature, and reasoning-budget settings are not exposed to me. This is a text-only audit; no code, linked files, or external sources were inspected.**

## 1. MISMATCHES

Path abbreviations below expand to these exact supplied paths:

- **S:** `Version 27 one-pager (text of the artifact, 11-17-2026)`
- **I:** `experiments/README.md`
- **F:** `paper_latex_and_notes/notes/2026-11-16_sanmi_feedback_corr_vs_ce.md`
- **E9:** `experiments/09_open_weight_real_judges_native_prob_ce_vs_corr_start_16Nov2026`
- **E10:** `experiments/10_frontier_ai_judges_native_prob_and_prompt_search_ce_vs_corr_start_16Nov2026`
- **D9:** `E9/RESULTS_open_weight_ce_vs_corr.md`
- **D10:** `E10/RESULTS_frontier_ce_vs_corr.md`

### Contradictions, misleading labels, and documented deviations

| Issue | First side: exact quotation and file | Second side: exact quotation and file | Verdict |
|---|---|---|---|
| Synthetic experiment classified as human data | “**02, 11–14 (real ratings)**” — **S**, notation table | “**13 (synthetic)**” — **S**, same table | Wrong range. Write **02, archived 11, 12, 14**. |
| Human small-sample experiments called untested | “**small-sample gains on human ratings remain untested**” — **S** | “**3 families (~ 31.4 ratings)**”: CE `0.1848`, Pearson `0.1814`; “**5 families (~ 51.7 ratings)**”: CE `0.1771`, Pearson `0.1815` — **D10**, learning curve | Stale statement. Retrospective human small-n comparisons exist; prospective confirmation does not. |
| All crossovers unchanged versus only RPS crossovers unchanged | “**Complete; held-out crossovers unchanged under these settings; no global optimum certificate**” — **I**, experiment 07 | “**On held-out test CE, two scenario-A crossovers move by one grid step.**” — **S** | README must specify **RPS** crossovers. |
| Blanket native-probability CE victory contradicts raw-logprob result | “**native six-category probabilities beat the constant marginal CE baseline**” — **E9/results.md**, TLDR-end | “**Exp 09 Conditional Digit-Logprob Pool**”: constant `0.1857 / 1.887`, CE-mix `0.1712 / 2.273` — **D9**, §1.2 | Raw digit-logprob CE is **0.386 nats worse** than constant. Verbal and calibrated harness results cannot stand for every native-probability method. |
| Blanket single-judge victory has a counterexample | “**both ④ CE and ④ RPS beat ② Pearson on mixtures AND single-judge selection**” — **S** | “**Multi-Model x P0..P5 (Digit Logprob)**”: Corr-single `0.2886 / 8.713`, CE-single `0.3239 / 3.851` — **D9**, §2.1 | CE-selected single has **0.0353 higher RPS** here. Restrict the summary to the particular pools that support it. |
| AST claim versus regex implementation | “**abstract syntax tree (AST) obligation extraction**” — **S** | “**The legacy abstract syntax tree (AST) scaffold is a regex declaration summary, without Lean elaboration.**” — **D9**, provenance paragraph | For experiment 09, call it a regex scaffold. The existing label exaggerates what ran. |
| Registered tuning grid versus executed grid | “**frozen five-fold/five-value grid**” — **D10**, Figure 3 caption | “**Actual inner selection uses three folds and ridge c in {0.1, 1, 10}**” — **D10**, same caption | A disclosed protocol deviation—not a transcription error. Do not describe the executed analysis as wholly matching the frozen protocol. |
| Model identifiers versus executed models | “**legacy keys `H2_ast_gpt61__*` and `H3_twostep_gpt61__*`**” — **E10/README.md** | “**executed `gpt-5.4-mini`**”; “**`H1_single_mistral35__*` / `H2_ast_mistral35__*` executed `codestral-2508`**” — **E10/README.md** | Already disclosed, but machine-readable names are misleading. Any figure labels inherited from them need checking. |

### Numbers that are unreconciled—not established arithmetic errors

| Issue | First source | Second source | What can actually be concluded |
|---|---|---|---|
| Human benchmark definitions/results differ | “**RPS = `0.1113`, pairwise Pearson = `0.805`, Spearman = `0.798`**” — **D9**, §1.1 | “**Spearman `0.7642`, `ICC(A,1)` `0.7589`, RPS floor `0.0455`**” — **D10**, §2 | Both appear under leave-one-rater-out/human-ceiling labels. Observed LORO loss and an estimated disagreement floor can differ legitimately. Supply formulas and weighting; these numbers are not interchangeable. |
| Theorem-count benchmarks differ | “**theorem-count baseline Spearman = `0.546`**” — **D9**, §1.1 heading | “**All 75: `0.6014`; Ordinary 60: `0.3024`**” for the “**Theorem-count ratio dumb baseline**” — **D10**, §2 | Possibly count versus ratio or another implementation difference. The bundle does not reconcile them. |
| Open-weight single-selection headline lacks corroborating table | “**Single RPS +0.0352 [+0.0098, +0.0639]**”; “**0.1488 vs 0.1840**” — **S** | **no second source in this bundle** | Difference checks: `0.1840 − 0.1488 = 0.0352`. Pool definition and interval cannot be independently checked here. |
| Frontier single-selection interval lacks corroboration | “**+0.0174 [+0.0016, +0.0413]**” — **S** | **D10** gives `0.1426` and `0.1252`, but **no second source in this bundle** for that paired interval | Point difference checks. Interval does not. |
| Paper readiness, deadline, fresh-rating status | “**due 11-20-2026, 06:00 UTC**”; “**zero warnings/leaks**”; “**0/100 scored as of 11-17-2026**” — **S** | **no second source in this bundle** | These are assertions, not audited deliverables. |

**Checked:** supplied 09/10 differences, signs, interval containment, displayed MSE sums allowing rounding, 15 frontier configurations across seven models, and the prompt split totals: `27+18+30=75`, `69+45+81=195`. Nothing unambiguously wrong found.

**Cost checks:** `$0.50+$10.87+$2.97+$3.51=$17.85`; `$17.8511` rounds to `$17.85` and remains below `$18`. Nothing found. `$0.00` for open-weight work means cash charged, not zero computational cost.

The November 20 entries for experiments 15/16 do **not** contradict their absence from a November 17 brief. Primary results for those experiments, active toy 11, and most earlier experiments are not supplied.

## 2. OVERCLAIMS

- **Mechanism is not isolated.**  
  Claim: “**three follow-ups resolve the mechanism behind the historical 3-sample tie**” and “**Switching to native 6-category probabilities … flips both mixture and single-judge selection**” — **S**.  
  Supported: different pools perform differently. Models, prompts, harnesses, pool membership and sometimes splitting change together. D9/D10 explicitly admit this. A matched representation/smoothing intervention is still missing.

- **“Lost” overstates experiment 12.**  
  Claim: “**CE single-judge selection lost (−0.025 in exp 12, −0.032 in exp 14)**” — **S**.  
  Experiment 12’s interval is `[-0.0570,+0.0005]`: direction favors correlation, but the interval includes zero. Experiment 14’s reported interval excludes zero. Do not merge their evidential strength.

- **Native RPS single-selection superiority is incompletely supported here.**  
  Claim: “**train-CE … and train-RPS … single judges both beat train-Pearson**” — **S**.  
  D10 supports lower **point estimates** for both. This bundle supplies no paired Native-15 Pearson-minus-RPS interval. Do not infer it from overlapping or non-overlapping marginal intervals.

- **The direct-weight-training conclusion hides the strongest competing method.**  
  Claim: “**Weights & score-then-calibrate strongly favor proper loss**” — **S**.  
  On Qwen3-4B, **Pearson-LoRA + CE head** scores `0.1412`, versus validation-selected **CE-LoRA** at `0.1997`; the paired difference excludes zero. Direct CE-LoRA scores approximately `0.193–0.201` across the two models, without a matched five-fold constant baseline. These results support **proper calibration**, not an uncomplicated recommendation to train transformer weights directly by CE instead of correlation.

- **The “1D bottleneck” explanation is not established.**  
  Claim: “**still hits the 1D bottleneck whenever disagreement varies (A, B)**” — **S**.  
  In A, `E[R|x]=2.5+0.4x` identifies `x`. B’s mean is also strictly increasing on the tested domain. Thus an exact scalar mean can encode everything needed to recover `κ(x)`. The tested **linear-logit calibration head** may be too restrictive; dimensionality alone does not prove information loss.

- **The low-data threshold needs its objective/tuning qualifier.**  
  Claim: “**crossing at 12 ratings in targets B–D**” — **S**.  
  That describes fixed-ridge comparisons. Your own table gives **48 ratings for tuned D**, and **96 rather than 192 for tuned A**. These are tested-grid crossovers, not universal sample requirements.

- **“Sanmi’s prediction confirmed” is not a valid verdict.**  
  Claim: “**Weights vs prompts/harnesses (Sanmi's prediction confirmed)**” — **S**.  
  His supplied feedback asks for coarse prompt/harness experiments and notes historical single-selection results favoring correlation. Uncertain validation contrasts do not confirm a directional prediction. The frontier train-selected prompt contrast actually favors correlation: `−0.0744 [−0.2173,−0.0192]`.

- **Preregistration is narrower than the presentation suggests.**  
  Experiment 08’s new sweep can be preregistered, but A/B results already existed in archived experiment 13. Experiment 10 deviated from its frozen tuning grid. Experiment 15 explicitly says “**a smoke run was read before the README**” — **I**; that is exploratory, not confirmation. The supplied text correctly labels it exploratory. No evidence here establishes fraudulent preregistration.

- **Feedback-response chronology is not demonstrated.**  
  Experiments 09/10 say verified at **15:00 PST** on November 16; the feedback says **5:15 PM**, without a timezone. Experiment 04 ran November 13. Do not claim these timestamps establish that the experiments were initiated in response to that feedback.

**Retrospective status:** checked S, D9 and D10: they explicitly acknowledge reused ratings. Nothing found falsely claiming an actual fresh human test. Keep “zero-exposure” restricted to the within-run proposal/fitting firewall.

## 3. DONE VS NOT DONE

**Status means established by this bundle, not what might exist on disk.**

| Sanmi concern | Status | Evidence and remaining gap |
|---|---|---|
| **1. RPS-selected arm** | **Partly done** | **D10 Table 1** includes Hist-5, Native-15 and Combined-20 RPS-selected singles/mixtures. Important omitted counterexample: Hist-5 RPS-single is **0.2154**, worse than Pearson-single **0.1793**. The exact archived 12/14 pool/split follow-up is described as experiment 16 in **I**, but its results are absent. |
| **2. Toy baselines and calibration** | **Partly done / verified** | **S’s experiment 11 table** reports MSE, tuned cross-fitted heads, constant and truth; **D9 §4/D10 learning curve** establish real calibrated-head comparisons. The toy 11 results file is absent. A flexible scalar calibrator is still needed before asserting a 1D impossibility. At `n=2048`, the reported calibrated-Pearson gaps are only about **3.9% and 2.7% of irreducible RPS** in A/B. |
| **3. Small-n overfitting** | **Partly done** | Experiments **07/08/11 in S**, **D9 §4.3**, **D10 Table 2/learning curve** address budget, restarts, tuning and capacity. Pearson nonconvergence remains; frontier learning-curve intervals are conditional split variability. No convergence-robust, independently validated human sample threshold is established. |
| **4. Item signal versus rater severity** | **Partly done; not settled** | **S** acknowledges the problem. **I**, experiment 15, reports an exploratory additive artifact+rater-offset analysis, but supplies no results. Need rater overlap/connectivity, assumptions, uncertainty and severity-corrected comparisons. Two–three ratings alone do not prove separation impossible under every crossed design. |
| **5. References** | **Done in the brief** | **S Appendix D** attributes Lin, McGraw–Wong and Linacre and recognizes proper-loss psychometric models. Actual bibliography accuracy and paper integration are not independently checkable here. |
| **Drop “don’t need psychometrics”** | **Partly done** | Main prose is corrected; the repository name remains `judges-dont-need-psychometrics-or-corr`. Stop displaying that slogan as the project’s framing. |
| **Prompts/harnesses versus weights** | **Done as retrospective coverage** | **D9 §§2–4**, **D10 §§3–5** contain the requested experiments. Missing: robust repeated prompt-search evaluation, matched search spaces and fresh data—not more claims that the prediction was confirmed. |

## 4. COVERAGE

| Experiments | Frontier models | Open-weight models | Synthetic-only? | What is missing |
|---|---|---|---|---|
| **00, 01, 06, 07, 08, active 11, archived 13** | None | None | **Yes.** Analytic forecasts or 78-weight softmax toys; not LLM training | Primary result files absent here; flexible calibration and convergence-robust comparisons |
| **05** | No executed real-model test documented | None documented | Analytic/numerical work; real VeriBench test listed as not run | Executed real comparison |
| **04** | Historical judge analysis includes `H0_opus`; exact model versions not supplied | None established | **No:** synthetic counterexamples/G-study plus historical human analysis | Primary results; actual crossed-rater validation of allocation assumptions |
| **02, archived 11/12** | Cached `H0_opus`, `H0_sol`, `H1_sol`, `H2_opus`, `H2_sol` | None documented | No | Exact underlying model versions; calibrated scalar baselines; fresh families |
| **Archived 14** | Five historical configurations | 39 open-weight **judge configurations**; full model roster absent | No | Do not call these 39 models; matched representation analysis |
| **09** | None | Qwen3-4B-Instruct-2507; Qwen2.5-Coder-7B-Instruct; Qwen3-30B-A3B; Qwen3-14B; IDs `q25c_14b`, `ds_r1_qwen_7b`. Claims ten models, but no complete roster supplied | No | Full roster; matched LoRA constants; P2-exposure sensitivity; reliable full digit probabilities; fresh ratings |
| **10** | GPT-6 Astra; GPT-6.1 Sol/Terra; GPT-5.6 Sol/Terra; GPT-5.4 mini; Codestral 25.08 | None tested as an open-weight training arm | No | No frontier transformer-weight training; repeated prompt-search splits; matched mechanism experiment; fresh ratings |
| **15/16** | Cached judge analyses described; no new inference established | Cached pools potentially included in 16 | No: existing human ratings | Actual results files. README descriptions are not completed evidence |

## 5. TOP 5 NEXT STEPS

Costs below are **planning estimates**, not measured costs.

1. **Run one genuinely untouched, byte-verified human test.** Freeze one primary comparison before labels arrive; use new task families and overlapping raters so severity can also be examined.  
   **Cost:** approximately 240 ratings, 40–80 expert-hours; **$2–8k** assuming $50–100/hour, plus modest model costs. **New ratings: yes.**

2. **Make calibrated correlation the serious comparator.** Same folds, features, tuning budget and calibration family for correlation, MSE and CE; add matched LoRA constants and an untrained-score-plus-calibration baseline. Test nonlinear scalar calibration in the toys.  
   **Cost:** CPU work plus roughly 1–3 cluster-days; **$0 cash if existing free GPUs**. **New ratings: no.**

3. **Isolate the claimed empty-bin mechanism.** Same model, prompt, artifacts and folds; compare three votes, more votes, smoothing/floor choices and native probabilities. Finish and publish experiment 16 rather than citing its README.  
   **Cost:** cached sweep approximately **$0**; matched new inference roughly **$20–100**. **New ratings: no.**

4. **Replace precision theater with defensible uncertainty.** Refit under family-level resampling; distinguish sampling uncertainty from split Monte Carlo error; declare primary contrasts and multiplicity handling. Add P2 exclusion, equal search spaces and fallback sensitivity.  
   **Cost:** CPU/cluster time, approximately **$0 additional cash**. **New ratings: no.**

5. **Freeze an auditable defense packet.** One generated result ledger linking each claim to its pool, split, estimand and source; include missing 11/15/16 reports, reconcile human benchmarks, and delete unsupported mechanism/“confirmed” language.  
   **Cost:** **1–2 working days**, approximately **$0**. **New ratings: no.**