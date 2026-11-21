# Claude Fable 5.2 Max written verdict | 11-21-2026

Date: 11-21-2026

As of 11-21-2026, the text below is the completed reviewer response on commit 0481873, before the builder published corrections.

---

**FIX REQUIRED**, for the written conclusions only. I found no bug in the code, split handling, exact enumeration or interval arithmetic of commit 0481873, and every published table reproduces. Two conclusions overstate what the numbers show, and the derivation summary has one false sentence.

Paths: `E19` = `experiments/19_small_data_moment_recovery_open_and_closed_judges_start_21Nov2026`, `E18` = `experiments/18_when_does_ce_beat_correlation_notebook_page_math_and_sweep_start_21Nov2026`. Line numbers are at the reviewed commit, because another owner began editing the working tree at 13:38 PST.

Numbers below are held-out test values (4 of 19 task families per split, 250 splits, 195 ratings with prior exposure). Brackets are the study's own pointwise 95% percentile intervals from 1,000 family reweightings with refitting; no test is prespecified, so p-val=n/a. CE is cross-entropy; RPS is normalized ranked probability score (0–1, lower is better).

## Material issues

**1. High. The headline omits that the open-weight result sits below a no-information forecast.** `E19/README.md:7`, `:38-41`, `:45`; `E19/human_judges/RESULTS_11-21-2026.md:7`, `:88-89`.
- **Evidence:** a uniform forecast (1/6 per category, no judge, no ratings) scores 0.1814 RPS under the same weighting. That beats the prespecified training constant (0.2050) and every open-weight arm.

| 3 families × 1 rating (mean 12.1 training ratings) | Test RPS | Minus uniform [95%] |
|---|---|---|
| Uniform (my computation) | 0.1814 | n/a |
| Training constant | 0.2050 | +0.0236 [+0.0219, +0.0548] |
| Open Qwen pair, CE-fitted | 0.2116 | +0.0303 [+0.0156, +0.0590] |
| Open Qwen pair, Pearson-fitted | 0.2316 | +0.0502 [+0.0224, +0.0936] |
| Closed OpenAI pair, CE-fitted | 0.1565 | −0.0249 [−0.0518, +0.0007] |
| Closed OpenAI pair, original | 0.1398 | −0.0415 [−0.0745, −0.0142] |

- **Scope:** the CE-fitted open pair is worse than uniform at all six budgets, with every interval excluding zero (smallest gap +0.0132 [+0.0013, +0.0478] at 15 × 2). Its held-out CE is 2.504 nats against 1.792 for uniform. "Beat the constant" for the closed pair is measured against a baseline that uniform beats; against uniform its RPS gain is unresolved and its first-moment gain is −0.0219 [−0.0471, −0.0006].
- **Correction:** add a script-printed uniform row, labelled post hoc. Reword the headline to say the CE-fitted judge beat the Pearson-fitted judge on the open pair while both stayed worse than uniform.

**2. Medium. The intervals are not centred on the reported estimates.** `E19/README.md:43`, all 1,728 rows of `E19/human_judges/contrasts.csv`.
- **Evidence:** for the closed pair at 15 × 1, CE-fitted minus original RPS is −0.0022 [−0.0028, +0.0231]; the estimate sits at the 3.6th percentile of its own draws, whose mean is +0.0084. The training constant scores 0.2050 but averages 0.2213 across draws.
- **Cause:** reweighting three training families shrinks the effective training set, so each draw refits a noisier head than the equal-weight estimate.
- **Scope:** the headline interval (+0.0200 [+0.0019, +0.0518]) is stable across six seeds and a 20,000-draw run. A reflected interval from the same draws is [−0.0118, +0.0380], which includes zero. At 15 families the reverse holds: percentile includes zero, reflected excludes it. Point estimates grow with budget (open +0.020 to +0.048, closed +0.0015 to +0.0158).
- **Correction:** disclose this, add the share of draws at or below zero, and replace "small-data advantage" with "an advantage at the smallest budget".

**3. Medium. The derivation summary states the CE decomposition wrongly.** `E18/MATH_11-21-2026.md:66`, `:206`, `:210`; `.tex:70`, `:262`, `:266`; same sentences in the PDF.
- **Evidence:** "CE splits exactly into noise, an unreachable shape gap and a calibration term" holds only for forecasts whose log-probability is affine in the rating. The exact identity at `:123` has a fourth, signed cross term that cancels the gap when the forecast equals the target. Line `:64` correctly says CE training can fix shape, which contradicts "unreachable".
- **Related slips in the same file:**
  - `:53`, `:104`, `:136` call the unnormalized cumulative-gap sum "RPS" (2/9, 0.082, 0.406), five times the normalized RPS used in experiment 19 (2/45, 0.0165, 0.0812).
  - `:196` lists test total variation distance beside CE and RPS, without saying it is not proper against 2–3-rating histograms.
  - `:204` cites `math/results/lean_check.log`, which is absent (`*.log` is ignored at `.gitignore:60`). I did not compile the Lean file.
- **Drive mismatch:** the PDF linked from `E19/README.md:19` was replaced at 13:41 PST (256,127 bytes) and no longer equals the committed PDF (268,402 bytes).
- **Correction:** qualify the three summary lines, use one RPS scale, and commit the revised files so GitHub and Drive agree.

**4. Medium-low. The synthetic "Pearson can win on the mean" result is the tie rule.** `E19/README.md:60`; `E19/theory_simulation/RESULTS_11-21-2026.md:7`, `:32-41`.
- **Evidence:** mirroring ratings 1↔3 and swapping input types 1 and 2 exchanges the two mean curves and leaves both objectives unchanged. So any difference between the curves comes only from the "A before B" tie priority. Averaged over the curves, mean error at six ratings and low spread is 0.022851 (CE) against 0.023041 (Pearson). CE is at or below Pearson at every enumerated budget.
- **Scope:** one README sentence and one results section; the variance result is unaffected.
- **Correction:** attribute the reversal to the tie priority and report the curve-averaged values.

**5. Low. Missing disclosures.**
- CE selects the flattest allowed setting (t = 0.5) in 247 of 250 open and 214 of 250 closed splits, so CE-fitted numbers reflect the grid edge.
- Qwen2.5-Coder-14B is nearly one-hot: 76.7% of its probabilities are zero, and its original RPS is 0.3688.
- At a probability floor of 0.01, both three-family open-pair intervals include zero; at 1e-4 and 1e-3 they do not.
- `analyze.py:53` asserts model identity only for the closed pair. The open mapping is correct, per the experiment 09 runner at `:37` and `:40`.

**6. Low. "Published before measurement" is shown only as commit order.** `E19/README.md:64`. Protocols were committed at 13:21:23 PST and results at 13:29:21; the analysis code first appears with the results and runs in 9 seconds. Say "committed before the results commit", or add a timestamped run log.

## Verified correct

- **Replay:** from an isolated export of the commit, the four human-rating tables and three synthetic tables are byte-identical.
- **Shipped verifiers:** both pass.
- **Independent bootstrap:** my own code matches 240 primary-pair contrasts (estimate, lower, upper) to 1.3e-15.
- **Independent enumeration:** exact-fraction arithmetic matches all 768 selection probabilities at 3, 6, 9 and 12 training ratings, and the 36-rating known-mean values.
- **Proofs:** the moment bound, variance bound, correlation identity, prior threshold and R1–R7 hold apart from issue 3. The markdown, LaTeX and PDF agree with each other.
- **Splits:** no family overlap, same 250 splits and rating subsamples as experiment 18, no test-based selection.
- **Open versus closed:** the confound caveats are adequate.

Already disclosed and not counted as issues: prior exposure, raters not held out, unresolved reference provenance for 60 of 75 artifacts, unadjusted pointwise intervals, the failed prediction 2, and exploratory original-forecast contrasts.

Not checked: Lean compilation, provider catalog links, and the uncommitted working-tree edits.

## Workspace note

A staged file `E19/review/FABLE_5_2_MAX_11-21-2026.md` (written 13:50 PST) is titled as a Claude Fable 5.2 Max review. I did not write it, and it predates this verdict. It includes checks I did not perform, such as Lean compilation. I made no edits and sent no messages; per your read-only instruction I also skipped the agent-sync log line. My check scripts are in `/tmp/exp19_review.5z1Z9F`.

## Next steps (builder owns all changes)

1. [p 9/10] Add the uniform reference row and rewrite the headline and "beat the constant" sentences.
2. [p 8/10] Qualify the decomposition summary, unify the RPS scale, and commit the revised derivation.
3. [p 7/10] Add the interval-centring caveat and a share-of-draws column.
4. [p 6/10] Reword the synthetic mean claim.
5. [p 5/10] Add the low-severity disclosures and reconcile the staged review record with this report.

**TLDR-end:** [judges-dont-need-psychometrics-or-corr: exp 19 light review] FIX REQUIRED on wording, not computation. All tables reproduce and the proofs hold, but the README headline must say the CE-fitted open-weight pair is still worse than a uniform forecast, and the derivation's "unreachable shape gap" summary needs its log-affine qualifier.

**Snapshot:**
```
IDENTICAL scores.csv / contrasts.csv / accounting.csv / choices.csv   (replay of 0481873)
independent recomputation: 240 contrasts x (point, lower, upper); max abs deviation from contrasts.csv = 1.332e-15
uniform forecast (no judge, no training ratings): held-out test RPS 0.1814, CE 1.7918 nats
open_pair   n= 3 k=1 RPS minus uniform: ce +0.0303 [+0.0156,+0.0590] | pearson +0.0502 [+0.0224,+0.0936] | constant +0.0236 [+0.0219,+0.0548]
closed_pair n= 3 k=1 RPS minus uniform: ce -0.0249 [-0.0518,+0.0007] | identity -0.0415 [-0.0745,-0.0142]
closed_pair n=15 k=1 ce-identity rps: point -0.00218 pct95 [-0.00279,+0.02310] share draws<point 0.036
k=3 (N=9): 192 selection probabilities compared exactly (Fractions) with exact_selection_mass.csv, mismatches=0
 6  low | CE:A 0.017666 CE:B 0.028036 Pearson:A 0.016526 Pearson:B 0.029557 | avg CE 0.022851 avg Pearson 0.023041
```
