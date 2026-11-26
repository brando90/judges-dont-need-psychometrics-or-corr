# Experiment 11: verified QA corrections (11-26-2026)

**Date: 11-26-2026.** Codex CLI, GPT-6 Astra ultra (Vals account). CPU Python/raw-output checks; no Lean checking, no new model calls. Original protocols, result tables and raw outputs remain preserved.

**g2-H1 — CONFIRMED, MSE baseline verdict.** Old headline says MSE “beats `constant` across all four scenarios,” and prediction 3 is “Confirmed” for both `mse_fixed` and `mse_valrps`. Corrected 11-26-2026 verdict: **partly supported**. At 2,048 training tasks in scenario A, fixed-ridge mean-squared-error (MSE) training has RPS excess **0.018541735107**, versus constant **0.009551171834**; fixed MSE minus constant is **+0.008990563272 [0.008162369356, 0.009812701132]**, so fixed MSE is worse. Validation-tuned MSE has RPS excess **0.007590648402** in A; the original **0.0076–0.0118** winning range applies to validation-tuned MSE only. The evidence does not support a joint fixed-and-tuned baseline victory. Original wording, predictions and scored tables remain preserved.

**g2-H2 — CONFIRMED, calibrated-head baseline criterion.** Old prediction verdict: “Mostly confirmed … eliminates it by `n = 32`”; old headline: “no interval computed.” Corrected 11-26-2026 verdict: **reduction supported; elimination by 32 tasks is not established**. In scenario A at 32 tasks the tuned Pearson head has RPS excess **0.012532210319**, constant **0.011907058734**; head minus constant is **+0.000625151585 [0.000193884974, 0.001265643217]**, wholly above zero. An interval was computed and is saved in `results/summary.json`; phase 2 reproduces its exact endpoints from raw seed scores. Cross-fitting and slope tuning reduce the spike but do not satisfy the stated constant-baseline criterion in this cell.

Both checks use the same 100 seeds, 500 held-out test tasks/seed and 10,000-draw paired 95% seed-percentile bootstrap as the original report. Ranked probability score (RPS) excess is mean squared cumulative error versus known truth [0,1], lower better; cross-entropy (CE) excess is KL(true || forecast), nats. Positive same-order RPS differences above mean the named judge is worse than constant. The fits were already saved; phase 2 does not rerun the sweep or alter a locked prediction.

**g2-M3 — CONFIRMED evidence limitation; historical provenance remains open.** Old README: “Pre-registration … Written before executing”; old reported maximum `check_grad <= 2.41e-7`. README and raw sweep first appear together in commit `06054e3`, with no earlier freeze/smoke/launch receipt preserved. Current checks reproduce gradient errors **1.68e-7, 2.49e-7, 8.82e-8** and marginal-limit error **1.09e-9**. The current maximum satisfies `<2.5e-7`; it cannot disprove the historical 2.41e-7 or establish pre-execution timing. The preregistration is left intact with this adjacent dated caveat.

## Recomputed evidence and matched metric panel (11-26-2026)

Date: 11-26-2026. Read-only CPU reconstruction; no model calls.

g2-M3: first README appearance: 06054e3 2026-11-17 18:28:59 -0800
First raw sweep appearance: 06054e3 2026-11-17 18:28:59 -0800
No earlier freeze/smoke receipt is preserved here. A current gradient check cannot establish historical pre-registration or historical maximum 2.41e-7.
check_grad errors: ce_exact=1.68e-07, mse=2.49e-07, stage2_head=8.82e-08
Stage-2 c_b=1e8 max |q - P.mean(0)| = 1.09e-09, max|b| = 4.08e-09


Reproduce the evidence with `python3 docs/qa_11-25-2026/phase2/verify_g1_g2_remaining.py 11` from the repository root. HIGH-item checks are separately saved as `verify_g1-H1.py` (experiment 05), `verify_g2-H1.py` / `verify_g2-H2.py` (experiment 11), and `verify_g2-H3.py` (experiment 07), with their printed `.txt` outputs.
