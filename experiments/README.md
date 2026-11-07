# Experiments

All experiments are **planned / in progress**. Don't add results here unless code in this folder actually produced them.

Origin: [Fathom call, 2026-11-07](https://fathom.video/calls/893310061). Notes are in [`../paper_latex_and_notes/notes/fathom_call_11-07-2026/`](../paper_latex_and_notes/notes/fathom_call_11-07-2026/notes.md).

| Folder | What | Status |
|---|---|---|
| `01_synthetic_pearson_vs_kl_squashed_judge_0to5/` | Synthetic 0–5 scores: high Pearson/Spearman, but large KL(P_human ‖ P_judge) | runnable |
| `02_kl_vs_corr_judge_fit_theorem_proving_human_ratings/` (todo) | Fit/evaluate LLM judges on human ratings of theorem-proving agent outputs by KL; report normalized MSE | waiting on human data |

## 01: toy counterexample

```bash
pip install numpy scipy
python experiments/01_synthetic_pearson_vs_kl_squashed_judge_0to5/kl_vs_corr.py
```

Output (seed 0, n=2000):

```
A: noisy faithful      pearson=0.900 spearman=0.900 KL(P_h||P_j)= 0.001 nMSE=0.023
B: inflated/squashed   pearson=0.877 spearman=0.878 KL(P_h||P_j)= 3.420 nMSE=0.216
```

The two judges have nearly the same correlation, but judge B only ever outputs 4 or 5, and KL exposes that.

## 02: theorem-proving judges (planned)

- Data: human scores r ∈ {0..5} for (agent output y, gold reference y*) pairs, with multiple raters per item where possible so we can compute per-item KL.
- Judges: LLM judges p_θ(r | y, y*), with θ = prompt/rubric/temperature/calibration.
- Objective: minimize E[KL(p_human ‖ p_θ)], compared against maximizing Pearson/Spearman.
- Report: KL, normalized MSE, and correlation for reference.
- Open question: how many human-rated pairs are needed.
