# Experiment 04: verified QA corrections (11-26-2026)

**Date: 11-26-2026.** Codex CLI, GPT-6 Astra ultra (Vals account). CPU Python/raw-output checks; no Lean checking, no new model calls. Original protocols, result tables and raw outputs remain preserved.

**g1-M4 — CONFIRMED evidence limitation; reconstruction remains open.** Old Phase C claim: “+0.082 [0.004, 0.168]” RPS gap. Corrected interpretation 11-26-2026: subtracting saved aggregate scores reproduces **+0.082221542223**, and saved ranking orders give Kendall **−0.4**. Private experiment 02 ratings/predictions are absent; individual-rating scoring and the 10,000-draw task-bootstrap interval cannot be independently reconstructed. The estimate is not shown wrong and remains unchanged.

**g1-L3 — CONFIRMED, integration status.** Old text: “decide whether to add a main-body cross-reference” / “main-body cross-reference remains optional.” Corrected 11-26-2026: the main-body cross-reference was added on 11-17-2026 and exists in `paper_latex_and_notes/AAAI_2027_EvalAIScientists_JudgesKL/3-problem.tex` at the construct-validity paragraph. The reliability appendix is already integrated; the harder Phase B follow-up remains unrun.

**g1-L7 — CONFIRMED, metric order.** Old A1 table leads with Pearson/Spearman/Kendall before distribution scores, and Phase C leads with repeatability. Corrected 11-26-2026: the companion panel below places ranked probability score (RPS), cross-entropy (CE) and total variation distance (TVD) before secondary Spearman and Pearson. A1 scores are exact population expectations over 11 synthetic artifacts, reconstructed from rational probabilities; Phase C values are labeled aggregate-only transcriptions. RPS averages five squared cumulative errors [0,1], CE is negative log probability (nats), TVD is half the absolute probability difference [0,1], all lower better. A1 uses the true distribution as its reference; Phase C scores individual ratings and empirical rating histograms. Correlations compare expected ratings with the corresponding target means, range [-1,1], higher better. Phase C `test` is the historical 30-artifact/10-task split; `all75` is a retrospective all-data diagnostic, not an independent new test.

## Recomputed evidence and matched metric panel (11-26-2026)

Date: 11-26-2026. Read-only CPU reconstruction; no model calls.

g1-M4: aggregate RPS gap H0_opus minus H0_sol = 0.08222154222334213
Aggregate ranking Kendall tau = -0.39999999999999997
Private Phase C dataset missing; scores and 10,000-draw intervals remain aggregate-only.
g1-L7: A1 population scores recomputed from rational probabilities; Phase C table transcribed from aggregate JSON with provenance limitation above.

| Dataset / judge | RPS | CE (nats) | TVD | Spearman | Pearson |
|---|---:|---:|---:|---:|---:|
| A1 population / truth | 0.116707954 | 1.403971001 | 0.000000000 | 1.000000000 | 1.000000000 |
| A1 population / forecast | 0.116726547 | 1.405603049 | 0.014755629 | 1.000000000 | 1.000000000 |
| A1_max population / forecast | 0.133397826 | 2.397834311 | 0.445986821 | 1.000000000 | 1.000000000 |
| Phase C test / baseline_train_marginal | 0.180477702 | 1.800402044 | 0.661538559 | not recorded | not recorded |
| Phase C test / H0_sol | 0.190739645 | 9.132984041 | 0.672221522 | 0.593001961 | 0.663652243 |
| Phase C test / H1_sol | 0.212220869 | 9.122740319 | 0.666665700 | 0.767431184 | 0.799192223 |
| Phase C test / H2_sol | 0.203331998 | 8.446424404 | 0.627776978 | 0.811447672 | 0.840346874 |
| Phase C test / H0_opus | 0.272961187 | 10.373840949 | 0.749998767 | 0.659087512 | 0.581912811 |
| Phase C test / H2_opus | 0.195924644 | 8.709692721 | 0.638888022 | 0.839711956 | 0.861880690 |
| Phase C all75 / baseline_train_marginal | 0.175121610 | 1.777724842 | 0.661880491 | not recorded | not recorded |
| Phase C all75 / H0_sol | 0.185480477 | 8.276143710 | 0.606666253 | 0.548138060 | 0.616390721 |
| Phase C all75 / H1_sol | 0.194369156 | 8.491952606 | 0.635554916 | 0.680403898 | 0.716912560 |
| Phase C all75 / H2_sol | 0.199702411 | 8.432332759 | 0.619999413 | 0.664804775 | 0.692887867 |
| Phase C all75 / H0_opus | 0.216887548 | 8.650340936 | 0.642221676 | 0.562136224 | 0.609330346 |
| Phase C all75 / H2_opus | 0.179258073 | 8.097932666 | 0.599999467 | 0.769241233 | 0.777381067 |


Reproduce the evidence with `python3 docs/qa_11-25-2026/phase2/verify_g1_g2_remaining.py 4` from the repository root. HIGH-item checks are separately saved as `verify_g1-H1.py` (experiment 05), `verify_g2-H1.py` / `verify_g2-H2.py` (experiment 11), and `verify_g2-H3.py` (experiment 07), with their printed `.txt` outputs.
