# Experiment 15: verified QA corrections (11-26-2026)

**Date: 11-26-2026.** Codex CLI, GPT-6 Astra ultra (Vals account). CPU Python/raw-output checks; no Lean checking, no new model calls. Original protocols, result tables and raw outputs remain preserved.

**g2-M4 — CONFIRMED evidence limitation; rating-level reconstruction remains open.** Old estimates include rater offsets, bootstrap intervals, variance components, concordance and overlap counts. Corrected interpretation 11-26-2026: the nine result rows agree with saved aggregate JSON, and component arithmetic gives variance share **0.021001799072**, residual SD **0.697695966032**, and rater-offset SD **0.127191893522**; counts give 75 artifacts and 195 ratings. Private individual ratings and the cached 44-judge input pool are absent. Those aggregates cannot reconstruct the original rater bootstrap or the artifact SD **1.438989469**. No mismatch is established in these unavailable calculations.

**g2-M5 — CONFIRMED provenance gap; historical timing remains open.** Old text: “200-draw smoke run”, “11-20-2026 22:24 to 22:25, 25 s”, `results/run.log`, `results/launcher.log`. Corrected 11-26-2026: neither named log nor smoke receipt is preserved; the JSON records **21.820522070 s** compute time. This does not contradict a 25 s wall-clock duration under different timing boundaries, and it cannot verify the exact start/end or smoke history. Original figures remain recorded as historical claims.

**g2-L3 — CONFIRMED, score-first reporting.** Old headline: “Kendall tau 0.996 (Spearman), 0.998 (Pearson), 1.000 (RPS).” Corrected 11-26-2026: **RPS ranking agreement is 1.000000000**, best judge `H2_opus` has **RPS 0.143104707** both before and after severity adjustment; secondary ranking agreement is **0.995771670** for Spearman and **0.997885835** for Pearson. The full saved 44-judge panel below puts ranked probability score (RPS), cross-entropy (CE) and total variation distance (TVD) before correlations. CE/TVD were not saved for this analysis and are marked unavailable. RPS is mean squared cumulative error over five thresholds [0,1], lower better, against the same 75-artifact empirical human histograms with the experiment's 0.05 floor. Rounded severity adjustment leaves all saved RPS values identical. This retrospective all-data analysis is not a new held-out test. Correlations compare forecast means with raw or severity-adjusted human means, range [-1,1], higher better.

## Recomputed evidence and matched metric panel (11-26-2026)

Date: 11-26-2026. Read-only CPU reconstruction; no model calls.

g2-M4: variance share from summaries= 0.02100179907205768
Residual SD= 0.6976959660317301 ; rater-offset SD= 0.1271918935222591
Counts from saved rater counts: 195 ; artifacts: 75
g2-M5: saved compute runtime seconds = 21.82052206993103 ; run.log exists = False ; launcher.log exists = False
Private ratings/judge pool and smoke/launch receipt absent; rating-level estimates, intervals and wall-clock boundaries remain unverifiable.
g2-L3: rps ranking Kendall tau 0.9999999999999999
g2-L3: spearman ranking Kendall tau 0.9957716701902748
g2-L3: pearson ranking Kendall tau 0.9978858350951374
All 44 cached judges, same 75 artifacts; rounded severity adjustment leaves every saved RPS unchanged. CE/TVD absent in this analysis; correlation raw -> adjusted follows RPS raw -> adjusted.

| Dataset / judge | RPS | CE (nats) | TVD | Spearman | Pearson |
|---|---:|---:|---:|---:|---:|
| H0_sol | 0.154521531 → 0.154521531 | not recorded | not recorded | 0.548138060 → 0.548369905 | 0.616390721 → 0.612969325 |
| H0_opus | 0.173796451 → 0.173796451 | not recorded | not recorded | 0.562136224 → 0.563110078 | 0.609330346 → 0.608449300 |
| H1_sol | 0.156443444 → 0.156443444 | not recorded | not recorded | 0.680403898 → 0.681637940 | 0.716912560 → 0.718399325 |
| H2_sol | 0.158941770 → 0.158941770 | not recorded | not recorded | 0.664804775 → 0.665901875 | 0.692887867 → 0.694420106 |
| H2_opus | 0.143104707 → 0.143104707 | not recorded | not recorded | 0.769241233 → 0.770803570 | 0.777381067 → 0.778873665 |
| glm4_9b_H0 | 0.387685090 → 0.387685090 | not recorded | not recorded | -0.438604923 → -0.440578133 | -0.585108040 → -0.580048438 |
| glm4_9b_H1 | 0.378735620 → 0.378735620 | not recorded | not recorded | -0.564405207 → -0.566180598 | -0.643838784 → -0.640872605 |
| glm4_9b_H2 | 0.328666372 → 0.328666372 | not recorded | not recorded | -0.036396779 → -0.037176258 | -0.017400726 → -0.018067453 |
| granite33_8b_H0 | 0.240889082 → 0.240889082 | not recorded | not recorded | 0.206833923 → 0.209511673 | 0.246010266 → 0.246095407 |
| granite33_8b_H2 | 0.265329597 → 0.265329597 | not recorded | not recorded | -0.170291448 → -0.171159732 | -0.163067366 → -0.165554574 |
| hermes3_8b_H0 | 0.251250853 → 0.251250853 | not recorded | not recorded | 0.255414791 → 0.252674060 | 0.178427580 → 0.173841406 |
| hermes3_8b_H1 | 0.248720569 → 0.248720569 | not recorded | not recorded | 0.357035433 → 0.355524964 | 0.312205323 → 0.307243685 |
| llama31_8b_H0 | 0.255805793 → 0.255805793 | not recorded | not recorded | 0.289958855 → 0.284892432 | 0.257728963 → 0.253307709 |
| llama31_8b_H1 | 0.239513169 → 0.239513169 | not recorded | not recorded | 0.204746945 → 0.202746244 | 0.062245173 → 0.061950995 |
| llama31_8b_H2 | 0.265524424 → 0.265524424 | not recorded | not recorded | -0.113028897 → -0.114357921 | -0.122082969 → -0.122766846 |
| mistral7b_v03_H0 | 0.310052397 → 0.310052397 | not recorded | not recorded | -0.070848491 → -0.072194114 | 0.065503584 → 0.069331127 |
| mistral7b_v03_H1 | 0.293141179 → 0.293141179 | not recorded | not recorded | -0.089785746 → -0.088686865 | 0.228872046 → 0.231742883 |
| mistral7b_v03_H2 | 0.248951069 → 0.248951069 | not recorded | not recorded | 0.271880748 → 0.269767746 | 0.203630459 → 0.203487762 |
| olmo3_7b_H0 | 0.294660989 → 0.294660989 | not recorded | not recorded | 0.198605725 → 0.195306796 | 0.211203856 → 0.208292630 |
| olmo3_7b_H1 | 0.290786449 → 0.290786449 | not recorded | not recorded | 0.308528985 → 0.307807263 | 0.344968382 → 0.343877379 |
| q25_7b_H0 | 0.303169581 → 0.303169581 | not recorded | not recorded | 0.021698031 → 0.018998618 | 0.045528447 → 0.044683844 |
| q25_7b_H1 | 0.280861156 → 0.280861156 | not recorded | not recorded | 0.275778141 → 0.274780777 | 0.194353187 → 0.195320243 |
| q25_7b_H2 | 0.283060109 → 0.283060109 | not recorded | not recorded | -0.036386772 → -0.038652479 | 0.030274081 → 0.031051710 |
| q25c_14b_H0 | 0.326675785 → 0.326675785 | not recorded | not recorded | -0.362922987 → -0.364485643 | -0.364555377 → -0.362885920 |
| q25c_14b_H1 | 0.300938282 → 0.300938282 | not recorded | not recorded | -0.272329173 → -0.275272996 | -0.247296838 → -0.247527756 |
| q25c_14b_H2 | 0.281320966 → 0.281320966 | not recorded | not recorded | -0.144226990 → -0.147284588 | -0.138317294 → -0.138084552 |
| q25c_7b_H0 | 0.283977744 → 0.283977744 | not recorded | not recorded | -0.160473050 → -0.162956895 | -0.139498449 → -0.140243413 |
| q25c_7b_H1 | 0.274026193 → 0.274026193 | not recorded | not recorded | -0.067955612 → -0.069872436 | -0.028968173 → -0.028153642 |
| q25c_7b_H2 | 0.311968959 → 0.311968959 | not recorded | not recorded | -0.215561362 → -0.216525171 | -0.290524142 → -0.289053186 |
| q3_14b_H0 | 0.280709740 → 0.280709740 | not recorded | not recorded | 0.138422277 → 0.137305818 | 0.199946527 → 0.200371550 |
| q3_14b_H1 | 0.241087321 → 0.241087321 | not recorded | not recorded | 0.172163530 → 0.172393217 | 0.143615200 → 0.143667913 |
| q3_14b_H2 | 0.319010525 → 0.319010525 | not recorded | not recorded | -0.252500908 → -0.255567727 | -0.408728304 → -0.407644418 |
| q3_30b_a3b_awq_H0 | 0.336329196 → 0.336329196 | not recorded | not recorded | 0.033299680 → 0.033296341 | 0.065468705 → 0.063434287 |
| q3_30b_a3b_awq_H1 | 0.335265426 → 0.335265426 | not recorded | not recorded | 0.088717483 → 0.087835713 | 0.059399383 → 0.057192037 |
| q3_30b_a3b_awq_H2 | 0.334698816 → 0.334698816 | not recorded | not recorded | 0.149948692 → 0.150767426 | 0.204341445 → 0.203586341 |
| q3_4b_2507_H0 | 0.300407791 → 0.300407791 | not recorded | not recorded | 0.140986593 → 0.142600025 | 0.070808567 → 0.074365126 |
| q3_4b_2507_H1 | 0.317249340 → 0.317249340 | not recorded | not recorded | 0.186070486 → 0.187492713 | 0.107227201 → 0.111116304 |
| q3_4b_2507_H2 | 0.260690165 → 0.260690165 | not recorded | not recorded | 0.389884763 → 0.389395179 | 0.406673245 → 0.406281412 |
| q3_8b_H0 | 0.326414366 → 0.326414366 | not recorded | not recorded | -0.029875488 → -0.028474863 | -0.055486124 → -0.052799614 |
| q3_8b_H1 | 0.395593235 → 0.395593235 | not recorded | not recorded | -0.253052138 → -0.253478229 | -0.337507048 → -0.336392276 |
| q3_8b_H2 | 0.362439083 → 0.362439083 | not recorded | not recorded | -0.380277954 → -0.381095624 | -0.383085985 → -0.380979452 |
| seed_coder_8b_H0 | 0.305041288 → 0.305041288 | not recorded | not recorded | 0.047670820 → 0.046459203 | 0.070242323 → 0.068089968 |
| seed_coder_8b_H1 | 0.313787218 → 0.313787218 | not recorded | not recorded | -0.001283568 → -0.003200773 | -0.086992401 → -0.087345892 |
| seed_coder_8b_H2 | 0.265645933 → 0.265645933 | not recorded | not recorded | 0.083742799 → 0.079600371 | 0.133315216 → 0.130500090 |


Reproduce the evidence with `python3 docs/qa_11-25-2026/phase2/verify_g1_g2_remaining.py 15` from the repository root. HIGH-item checks are separately saved as `verify_g1-H1.py` (experiment 05), `verify_g2-H1.py` / `verify_g2-H2.py` (experiment 11), and `verify_g2-H3.py` (experiment 07), with their printed `.txt` outputs.
