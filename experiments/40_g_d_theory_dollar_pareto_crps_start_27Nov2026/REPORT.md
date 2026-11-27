# Experiment 40 report

**Doc link:** https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/40_g_d_theory_dollar_pareto_crps_start_27Nov2026/REPORT.md

Numbers below are written by `run_all.py` from `results.json`. Read `HOW_TO_READ_RESULTS.md` before quoting one.

## Machine checks

- Lean 4.36.0, `lake build`, zero `sorry`. Axiom receipt: `lean_check.txt`. Standard axioms only.
- `pytest` in this folder: witness identities, random Brier and RPS decompositions, G-study versus balanced ANOVA.
- Bootstrap B=1000, item-clustered percentile 95% intervals, seed 1, fixed head.

## Executive findings

On the large public sets, once the nominal budget reaches a few hundred ratings the held-out cross-entropy of this linear head is within a few thousandths of a nat across k, and the item-clustered 95% interval of the D-study-minus-oracle difference covers 0. D-study still names a specific k (17 on DICES-990; 10 on POPQUORN Offensiveness, where train-split Φ(10) = 0.7875 and the 0.80 target is not reached). At small budgets that k is often infeasible, because N = B/k drops below 4 prompts. The detectable misallocation is VeriBench `native_main` on the sealed `exp27_split`: at B=64, D-study k=2 has held-out test cross-entropy 0.081 nats worse than oracle k=3, interval [0.013, 0.141].

The pre-registered +0.05 nat tolerance versus the full-train head is reached at B=32 or B=64 on every set except SummEval expert, which reaches it at B=128. At the stated crowd prices that is a few dollars; at the stated expert prices it is a few hundred to a few thousand dollars. The $15,000–$75,000 researcher-plus-GPU anchor is not a cost measured here.

## Where D-study and the proper score agree

- `summeval_expert`, largest grid budget B=4096, held-out test prompts n=1280: D-study and the test-oracle allocation are the same k. Held-out test cross-entropy of the D-study allocation minus the oracle allocation is 0.0000 [0.0000, 0.0000] (positive means the D-study allocation is worse). D-study test CE 1.2084 [1.1688, 1.2466]; oracle test CE 1.2084 [1.1688, 1.2466].
- `summeval_crowd`, largest grid budget B=4096, held-out test prompts n=1280: D-study and the test-oracle allocation are not the same k (D-study 5, oracle 3). Held-out test cross-entropy of the D-study allocation minus the oracle allocation is 0.0010 [-0.0006, 0.0025] (positive means the D-study allocation is worse). D-study test CE 1.3260 [1.2997, 1.3528]; oracle test CE 1.3250 [1.2997, 1.3514].
- `popquorn_offensiveness`, largest grid budget B=4096, held-out test prompts n=300: D-study and the test-oracle allocation are the same k. Held-out test cross-entropy of the D-study allocation minus the oracle allocation is 0.0000 [0.0000, 0.0000] (positive means the D-study allocation is worse). D-study test CE 1.2980 [1.2411, 1.3599]; oracle test CE 1.2980 [1.2411, 1.3599].
- `popquorn_politeness`, largest grid budget B=4096, held-out test prompts n=745: D-study and the test-oracle allocation are not the same k (D-study 6, oracle 2). Held-out test cross-entropy of the D-study allocation minus the oracle allocation is 0.0018 [-0.0012, 0.0049] (positive means the D-study allocation is worse). D-study test CE 1.5302 [1.5169, 1.5441]; oracle test CE 1.5284 [1.5125, 1.5442].
- `dices990`, largest grid budget B=4096, held-out test prompts n=198: D-study and the test-oracle allocation are not the same k (D-study 17, oracle 8). Held-out test cross-entropy of the D-study allocation minus the oracle allocation is 0.0000 [-0.0034, 0.0034] (positive means the D-study allocation is worse). D-study test CE 0.7444 [0.7076, 0.7809]; oracle test CE 0.7443 [0.7098, 0.7786].
- `chaosnli_snli_mnli`, largest grid budget B=4096, held-out test prompts n=624: D-study and the test-oracle allocation are not the same k (D-study 5, oracle 8). Held-out test cross-entropy of the D-study allocation minus the oracle allocation is 0.0015 [-0.0004, 0.0036] (positive means the D-study allocation is worse). D-study test CE 1.0546 [1.0421, 1.0685]; oracle test CE 1.0531 [1.0407, 1.0675].
- `veribench_fsc_native_main`, largest grid budget B=64, held-out test prompts n=40: D-study and the test-oracle allocation are not the same k (D-study 2, oracle 3). Held-out test cross-entropy of the D-study allocation minus the oracle allocation is 0.0813 [0.0128, 0.1406] (positive means the D-study allocation is worse). D-study test CE 0.7348 [0.5814, 0.9411]; oracle test CE 0.6535 [0.4547, 0.9124].

## Dollar frontier

Tolerance was fixed in `PROTOCOL.md`: 0.05 nats of cross-entropy, 0.02 of sqrt(RPS), above the full-train head.

- `summeval_expert`: full-train held-out CE 1.2097. Smallest frontier B within +0.05 nats: 128. Crowd dollars at $0.05/$0.10/$0.20: {'0.05': 6.4, '0.10': 12.8, '0.20': 25.6}. Expert dollars at $5/$10/$20: {'5.00': 640.0, '10.00': 1280.0, '20.00': 2560.0}. sqrt(RPS) B: 32.
- `summeval_crowd`: full-train held-out CE 1.3265. Smallest frontier B within +0.05 nats: 32. Crowd dollars at $0.05/$0.10/$0.20: {'0.05': 1.6, '0.10': 3.2, '0.20': 6.4}. Expert dollars at $5/$10/$20: {'5.00': 160.0, '10.00': 320.0, '20.00': 640.0}. sqrt(RPS) B: 32.
- `popquorn_offensiveness`: full-train held-out CE 1.2983. Smallest frontier B within +0.05 nats: 64. Crowd dollars at $0.05/$0.10/$0.20: {'0.05': 3.2, '0.10': 6.4, '0.20': 12.8}. Expert dollars at $5/$10/$20: {'5.00': 320.0, '10.00': 640.0, '20.00': 1280.0}. sqrt(RPS) B: 32.
- `popquorn_politeness`: full-train held-out CE 1.5281. Smallest frontier B within +0.05 nats: 32. Crowd dollars at $0.05/$0.10/$0.20: {'0.05': 1.6, '0.10': 3.2, '0.20': 6.4}. Expert dollars at $5/$10/$20: {'5.00': 160.0, '10.00': 320.0, '20.00': 640.0}. sqrt(RPS) B: 32.
- `dices990`: full-train held-out CE 0.7430. Smallest frontier B within +0.05 nats: 32. Crowd dollars at $0.05/$0.10/$0.20: {'0.05': 1.6, '0.10': 3.2, '0.20': 6.4}. Expert dollars at $5/$10/$20: {'5.00': 160.0, '10.00': 320.0, '20.00': 640.0}. sqrt(RPS) B: 32.
- `chaosnli_snli_mnli`: full-train held-out CE 1.0529. Smallest frontier B within +0.05 nats: 32. Crowd dollars at $0.05/$0.10/$0.20: {'0.05': 1.6, '0.10': 3.2, '0.20': 6.4}. Expert dollars at $5/$10/$20: {'5.00': 160.0, '10.00': 320.0, '20.00': 640.0}. sqrt(RPS) B: 32.
- `veribench_fsc_native_main`: full-train held-out CE 0.7049. Smallest frontier B within +0.05 nats: 32. Crowd dollars at $0.05/$0.10/$0.20: {'0.05': 1.6, '0.10': 3.2, '0.20': 6.4}. Expert dollars at $5/$10/$20: {'5.00': 160.0, '10.00': 320.0, '20.00': 640.0}. sqrt(RPS) B: 32.

The $15,000–$75,000 researcher-plus-GPU band is a stated anchor for 1–3 researcher-months. This run did not measure that cost. The linear head is not a GPU low-rank adaptation run.

## R_v1

Pinned loader: `{"status": "unavailable", "error": "HTTPError: HTTP Error 404: Not Found", "what_was_attempted": "experiment 36 load_veribench_fsc_rv1('all5') at veribench@d1c47355"}`.

trustworthy-theorem-judges efdc8c3 experiments/28_.../data_11-26-2026/human_ratings_old_11-26-2026.csv. Freeze note in the file: exp 100 / exp 21 / exp 27 expert ratings. Rater identity was removed and the three scores were sorted, so this is not a crossed person G-study and it is not a byte-verified read of veribench@d1c47355. The experiment 36 loader's items.jsonl returned HTTP 404 from this machine.

## Synthetic K sweep

0.7 Normal(100 u, 8^2) + 0.3 Normal(100 (1-u), 12^2) on {0,...,100}, u = (i+0.5)/n

- K=2: point-estimate k* = 1; CI k* = 2.
- K=3: point-estimate k* = 1; CI k* = 1.
- K=5: point-estimate k* = 1; CI k* = 1.
- K=10: point-estimate k* = 1; CI k* = 1.
- K=20: point-estimate k* = 1; CI k* = 1.
- K=101: point-estimate k* = 1; CI k* = 1.

## GPU bridge

Proper-score means and intervals in the quoted rows are copied from RESULTS_11-26-2026.md at efdc8c3. They were not recomputed: the JSON stores probabilities, not the human ratings those scores were computed against. The JSON checks are recomputed here.

- `qwen3_4b_logit_6144.json` n=175, probability-sum error 2.000e-06, expected-score error 6.000e-06.
- `qwen3_8b_logit_6144.json` n=175, probability-sum error 2.000e-06, expected-score error 5.000e-06.

## Caveats

- ChaosNLI ranked probability score uses the file's label order (entailment, neutral, contradiction). That order is imposed.
- SummEval and ChaosNLI have no person id. Their rater component is unidentified on purpose.
- The test-oracle k is not a deployable selection. The validation-chosen k is the one that does not look at test labels.
- Intervals do not retrain the head.
- WMT direct assessment at `RicardoRei/wmt-da-human-evaluation` (commit 301de385, train.csv 638,191,673 bytes) was reachable and is a mean score plus an annotator count. Individual rater rows are not in that file, so it was not used as multi-rater data.
