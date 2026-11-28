# How are LLM judges validated in practice? A literature check of the "broken ruler" premise (27 Nov 2026)

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/43_llm_judge_validation_practice_survey_start_27Nov2026/README.md>

**Earlier 37-paper Google Doc copy (private to Brando; this README is the updated 49-paper canonical text):** <https://docs.google.com/document/d/1xyjd_VDgXnGued5w2PdK8z4M6sy_Ix1bAdmiw-V0VkI/edit>

**Merged framing Doc (Gemini's synthesis, used as the merged answer; not edited here):** <https://docs.google.com/document/d/1Pt4NFmrYn3aclbQqx6_YGIUXS9bQappZsY08IEEBlNo/edit>

**TLDR:** The premise is **partly true**. Correlation with human scores (Pearson, Spearman, Kendall) is the near-universal validation statistic for *direct-score* judges, automatic metrics and judge-based benchmarks (29 of 30 such papers coded here; the exception is an IRT reliability paper). It is **not** the main statistic for *pairwise-preference* judges and reward models, which report accuracy or agreement against a single human label (13 of 13 pairwise papers report agreement, accuracy or kappa). What is truly widespread is narrower and stronger for the paper: 43 of 49 papers score the judge against **one aggregated or forced-choice human label per item** with a point-agreement statistic, and do not report a proper score for the human rating distribution. Phrase the target as "correlation and majority-label agreement", not "correlation" alone. A 2025–2026 psychometric-reliability literature (kappa, IRT, G-theory) exists but is still a minority practice in this sample (1 of 49 papers coded R). Closest distributional prior work is Lyu, Do and Pergola (27 Oct 2026), who already evaluate ordinal LLM judges with the ranked probability score (RPS) and nominal judges with cross-entropy (CE).

## 1. Question

Brando (27 Nov 2026): the paper argues that correlation with humans is a broken ruler for validating large language model (LLM) judges, "assuming that is the way [we] do LLM judging validation in ML/AI. Verify that ... Otherwise we'd break a ruler people aren't using." This note checks that premise against the published record. It is a literature check, not an experiment; it produces no new numerical results about judges.

## 2. Method

1. **Sample.** 49 papers or tools that propose or validate an LLM judge, an automatic text-quality metric, or a judge-based benchmark, plus background papers on human label disagreement and surveys or scoping reviews cited in Section 5. The sample is the canonical set named in the brief, papers found by targeted search (arXiv, ACL Anthology, OpenReview), the AlpacaEval family, and verified entries from a later literature sweep. It is a **purposive convenience sample**, not a systematic review: it over-represents highly cited 2020–2026 work and under-represents applied and industry papers. Percentages below describe this sample, not the field.
2. **Verification.** Every paper's title, authors and date were retrieved from the arXiv API, the ACL Anthology or OpenReview search. Venues come from the arXiv comment or journal-ref field, OpenReview, Crossref, the ACL Anthology or the PDF footer. The metric column was read from each paper's full text (PDF converted with `pdftotext`, then searched for the metric names and read in context), not from memory, except where a source is marked as a GitHub README (AlpacaEval). No paper was read end to end.
3. **Coding.** Each paper gets every metric family it reports for judge-versus-human (or metric-versus-human) validation:
   - **C**, correlation: Pearson, Spearman, Kendall, at sample, summary or system level.
   - **A**, raw agreement: accuracy, percent agreement, precision, recall, F1.
   - **K**, chance-corrected agreement: Cohen or Fleiss kappa, Scott's pi, Krippendorff alpha used as agreement with a reference label.
   - **P**, a proper score or distributional divergence against *item-level* human ratings: cross-entropy (CE) or negative log-likelihood, Kullback–Leibler (KL) or Jensen–Shannon divergence, Brier score or mean squared error (MSE) against a soft label, ranked probability score (RPS).
   - **Ps**, a proper score at the system (model-pair) level only.
   - **R**, psychometric reliability of the judge as an instrument (item-response-theory reliability or consistency, generalizability-theory coefficients), not agreement with a reference label.

   Judge type: **D** direct-score judge or metric, **S** benchmark validated against a human leaderboard, **W** pairwise-preference judge or reward model, **X** disagreement- or distribution-aware validation.
4. **Reproducibility.** [`papers.py`](papers.py) holds the coded rows; [`tally.py`](tally.py) prints every count in Section 3 and the table in Section 4 (`python3 tally.py`; deterministic, run twice with identical output).

## 3. Tally (49 judge, metric or benchmark papers)

Counts are papers reporting at least one metric of the family, so rows overlap.

| Metric family | Papers | Share |
|---|---|---|
| Correlation (C) | 33 | 67% |
| Raw agreement / accuracy (A) | 24 | 49% |
| Chance-corrected agreement (K) | 7 | 14% |
| Item-level proper score or divergence vs. human distribution (P) | 6 | 12% |
| System-level proper score only (Ps) | 2 | 4% |
| Psychometric reliability of the judge (R: IRT or G-theory) | 1 | 2% |
| Correlation is the *only* family reported | 18 | 37% |
| **No item-level proper score** | **43** | **88%** |

By judge type (group is the first letter of the type code, so `D+W` counts as D and `S+W` as S):

| Judge type | Papers | C | A | K | P | Ps | R |
|---|---|---|---|---|---|---|---|
| Direct-score judges and automatic metrics (D) | 25 | 24 | 6 | 2 | 0 | 0 | 1 |
| Judge-based benchmarks vs. a human leaderboard (S) | 5 | 5 | 3 | 0 | 0 | 2 | 0 |
| Pairwise judges and reward models (W) | 13 | 1 | 12 | 3 | 0 | 0 | 0 |
| Distribution-aware (X) | 6 | 3 | 3 | 2 | 6 | 0 | 0 |

Read: correlation is the default whenever the judge emits a score (29/30 of D and S papers; the exception is Choi et al. 2026, which reports IRT reliability rather than correlation). Agreement or accuracy is the default whenever it emits a preference (12/13 W papers report A; all 13 report A or K). Proper scores against the human rating distribution appear in six 2025–2026 papers, most written to criticize or replace single-label validation.

The AlpacaEval family is now in the table: AlpacaFarm (Dubois et al., NeurIPS 2023) validates simulated annotators by agreement with a human majority (65% vs. 66% for a held-out human); the AlpacaEval repo (Tatsu lab, no standalone paper) selects and reports evaluators by human agreement on the AlpacaFarm cross-annotation set, plus Spearman and Pearson vs. a 9-model human leaderboard; Length-Controlled AlpacaEval (already present) reports Spearman with Chatbot Arena.

## 4. Paper-by-paper table

Venue "arXiv" means no peer-reviewed venue was found. The metric column paraphrases the paper's own text. The AlpacaEval row is a benchmark/tool, not a paper.

| # | Paper | Year | Venue | Judge type | Validation metric(s) reported | Codes | Link |
|---|---|---|---|---|---|---|---|
| 1 | G-Eval (Liu et al.) | 2023 | EMNLP 2023 | D | Spearman, Kendall (summary-level, SummEval; Topical-Chat) | C | [2303.16634](https://arxiv.org/abs/2303.16634) |
| 2 | GPTScore (Fu et al.) | 2024 | NAACL 2024 | D | Spearman (primary correlation measure) | C | [2302.04166](https://arxiv.org/abs/2302.04166) |
| 3 | Can LLMs be an alternative to human evaluations? (Chiang & Lee) | 2023 | ACL 2023 | D | Kendall tau vs. teacher ratings; Krippendorff alpha only for human IAA | C | [2305.01937](https://arxiv.org/abs/2305.01937) |
| 4 | Prometheus (Kim et al.) | 2024 | ICLR 2024 | D | Pearson with human and GPT-4 scores; human pairwise preference of feedback | C | [2310.08491](https://arxiv.org/abs/2310.08491) |
| 5 | Prometheus 2 (Kim et al.) | 2024 | EMNLP 2024 | D+W | Pearson, Spearman, Kendall (direct); accuracy (pairwise) | C,A | [2405.01535](https://arxiv.org/abs/2405.01535) |
| 6 | FLASK (Ye et al.) | 2024 | ICLR 2024 | D | Pearson, Spearman, Kendall between model-based and human scores | C | [2307.10928](https://arxiv.org/abs/2307.10928) |
| 7 | BiGGen Bench (Kim et al.) | 2025 | NAACL 2025 | D | Pearson between evaluator LMs and human scores | C | [2406.05761](https://arxiv.org/abs/2406.05761) |
| 8 | Calibrating LLM-Based Evaluator / AutoCalibrate (Liu et al.) | 2024 | LREC-COLING 2024 | D | Spearman, Kendall (summary-level); 'calibration' here means aligning to humans, not probability calibration | C | [2309.13308](https://arxiv.org/abs/2309.13308) |
| 9 | LLM-Rubric (Hashemi et al.) | 2024 | ACL 2024 | D | RMSE plus Pearson, Spearman, Kendall vs. individual judges' answers | C | [2501.00274](https://arxiv.org/abs/2501.00274) |
| 10 | MLLM-as-a-Judge (Chen et al.) | 2024 | ICML 2024 | D+W | Pearson (scoring); agreement (pair, batch) | C,A | [2402.04788](https://arxiv.org/abs/2402.04788) |
| 11 | LLMs instead of Human Judges? JUDGE-BENCH (Bavaresco et al.) | 2025 | ACL 2025 | D | Cohen kappa (categorical), Spearman (graded), vs. aggregated human labels | C,K | [2406.18403](https://arxiv.org/abs/2406.18403) |
| 12 | Judge's Verdict (Han et al.) | 2025 | arXiv (under review, ICLR 2026) | D | Pearson r >= 0.80 filter, then Cohen kappa | C,K | [2510.09738](https://arxiv.org/abs/2510.09738) |
| 13 | CheckEval (Lee et al.) | 2025 | EMNLP 2025 | D | Pearson, Spearman, Kendall vs. human ratings (sample-level on SummEval; turn-level on Topical-Chat; QAGS in appendix). The judge itself uses binary checklist items aggregated to a yes-rate, not a raw Likert. | C | [2403.18771](https://arxiv.org/abs/2403.18771) |
| 14 | LAGER / Beyond the Surface (Lai et al.) | 2025 | NeurIPS 2025 | D | Spearman on FLASK, HelpSteer and BiGGen (main reported metric); Pearson in Table 5. LAGER is the method name in a paper titled 'Beyond the Surface: Enhancing LLM-as-a-Judge Alignment with Human via Internal Representations'. | C | [2508.03550](https://arxiv.org/abs/2508.03550) |
| 15 | Evaluating Scoring Bias in LLM-as-a-Judge (Li, Dou, Shao, Chen & Hu) | 2026 | DASFAA 2026 | D | Spearman and Pearson vs. human gold scores on BiGGen Bench and FLASK (Tables 2 and 4; the paper calls these 'accuracy metrics') | C | [2506.22316](https://arxiv.org/abs/2506.22316) |
| 16 | SummEval (Fabbri et al.) | 2021 | TACL 2021 | D | Kendall tau (system level) | C | [2007.12626](https://arxiv.org/abs/2007.12626) |
| 17 | BERTScore (Zhang et al.) | 2020 | ICLR 2020 | D | |Pearson| and Kendall (WMT18 standard) | C | [1904.09675](https://arxiv.org/abs/1904.09675) |
| 18 | BLEURT (Sellam et al.) | 2020 | ACL 2020 | D | Kendall tau and Pearson (WMT17/18/19) | C | [2004.04696](https://arxiv.org/abs/2004.04696) |
| 19 | COMET (Rei et al.) | 2020 | EMNLP 2020 | D | WMT Kendall-tau-like (concordant minus discordant pairs) | C | [2009.09025](https://arxiv.org/abs/2009.09025) |
| 20 | UniEval (Zhong et al.) | 2022 | EMNLP 2022 | D | Spearman, Kendall (summary-level) | C | [2210.07197](https://arxiv.org/abs/2210.07197) |
| 21 | WMT23 Metrics Shared Task (Freitag et al.) | 2023 | WMT 2023 | D | System: pairwise accuracy + Pearson; segment: Pearson + tie-calibrated pairwise accuracy (acc*eq) | C,A | [ACL:2023.wmt-1.51](https://aclanthology.org/2023.wmt-1.51/) |
| 22 | WMT25 Metrics Shared Task | 2025 | WMT 2025 | D | Official meta-metrics follow WMT24: system-level Soft Pairwise Accuracy (SPA; Thompson et al. 2024) and segment-level tie-calibrated acc*eq. Pearson and Kendall are discussed as traditional options, not as the official primary metrics. | A,C | [ACL:2025.wmt-1.24](https://aclanthology.org/2025.wmt-1.24/) |
| 23 | GEMBA (Kocmi & Federmann) | 2023 | EAMT 2023 | D | System-level pairwise accuracy; segment-level Kendall | C,A | [2302.14520](https://arxiv.org/abs/2302.14520) |
| 24 | AlpacaFarm (Dubois, Li, Taori et al.) | 2023 | NeurIPS 2023 (Spotlight) | W | Agreement of the simulated evaluator p_sim^eval with the majority vote of 3 human annotators is 65%, vs 66% for a held-out human; some single GPT-4 prompts reach 68%. The training annotator has lower raw agreement because of 25% label-flip noise. | A | [2305.14387](https://arxiv.org/abs/2305.14387) |
| 25 | AlpacaEval (tatsu-lab repo and leaderboard; no standalone paper) | 2023 | GitHub benchmark/tool (Tatsu lab) | S+W | Evaluators are compared on the AlpacaFarm human cross-annotation set (~650 instructions, 4 human annotations each). Headline metric is human agreement (leave-one-out accuracy vs. the mode of the other 3). The suggested evaluator alpaca_eval_gpt4 scores 69.2 agreement (humans 65.7), with Spearman 0.97 and Pearson 0.93 vs. a 9-model human leaderboard; the README warns that correlation with only 9 models 'is not very reliable'. Length-controlled win-rate correlation with Chatbot Arena is reported in the already-listed COLM 2024 paper. | A,C | [GH:tatsu-lab/alpaca_eval](https://github.com/tatsu-lab/alpaca_eval) |
| 26 | Length-Controlled AlpacaEval (Dubois et al.) | 2024 | COLM 2024 | S | Spearman with Chatbot Arena (0.94 to 0.98) | C | [2404.04475](https://arxiv.org/abs/2404.04475) |
| 27 | WildBench (Lin et al.) | 2025 | ICLR 2025 | S | Pearson with Chatbot Arena Elo (0.98 / 0.95) | C | [2406.04770](https://arxiv.org/abs/2406.04770) |
| 28 | Arena-Hard / BenchBuilder (Li et al.) | 2025 | ICML 2025 | S | Separability and agreement with confidence, Spearman, Pearson, Pair Rank Brier score (system pairs) | C,A,Ps | [2406.11939](https://arxiv.org/abs/2406.11939) |
| 29 | Preference Proxy Evaluations / PPE (Frick et al.) | 2025 | ICLR 2025 | S | Accuracy on human preference and verifiable-correctness pairs; Spearman, Kendall and row-wise Pearson vs. Chatbot Arena rankings/win-rate matrices; Separability, Confidence Agreement and Brier score (as in Arena-Hard) at the system level | C,A,Ps | [2410.14872](https://arxiv.org/abs/2410.14872) |
| 30 | MT-Bench & Chatbot Arena judge (Zheng et al.) | 2023 | NeurIPS 2023 D&B | W | Agreement = P(random individual of each type agree); >80% | A | [2306.05685](https://arxiv.org/abs/2306.05685) |
| 31 | Chatbot Arena (Chiang et al.) | 2024 | ICML 2024 | W | Pairwise agreement rate (crowd, GPT-4 judge, experts); leaderboard is a Bradley-Terry fit minimizing binary cross-entropy | A | [2403.04132](https://arxiv.org/abs/2403.04132) |
| 32 | PandaLM (Wang et al.) | 2024 | ICLR 2024 | W | Accuracy, precision, recall, F1 vs. human annotation | A | [2306.05087](https://arxiv.org/abs/2306.05087) |
| 33 | JudgeLM (Zhu et al.) | 2025 | ICLR 2025 | W | Agreement, precision, recall, F1 vs. GPT-4 teacher and human labels | A | [2310.17631](https://arxiv.org/abs/2310.17631) |
| 34 | Auto-J (Li et al.) | 2024 | ICLR 2024 | W | Pairwise agreement with human labels; Pearson/Spearman in a best-of-N analysis | A,C | [2310.05470](https://arxiv.org/abs/2310.05470) |
| 35 | LLMs are not Fair Evaluators (Wang et al.) | 2024 | ACL 2024 | W | Accuracy and kappa vs. human pairwise judgments | A,K | [2305.17926](https://arxiv.org/abs/2305.17926) |
| 36 | RewardBench (Lambert et al.) | 2025 | Findings NAACL 2025 | W | Accuracy (chosen scored above rejected) | A | [2403.13787](https://arxiv.org/abs/2403.13787) |
| 37 | RewardBench 2 (Malik et al.) | 2026 | ICLR 2026 | W | Accuracy in a best-of-4 format (one chosen, three rejected) | A | [2506.01937](https://arxiv.org/abs/2506.01937) |
| 38 | JudgeBench (Tan et al.) | 2025 | ICLR 2025 | W | Accuracy on objectively-labelled pairs | A | [2410.12784](https://arxiv.org/abs/2410.12784) |
| 39 | Judging the Judges (Thakur et al.) | 2025 | GEM 2025 workshop | W | Percent agreement and Scott's pi (Cohen kappa in an earlier version) | A,K | [2406.12624](https://arxiv.org/abs/2406.12624) |
| 40 | Who Validates the Validators? EvalGen (Shankar et al.) | 2024 | UIST 2024 | W | Alignment of evaluator verdicts with user grades (accuracy-type) | A | [2404.12272](https://arxiv.org/abs/2404.12272) |
| 41 | No Free Labels (Krumdick et al.) | 2026 | COLM 2026 (per PDF footer) | W+D | Cohen kappa (F1 in appendix) vs. expert labels | K | [2503.05061](https://arxiv.org/abs/2503.05061) |
| 42 | LongJudgeBench (Chen, Dong, Li et al.) | 2026 | EMNLP 2026 | D+W | Accuracy is the main metric (agreement with reference judgments, averaged over query groups); also Spearman and Kendall tau_b. Covers pointwise, pairwise and listwise protocols. | A,C | [2606.01629](https://arxiv.org/abs/2606.01629) |
| 43 | Beyond correlation (Elangovan et al.) | 2025 | ICLR 2025 | X | Stratify by human-label uncertainty; binned Jensen-Shannon divergence; critiques correlation, Krippendorff alpha, Randolph kappa | P,C,K | [2410.03775](https://arxiv.org/abs/2410.03775) |
| 44 | Validating LLM-as-a-Judge under Rating Indeterminacy (Guerdan et al.) | 2025 | NeurIPS 2025 | X | KL and CE vs. human distribution, response-set MSE; forced-choice agreement selects judges up to 31% worse | P,A,K | [2503.05965](https://arxiv.org/abs/2503.05965) |
| 45 | Distributional LLM-as-a-Judge (Chen et al.) | 2025 | NeurIPS 2025 | X | KL to human rating distribution; accuracy vs. majority | P,A | [2505.12301](https://arxiv.org/abs/2505.12301) |
| 46 | Improving LLM-as-a-Judge Inference with the Judgment Distribution (Wang, Zhang & Choi) | 2025 | Findings EMNLP 2025 | X | Accuracy plus MSE vs. averaged human preference (= Brier score for binary targets); Spearman of std. devs. Distinct from Li et al. 'Judging with Confidence' (Findings EMNLP 2026). | P,A,C | [2503.03064](https://arxiv.org/abs/2503.03064) |
| 47 | Judging with Confidence: Calibrating Autoraters to Preference Distributions (Li et al.) | 2026 | Findings EMNLP 2026 | X | Pairwise autoraters calibrated by SFT or RL with Brier and logarithmic rewards against a preference distribution. The paper states that a thorough empirical study of ordinal/Likert (multi-class Brier or CE) is left as future work. Published at Findings EMNLP 2026 (ACL Anthology 2026.findings-emnlp.2093); the ICLR 2026 OpenReview forum still reads 'Submitted to ICLR 2026'. | P | [2510.00263](https://arxiv.org/abs/2510.00263) |
| 48 | Human-Residual Steering (Lyu, Do & Pergola) | 2026 | arXiv (27 Oct 2026) | X | Extracts next-token logits over rating verbalizers from a frozen judge. Trains with ranked probability score (RPS) for ordinal labels and cross-entropy (CE) for nominal labels. Evaluates Jensen-Shannon distance as the primary metric, plus annotation-level RPS (ordinal) or Brier (nominal), and Spearman and MAE vs. aggregate human scores (mean Spearman 0.449 to 0.579). Datasets: SummEval, Topical-Chat, FLASK, HelpSteer2, HelpSteer3, ChaosNLI, DynaSent, Scruples. | P,C | [2610.37868](https://arxiv.org/abs/2610.37868) |
| 49 | Diagnosing the Reliability of LLM-as-a-Judge via Item Response Theory (Choi et al.) | 2026 | ICML 2026 | D | Fits an IRT Graded Response Model (GRM). Reports prompt consistency as a coefficient of variation (CV) and marginal reliability (rho), then human alignment. Empirical judges include G-Eval on SummEval, Topical-Chat and HelpSteer-2, and VIEScore on ImageHub. | R | [2602.00521](https://arxiv.org/abs/2602.00521) |

## 5. Independent evidence on prevalence

These sources counted metrics themselves; they are larger or differently selected than the sample above.

1. **Rao & Callison-Burch (2026),** "Agreement Metrics for LLM-as-Judge Evaluation: What to Report and Why", [arXiv:2606.00093](https://arxiv.org/abs/2606.00093). Their coding of 24 recent LLM-as-judge papers "finds 11 graded or continuous evaluations, most reporting correlations against human ratings, 7 pairwise-preference evaluations with varying tie conventions, 3 binary-verdict evaluations, 2 judge ensembles, and 1 scale-matched mix." They describe it as a descriptive, non-exhaustive scan coded by one author. They recommend weighted kappa for 1–5 ordinal scales. The authors are Delip Rao and Chris Callison-Burch, not anonymous.
2. **Lee, Yun & Jung (preprint, under review),** "RULE-LLM reporting framework ...", [doi:10.21203/rs.3.rs-11611263/v1](https://doi.org/10.21203/rs.3.rs-11611263/v1). The public abstract says 91 medical LLM-judge studies were included and 65.9% reported agreement with human evaluators "with widely varying metrics." A per-metric breakdown circulating in a search excerpt (kappa 12, Pearson 7, and so on) was **not re-read in the full text and is not used below**.
3. **Li et al. (2026),** "LLM-as-a-Judge in Healthcare: A Scoping Analysis ...", [arXiv:2605.25273](https://arxiv.org/abs/2605.25273). 134 studies. Its metric taxonomy splits "association" metrics (Pearson, Spearman, Kendall, R²) from "classification" metrics (agreement rate, accuracy, F1, kappa, Krippendorff alpha, Gwet's AC1), and it synthesizes agreement rate, Cohen's kappa and correlation as "three widely reported metrics."
4. **Surveys.** Gu et al. (2024), [arXiv:2411.15594](https://arxiv.org/abs/2411.15594), list agreement, Cohen's kappa and Spearman correlation under "Evaluation of Agreement with Human Judgments." Li et al. (2024), [arXiv:2412.05579](https://arxiv.org/abs/2412.05579), list "Accuracy, Pearson, Spearman, Kendall's Tau, Cohen's Kappa, ICC" as the meta-evaluation metrics. A case-insensitive full-text search of those two surveys plus [2605.25273](https://arxiv.org/abs/2605.25273) and [2606.00093](https://arxiv.org/abs/2606.00093) found no occurrence of "Brier", "cross-entropy", "log loss" or "proper scor". Li et al. (EMNLP 2025), [arXiv:2411.16594](https://arxiv.org/abs/2411.16594), was opened but not included in that four-file search.
5. **Statements by critics of current practice.** Elangovan et al. (ICLR 2025, [arXiv:2410.03775](https://arxiv.org/abs/2410.03775)): "The effectiveness of automatic evaluation of generative models is typically measured by comparing the labels generated via automation with labels by humans using correlation metrics." Guerdan et al. (NeurIPS 2025, [arXiv:2503.05965](https://arxiv.org/abs/2503.05965)): "To validate such judge systems, evaluators assess human–judge agreement by first collecting multiple human ratings for each item in a validation corpus, then aggregating the ratings into a single, per-item gold label rating." Arena-Hard (ICML 2025, [arXiv:2406.11939](https://arxiv.org/abs/2406.11939)) notes Pearson and Spearman are "commonly used in benchmarks such as AlpacaEval ... to measure correlation to human preference ranking."
6. **This repository's earlier check** (AGENTS.md, 11-09-2026, about 90 papers by abstract and key passages) reached the same conclusion: "the usual justification of an LLM judge is agreement with humans on a test set (accuracy, kappa, Pearson/Spearman/Kendall)."

## 6. Pairwise versus direct-score judges, and whether the critique reaches accuracy and kappa

1. **Two regimes.** Direct-score judges (G-Eval, Prometheus, FLASK, BiGGen, JUDGE-BENCH graded tasks, CheckEval, LAGER) and automatic metrics (BERTScore, BLEURT, COMET, UniEval, the WMT metrics task) are validated by correlation. Pairwise judges and reward models (MT-Bench judge, PandaLM, JudgeLM, Auto-J, RewardBench, JudgeBench, AlpacaFarm) are validated by accuracy or agreement. Categorical and medical judges lean on Cohen's kappa. Pairwise judges, reward models and arena-style benchmarks are a large share of current judge use, so "correlation is how people validate judges" is **false for a large share of practice.**
2. **Two levels.** AlpacaEval, WildBench, Arena-Hard and PPE correlate or otherwise compare *model or reward-model rankings* with the Chatbot Arena leaderboard (system level), not only item-level judgments. The paper's item-level argument concerns forecasting individual ratings; at the system level the analogous proper target is the pairwise win probability, which Arena-Hard and PPE already score with a Brier score. The Chatbot Arena leaderboard itself is a Bradley–Terry fit whose coefficients "minimize the binary cross-entropy" ([arXiv:2403.04132](https://arxiv.org/abs/2403.04132)), so the field's gold standard for preferences is already estimated with a proper loss. WMT25's official system-level meta-metric is Soft Pairwise Accuracy (SPA; already official in WMT24); the official segment-level metric is tie-calibrated acc*eq, not SPA.
3. **Does the critique apply to accuracy and kappa?** Yes, by the same logic, though not by the same theorem.
   - Accuracy against the majority label depends on a forecast q only through argmax_r q(r). Two forecasts with the same mode tie, however different their probabilities. MT-Bench's agreement (the probability that a random judge verdict and a random human verdict match) behaves the same way: for a hard verdict r its expectation is p(r), maximized by the mode. Kappa and Scott's pi are computed from the same hard labels. None of them is a strictly proper score for the human rating distribution.
   - Wang, Zhang and Choi (Findings EMNLP 2025, [arXiv:2503.03064](https://arxiv.org/abs/2503.03064)) make exactly this point: "Accuracy incentivizes predicting a winner instead of a tie as long as oracle confidence is over 50%. In contrast, expected MSE is optimized by exactly predicting the oracle confidence," and they note MSE against a binary target "is equivalent to the Brier score."
   - Guerdan et al. report that forced-choice agreement metrics select judge systems "performing as much as 31% worse" than their response-set approach, and that among forced-choice metrics the distributional ones (KL, CE in the human-to-judge direction) "perform best."
   - Status of these statements: the argmax point is a standard property, argued here by hand and **not** machine-checked in this repository's Lean files. The repository's Lean results cover correlation and mean-matching forecasts (experiments 33, 35, 36), not accuracy or kappa.
4. **Where the premise is weakest.** (a) Pairwise judges and reward models, where the claim must be about majority-label accuracy rather than correlation. (b) System-level benchmark validation, where the object is a ranking and correlation is a defensible summary of ranking agreement. (c) Objective-answer judging (JudgeBench, No Free Labels, Judge's Verdict) where there is one correct label and little genuine disagreement; there accuracy is close to proper because the human distribution is nearly a point mass. (d) The critique has precedent: Pavlick and Kwiatkowski (2019), ChaosNLI (2020), Uma et al. (2021) and the 2025–2026 papers in group X already argue for distributional evaluation, so the paper should cite them rather than claim novelty for the idea.

## 7. Closest prior work and novelty risk

**Lyu, Do and Pergola (27 Oct 2026),** "Human-Residual Steering: Aligning Frozen LLM Judges with Human Judgment Distributions", [arXiv:2610.37868](https://arxiv.org/abs/2610.37868), is the closest prior work found.

Verified from the PDF: they extract next-token logits over single-token rating verbalizers from a frozen judge; train with RPS for ordinal labels and CE for nominal labels; evaluate Jensen–Shannon distance as the **primary** metric, plus annotation-level RPS or Brier, and Spearman and mean absolute error versus aggregate human scores. Mean Spearman rises from 0.449 to 0.579. Datasets include SummEval, Topical-Chat, FLASK, HelpSteer2, HelpSteer3, ChaosNLI, DynaSent and Scruples. Baselines include temperature scaling, Dirichlet calibration, linear and MLP output mappers, LoRA distributional fine-tuning after Chen et al. (2025), and NAPHA (Steindl et al. 2026).

It is a representation-steering method paper, not an evaluation-protocol or identification paper. It still reports Spearman gains as a validation result. Do **not** claim that this repository was first to compute RPS on an ordinal LLM judge.

**Li et al.,** "Judging with Confidence: Calibrating Autoraters to Preference Distributions", [arXiv:2510.00263](https://arxiv.org/abs/2510.00263), is a **different** paper from Wang, Zhang and Choi (Findings EMNLP 2025, [arXiv:2503.03064](https://arxiv.org/abs/2503.03064)). Li et al. calibrate pairwise autoraters with Brier and logarithmic rewards and leave a thorough ordinal/Likert study as future work. The published venue is **Findings of EMNLP 2026** ([ACL Anthology 2026.findings-emnlp.2093](https://aclanthology.org/2026.findings-emnlp.2093/)). The ICLR 2026 OpenReview forum still reads "Submitted to ICLR 2026"; acceptance at ICLR 2026 was **not** confirmed and should not be claimed.

**2025 allies** (already in the table): Elangovan et al. (ICLR 2025), Guerdan et al. (NeurIPS 2025), Chen et al. "Distributional LLM-as-a-Judge" (NeurIPS 2025), and Wang, Zhang and Choi (Findings EMNLP 2025). Plus the earlier disagreement lineage in Section 8.

A recommended niche, if the paper needs one: a measurement and decision-theory argument that correlation, majority-label agreement **and** psychometric reliability (kappa, IRT, G-theory) do not certify a judge as a forecaster of p*(y | c), with proper scores (RPS, CE, Brier) as the evaluation target, including judge-selection reversals. That is not what Lyu et al. or Li et al. set out to do.

## 8. Where distributional (proper-score) evaluation already exists

| Paper | Year | Venue | What it evaluates against | Metric | Link |
|---|---|---|---|---|---|
| Pavlick & Kwiatkowski, "Inherent Disagreements in Human Textual Inferences" | 2019 | TACL 7:677–694 | Distribution of NLI judgments | Argues for "a refined evaluation objective that requires models to explicitly capture the full distribution of plausible human judgments" | [Q19-1043](https://aclanthology.org/Q19-1043/) |
| Nie, Zhou & Bansal, ChaosNLI | 2020 | EMNLP 2020 | 100 annotations per NLI item | Jensen–Shannon distance and KL between model softmax and human distribution | [2010.03532](https://arxiv.org/abs/2010.03532) |
| Uma, Fornaciari, Hovy, Paun, Plank & Poesio, "Learning from Disagreement: A Survey" | 2021 | JAIR 72:1385–1470 | Multi-annotator NLP and vision datasets | Hard and soft evaluation; finds rankings of methods depend on the evaluation chosen | [jair.1.12752](https://doi.org/10.1613/jair.1.12752) |
| Uma et al., "SemEval-2021 Task 12: Learning with Disagreements" | 2021 | SemEval 2021 | Soft labels per item | Hard F1 plus soft cross-entropy against the annotator distribution | [2021.semeval-1.41](https://aclanthology.org/2021.semeval-1.41/) |
| Plank, "The 'Problem' of Human Label Variation" | 2022 | EMNLP 2022 | Position paper | Argues against single ground truth in data, modelling and evaluation | [2211.02570](https://arxiv.org/abs/2211.02570) |
| Aroyo et al., "DICES Dataset" | 2023 | NeurIPS 2023 (Datasets and Benchmarks) | Safety ratings with high replication per item, as distributions across demographics | Dataset (enables distributional evaluation) | [2306.11247](https://arxiv.org/abs/2306.11247) |
| Elangovan et al., "Beyond correlation" | 2025 | ICLR 2025 | Human label uncertainty in judge validation | Binned Jensen–Shannon divergence; stratification | [2410.03775](https://arxiv.org/abs/2410.03775) |
| Guerdan et al., "Validating LLM-as-a-Judge Systems under Rating Indeterminacy" | 2025 | NeurIPS 2025 | Multi-label response sets | KL, CE, response-set MSE | [2503.05965](https://arxiv.org/abs/2503.05965) |
| Chen et al., "Distributional LLM-as-a-Judge" | 2025 | NeurIPS 2025 | Empirical human rating distributions | Trains with KL plus CE; evaluates KL and accuracy | [2505.12301](https://arxiv.org/abs/2505.12301) |
| Wang, Zhang & Choi, "Improving LLM-as-a-Judge Inference with the Judgment Distribution" | 2025 | Findings EMNLP 2025 | Averaged human preferences | MSE (Brier) plus accuracy | [2503.03064](https://arxiv.org/abs/2503.03064) |
| Li et al., "Judging with Confidence" | 2026 | Findings EMNLP 2026 | Pairwise preference distribution | Brier and log rewards; ordinal Likert left as future work | [2510.00263](https://arxiv.org/abs/2510.00263) |
| Lyu, Do & Pergola, "Human-Residual Steering" | 2026 | arXiv, 27 Oct 2026 | Individual human rating distributions | RPS (ordinal), CE (nominal); JSD primary; also Spearman | [2610.37868](https://arxiv.org/abs/2610.37868) |

Distributional evaluation of *classifiers* against multi-annotator labels is an established NLP niche since 2019–2021; its use for *LLM-judge validation* appears from 2025 and remains a minority practice (6 of 49 coded papers).

## 9. Checked claims from the live literature sweep (Gemini)

A separate literature sweep proposed additional papers and venues. Each claim was checked against a primary source. The merged framing Doc at [1Pt4NFmrYn3aclbQqx6_YGIUXS9bQappZsY08IEEBlNo](https://docs.google.com/document/d/1Pt4NFmrYn3aclbQqx6_YGIUXS9bQappZsY08IEEBlNo/edit) should be read with the corrections below; that Doc was **not** edited.

Verified and added to `papers.py` where they were missing: CheckEval (Lee et al., EMNLP 2025); LAGER / "Beyond the Surface" (Lai et al., NeurIPS 2025); Evaluating Scoring Bias (Li et al., DASFAA 2026); PPE (Frick et al., ICLR 2025); WMT25 Metrics Shared Task; LongJudgeBench (Chen et al., EMNLP 2026); RewardBench 2 (Malik et al., ICLR 2026); AlpacaFarm; the AlpacaEval repo; Lyu, Do and Pergola (arXiv, 27 Oct 2026); Li et al. "Judging with Confidence" (Findings EMNLP 2026); Choi et al. IRT-GRM (ICML 2026).

Claim-by-claim:

| Claim | Status | Fix |
|---|---|---|
| Li et al., "Judging with Confidence" ([arXiv:2510.00263](https://arxiv.org/abs/2510.00263)) was accepted at ICLR 2026, and is not Findings EMNLP | **False** | Published venue is Findings of EMNLP 2026 ([2026.findings-emnlp.2093](https://aclanthology.org/2026.findings-emnlp.2093/)). The ICLR 2026 OpenReview forum still says "Submitted to ICLR 2026". This paper is also **not** the same as Wang, Zhang and Choi (Findings EMNLP 2025, [2503.03064](https://arxiv.org/abs/2503.03064)). |
| Lyu, Do and Pergola ([arXiv:2610.37868](https://arxiv.org/abs/2610.37868), 27 Oct 2026) use rating-token logits and RPS/CE on SummEval, FLASK, HelpSteer2/3, ChaosNLI and related sets, and report mean Spearman 0.449 to 0.579 | **Verified** | Add that Jensen–Shannon distance is their primary metric, and that Topical-Chat, DynaSent and Scruples are also in the suite. The merged Doc's arXiv URL is truncated in one place to `2610.3786` (missing the final 8). |
| Norman et al., "Reliability without Validity" ([arXiv:2606.19544](https://arxiv.org/abs/2606.19544)) was withdrawn by arXiv admin on 3 Nov 2026 | **Verified** | This survey never cited it; do not add it. The merged Doc correctly warns against citing it. |
| "Correlation Is Not Agreement" (OpenReview [67Z2nGMzNI](https://openreview.net/forum?id=67Z2nGMzNI)) proposes Alpha Parity (Krippendorff's alpha) after auditing 3.06M judgments on WMT, SummEval and QAGS | **Partly verified** | Title and forum id exist; OpenReview venue is "ACL ARR 2026 May Submission". The 3.06M-judgment and Alpha Parity details were **not** confirmed (OpenReview PDF was blocked). Do not treat those details as established, and do not code the paper in `papers.py` until the PDF is read. |
| Choi et al. ([arXiv:2602.00521](https://arxiv.org/abs/2602.00521)) is ICML 2026 and fits an IRT graded response model on SummEval/VIEScore | **Verified, incomplete** | Also evaluates Topical-Chat and HelpSteer-2, not only SummEval and VIEScore. |
| "From LLM Judge Scores to Evaluation Design: A Generalizability-Theory Framework" (OpenReview [YawwpIWqTi](https://openreview.net/forum?id=YawwpIWqTi)) is a 2026 G-theory paper | **Incomplete / misleading as cited** | OpenReview venue is "ICLR 2027 Conference Withdrawn Submission". Authors are François Frankart and Maxime Cordy. Do not cite it as a published paper; it is not in `papers.py`. |
| Gtheory4LLM on CRAN (1 Nov 2026) | **Verified** | Package by Jin Liu, [CRAN Gtheory4LLM](https://cran.r-project.org/package=Gtheory4LLM). A software package, not a validation experiment, so it is not a coded row. |
| Rao and Callison-Burch ([arXiv:2606.00093](https://arxiv.org/abs/2606.00093)), May 2026, coded 24 papers as described | **Verified** | Not an anonymous paper. |
| WMT25 shifted its primary *segment-level* metric to Soft Pairwise Accuracy (SPA) and acc* | **Partly false** | SPA is the official *system-level* meta-metric (already official in WMT24). The official segment-level metric is tie-calibrated acc*eq. WMT25 says it follows the same approach as last year. |
| LAGER is an arXiv 2025 paper; scoring-bias is "arXiv 2025/2026"; RewardBench 2 is "2025" | **Venues incomplete** | LAGER is the method name in Lai et al., **NeurIPS 2025**. Scoring bias is Li et al., **DASFAA 2026**. RewardBench 2 is Malik et al., **ICLR 2026**. |
| The 2025–2026 literature has rebounded into psychometric reliability as the replacement ruler | **Overstated** | The thread is real (Choi; Rao and Callison-Burch recommend weighted kappa; an ARR submission and a withdrawn ICLR 2027 G-theory paper exist). In this sample only 1 of 49 coded papers reports IRT/G-theory reliability (R). Correlation and majority-label agreement remain the dominant practice. |

## 10. Verdict and recommended phrasing

1. **Verdict: partly true, overstated as worded.** "Correlation is how the field validates judges" is true for direct-score judges, automatic metrics and judge-based benchmarks (29/30 in this sample), and false for pairwise judges and reward models, which use accuracy or agreement (13/13). The broader claim, that judges are validated by a point-agreement statistic against one aggregated human label per item and almost never by a proper score against the human rating distribution, holds in 43 of 49 coded papers.
2. **Recommended phrasing for the paper:** "LLM judges are validated almost exclusively by agreement with aggregated human labels: correlation for scored outputs and accuracy, percent agreement or kappa for preferences and categories. None of these is a proper scoring rule for the conditional distribution of human ratings." Then: correlation is the headline case because it is the default for scored judges and is invariant to affine maps; majority-label accuracy is the same failure for preference judges, because it sees only the mode. Kappa, IRT reliability and G-theory coefficients measure agreement or reproducibility, not forecast quality, so they are not scoring rules for p*(y | c).
3. **Do not write** that correlation is the only or universal validation method, that nobody evaluates distributions (Section 8), that the accuracy result is proved in this repository (Section 6.3), that Li et al. (2510.00263) is an ICLR 2026 paper, or that this work was first to compute RPS on an LLM judge.
4. **Cite as allies:** Elangovan et al. 2025, Guerdan et al. 2025, Chen et al. 2025, Wang, Zhang and Choi 2025, Li et al. 2026 (Findings EMNLP), Lyu, Do and Pergola 2026, plus Pavlick and Kwiatkowski 2019, Nie et al. 2020, Uma et al. 2021 and Plank 2022.

## 11. Limits of this check

- Convenience sample of 49 papers and tools chosen from canonical, highly cited and later-sweep work; not a systematic review, no second coder, no inter-coder reliability.
- Metric coding is from targeted full-text search, not end-to-end reading; a metric mentioned only in an appendix may be missed.
- Venues for Judge's Verdict (under review) and No Free Labels (COLM 2026 per the PDF footer) were not confirmed on a proceedings site.
- "Correlation Is Not Agreement" (OpenReview 67Z2nGMzNI) was not coded, because the PDF could not be read.
- The RULE-LLM per-metric counts remain unread in the full text.

## TL;DR

The ruler is in use, but it has two faces. Scored judges and metrics are validated by correlation (29/30 such papers here); preference judges and reward models by accuracy or agreement (13/13 report agreement, accuracy or kappa). Across all 49 coded papers, 43 compare the judge with one aggregated or forced-choice human label and report no proper score against the human rating distribution; six 2025–2026 papers do. The paper should attack "correlation and majority-label agreement", cite the 2019–2026 disagreement and distributional literature as allies — especially Lyu, Do and Pergola (Oct 2026), who already use RPS and CE — and keep the accuracy argument labelled as argued, not Lean-checked.
