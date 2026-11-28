# How are LLM judges validated in practice? A literature check of the "broken ruler" premise (27 Nov 2026)

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/43_llm_judge_validation_practice_survey_start_27Nov2026/README.md>

**Google Doc copy (private to Brando):** <https://docs.google.com/document/d/1xyjd_VDgXnGued5w2PdK8z4M6sy_Ix1bAdmiw-V0VkI/edit>

**TLDR:** The premise is **partly true**. Correlation with human scores (Pearson, Spearman, Kendall) is the near-universal validation statistic for *direct-score* judges, automatic metrics and judge-based benchmarks (22 of 22 such papers coded here). It is **not** the main statistic for *pairwise-preference* judges and reward models, which report accuracy or agreement against a single human label (11 of 11 pairwise papers report agreement, accuracy or kappa). What is truly universal is narrower and stronger for the paper: 33 of 37 papers score the judge against **one aggregated or forced-choice human label per item** with a point-agreement statistic, and none of those statistics is a proper score for the human rating distribution. Phrase the target as "correlation and majority-label agreement", not "correlation" alone.

## 1. Question

Brando (27 Nov 2026): the paper argues that correlation with humans is a broken ruler for validating large language model (LLM) judges, "assuming that is the way [we] do LLM judging validation in ML/AI. Verify that ... Otherwise we'd break a ruler people aren't using." This note checks that premise against the published record. It is a literature check, not an experiment; it produces no new numerical results about judges.

## 2. Method

1. **Sample.** 37 papers that propose or validate an LLM judge, an automatic text-quality metric, or a judge-based benchmark, plus 6 background papers on human label disagreement and 6 surveys or scoping reviews. The sample is the canonical set named in the brief plus papers found by targeted search (arXiv, ACL Anthology, OpenReview). It is a **purposive convenience sample**, not a systematic review: it over-represents highly cited 2020–2025 work and under-represents applied and industry papers. Percentages below describe this sample, not the field.
2. **Verification.** Every paper's title, authors and date were retrieved from the arXiv API or the ACL Anthology. Venues come from the arXiv comment field, OpenReview, Crossref or the PDF footer. The metric column was read from each paper's full text (PDF converted with `pdftotext`, then searched for the metric names and read in context), not from memory. No paper was read end to end.
3. **Coding.** Each paper gets every metric family it reports for judge-versus-human (or metric-versus-human) validation:
   - **C**, correlation: Pearson, Spearman, Kendall, at sample, summary or system level.
   - **A**, raw agreement: accuracy, percent agreement, precision, recall, F1.
   - **K**, chance-corrected agreement: Cohen or Fleiss kappa, Scott's pi, Krippendorff alpha.
   - **P**, a proper score or distributional divergence against *item-level* human ratings: cross-entropy (CE) or negative log-likelihood, Kullback–Leibler (KL) or Jensen–Shannon divergence, Brier score or mean squared error (MSE) against a soft label.
   - **Ps**, a proper score at the system (model-pair) level only.
   
   Judge type: **D** direct-score judge or metric, **S** benchmark validated against a human leaderboard, **W** pairwise-preference judge or reward model, **X** disagreement- or distribution-aware validation.
4. **Reproducibility.** [`papers.py`](papers.py) holds the coded rows; [`tally.py`](tally.py) prints every count in Section 3 and the table in Section 4 (`python3 tally.py`; deterministic, run twice with identical output).

## 3. Tally (37 judge or metric papers)

Counts are papers reporting at least one metric of the family, so rows overlap.

| Metric family | Papers | Share |
|---|---|---|
| Correlation (C) | 25 | 68% |
| Raw agreement / accuracy (A) | 18 | 49% |
| Chance-corrected agreement (K) | 7 | 19% |
| Item-level proper score or divergence vs. human distribution (P) | 4 | 11% |
| System-level proper score only (Ps) | 1 | 3% |
| Correlation is the *only* family reported | 15 | 41% |
| **No item-level proper score** | **33** | **89%** |

By judge type:

| Judge type | Papers | C | A | K | P |
|---|---|---|---|---|---|
| Direct-score judges and automatic metrics (D) | 19 | 19 | 4 | 2 | 0 |
| Judge-based benchmarks vs. a human leaderboard (S) | 3 | 3 | 1 | 0 | 0 (1 Ps) |
| Pairwise judges and reward models (W) | 11 | 1 | 10 | 3 | 0 |
| Distribution-aware (X), all from 2025 | 4 | 2 | 3 | 2 | 4 |

Read: correlation is the default whenever the judge emits a score (22/22 of D and S papers); agreement or accuracy is the default whenever it emits a preference (10/11 W papers report A; all 11 report A or K). Proper scores against the human rating distribution appear only in four 2025 papers that were written specifically to criticize single-label validation.

## 4. Paper-by-paper table

Venue "arXiv" means no peer-reviewed venue was found. The metric column paraphrases the paper's own text.

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
| 13 | SummEval (Fabbri et al.) | 2021 | TACL 2021 | D | Kendall tau (system level) | C | [2007.12626](https://arxiv.org/abs/2007.12626) |
| 14 | BERTScore (Zhang et al.) | 2020 | ICLR 2020 | D | |Pearson| and Kendall (WMT18 standard) | C | [1904.09675](https://arxiv.org/abs/1904.09675) |
| 15 | BLEURT (Sellam et al.) | 2020 | ACL 2020 | D | Kendall tau and Pearson (WMT17/18/19) | C | [2004.04696](https://arxiv.org/abs/2004.04696) |
| 16 | COMET (Rei et al.) | 2020 | EMNLP 2020 | D | WMT Kendall-tau-like (concordant minus discordant pairs) | C | [2009.09025](https://arxiv.org/abs/2009.09025) |
| 17 | UniEval (Zhong et al.) | 2022 | EMNLP 2022 | D | Spearman, Kendall (summary-level) | C | [2210.07197](https://arxiv.org/abs/2210.07197) |
| 18 | WMT23 Metrics Shared Task (Freitag et al.) | 2023 | WMT 2023 | D | System: pairwise accuracy + Pearson; segment: Pearson + tie-calibrated pairwise accuracy (acc*eq) | C,A | [ACL:2023.wmt-1.51](https://aclanthology.org/2023.wmt-1.51/) |
| 19 | GEMBA (Kocmi & Federmann) | 2023 | EAMT 2023 | D | System-level pairwise accuracy; segment-level Kendall | C,A | [2302.14520](https://arxiv.org/abs/2302.14520) |
| 20 | Length-Controlled AlpacaEval (Dubois et al.) | 2024 | COLM 2024 | S | Spearman with Chatbot Arena (0.94 to 0.98) | C | [2404.04475](https://arxiv.org/abs/2404.04475) |
| 21 | WildBench (Lin et al.) | 2025 | ICLR 2025 | S | Pearson with Chatbot Arena Elo (0.98 / 0.95) | C | [2406.04770](https://arxiv.org/abs/2406.04770) |
| 22 | Arena-Hard / BenchBuilder (Li et al.) | 2025 | ICML 2025 | S | Separability and agreement with confidence, Spearman, Pearson, Pair Rank Brier score (system pairs) | C,A,Ps | [2406.11939](https://arxiv.org/abs/2406.11939) |
| 23 | MT-Bench & Chatbot Arena judge (Zheng et al.) | 2023 | NeurIPS 2023 D&B | W | Agreement = P(random individual of each type agree); >80% | A | [2306.05685](https://arxiv.org/abs/2306.05685) |
| 24 | Chatbot Arena (Chiang et al.) | 2024 | ICML 2024 | W | Pairwise agreement rate (crowd, GPT-4 judge, experts); leaderboard is a Bradley-Terry fit minimizing binary cross-entropy | A | [2403.04132](https://arxiv.org/abs/2403.04132) |
| 25 | PandaLM (Wang et al.) | 2024 | ICLR 2024 | W | Accuracy, precision, recall, F1 vs. human annotation | A | [2306.05087](https://arxiv.org/abs/2306.05087) |
| 26 | JudgeLM (Zhu et al.) | 2025 | ICLR 2025 | W | Agreement, precision, recall, F1 vs. GPT-4 teacher and human labels | A | [2310.17631](https://arxiv.org/abs/2310.17631) |
| 27 | Auto-J (Li et al.) | 2024 | ICLR 2024 | W | Pairwise agreement with human labels; Pearson/Spearman in a best-of-N analysis | A,C | [2310.05470](https://arxiv.org/abs/2310.05470) |
| 28 | LLMs are not Fair Evaluators (Wang et al.) | 2024 | ACL 2024 | W | Accuracy and kappa vs. human pairwise judgments | A,K | [2305.17926](https://arxiv.org/abs/2305.17926) |
| 29 | RewardBench (Lambert et al.) | 2025 | Findings NAACL 2025 | W | Accuracy (chosen scored above rejected) | A | [2403.13787](https://arxiv.org/abs/2403.13787) |
| 30 | JudgeBench (Tan et al.) | 2025 | ICLR 2025 | W | Accuracy on objectively-labelled pairs | A | [2410.12784](https://arxiv.org/abs/2410.12784) |
| 31 | Judging the Judges (Thakur et al.) | 2025 | GEM 2025 workshop | W | Percent agreement and Scott's pi (Cohen kappa in an earlier version) | A,K | [2406.12624](https://arxiv.org/abs/2406.12624) |
| 32 | Who Validates the Validators? EvalGen (Shankar et al.) | 2024 | UIST 2024 | W | Alignment of evaluator verdicts with user grades (accuracy-type) | A | [2404.12272](https://arxiv.org/abs/2404.12272) |
| 33 | No Free Labels (Krumdick et al.) | 2026 | COLM 2026 (per PDF footer) | W+D | Cohen kappa (F1 in appendix) vs. expert labels | K | [2503.05061](https://arxiv.org/abs/2503.05061) |
| 34 | Beyond correlation (Elangovan et al.) | 2025 | ICLR 2025 | X | Stratify by human-label uncertainty; binned Jensen-Shannon divergence; critiques correlation, Krippendorff alpha, Randolph kappa | P,C,K | [2410.03775](https://arxiv.org/abs/2410.03775) |
| 35 | Validating LLM-as-a-Judge under Rating Indeterminacy (Guerdan et al.) | 2025 | NeurIPS 2025 | X | KL and CE vs. human distribution, response-set MSE; forced-choice agreement selects judges up to 31% worse | P,A,K | [2503.05965](https://arxiv.org/abs/2503.05965) |
| 36 | Distributional LLM-as-a-Judge (Chen et al.) | 2025 | NeurIPS 2025 | X | KL to human rating distribution; accuracy vs. majority | P,A | [2505.12301](https://arxiv.org/abs/2505.12301) |
| 37 | Improving LLM-as-a-Judge Inference with the Judgment Distribution (Wang et al.) | 2025 | Findings EMNLP 2025 | X | Accuracy plus MSE vs. averaged human preference (= Brier score for binary targets); Spearman of std. devs. | P,A,C | [2503.03064](https://arxiv.org/abs/2503.03064) |

## 5. Independent evidence on prevalence

These sources counted metrics themselves; they are larger or differently selected than the sample above.

1. **Rao & Callison-Burch (2026),** "Agreement Metrics for LLM-as-Judge Evaluation: What to Report and Why", [arXiv:2606.00093](https://arxiv.org/abs/2606.00093). Their coding of 24 recent LLM-as-judge papers "finds 11 graded or continuous evaluations, most reporting correlations against human ratings, 7 pairwise-preference evaluations with varying tie conventions, 3 binary-verdict evaluations, 2 judge ensembles, and 1 scale-matched mix." They describe it as a descriptive, non-exhaustive scan coded by one author.
2. **Lee, Yun & Jung (preprint, under review),** "RULE-LLM reporting framework for large language model-as-a-judge evaluation in medicine developed from a scoping review", [doi:10.21203/rs.3.rs-11611263/v1](https://doi.org/10.21203/rs.3.rs-11611263/v1). The abstract says 91 medical LLM-judge studies were included and 65.9% reported agreement with human evaluators "with widely varying metrics." A search-engine excerpt of the full text (not re-read by me; **unverified**) gives, among the 60 studies reporting agreement: Cohen's kappa 12 (20.0%), Pearson 7 (11.7%), Spearman, Kendall and intraclass correlation 6 each (10.0%).
3. **Li et al. (2026),** "LLM-as-a-Judge in Healthcare: A Scoping Analysis of Applications, Methods, and Human Alignment", [arXiv:2605.25273](https://arxiv.org/abs/2605.25273). 134 studies. Its metric taxonomy splits "association" metrics (Pearson, Spearman, Kendall, \(R^2\)) from "classification" metrics (agreement rate, accuracy, F1, kappa, Krippendorff alpha, Gwet's AC1), and it synthesizes agreement rate, Cohen's kappa and correlation as "three widely reported metrics."
4. **Surveys.** Gu et al. (2024), [arXiv:2411.15594](https://arxiv.org/abs/2411.15594), list agreement, Cohen's kappa and Spearman correlation under "Evaluation of Agreement with Human Judgments." Li et al. (2024), [arXiv:2412.05579](https://arxiv.org/abs/2412.05579), list "Accuracy, Pearson, Spearman, Kendall's Tau, Cohen's Kappa, ICC" as the meta-evaluation metrics. Neither lists a proper scoring rule (Section 8 gives the full-text search). Li et al. (2024/EMNLP 2025), [arXiv:2411.16594](https://arxiv.org/abs/2411.16594), was opened but its metric discussion was not coded.
5. **Statements by critics of current practice.** Elangovan et al. (ICLR 2025, [arXiv:2410.03775](https://arxiv.org/abs/2410.03775)): "The effectiveness of automatic evaluation of generative models is typically measured by comparing the labels generated via automation with labels by humans using correlation metrics." Guerdan et al. (NeurIPS 2025, [arXiv:2503.05965](https://arxiv.org/abs/2503.05965)): "To validate such judge systems, evaluators assess human–judge agreement by first collecting multiple human ratings for each item in a validation corpus, then aggregating the ratings into a single, per-item gold label rating." Arena-Hard (ICML 2025, [arXiv:2406.11939](https://arxiv.org/abs/2406.11939)) notes Pearson and Spearman are "commonly used in benchmarks such as AlpacaEval ... to measure correlation to human preference ranking."
6. **This repository's earlier check** (AGENTS.md, 11-09-2026, about 90 papers by abstract and key passages) reached the same conclusion: "the usual justification of an LLM judge is agreement with humans on a test set (accuracy, kappa, Pearson/Spearman/Kendall)."

## 6. Pairwise versus direct-score judges, and whether the critique reaches accuracy and kappa

1. **Two regimes.** Direct-score judges (G-Eval, Prometheus, FLASK, BiGGen, JUDGE-BENCH graded tasks) and automatic metrics (BERTScore, BLEURT, COMET, UniEval, the WMT metrics task) are validated by correlation. Pairwise judges and reward models (MT-Bench judge, PandaLM, JudgeLM, Auto-J, RewardBench, JudgeBench) are validated by accuracy or agreement. Categorical and medical judges lean on Cohen's kappa. Pairwise judges, reward models and arena-style benchmarks are a large share of current judge use, so "correlation is how people validate judges" is **false for a large share of practice.**
2. **Two levels.** AlpacaEval, WildBench and Arena-Hard correlate *model rankings* with the Chatbot Arena leaderboard (system level), not item-level judgments. The paper's item-level argument concerns forecasting individual ratings; at the system level the analogous proper target is the pairwise win probability, which Arena-Hard already scores with a Brier score. The Chatbot Arena leaderboard itself is a Bradley–Terry fit whose coefficients "minimize the binary cross-entropy" ([arXiv:2403.04132](https://arxiv.org/abs/2403.04132)), so the field's gold standard for preferences is already estimated with a proper loss.
3. **Does the critique apply to accuracy and kappa?** Yes, by the same logic, though not by the same theorem.
   - Accuracy against the majority label depends on a forecast \(q\) only through \(\arg\max_r q(r)\). Two forecasts with the same mode tie, however different their probabilities. MT-Bench's agreement (the probability that a random judge verdict and a random human verdict match) behaves the same way: for a hard verdict \(r\) its expectation is \(p(r)\), maximized by the mode. Kappa and Scott's pi are computed from the same hard labels. None of them is a strictly proper score for the human rating distribution.
   - Wang et al. (Findings EMNLP 2025, [arXiv:2503.03064](https://arxiv.org/abs/2503.03064)) make exactly this point: "Accuracy incentivizes predicting a winner instead of a tie as long as oracle confidence is over 50%. In contrast, expected MSE is optimized by exactly predicting the oracle confidence," and they note MSE against a binary target "is equivalent to the Brier score."
   - Guerdan et al. report that forced-choice agreement metrics select judge systems "performing as much as 31% worse" than their response-set approach, and that among forced-choice metrics the distributional ones (KL, CE in the human-to-judge direction) "perform best."
   - Status of these statements: the argmax point is a standard property, argued here by hand and **not** machine-checked in this repository's Lean files. The repository's Lean results cover correlation and mean-matching forecasts (experiments 33, 35, 36), not accuracy or kappa.
4. **Where the premise is weakest.** (a) Pairwise judges and reward models, where the claim must be about majority-label accuracy rather than correlation. (b) System-level benchmark validation, where the object is a ranking and correlation is a defensible summary of ranking agreement. (c) Objective-answer judging (JudgeBench, No Free Labels, Judge's Verdict) where there is one correct label and little genuine disagreement; there accuracy is close to proper because the human distribution is nearly a point mass. (d) The critique has precedent: Pavlick & Kwiatkowski (2019), ChaosNLI (2020), Uma et al. (2021) and the 2025 papers in group X already argue for distributional evaluation, so the paper should cite them rather than claim novelty for the idea.

## 7. Where distributional (proper-score) evaluation already exists

| Paper | Year | Venue | What it evaluates against | Metric | Link |
|---|---|---|---|---|---|
| Pavlick & Kwiatkowski, "Inherent Disagreements in Human Textual Inferences" | 2019 | TACL 7:677–694 | Distribution of NLI judgments | Argues for "a refined evaluation objective that requires models to explicitly capture the full distribution of plausible human judgments" | [Q19-1043](https://aclanthology.org/Q19-1043/) |
| Nie, Zhou & Bansal, "What Can We Learn from Collective Human Opinions on Natural Language Inference Data?" (ChaosNLI) | 2020 | EMNLP 2020 | 100 annotations per NLI item | Jensen–Shannon distance and KL between model softmax and human distribution | [2010.03532](https://arxiv.org/abs/2010.03532) |
| Uma, Fornaciari, Hovy, Paun, Plank & Poesio, "Learning from Disagreement: A Survey" | 2021 | JAIR 72:1385–1470 | Multi-annotator NLP and vision datasets | Hard and soft evaluation; finds rankings of methods depend on the evaluation chosen | [jair.1.12752](https://doi.org/10.1613/jair.1.12752) |
| Uma et al., "SemEval-2021 Task 12: Learning with Disagreements" | 2021 | SemEval 2021 | Soft labels per item | Hard F1 plus soft cross-entropy against the annotator distribution | [2021.semeval-1.41](https://aclanthology.org/2021.semeval-1.41/) |
| Plank, "The 'Problem' of Human Label Variation" | 2022 | EMNLP 2022 | Position paper | Argues against single ground truth in data, modelling and evaluation | [2211.02570](https://arxiv.org/abs/2211.02570) |
| Aroyo et al., "DICES Dataset" | 2023 | NeurIPS 2023 (Datasets and Benchmarks) | Safety ratings with high replication per item, as distributions across demographics | Dataset (enables distributional evaluation) | [2306.11247](https://arxiv.org/abs/2306.11247) |
| Elangovan et al., "Beyond correlation" | 2025 | ICLR 2025 | Human label uncertainty in judge validation | Binned Jensen–Shannon divergence; stratification | [2410.03775](https://arxiv.org/abs/2410.03775) |
| Guerdan et al., "Validating LLM-as-a-Judge Systems under Rating Indeterminacy" | 2025 | NeurIPS 2025 | Multi-label response sets | KL, CE, response-set MSE | [2503.05965](https://arxiv.org/abs/2503.05965) |
| Chen et al., "Distributional LLM-as-a-Judge" (arXiv title "Beyond Single-Point Judgment") | 2025 | NeurIPS 2025 | Empirical human rating distributions | Trains with KL plus CE; evaluates KL and accuracy | [2505.12301](https://arxiv.org/abs/2505.12301) |
| Wang et al., "Improving LLM-as-a-Judge Inference with the Judgment Distribution" | 2025 | Findings EMNLP 2025 | Averaged human preferences | MSE (Brier) plus accuracy | [2503.03064](https://arxiv.org/abs/2503.03064) |

So: distributional evaluation of *classifiers* against multi-annotator labels is an established NLP niche since 2019–2021; its use for *LLM-judge validation* appears only in 2025 and is still a minority practice.

## 8. Verdict and recommended phrasing

1. **Verdict: partly true, overstated as worded.** "Correlation is how the field validates judges" is true for direct-score judges, automatic metrics and judge-based benchmarks, and false for pairwise judges and reward models, which use accuracy or agreement. The broader claim, that judges are validated by a point-agreement statistic against one aggregated human label per item and almost never by a proper score against the human rating distribution, holds in 33 of 37 coded papers. The full texts of two surveys and two prevalence scans ([2411.15594](https://arxiv.org/abs/2411.15594), [2412.05579](https://arxiv.org/abs/2412.05579), [2605.25273](https://arxiv.org/abs/2605.25273), [2606.00093](https://arxiv.org/abs/2606.00093)) contain no occurrence of "Brier", "cross-entropy", "log loss" or "proper scor[ing rule]" (case-insensitive search).
2. **Recommended phrasing for the paper:** "LLM judges are validated almost exclusively by agreement with aggregated human labels: correlation for scored outputs and accuracy, percent agreement or kappa for preferences and categories. None of these is a proper scoring rule for the conditional distribution of human ratings." Then: correlation is the headline case because it is the default for scored judges and is invariant to affine maps; majority-label accuracy is the same failure for preference judges, because it sees only the mode.
3. **Do not write** that correlation is the only or universal validation method, that nobody evaluates distributions (Section 7), or that the accuracy result is proved in this repository (Section 6.3).
4. **Cite as allies:** Elangovan et al. 2025, Guerdan et al. 2025, Chen et al. 2025, Wang et al. 2025, plus Pavlick & Kwiatkowski 2019, Nie et al. 2020, Uma et al. 2021 and Plank 2022.

## 9. Limits of this check

- Convenience sample of 37 papers chosen from canonical and highly cited work; not a systematic review, no second coder, no inter-coder reliability.
- Metric coding is from targeted full-text search, not end-to-end reading; a metric mentioned only in an appendix may be missed.
- Venues for Judge's Verdict (under review) and No Free Labels (COLM 2026 per the PDF footer) were not confirmed on a proceedings site.
- The RULE-LLM per-metric counts are from a search excerpt of the full text and are marked unverified.

## TL;DR

The ruler is in use, but it has two faces. Scored judges and metrics are validated by correlation (22/22 such papers here); preference judges and reward models by accuracy or agreement (11/11 report agreement, accuracy or kappa). Across all 37 coded papers, 33 compare the judge with one aggregated or forced-choice human label and report no proper score against the human rating distribution; only four 2025 papers do. The paper should attack "correlation and majority-label agreement", cite the 2019–2025 disagreement literature as allies, and keep the accuracy argument labelled as argued, not Lean-checked.
