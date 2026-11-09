# Primary-source citation checks

Checked 11-08-2026. The current paper uses the sources below. Earlier unused bibliography entries were removed from the current draft; prior revisions preserve them. No citation is evidence of human generalization for the new synthetic study.

| Citation | Primary source inspected | What it supports |
| --- | --- | --- |
| Gneiting and Raftery (2007), *Strictly Proper Scoring Rules, Prediction, and Estimation*, JASA 102(477), 359–378, DOI 10.1198/016214506000001437 | [Author-hosted published paper](https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf), categorical Examples 1 and 3 and the threshold-score construction | Logarithmic and Brier properness; the ranked probability score from cumulative thresholds. The study uses loss signs and explicitly states its averaging conventions. |
| Liu et al. (2023), *G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment* | [ACL Anthology paper and official metadata](https://aclanthology.org/2023.emnlp-main.153/) | Correlation with human judgments is an evaluation measure in G-Eval; the paper does not establish that every judge study uses only correlation. |
| Zheng et al. (2023), *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena* | [Author manuscript, v4](https://arxiv.org/abs/2306.05685v4), matched to the NeurIPS 2023 Datasets and Benchmarks entry | Human-preference agreement and documented judge biases. Agreement is not misdescribed as a correlation-only validation. |
| Cronbach and Meehl (1955), *Construct Validity in Psychological Tests*, Psychological Bulletin 52(4), 281–302, DOI 10.1037/h0040957 | [Reprint of the original article](https://www.yorku.ca/pclassic/Cronbach/construct.htm), metadata checked against the publication record | Construct validity concerns evidence supporting an interpretation, not merely one coefficient. This prevents the paper from claiming that failure of correlation rejects psychometrics as a whole. |
| Companion historical and weighted-training reports (2026) | [Exact report revisions and SHA-256 receipts](results/public_ttj_evidence.json) | Public aggregate numerical evidence only; null findings, optimizer failures and invalid resampling draws are retained. No private source data were accessed or imported. |

The shared-mean construction, loss-excess identities and scalar squared-error decomposition are derived directly in the study and checked numerically in `test_study.py` and `verify_results.py`; they are not claims of new scoring-rule theory. Terminology is consistent: conditional rating distribution, conditional mean, individual-rating loss, marginal histogram, and selection tie have separate meanings.

The paper follows a scoped countervailing-wisdom position-paper structure. It challenges correlation alone as a distribution-identification criterion, supported by a controlled construction; it does not claim a new paradigm or empirical human superiority.

**TLDR-end:** [judge objectives: citations] Every source used in the revised paper was checked against a primary text or exact public report revision. The claims stay within what those sources and the synthetic data establish.
