# judges-dont-need-psychometrics-or-corr

**Thesis.** To check whether an LLM (or agent) judge "works", stop computing Pearson/Spearman correlation with human scores and stop borrowing psychometric validity machinery. A judge is a probability distribution over scores. So treat judge evaluation and judge fitting the way ML treats everything else: **match the human score distribution, i.e. minimize a KL divergence.**

> "If humans have some taste, just match the taste distribution. That's it."

Status: **early, position-paper stage.** No empirical results yet. The toy script in `experiments/01_synthetic_pearson_vs_kl_squashed_judge_0to5/` is the only number-producing code so far.

---

## 1. The problem

The usual way to validate a judge: collect human scores and judge scores on the same items, make a scatter plot, and report a correlation.

- **Pearson** only measures *linear* association. If the relationship isn't linear in the features you happened to plot, it tells you something misleading.
- **Spearman / Kendall** only measure *rank order*. They tell you whether the ordering is right, not whether the scores are.
- Together they still capture only **some** distributional properties of human judgment, not all of it. A judge can be highly correlated with humans while assigning a completely different score distribution (for example, giving everything a 4 or a 5). Correlation is also easy to over-read and invites spurious conclusions.
- **Psychometrics.** A common response is to import validity and reliability metrics from psychometrics. The argument here is that we don't need them. Fitting and evaluating a conditional distribution is a standard ML problem with mature tools.

Analogy from the call: means and variances are the first two "features" of a distribution. Keep adding moments and you pin down the whole PMF/PDF. Correlation fixates on one summary. KL compares the whole distribution.

## 2. Proposal: a judge is a distribution, so fit it with KL

**Judge.** Given an agent output $y$ and a gold reference $y^{\star}$, a judge is a distribution over a score $r$:

$$p_\theta(r \mid y, y^{\star}), \qquad r \in \lbrace 0,\dots,5 \rbrace \text{ (or } [0,1]).$$

Here $\theta$ is anything you can search over: prompt, rubric, temperature, weights, or calibration parameters.

**What people do now:**

$$\theta^{\star} = \arg\max_\theta \ \mathrm{corr}\big(r_{\text{judge}}(\theta),\ r_{\text{human}}\big)$$

**What we argue for:**

$$\theta^{\star} = \arg\min_\theta \ \mathbb{E}_{(y,y^{\star})}\Big[ D_{\mathrm{KL}}\big(p_{\text{human}}(\cdot \mid y,y^{\star})\ \Vert\ p_\theta(\cdot \mid y,y^{\star})\big) \Big]$$

Practical variants we plan to support:

- **Per-item KL** when several humans rate the same $(y, y^{\star})$, so $p_{\text{human}}$ is the empirical histogram for that item.
- **Pooled KL** between the marginal score histograms when there is only one rating per item. This is weaker, since it ignores the pairing, but it is cheap.
- **Smoothing** (additive/Laplace) so that KL stays finite when a judge never uses some score.
- **Optimization.** Reuse the standard toolbox built for KL objectives (cross-entropy, SGD/Adam) instead of inventing a new metric.

**Reporting.** KL is the right *objective*, but it isn't intuitive for people to read. For bounded scores we plan to also report **MSE on normalized scores** (for example, divide a 0–5 score by 5). Unlike classification accuracy, MSE penalizes judge=0 vs human=5 much more than judge=4 vs human=5. Exactly which metric to report is still an open question from the call.

## 3. Planned experiments

1. **Toy counterexamples** (`experiments/01_synthetic_pearson_vs_kl_squashed_judge_0to5/`): a judge with Pearson ≈ 0.88 whose score distribution is far from the human one (large KL), next to a judge with similar correlation and near-zero KL. Run `python experiments/01_synthetic_pearson_vs_kl_squashed_judge_0to5/kl_vs_corr.py`.
2. **Theorem-proving agent judges**: apply KL-based judge fitting and evaluation to the human judgments being collected on whether Brando's theorem-proving agents write good theorems/formalizations. Human data still needs to be collected. An open sub-question is how many human-rated $(y, y^{\star})$ pairs are needed.

See [`experiments/README.md`](experiments/README.md).

## 4. Repo layout

The repo follows the [agents-config](https://github.com/brando90/agents-config) research layout (Rule 49): an `experiments/` bucket, a `paper_latex_and_notes/` bucket, and later `src/`.

```
.
├── README.md
├── AGENTS.md / CLAUDE.md                       # identical agent entry points (route to ~/agents-config)
├── experiments/
│   ├── README.md                               # experiment index
│   └── 01_synthetic_pearson_vs_kl_squashed_judge_0to5/   # kl_vs_corr.py
└── paper_latex_and_notes/
    ├── AAAI_2027_EvalAIScientists_JudgesKL/    # position-paper draft: 00_main_aaai_2027.tex (+ PDF)
    └── notes/fathom_call_11-07-2026/           # notes + quotes from the originating call
```

## 5. Source

The ideas come from a discussion on 2026-11-07. Notes are in [`paper_latex_and_notes/notes/fathom_call_11-07-2026/notes.md`](paper_latex_and_notes/notes/fathom_call_11-07-2026/notes.md).

Target venue: [Effective and Efficient Evaluation for AI Scientists, AAAI-27 workshop](https://eval-ai-scientists.github.io/aaai27/) (position paper, 2–4 pages).

---

_Drafted with AI assistance (Grok Bot)._
