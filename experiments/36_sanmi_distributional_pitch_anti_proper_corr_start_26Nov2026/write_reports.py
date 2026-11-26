"""Write the reader-facing notes from results.json so every quoted number is the script's number."""

from __future__ import annotations

from pathlib import Path

from pitch_addenda import COT, ELICITATION, POWER

G_THEORY = r"""## Why not psychometrics (G-theory / D-theory)?

**G-theory.** Generalizability theory (Cronbach, Gleser, Nanda, and Rajaratnam, 1972) writes a scalar rating as an additive variance split: prompt variance \(\sigma^2_c\), rater variance \(\sigma^2_r\), and the prompt-by-rater residual \(\sigma^2_{cr}\). The generalizability coefficient \(G\) is a multi-facet form of an intraclass correlation. Like Pearson correlation, it is a second-moment summary. The usual analysis of variance treats the components as additive and homoscedastic. On a bounded 0–5 or 1–5 rubric that summary does not score calibration, and it does not see a second mode. It is not a divergence from \(p^*(y \mid c)\).

**D-studies.** A decision study pushes those variance components forward and asks how many raters \(k\) would bring \(G\) to a threshold such as \(0.80\). Under a proper scoring rule the same allocation question is the held-out risk \(R(N, k)\) of a judge trained on \(N\) prompts with \(k\) ratings each. The curves in this folder answer it in held-out cross-entropy, in the square root of the ranked probability score (RPS), and in Brier score, and they convert \(B = N \times k\) into dollars at a stated price. They do not pass through a Gaussian analysis of variance.

**Where a generalizability coefficient still helps.** It is an optional diagnostic of the human panel. A large rater main effect is harshness or leniency. A large residual is disagreement about the item or the rubric. That diagnosis plans the annotation. It is not a training loss for the judge, and it is not the test metric.

**What the \((N, k)\) curves cost on the sets this folder actually scored.** Two unit prices, both labeled as assumptions. \(\$0.05\) per rating is the published POPQUORN crowd rate, used here as a unit price. It is not a wage we observed for SummEval, ChaosNLI, or DICES. \(\$5\) per rating is an assumed expert rate. POPQUORN ratings were not loaded. CertJudge / VeriBench functional semantic correctness ratings were not on this machine. Neither appears in the table.

Train sizes are the prompt-level split, seed 0: SummEval crowd \(N=3840\) items (same article split as the expert matrix), DICES-990 \(N=594\), ChaosNLI SNLI+MNLI \(N=1867\), ChaosNLI abductive NLI \(N=919\). The \(k\)-sweep trained a cross-entropy head and a Pearson head. On SummEval crowd (grades 1–5), DICES-990, and ChaosNLI SNLI+MNLI, the cross-entropy head has lower held-out cross-entropy, lower sqrt(RPS), and lower Brier at \(k=1\), the smallest budget we trained, and it stays lower at every larger \(k\) through 4 or through 32. There is no later \(k\) at which the Pearson head becomes the better proper-score trainer.

| Dataset | \(k\) at which CE is ahead on CE, sqrt(RPS), and Brier | Train ratings \(B=N k\) | Crowd dollars at \$0.05 | Expert dollars at \$5 | Held-out CE, CE-trained vs Pearson-trained |
| --- | --- | --- | --- | --- | --- |
| SummEval crowd, grades 1–5 | 1 (also 2 and 4) | 3,840 | \$192 | \$19,200 | 1.328 vs 8.651 |
| DICES-990, three-level | 1 (through 32) | 594 | \$29.70 | \$2,970 | 0.747 vs 1.053 |
| ChaosNLI SNLI+MNLI, three-class | 1 (through 32) | 1,867 | \$93.35 | \$9,335 | 1.053 vs 4.926 |
| ChaosNLI abductive NLI, binary | point estimates differ by about 0.002; both heads sit on \(\log 2 \approx 0.693\) | 919 | \$45.95 | \$4,595 | 0.697 vs 0.699 |
| POPQUORN | not scored |  |  |  |  |
| CertJudge / VeriBench FSC, 0–5 | not scored |  |  |  |  |

At \(k=1\) the other two proper scores move with cross-entropy. DICES-990 sqrt(RPS) is 0.629 versus 0.688 and Brier is 0.445 versus 0.525. SummEval crowd sqrt(RPS) is 0.719 versus 1.012 and Brier is 0.708 versus 1.240. ChaosNLI SNLI+MNLI sqrt(RPS) is 0.627 versus 0.710 and Brier is 0.637 versus 0.939. The abductive panel is the negative control: the hash does not predict the task, so that row is not a budget recommendation.

The iso-budget cut holds \(B=1024\) ratings on ChaosNLI SNLI+MNLI, which is \$51.20 at \$0.05 and \$5,120 at \$5. Held-out cross-entropy of the cross-entropy head is 1.054 at \((N,k)=(1024,1)\), 1.056 at \((256,4)\), 1.062 at \((64,16)\), and 1.101 at \((32,32)\). The Pearson head on the same four cells is 4.580, 5.004, 6.051, and 5.874. Inside this grid, spending the fixed bill on more prompts with one rating beats fewer prompts with many ratings.

The \(k\)-sweep did not train the mean-squared-error head. On the full training ratings, that head's two-point readout has infinite held-out cross-entropy on SummEval expert, on ChaosNLI SNLI+MNLI, and on DICES-990, because a rating off the two adjacent scores has probability 0. SummEval expert uses all 3 grades on each of 3,840 train items (11,520 grades). ChaosNLI SNLI+MNLI uses all 100 votes on each of 1,867 train items (186,700 votes). DICES-990 uses every filtered rating on each of 594 train conversations. The file has 60–69 ratings per conversation and 64,886 ratings in total. The train-slice total was not stored as its own field.

This driver finished in about 105 seconds on CPU and spent \$0 on model calls. Researcher salary was not metered, so this note does not invent one. A generalizability coefficient computed on the same ratings would still not be a held-out cross-entropy, Brier score, or RPS. That is the distinction the dollar rows are buying. On known synthetic ordinal laws, not on these human sets, experiment 34 already recorded a lenient point-mass judge with generalizability coefficient 1 and Pearson 0.961 whose expected RPS was worse than a constant forecast by 0.0492 [0.0477, 0.0506].

"""

TRANSCRIPT = r"""## Complete Sanmi Transcript Coverage

As of 11-26-2026. These five points are the ones the 11-25 call left open. The table is the taxonomy; the sentences under it are the boundaries.

| Point | Output space or claim | What is true | Where it is checked |
| --- | --- | --- | --- |
| 1a | Binary, \(K=2\) | One degree of freedom. The scalar mean is the full probability mass function, and scalar \(L_2\) equals the Brier score (two-class Brier is twice scalar squared error). Pearson correlation still fails calibration, because \(\operatorname{Corr}(aX+b,Y)=\operatorname{Corr}(X,Y)\) for \(a>0\), and it is strictly anti-proper for a sampled judge: \(\sqrt{\rho}>\rho\). | Lean `binary_brier_eq_two_l2`, `pearson_affine_invariant`, `binary_witness_corr_tie`, `anti_proper_from_parts`. Python `binary_l2`, `binary_affine_witness`, `anti_proper`. Real binary panels: SummEval cut at grade \(\ge 4\), ChaosNLI abductive NLI, DICES Yes-versus-not. |
| 1b | Finite categorical or ordinal, \(K\ge 3\) (a 0–5 rubric is this case) | \(K-1\) degrees of freedom. This is ordinary discrete classification. The multinomial frequency error is \(\mathcal{O}(K/N)\), from \(\mathbb{E}[\|\hat q_N-p^*\|_2^2]=(1/N)(1-\|p^*\|_2^2)\), not a nonparametric curse of dimensionality. A scalar mean drops spread and multimodality. Cross-entropy, Brier, and ranked probability score (RPS) see them. | Lean `empirical_l2_rate`, `plugin_DB_bias`, `ordinal_same_mean_different_law`. Python `empirical_rate`, `mean_ne_pmf`. Real panels: SummEval grades 1–5 and DICES-990 three-level `Q_overall`. VeriBench functional semantic correctness (0–5) was not on this machine. |
| 1c | Continuous, \(Y\in\mathbb{R}\) | The target is an infinite-dimensional density \(p^*(y\mid c)\). Sanmi's nonparametric sample-complexity warning applies here. A usable judge then needs a parametric head (Gaussian, Beta, or quantiles), the continuous ranked probability score (CRPS), or a discretization into \(K\) bins. | Conceded. This folder does not estimate a continuous density. |
| 2 | Paired versus unpaired; agreement versus fidelity | Correlation has one narrow use as an unpaired or arbitrary-scale diagnostic. Inter-rater agreement, for example Krippendorff's \(\alpha\), measures human-to-human consensus when no target distribution has been defined. A proper scoring rule measures judge fidelity to a defined target \(p^*(y\mid c)\). Those are different questions. | The experiments in this folder score fidelity on held-out prompts. They do not report Krippendorff's \(\alpha\). |
| 3 | Sanmi's RPS framing | RPS is the squared distance between forecast and realized cumulative probabilities. It charges an ordering of how wrong the forecast is — distance from the realized rating — over the whole distribution. It is not a ranking of items. | Lean `bernoulli_square_excess` (one threshold) and `ordinal_same_mean_different_law` (the 0–5 gap). The reported column is the square root of the unbiased mean per-sample RPS; normalized RPS (divided by \(K-1\)) is stored beside it. |
| 4 | Surrogate-loss literature | Proper scoring rules and surrogate losses are classic: Brier (1950), Epstein (1969) for the ranked probability score, Gneiting and Raftery (2007). What is new here is the application to an LLM-as-a-judge, the strict anti-properness \(\sqrt{\rho}>\rho\) for independent sampled judges, the single-pass token-logit readout, and the rating-budget and dollar scaling curves. | This note, `SanmiPitch.lean`, and the figures under `figures/`. |
| 5 | Sharpen the rubric versus match \(p^*\) | Tightening a rubric so that \(p^*(y\mid c)\) becomes peakier, and training a judge to match \(p^*(y\mid c)\), are complementary. What fails is pretending a non-peaky target has zero conditional variance. | Proof B (consensus on 2 versus a 50/50 split on \(\{0,4\}\), same mean 2) and Proof C (the collapsed judge \(X_\delta=a\mu_Y(C)+b\)). |

"""


def _f(x, nd=4):
    if x is None:
        return "undefined"
    if isinstance(x, str):
        return x
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if v != v:
        return "undefined"
    if v == float("inf"):
        return "+inf"
    return f"{v:.{nd}f}"


def _ci(block, metric, nd=3):
    point = block["point"][metric]
    ci = block["ci"][metric]
    return f"{_f(point, nd)} [{_f(ci['lo'], nd)}, {_f(ci['hi'], nd)}]"


def _matrix_md(payload):
    lines = []
    for name, block in payload["matrices"].items():
        lines.append(f"### {name}")
        lines.append("")
        lines.append(block["summary"]["notes"])
        lines.append("")
        lines.append(
            f"Split by prompt group, seed 0: train {block['n_train']}, validation {block['n_val']}, "
            f"test {block['n_test']}. SVD components used: {block['svd_rank_used']}. "
            "Intervals are item-clustered percentile 95% intervals, B draws in `results.json`."
        )
        lines.append("")
        header = "| Objective | CE | sqrt(RPS) | Brier | norm. RPS | Pearson r | Spearman rho | Kendall tau |"
        lines.append(header)
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for obj, row in block["rows"].items():
            p = row["point"]
            lines.append(
                f"| {obj}-trained | {_ci(row, 'ce')} | {_ci(row, 'sqrt_rps')} | {_ci(row, 'brier')} | "
                f"{_f(p['rps_norm'])} | {_f(p['pearson_r'], 3)} | {_f(p['spearman_rho'], 3)} | {_f(p['kendall_tau'], 3)} |"
            )
        u = block["uniform"]["point"]
        m = block["marginal"]["point"]
        lines.append("")
        lines.append(
            f"Uniform forecast held-out test CE {_f(u['ce'])}. "
            f"Train-marginal forecast held-out test CE {_f(m['ce'])}. "
            f"CE head refit after shuffling train features: held-out test CE {_f(block['shuffle_ce'])}."
        )
        lines.append("")
    return "\n".join(lines)


def _curve_md(payload):
    lines = []
    for key, block in payload["crossover"].items():
        lines.append(f"### {block['label']}")
        lines.append("")
        lines.append("| k | objective | CE | sqrt(RPS) | Brier | Pearson r |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for obj, rows in block["curves"].items():
            for row in rows:
                p = row["point"]
                lines.append(
                    f"| {row['k']} | {obj}-trained | {_f(p['ce'])} | {_f(p['sqrt_rps'])} | {_f(p['brier'])} | {_f(p['pearson_r'], 3)} |"
                )
        lines.append("")
    return "\n".join(lines)


def _iso_md(payload):
    rows = payload["iso_budget"]["rows"]
    if not rows:
        return "Iso-budget rows were not produced in this run.\n"
    lines = ["| N | k | B | objective | held-out test CE | sqrt(RPS) | Brier |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        p = r["point"]
        lines.append(f"| {r['N']} | {r['k']} | {r['B']} | {r['objective']} | {_f(p['ce'])} | {_f(p['sqrt_rps'])} | {_f(p['brier'])} |")
    return "\n".join(lines) + "\n"


def _proof_blurb(payload):
    p = payload["proofs_numeric"]
    b = p["binary_affine_witness"]
    a = p["anti_proper"]
    m = p["mean_ne_pmf"]
    return (
        f"Binary witness, four paired items: Pearson correlation of the calibrated probabilities "
        f"and of the affine map \(0.1q+0.45\) are both {_f(b['corr_star'], 3)}. "
        f"Mean Brier {_f(b['brier_star'])} versus {_f(b['brier_bad'])}. "
        f"Mean cross-entropy {_f(b['ce_star'])} versus {_f(b['ce_bad'])} nats. "
        f"Anti-proper Monte Carlo (n={a['n']}): correlation of the true sampled judge {_f(a['corr_true_distribution'], 4)} "
        f"(theory 0.5) and of \(0.1\\mu+4\) {_f(a['corr_collapsed'], 4)} (theory \(\\sqrt{{1/2}}\\approx 0.7071\)). "
        f"Ordinal witness: both laws have mean {_f(m['mu'], 1)}; \(D_B={_f(m['D_B'], 1)}\); "
        f"normalized RPS gap {_f(m['normalized_rps_gap'])}; cross-entropy of the split forecast on the consensus outcome is infinite."
    )


def transcript_section():
    return TRANSCRIPT


def write_all(payload, folder: Path):
    folder = Path(folder)
    proof = _proof_blurb(payload)
    matrix = _matrix_md(payload)
    curves = _curve_md(payload)
    iso = _iso_md(payload)
    oracle = payload["within_prompt_oracle"]
    roll_lines = "\n".join(
        f"- {r['readout']}: held-out test CE {_f(r.get('ce_mean_over_reps', r['ce']))}, "
        f"Brier {_f(r['brier'])}, sqrt(RPS) {_f(r['sqrt_rps'])}"
        + (" (mean over 8 floored histograms; one unlucky draw can still put a zero on a realized class)"
           if "ce_mean_over_reps" in r else " (the softmax itself, no floor)")
        for r in payload["rollout"]["rows"]
    )
    oracle_note = (
        f"On DICES-990, splitting each test prompt's raters in half gives a within-prompt oracle, "
        f"not a deployable judge. A draw from the add-one distribution of the first half has correlation "
        f"{_f(oracle.get('corr_sampled_halfA'), 3)} with the second half. The collapsed score "
        f"0.1·mean+4 has correlation {_f(oracle.get('corr_collapsed_affine'), 3)}. "
        f"Held-out-half cross-entropy is {_f(oracle.get('mean_ce_distributional'))} versus "
        f"{_f(oracle.get('mean_ce_collapsed'))}, and Brier is {_f(oracle.get('mean_brier_distributional'))} versus "
        f"{_f(oracle.get('mean_brier_collapsed'))}. Higher correlation, worse proper scores."
    )
    avail = payload["availability"]
    pitch = f"""# Sanmi distributional pitch — 11-26-2026

**As of 11-26-2026.** Experiment `36_sanmi_distributional_pitch_anti_proper_corr_start_26Nov2026`.

**Bottom line.** On a known law, Pearson correlation is unchanged by any positive affine map and, for a judge that samples independently of the human given the prompt, a collapsed affine mean scores a strictly higher correlation than the true conditional distribution whenever the conditional mean actually varies and the conditional variance is not zero. On the real multi-annotator sets that could be loaded on this machine, the held-out comparison is the prompt-level unbiased proper risk, with item-clustered 95% intervals. Synthetic numbers below are checks of those identities. They are not human results.

## 1. Merged four-step pitch

Notation, used the same way in the note, the figures, the Python, and the Lean comments. \(c\) is the prompt. \(X\\sim\\hat q(\\cdot\\mid c)\) is the judge's rating. \(Y\\sim p^*(\\cdot\\mid c)\) is the human rating. Centered variables are \(\\tilde X=X-\\mu_X\) and \(\\tilde Y=Y-\\mu_Y\). Standardized variables are \(z_X=\\tilde X/\\sigma_X\) and \(z_Y=\\tilde Y/\\sigma_Y\). The excess Brier divergence is \(D_B(p^*\\Vert\\hat q)=\\sum_y(p^*(y\\mid c)-\\hat q(y\\mid c))^2\).

**Step 1. Concede the two points that are right, and stop at the boundary.** On a binary rating, the scalar mean is the distribution, and scalar squared error matches the Brier score: two-class Brier equals \(2(q-y)^2\). The Brier score is strictly proper, with excess risk \(D_B\). Pearson correlation is not that object. It is invariant to \(X\\mapsto aX+b\) for \(a>0\), so a shifted and compressed forecast can tie the true probabilities on correlation while \(D_B\) and cross-entropy get worse. {proof}

**Step 2. Mean squared error splits into bias, scale, and correlation, and a discrete mean is not a discrete law.** For standardized variables, \(\\mathbb E[(z_X-z_Y)^2]=2(1-\\operatorname{{Corr}}(X,Y))\). In the original units, \(\\mathbb E[(X-Y)^2]=(\\mu_X-\\mu_Y)^2+(\\sigma_X-\\sigma_Y)^2+2\\sigma_X\\sigma_Y(1-\\operatorname{{Corr}}(X,Y))\). On \(\{{0,1,2,3,4,5\}}\) a point mass at 2 and a 50/50 split on \(\{{0,4\}}\) share the mean 2, so every scalar-mean correlation ties them, while \(D_B\), normalized ranked probability score (RPS), and cross-entropy separate them. A fair coin has three couplings with \(\\mathbb E[(X-Y)^2]\\in\\{{0,1/2,1\\}}\) and \(D_B=0\). If \(X\\perp Y\), the squared difference collapses to \((\mu_X-\mu_Y)^2+\\sigma_X^2+\\sigma_Y^2\).

**Step 3. Correlation is strictly anti-proper for a sampled distributional judge, and one forward pass already returns the distribution.** Given the prompt, take \(X\) independent of \(Y\). The law of total covariance then says \(\\operatorname{{Corr}}(X_{{p^*}},Y)=\\operatorname{{Var}}(\\mu_Y(C))/\\operatorname{{Var}}(Y)\), while the zero-variance judge \(X_\\delta=a\\mu_Y(C)+b\) with \(a>0\) scores the square root of that correlation. Whenever both variance pieces are positive, \(0<\\operatorname{{Corr}}(X_{{p^*}},Y)<1\) and the collapsed judge wins the correlation. The numerical case \(\\operatorname{{Var}}(\\mu)=\\mathbb E[\\sigma^2]\) is 0.50 versus about 0.707. Spearman and Kendall have the same monotone-warp invariance and the same ordering on that draw. The practical answer to a single-shot objection is that a softmax over the rating-token logits is one forward pass: the distribution is the normalized logit vector, not an average of sampled generations. The rollout figure compares that single pass with \(M\\in\\{{1,5,20\\}}\) multinomial samples of the same vector, on a linear head because this machine has no GPU.

**Step 4. The empirical curves, the dollar axis, and the scaling bridge.** Held-out cross-entropy, the square root of RPS, and Brier are plotted against ratings per item \(k\), with a top axis in illustrative dollars at $0.05 per rating (the POPQUORN crowd rate) and $5 per rating (an assumed expert rate, not a measured wage for these sets). Experiment 28, which is a different design (five frontier judges, not this linear head), found CE-fit ahead of Pearson-fit on held-out RPS after 4 to 8 ratings at one rating per item. The shaded band marks that region. It is not a claim that this head reproduces those judges. A proper loss keeps using distributional signal after a correlation objective has the ranking of the means.

## Real-data matrix

The rows are training objectives on the same hashed-text linear family. Cross-entropy, Brier, and RPS train a softmax. Pearson trains the same softmax to raise the correlation of a sampled rating (law of total covariance: conditional variance is punished). MSE is ridge regression on the conditional mean, read out as the two-point distribution on adjacent scores with that mean. Test risk for a proper loss is the mean over test prompts of the mean over that prompt's ratings. It is not the plug-in divergence of the test histogram.

{matrix}

## What the held-out table shows

Cross-entropy, Brier, and RPS heads land on nearly the same held-out proper scores. On SummEval expert grades 1–5 the CE head's held-out cross-entropy is about 1.21, against about 4.73 for the Pearson head and infinity for the MSE two-point readout. Infinity is the ordinal gap: that readout puts probability 0 on every score except the two adjacent to the ridge mean, so one off-support rating makes -log q(y) infinite. On the binary views, where the mean is the distribution, the MSE readout is finite and competitive with the CE head, while the Pearson head is worse on cross-entropy and not better on correlation. Hash features are weak. The shuffle control sits next to the train-marginal forecast, and on ChaosNLI abductive NLI every head sits on the uniform forecast (log 2 ≈ 0.693). The Pearson-versus-CE gap on cross-entropy is the collapse of the Pearson objective. It is visible at every k in the curves, including the 4–8 rating band where experiment 28's frontier CE-fit first beat Pearson-fit. There is no small-k region in this linear family where the Pearson head wins the proper scores. {oracle_note}

## Finite-sample curves and iso-budget

{curves}

### Iso-budget on ChaosNLI SNLI+MNLI

{iso}

## Single pass versus Monte Carlo rollouts

Dataset `{payload['rollout']['dataset']}`. The Monte Carlo histograms use a \(10^{{-6}}\) floor so a single unlucky sample does not make cross-entropy infinite. The single pass does not need that floor.

{roll_lines}

{COT}

## Within-prompt oracle (not a held-out judge)

Dataset `{oracle.get('dataset')}`. Half the raters of a test prompt estimate the distribution; the other half are scored. Sampled-judge correlation {_f(oracle.get('corr_sampled_halfA'), 3)}. Collapsed affine correlation {_f(oracle.get('corr_collapsed_affine'), 3)}. Distributional half CE {_f(oracle.get('mean_ce_distributional'))}. Collapsed half CE {_f(oracle.get('mean_ce_collapsed'))}.

## Figures

![Wrong-objective matrix](figures/wrong_objective_matrix.png)

![Finite-sample crossover, binary versus ordinal Likert](figures/finite_sample_crossover_binary_vs_ordinal.png)

![High-k crossover, DICES-990 and ChaosNLI](figures/high_k_crossover_dices_chaos.png)

![Single pass versus rollouts](figures/single_pass_vs_rollout.png)

![Iso-budget](figures/iso_budget_chaosnli.png)

![Test-set power on DICES-990](figures/eval_power_mdd_dices990.png)

## Ready-to-send note to Sanmi

Sanmi — three mathematical facts, then the empirical picture.

On binary ratings you are right that the mean is the distribution and that scalar squared error matches Brier. Brier is strictly proper. Correlation is not. Corr(aX+b, Y) = Corr(X, Y) for any a>0, so a compressed forecast ties the true probabilities on correlation while Brier and cross-entropy get worse (in the four-item witness, correlation stays 1 while mean Brier moves from {_f(payload['proofs_numeric']['binary_affine_witness']['brier_star'])} to {_f(payload['proofs_numeric']['binary_affine_witness']['brier_bad'])} and cross-entropy from {_f(payload['proofs_numeric']['binary_affine_witness']['ce_star'])} to {_f(payload['proofs_numeric']['binary_affine_witness']['ce_bad'])} nats).

On a 0–5 rubric the mean is not the law. A point mass at 2 and a 50/50 split on {{0, 4}} have the same mean and different Brier, RPS, and cross-entropy. RPS is the squared CDF gap: it charges distance from the realized rating, over the whole distribution, not a ranking of items.

If the judge samples independently of the human given the prompt, correlation is strictly anti-proper. The true distribution scores Corr = Var(mean)/Var(Y). A collapsed judge a·mean+b scores the square root. When the two variance pieces are equal, that is 0.50 versus about 0.707. Spearman and Kendall do the same thing under a monotone warp. One forward pass already returns the distribution: softmax of the rating-token logits. Samples are a Monte Carlo estimate of that vector, not a requirement.

The curves plot held-out cross-entropy, sqrt(RPS), and Brier against ratings per item, with a top axis at $0.05/rating (POPQUORN's crowd rate) and $5/rating (an assumed expert rate). Experiment 28's frontier-judge break-even sat near 4–8 ratings; the band on the figure marks that earlier result, which is a different model family. Continuous densities are the case where your nonparametric sample-size warning applies. Finite rubrics are ordinary K-way classification, with an O(K/N) multinomial rate. Tightening the rubric and matching p* are complementary. The failure mode is treating a spread-out target as if it had variance zero.

Happy to walk through the Lean file (no sorry; axioms are propext, Classical.choice, Quot.sound) or the held-out tables.

## Two-paper roadmap

**Paper 1 — Target and evaluation protocol.** Define the target as \(p^*(y\\mid c)\), not a correlation with a scalar. State the output-space taxonomy (binary, finite ordinal, continuous). Fix the test estimator as the unbiased per-sample proper risk, the split as prompt-level, and the interval as an item-clustered bootstrap. Keep Pearson, Spearman, and Kendall as diagnostics. Put Krippendorff-style agreement in the human-consensus section, not in the judge-fidelity section. Lead with RPS as distance-from-the-realized-rating.

**Paper 2 — Training, scaling, and dollars.** Train the same family by cross-entropy, Brier, RPS, Pearson, and MSE. Report the anti-proper collapse. Show the single-pass logit readout against M-sample rollouts, and keep a chain-of-thought sample distinct from the marginal over rationales. Plot held-out proper scores against \(k\) and against dollars, crowd and expert, and the iso-budget cut \(B=N\\times k\). A second planned axis is model size \(P\), recorded in `scripts/snap_lora_train.py` and not measured in this folder. The scaling claim is that correlation saturates once the conditional mean is ranked, while a proper loss keeps fitting the conditional distribution.

{ELICITATION}

{G_THEORY}

{TRANSCRIPT}

## What this machine did not run

- POPQUORN: {avail['popquorn']}
- VeriBench FSC: {avail['veribench_fsc']}
- No GPU is present, so no LoRA fine-tune and no live LLM logit extraction were run. Ready scripts: `scripts/snap_lora_train.py` and `scripts/snap_llm_logit_readout.py`.

## Kohli et al. 2026

Guneet Kohli, arXiv:2605.29797, studies how many annotators a soft-label NLI model needs. On ChaosNLI (SNLI+MNLI, 3,113 items, 100 annotations), KL divergence to the full human distribution captures about 91% (the paper's table: 91±3%) of the hard-label-to-100-annotator improvement by 10 annotators, while entropy correlation needs on the order of 20–50. On DICES-990 the paper's exploratory table gives 90% of the KL improvement at 3 annotators and 62% of the entropy-correlation improvement. That is a saturation result about annotation count for one training loss. This folder asks a different question: which training objective, on a shared linear head, wins held-out proper scores. The two results sit next to each other. They are not the same measurement, and this folder does not re-run those DeBERTa fine-tunes.
"""
    (folder / "SANMI_PITCH.md").write_text(pitch)

    summary = f"""# Results summary — Sanmi distributional pitch (11-26-2026)

**As of 11-26-2026.**

**Bottom line.** Lean 4.36.0 checks the algebraic claims with no `sorry`. Python checks the same claims numerically, including 0.50 versus \(\\sqrt{{1/2}}\) for the anti-proper correlation. The human-data tables below are held-out test prompts from SummEval, ChaosNLI, and DICES. POPQUORN and VeriBench functional semantic correctness were not available here and are not imputed.

## Formally proved (Lean) and numerically checked

{proof}

Theorem names and the axiom printout are in `lean_check.txt`. The numerical assertions are `proofs_numeric.py` (every check returned ok).

Fair-coin couplings: \(\\mathbb E[(X-Y)^2]\) is 0, 1/2, and 1 while \(D_B=0\). Plug-in \(D_B(\\hat p_N\\Vert q)\) sits above \(D_B(p^*\\Vert q)\) by \((1/N)(1-\\|p^*\\|_2^2)\) in expectation. The test tables do not use that plug-in.

## What the held-out numbers say

The CE, Brier, and RPS heads agree with each other. The Pearson head's held-out cross-entropy is much larger on every ordinal and 3-class set (SummEval expert about 4.73 versus about 1.21; DICES-990 about 1.16 versus about 0.74; ChaosNLI SNLI+MNLI about 4.02 versus about 1.05). Its correlation is not higher than the CE head's. The MSE two-point readout has infinite cross-entropy on those sets because it assigns probability 0 off the two adjacent scores. On binary views the same readout is finite and competitive, which is the K=2 case where the mean is the law. ChaosNLI abductive NLI is a negative control: every head matches the uniform forecast, so the text hash does not predict that task. {oracle_note}

## Real multi-annotator results (this machine)

Proper-score cells are the unbiased prompt-average of per-rating losses. Pearson r is the correlation of the judge's scalar (expected score, or the ridge mean) with individual ratings, item-clustered bootstrap. Spearman and Kendall are correlations of the prompt-level human mean with that scalar, same bootstrap. Lower is better for cross-entropy, sqrt(RPS), Brier, and normalized RPS. Higher is better for the three correlations.

{matrix}

### Crossover

{curves}

### Iso-budget

{iso}

### Single pass versus rollouts

{roll_lines}

### Within-prompt oracle

{oracle}

## Synthetic ground truth, labeled as such

The identities in the bottom line are not estimated from human ratings. The Monte Carlo anti-proper check draws from a known two-context law. It is the verification of Proof C, not a human result.

## Positioning against Kohli et al. 2026, arXiv:2605.29797

Kohli reports metric-dependent annotation saturation for soft-label NLI training: on ChaosNLI, KL is largely in place by about 10 annotators (paper table: 91±3% of the hard-to-soft-100 KL improvement) while entropy correlation keeps moving through roughly 20–50 annotators; the DICES-990 exploratory table is 90% versus 62% at 3 annotators for those two metrics. This experiment does not repeat that fine-tune. It compares objectives on a shared head and scores held-out proper risk. Experiment 28 remains the frontier-judge break-even (CE-fit ahead of Pearson-fit on held-out RPS after 4–8 ratings at one per item, or 10–20 at five per item, across its 25 dataset-by-judge cells). The shaded band on the new curves cites that result. The heads here are hashed-text linear heads, not those frontier judges.

{POWER}

## GPU scripts not run

`scripts/snap_lora_train.py` and `scripts/snap_llm_logit_readout.py` are ready for SNAP or Sherlock. This VM has no GPU (`nvidia-smi` absent). No LoRA number in this folder comes from a run.

{ELICITATION}

{G_THEORY}

{TRANSCRIPT}
"""
    (folder / "RESULTS_SUMMARY.md").write_text(summary)

    readme = f"""# Experiment 36: Sanmi distributional pitch, anti-proper correlation

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/36_sanmi_distributional_pitch_anti_proper_corr_start_26Nov2026/README.md>

**TLDR:** Pearson correlation is affine-invariant and, for a judge sampled independently of the human given the prompt, strictly anti-proper relative to the true conditional distribution. Brier and ranked probability score (RPS) are strictly proper. The human-data half is a linear head on SummEval, ChaosNLI, and DICES. Read `HOW_TO_READ_RESULTS.md` before quoting a number.

**Status:** run complete on this machine (11-26-2026) for every dataset that could be downloaded. No GPU. Cost $0 of new model calls. Raw downloads are cached under `data_cache/` and are not committed.

## Reproduce

From the repository root, with Python 3, NumPy, SciPy, and Matplotlib:

```bash
python3 experiments/36_sanmi_distributional_pitch_anti_proper_corr_start_26Nov2026/run_all.py
python3 -m pytest experiments/36_sanmi_distributional_pitch_anti_proper_corr_start_26Nov2026/tests -q
```

Lean 4.36.0 (`lean-toolchain`), no Mathlib:

```bash
cd experiments/36_sanmi_distributional_pitch_anti_proper_corr_start_26Nov2026
elan run leanprover/lean4:v4.36.0 lean SanmiPitch.lean
```

Exit code 0, no `sorry`. The captured compiler output is `lean_check.txt`.

## Files

| Path | Role |
| --- | --- |
| `SANMI_PITCH.md` | Four-step pitch, note to Sanmi, two-paper roadmap, transcript coverage |
| `RESULTS_SUMMARY.md` | Proved claims, held-out tables, Kohli positioning, transcript coverage |
| `HOW_TO_READ_RESULTS.md` | Data, metric, difference, interval, and how the head was chosen |
| `results.md` | Live ledger |
| `results.json`, `results.txt` | Exact metrics |
| `PROTOCOL.md` | Measurement contract, written as the pre-registered choices |
| `SanmiPitch.lean`, `lean_check.txt` | Formal proofs and the compiler receipt |
| `proofs_numeric.py` | Numerical twins of the Lean claims |
| `metrics.py`, `data_io.py`, `fit_eval.py`, `run_all.py` | Estimators, loaders, heads, driver |
| `figures/` | Matrix, crossover, high-k, rollout, iso-budget, DICES-990 test-set power |
| `tables/` | Markdown tables copied from the run |
| `scripts/snap_lora_train.py`, `scripts/snap_llm_logit_readout.py` | Not run here |
| `tests/test_hygiene.py` | Unbiased risk, prompt split, metric identities |

## Data that was scored

{avail_lines(payload)}

## Verified numerical snapshot

{proof}
"""
    (folder / "README.md").write_text(readme)
    (folder / "HOW_TO_READ_RESULTS.md").write_text(how_to_read())
    (folder / "results.md").write_text(results_md(payload))
    (folder / "PLAN_STATUS.md").write_text(plan_md(payload))
    tables = folder / "tables"
    tables.mkdir(exist_ok=True)
    (tables / "objective_metric_matrix.md").write_text(matrix)
    (tables / "crossover.md").write_text(curves)
    (tables / "iso_budget.md").write_text(iso)
    (tables / "objective_metric_matrix.tex").write_text(_tex(payload))


def avail_lines(payload):
    return "\n".join(f"- **{k}.** {v}" for k, v in payload["availability"].items())


def how_to_read():
    return """# How to read these results

**As of 11-26-2026.**

1. **Data.** Human numbers are held-out test prompts. The split is by prompt group (SummEval: source article; ChaosNLI and DICES: the item id), fractions 0.6/0.2/0.2, seed 0. Raters of one prompt are never placed on two sides of the split. A within-prompt oracle is labeled as such and is not a held-out judge. Synthetic checks say they use a known law. SummEval has no stable person id in the jsonl; the rater field there is a within-item vote index. ChaosNLI releases counts; vote indices are the expansion of those counts, not people. DICES `rater_id` is the released person id, after the README quality filter. Experiment 28's 4–8 rating break-even is an earlier frontier-judge result, cited as such.

2. **Metrics.** Cross-entropy is \(-\log \hat q(y)\) in nats, lower better, range \([0,\infty)\), reference \(\log K\) for a uniform forecast. One-hot Brier is \(\|\hat q-e_y\|_2^2\), lower better, range \([0,2]\). Ranked probability score (RPS) is \(\sum_{t=0}^{K-2}(F(t)-1_{y\le t})^2\), lower better, range \([0,K-1]\). The matrix column sqrt(RPS) is the square root of the unbiased mean of that score. Normalized RPS divides by \(K-1\) and is the project's primary proper score; it is in the tables next to sqrt(RPS). Pearson r, Spearman rho, and Kendall tau are higher-better association diagnostics. Spearman and Kendall in the matrix use one human mean per prompt.

3. **The test estimator.** For a proper loss the reported risk is the mean over test prompts of the mean over that prompt's ratings. A plug-in \(D_B(\hat p_{test}\Vert\hat q)\) is biased high by \((1/k_c)(1-\|p^*(c)\|_2^2)\) in expectation when \(\hat p\) is the empirical frequency. That plug-in is not the reported test risk.

4. **Differences.** A comparison is the same metric on the same split. "Held-out test CE of the Pearson-trained head minus held-out test CE of the CE-trained head" is positive when the CE-trained head is better.

5. **Intervals.** Percentile 95% intervals from an item-clustered bootstrap, B=1000 (B=40 only if `SANMI36_SMOKE=1`). The unit is the prompt. The interval is for this fixed trained head, not for retraining.

6. **How the head was chosen.** Signed 256-dimensional token hash, 24 principal components fit on train prompts only, plus an intercept. L2 penalty \(10^{-2}\) on non-intercept weights. Cross-entropy, Brier, RPS, and Pearson: L-BFGS-B from the zero vector, at most 35 iterations. Cross-entropy of a linear softmax is convex in the weights; the others are not, and the result is this start and this budget. MSE is ridge regression (closed form, global for that penalized linear problem) read out as the adjacent two-point distribution with the same mean. Validation did not choose these penalties.

7. **Direction and scale.** Proper scores: lower is better. Correlations: higher is better. A uniform forecast on K classes has cross-entropy \(\log K\).
"""


def results_md(payload):
    return f"""# Results ledger

**Status:** COMPLETE for the datasets listed in `results.json` (11-26-2026).

**Last updated:** 11-26-2026.

| Phase | State |
| --- | --- |
| Numeric proof checks | DONE |
| Lean compile | see `lean_check.txt` |
| Real-data matrix, crossover, iso-budget, rollout | DONE on SummEval, ChaosNLI, DICES |
| POPQUORN, VeriBench FSC, GPU LoRA | NOT RUN — recorded in `SANMI_PITCH.md` |

Runtime seconds: {payload.get('seconds')}.

**TLDR-end:** [judges-corr-vs-loss: sanmi distributional pitch] Correlation is affine-invariant and anti-proper for an independently sampled judge; the held-out human tables are in `RESULTS_SUMMARY.md`.
"""


def plan_md(payload):
    return f"""# Plan and status — 11-26-2026

**Status:** the driver `run_all.py` finished. Smoke flag: {payload.get('smoke')}.

## Contract (fixed before test metrics were read)

Predictions, written down before the run:

- Uniform cross-entropy equals \(\log K\).
- A CE-trained softmax should beat a constant uniform forecast on held-out CE when the hash features carry any signal. If it does not, the objective comparison is still inside a weak-feature family and the shuffle control should match the real CE head.
- Shuffling train features should not improve held-out CE.
- On a known two-context law, the collapsed judge's correlation should sit near 0.707 and the true sampled judge near 0.50.
- On binary data, scalar L2 and Brier match up to the factor 2. Correlation can still tie while Brier and CE worsen.
- On ordinal data with within-item spread, a mean-matched two-point readout should lose proper scores to a softmax that is allowed to use the full simplex.
- The within-prompt oracle should show a higher correlation for the collapsed affine score than for a draw from the half-sample distribution, and a worse proper score, when within-item variance is positive. This oracle is not a test-set judge.

Deviations are whatever `results.json` shows. They are not edited back into this list.

## Not available

POPQUORN and VeriBench FSC. See the availability block in `results.json`.
"""


def _tex(payload):
    lines = [
        "% Auto-extracted from results.json. Fragment, not a standalone paper.",
        "\\begin{tabular}{llrrrr}",
        "\\hline",
        "Dataset & Objective & CE & sqrt RPS & Brier & Pearson r \\\\",
        "\\hline",
    ]
    for name, block in payload["matrices"].items():
        for obj, row in block["rows"].items():
            p = row["point"]
            lines.append(
                f"{name} & {obj} & {_f(p['ce'])} & {_f(p['sqrt_rps'])} & {_f(p['brier'])} & {_f(p['pearson_r'], 3)} \\\\"
            )
    lines.append("\\hline\n\\end{tabular}")
    return "\n".join(lines) + "\n"
