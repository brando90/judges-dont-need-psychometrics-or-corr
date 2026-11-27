"""Static & dynamically formatted sections added after the main run."""

from __future__ import annotations

import json
from pathlib import Path

ELICITATION = r"""## Elicitation hierarchy

Sanmi's question was what accuracy, cross-entropy, and moments each identify. The rows below are ordered by how much of \(p^*(y \mid c)\) the population minimizer is forced to match. A loss is Bayes-consistent for a target functional when every population minimizer returns that functional. Bartlett, Jordan, and McAuliffe (2006) call the 0–1 case classification calibration. Gneiting and Raftery (2007) characterize strictly proper scoring rules: the Bayes act is the full distribution.

| Row | Elicitation | Bayes act | Properness |
| --- | --- | --- | --- |
| 1 | Hard 0–1 accuracy | A point mass on the mode, \(\delta_{\mathrm{mode}}\). Accuracy computed on a sample from the judge is still maximized by always emitting the mode. | Bayes-consistent for the mode only. The rest of the probability mass function is free. |
| 2 | Scalar mean squared error | The conditional mean \(\mu_Y(c)\). | Bayes-consistent for the mean. Silent on spread and on a second mode. |
| 3 | Pearson \(\operatorname{Corr}(X, Y)\) | An affine-uncalibrated mean. For a judge sampled independently of \(Y\) given \(c\), the maximizer collapses to \(X_\delta = a\mu_Y(c)+b\) with \(a>0\), and \(\sqrt{\rho}>\rho\). | Not strictly proper. Not Bayes-consistent for \(p^*\). |
| 4 | Cross-entropy or Brier | The full probability mass function, \(\hat q = p^*\). | Strictly proper and class-symmetric. A wrong class is not charged by how far it sits on the rubric. |
| 5 | Ranked probability score, or the continuous ranked probability score | The full cumulative distribution. The loss grows with the distance of the realized rating. | Strictly proper, and ordinal-distance-sensitive. The continuous ranked probability score is the version for a real-valued \(Y\). |

**Symmetric agreement versus asymmetric target matching.** An inter-rater coefficient treats the two scores as exchangeable. Swapping the raters does not change Pearson correlation, an intraclass correlation, Krippendorff's \(\alpha\), or a generalizability coefficient. A proper score does not swap. It scores a forecast \(\hat q\) against a draw from a target \(p^*(y \mid c)\). That is asymmetric target matching. The psychometric coefficient remains available as a description of the human panel. It is not the loss that trains the judge, and it is not the loss that says whether the judge matched \(p^*\).
"""

COT = r"""## Chain-of-thought logits, and a model-size axis that was not run

The single-pass numbers above are a linear softmax: one map from the prompt features to a distribution on the rating classes. A chain-of-thought judge is a different object. If the model first writes a rationale \(r\) and the rating logits are read at the end of that text, those logits are \(q(y \mid c, r)\), the rating distribution conditional on one sampled trajectory. The marginal the proper scores want is
\[
q(y \mid c) = \sum_r q(r \mid c)\, q(y \mid c, r).
\]
One rationale does not return that sum. Two procedures do. A direct forward pass, with the rating token next and no rationale, reads \(q(y \mid c)\) from the logits of the rating tokens. \(M\) independent chain-of-thought samples, \(M \in \{1, 5, 20\}\) in the rollout above, are a Monte Carlo estimate of the same marginal. The linear-head figure is that comparison without a live language model. `scripts/snap_llm_logit_readout.py` states the same split for a GPU node.

A second scaling axis is model size \(P\), beside the annotation budget \(B = N \times k\). The GPU script records a planned grid, Qwen3-4B and Llama-3.1-8B, and does not run it. No held-out number in this folder is a function of \(P\).
"""


def build_power_section(folder: Path) -> str:
    """Format the test-set power & MDD section directly from results_power_mdd.json."""
    power_path = folder / "results_power_mdd.json"
    if not power_path.is_file():
        return "## Test-set power and minimum detectable difference\n\nRun `eval_power_mdd.py` to populate `results_power_mdd.json`.\n"
    data = json.loads(power_path.read_text())

    def _fmt_summary(block, label):
        s = block["summary"]
        rows = block["rows"]
        n_avail = block["n_test_available"]
        lines = [f"### {label} (`N_test` max = {n_avail})", ""]
        for metric, mlabel in (("ce", "Cross-entropy"), ("brier", "Brier"), ("sqrt_rps", "sqrt(RPS)"), ("pearson_r", "Pearson r")):
            k1_rows = [r for r in rows if r["k_test"] == 1 and r["metric"] == metric]
            max_r = max(k1_rows, key=lambda r: r["n_test"]) if k1_rows else None
            hit = s[metric].get("smallest_n_at_k1_power80")
            if hit:
                desc = f"hits power ≥ 0.80 at `N_test = {hit['n_test']}` (power {hit['power']:.2f}, MDD {hit['mdd']:.4f})"
            elif max_r:
                desc = f"does NOT reach 0.80 at `k_test = 1` (at `N_test = {max_r['n_test']}`, power {max_r['power']:.2f}, MDD {max_r['mdd']:.4f}, median diff {max_r['median_difference']:.4f})"
            else:
                desc = "not evaluated"
            lines.append(f"- **{mlabel} at `k_test = 1`:** {desc}.")
        return "\n".join(lines)

    dices_txt = _fmt_summary(data, "DICES-990")
    extra = data.get("extra_datasets", {})
    extra_parts = []
    if "popquorn_offensiveness" in extra:
        extra_parts.append(_fmt_summary(extra["popquorn_offensiveness"], "POPQUORN Offensiveness"))
    if "popquorn_politeness" in extra:
        extra_parts.append(_fmt_summary(extra["popquorn_politeness"], "POPQUORN Politeness"))
    extra_block = "\n\n".join(extra_parts)

    return f"""## Test-set power and minimum detectable difference

This complements the unbiased test risk at one rating per prompt. The two judges are fixed: the cross-entropy head and the Pearson head, trained once on the train split (seed 0). Only the test design changes. A cell is \\(N_{{\\mathrm{{test}}}}\\) held-out items and \\(k_{{\\mathrm{{test}}}}\\) ratings from each. There are {data['n_outer_subsets']} random subsets, each scored with an {data['interval']} interval from {data['n_inner_bootstrap']} item-clustered resamples. Power is the fraction of subsets whose interval excludes 0; minimum detectable difference (MDD) is the median half-width of those intervals.

For proper scores, the difference is `{data['difference_proper']}`. For Pearson \\(r\\), `{data['difference_correlation']}`.

{dices_txt}

{extra_block}

The proper scores separate these two heads at \\(k_{{\\mathrm{{test}}}} = 1\\) with far fewer test prompts than correlation needs, because the proper-score gaps are large and the correlation gap is small. Figures: `figures/eval_power_mdd_dices990.png` and `figures/eval_power_mdd_popquorn_offensiveness.png`. Full DICES-990 grid: `tables/eval_power_mdd_dices990.md`.

![Test-set power on DICES-990](figures/eval_power_mdd_dices990.png)

![Test-set power on POPQUORN Offensiveness](figures/eval_power_mdd_popquorn_offensiveness.png)
"""
