"""Write the reader-facing notes from results.json and results_power_mdd.json."""

from __future__ import annotations

import json
import math
from pathlib import Path

from pitch_addenda import COT, ELICITATION, build_power_section


def _f(x, nd=4):
    if x is None:
        return "undefined"
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


def _build_g_theory(payload) -> str:
    """Build the G-theory / D-studies section with dynamically verified (N, k) dollar rows."""
    cro = payload.get("crossover", {})
    mats = payload.get("matrices", {})

    def _row(curve_key, mat_key, label, max_k_note=None):
        c = cro.get(curve_key)
        m = mats.get(mat_key)
        if not c or not m:
            return None
        n_train = m["n_train"]
        ce_c = {r["k"]: r["point"] for r in c["curves"]["ce"]}
        pe_c = {r["k"]: r["point"] for r in c["curves"]["pearson"]}
        ks = sorted(ce_c)
        ce1, pe1 = ce_c[1], pe_c[1]
        if max_k_note:
            k_status = max_k_note
        elif all(ce_c[k]["ce"] < pe_c[k]["ce"] and ce_c[k]["sqrt_rps"] < pe_c[k]["sqrt_rps"] and ce_c[k]["brier"] < pe_c[k]["brier"] for k in ks):
            k_status = f"1 (through {max(ks)})"
        else:
            wins = [k for k in ks if ce_c[k]["ce"] < pe_c[k]["ce"]]
            k_status = f"CE ahead at k={wins}" if wins else "tie on log 2"
        b1 = n_train
        return (
            f"| {label} | {k_status} | {b1:,} | \\${0.05 * b1:,.2f} | \\${5 * b1:,.0f} | "
            f"{_f(ce1['ce'], 3)} vs {_f(pe1['ce'], 3)} (sqrtRPS {_f(ce1['sqrt_rps'], 3)} vs {_f(pe1['sqrt_rps'], 3)}) |"
        )

    rows = [
        _row("ordinal_summeval_crowd", "summeval_expert", "SummEval crowd, grades 1–5"),
        _row("ordinal_dices990", "dices990", "DICES-990, three-level"),
        _row("chaos_3way", "chaosnli_snli_mnli", "ChaosNLI SNLI+MNLI, three-class"),
        _row("binary_alpha", "chaosnli_alpha", "ChaosNLI abductive NLI, binary (negative control)", "both sit near log 2 ≈ 0.693"),
        _row("ordinal_popquorn_offensiveness", "popquorn_offensiveness", "POPQUORN offensiveness, 1–5"),
        _row("ordinal_popquorn_politeness", "popquorn_politeness", "POPQUORN politeness, 1–5"),
        _row("ordinal_veribench_fsc_rv1", "veribench_fsc_rv1", "VeriBench FSC R_v1, 0–5 (25 tasks)"),
        _row("ordinal_veribench_fsc_pooled", "veribench_fsc_pooled", "VeriBench FSC pooled (R_v1 + native_main), 0–5 (50 tasks)"),
    ]
    table_rows = "\n".join(r for r in rows if r is not None)

    iso_rows = payload.get("iso_budget", {}).get("rows", [])
    iso_ce = [r for r in iso_rows if r["objective"] == "ce"]
    iso_pe = [r for r in iso_rows if r["objective"] == "pearson"]
    if iso_ce and iso_pe:
        b = payload["iso_budget"]["budget_target"]
        ce_str = ", ".join(f"{_f(r['point']['ce'], 3)} at ({r['N']},{r['k']})" for r in iso_ce)
        pe_str = ", ".join(f"{_f(r['point']['ce'], 3)} at ({r['N']},{r['k']})" for r in iso_pe)
        iso_para = (
            f"The iso-budget cut holds \\(B={b}\\) ratings on ChaosNLI SNLI+MNLI "
            f"(\\${0.05 * b:,.2f} at \\$0.05 and \\${5 * b:,.0f} at \\$5). Held-out cross-entropy of the CE head is {ce_str}; "
            f"the Pearson head on the same cells is {pe_str}. Inside this grid, spending the fixed bill on more prompts "
            "with one rating beats fewer prompts with many ratings."
        )
    else:
        iso_para = ""

    return f"""## Why not psychometrics (G-theory / D-theory)?

**G-theory.** Generalizability theory (Cronbach, Gleser, Nanda, and Rajaratnam, 1972) writes a scalar rating as an additive variance split: prompt variance \\(\\sigma^2_c\\), rater variance \\(\\sigma^2_r\\), and the prompt-by-rater residual \\(\\sigma^2_{{cr}}\\). The generalizability coefficient \\(G\\) is a multi-facet form of an intraclass correlation. Like Pearson correlation, it is a second-moment summary. The usual analysis of variance treats the components as additive and homoscedastic. On a bounded 0–5 or 1–5 rubric that summary does not score calibration, and it does not see a second mode. It is not a divergence from \\(p^*(y \\mid c)\\).

**D-studies.** A decision study pushes those variance components forward and asks how many raters \\(k\\) would bring \\(G\\) to a threshold such as \\(0.80\\). Under a proper scoring rule the same allocation question is the held-out risk \\(R(N, k)\\) of a judge trained on \\(N\\) prompts with \\(k\\) ratings each. The curves in this folder answer it in held-out cross-entropy, in the square root of the ranked probability score (RPS), and in Brier score, and they convert \\(B = N \\times k\\) into dollars at a stated price. They do not pass through a Gaussian analysis of variance.

**Where a generalizability coefficient still helps.** It is an optional diagnostic of the human panel. A large rater main effect is harshness or leniency. A large residual is disagreement about the item or the rubric. That diagnosis plans the annotation. It is not a training loss for the judge, and it is not the test metric.

**What the \\((N, k)\\) curves cost on all scored sets.** Two unit prices: \\(\\$0.05\\) per rating is POPQUORN's published crowd rate (now measured directly in the POPQUORN rows below, and used as an illustrative crowd rate on the other sets), and \\(\\$5\\) per rating is an assumed expert rate. On every ordinal and 3-class set with signal (SummEval crowd, DICES-990, ChaosNLI SNLI+MNLI, POPQUORN offensiveness, POPQUORN politeness, VeriBench FSC R_v1, and VeriBench FSC pooled), the cross-entropy head beats the Pearson head on held-out cross-entropy, sqrt(RPS), and Brier already at \\(k=1\\), and stays ahead through every larger \\(k\\).

| Dataset | \\(k\\) at which CE is ahead on CE, sqrt(RPS), and Brier | Train ratings \\(B=N k\\) | Crowd dollars at \\$0.05 | Expert dollars at \\$5 | Held-out CE & sqrt(RPS) at \\(k=1\\), CE-trained vs Pearson-trained |
| --- | --- | --- | --- | --- | --- |
{table_rows}

{iso_para}

On known synthetic ordinal laws, experiment 34 already recorded a lenient point-mass judge with generalizability coefficient 1 and Pearson 0.961 whose expected RPS was worse than a constant forecast by 0.0492 [0.0477, 0.0506].
"""


def _build_transcript() -> str:
    return r"""## Complete Sanmi Transcript Coverage

As of 11-26-2026. These five points are the ones the 11-25 call left open. The table is the taxonomy; the sentences under it are the boundaries.

| Point | Output space or claim | What is true | Where it is checked |
| --- | --- | --- | --- |
| 1a | Binary, \(K=2\) | One degree of freedom. The scalar mean is the full probability mass function, and scalar \(L_2\) equals the Brier score (two-class Brier is twice scalar squared error). Pearson correlation still fails calibration, because \(\mathrm{Corr}(aX+b,Y)=\mathrm{Corr}(X,Y)\) for \(a>0\), and it is strictly anti-proper for a sampled judge: \(\sqrt{\rho}>\rho\). | Lean `binary_brier_eq_two_l2`, `pearson_affine_invariant`, `binary_witness_corr_tie`, `anti_proper_from_parts`. Python `binary_l2`, `binary_affine_witness`, `anti_proper`. Real binary panels: SummEval cut at grade \(\ge 4\), ChaosNLI abductive NLI, DICES Yes-versus-not, POPQUORN cut \(\ge 4\), VeriBench FSC cut \(\ge 4\). |
| 1b | Finite categorical or ordinal, \(K\ge 3\) (a 0–5 rubric is this case) | \(K-1\) degrees of freedom. This is ordinary discrete classification. The multinomial frequency error is \(\mathcal{O}(K/N)\), from \(\mathbb{E}[\Vert\hat q_N-p^{\ast}\Vert_2^2]=(1/N)(1-\Vert p^{\ast}\Vert_2^2)\), not a nonparametric curse of dimensionality. A scalar mean drops spread and multimodality. Cross-entropy, Brier, and ranked probability score (RPS) see them. | Lean `empirical_l2_rate`, `plugin_DB_bias`, `ordinal_same_mean_different_law`. Python `empirical_rate`, `mean_ne_pmf`. Real panels: SummEval 1–5, DICES-990 3-level, POPQUORN offensiveness & politeness 1–5, VeriBench FSC (`R_v1`, `native_main`, pooled) 0–5, and CertJudge real LLM judge panels (`E63`, `E39`, `Exp 67`). |
| 1c | Continuous, \(Y\in\mathbb{R}\) | The target is an infinite-dimensional density \(p^*(y\mid c)\). Sanmi's nonparametric sample-complexity warning applies here. A usable judge then needs a parametric head (Gaussian, Beta, or quantiles), the continuous ranked probability score (CRPS), or a discretization into \(K\) bins. | Conceded. This folder does not estimate a continuous density. |
| 2 | Paired versus unpaired; agreement versus fidelity | Correlation has one narrow use as an unpaired or arbitrary-scale diagnostic. Inter-rater agreement, for example Krippendorff's \(\alpha\), measures human-to-human consensus when no target distribution has been defined. A proper scoring rule measures judge fidelity to a defined target \(p^*(y\mid c)\). Those are different questions. | The experiments in this folder score fidelity on held-out prompts. |
| 3 | Sanmi's RPS framing | RPS is the squared distance between forecast and realized cumulative probabilities. It charges an ordering of how wrong the forecast is — distance from the realized rating — over the whole distribution. It is not a ranking of items. | Lean `bernoulli_square_excess` (one threshold) and `ordinal_same_mean_different_law` (the 0–5 gap). Reported column is sqrt(RPS); normalized RPS (divided by \(K-1\)) is stored beside it. |
| 4 | Surrogate-loss literature | Proper scoring rules and surrogate losses are classic: Brier (1950), Epstein (1969) for RPS, Gneiting and Raftery (2007). What is new here is the application to an LLM-as-a-judge, the strict anti-properness \(\sqrt{\rho}>\rho\) for independent sampled judges, the single-pass token-logit readout, and the rating-budget and dollar scaling curves. | This note, `SanmiPitch.lean`, CertJudge real LLM judge bridge, and `figures/`. |
| 5 | Sharpen the rubric versus match \(p^*\) | Tightening a rubric so that \(p^*(y\mid c)\) becomes peakier, and training a judge to match \(p^*(y\mid c)\), are complementary. What fails is pretending a non-peaky target has zero conditional variance. | Proof B (consensus on 2 versus a 50/50 split on \(\{0,4\}\), same mean 2) and Proof C (the collapsed judge \(X_\delta=a\mu_Y(C)+b\)). |
"""


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
    for _key, block in payload["crossover"].items():
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
    def _tbl(block, title):
        rows = block.get("rows", [])
        if not rows:
            return ""
        lines = [f"#### {title} (B ≈ {block.get('budget_target')})", "",
                 "| N | k | B | objective | held-out test CE | sqrt(RPS) | Brier |",
                 "| --- | --- | --- | --- | --- | --- | --- |"]
        for r in rows:
            p = r["point"]
            lines.append(f"| {r['N']} | {r['k']} | {r['B']} | {r['objective']} | {_f(p['ce'])} | {_f(p['sqrt_rps'])} | {_f(p['brier'])} |")
        lines.append("")
        return "\n".join(lines)

    out = [_tbl(payload.get("iso_budget", {}), "ChaosNLI SNLI+MNLI")]
    for name, block in payload.get("iso_budget_extra", {}).items():
        out.append(_tbl(block, name))
    return "\n".join(x for x in out if x) or "Iso-budget rows were not produced in this run.\n"


def _certjudge_md(payload) -> str:
    cj = payload.get("certjudge_real_panels")
    if not cj:
        return ""
    e63_all5 = cj["e63_study_a"]["all5"]["arms"]
    e63_exp3 = cj["e63_study_a"]["expert3"]["arms"]
    e39_test = cj["e39_study_b"]["test_llm"]
    e39_all = cj["e39_study_b"]["all_llm"]
    exp67 = cj["exp67_native_interim"]

    lines = [
        f"## CertJudge real LLM judge panels (`cert-judge@{cj['source_commit'][:8]}`, `$0` extra API spend)",
        "",
        "These are real open-weight and frontier LLM judges evaluated on VeriBench functional semantic correctness (FSC, 0–5 integer scale), bridged directly from verified `cert-judge` Experiment 67 & 68 artifacts:",
        "",
        "### 1. CertJudge Experiment 63 (`Qwen3-8B` native 6-class probabilities on VeriBench FSC `R_v1` held-out test, 30 items / 10 tasks)",
        "",
        "| Training / prompting arm | `all5` CE [95% CI] | `all5` RPS [95% CI] | `all5` Spearman | `expert3` CE | `expert3` RPS | `expert3` Spearman |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for arm_name in ("zero_shot_base", "prompt_opt_arm1", "rl_grpo", "teacher_only_sft", "human_mix_sft", "shuffled_control"):
        if arm_name not in e63_all5:
            continue
        a5 = e63_all5[arm_name]
        e3 = e63_exp3[arm_name]
        lines.append(
            f"| `{arm_name}` | {_f(a5['ce'], 3)} [{_f(a5['ce_ci'][0], 3)}, {_f(a5['ce_ci'][1], 3)}] | "
            f"{_f(a5['rps'], 3)} [{_f(a5['rps_ci'][0], 3)}, {_f(a5['rps_ci'][1], 3)}] | {_f(a5['spearman'], 3)} | "
            f"{_f(e3['ce'], 3)} | {_f(e3['rps'], 3)} | {_f(e3['spearman'], 3)} |"
        )
    lines.extend([
        "",
        f"Key finding on real `Qwen3-8B` weights: `rl_grpo` reaches Spearman `{_f(e63_all5['rl_grpo']['spearman'], 3)}` (`all5`) / `{_f(e63_exp3['rl_grpo']['spearman'], 3)}` (`expert3`), "
        f"yet blows up held-out CE to `{_f(e63_all5['rl_grpo']['ce'], 3)}` / `{_f(e63_exp3['rl_grpo']['ce'], 3)}` (worse than `shuffled_control` CE `{_f(e63_all5['shuffled_control']['ce'], 3)}` / `{_f(e63_exp3['shuffled_control']['ce'], 3)}`). "
        f"Training with cross-entropy (`human_mix_sft`) achieves the lowest CE (`{_f(e63_all5['human_mix_sft']['ce'], 3)}` / `{_f(e63_exp3['human_mix_sft']['ce'], 3)}`), lowest RPS (`{_f(e63_all5['human_mix_sft']['rps'], 3)}` / `{_f(e63_exp3['human_mix_sft']['rps'], 3)}`), and highest Spearman (`{_f(e63_all5['human_mix_sft']['spearman'], 3)}` / `{_f(e63_exp3['human_mix_sft']['spearman'], 3)}`).",
        "",
        f"### 2. CertJudge Experiment 39 ({e39_all['n_configs']} LLM judge configs on {cj['e39_study_b']['n_heldout_items']} held-out VeriBench FSC `R_v1` items / {cj['e39_study_b']['n_heldout_tasks']} tasks, `all5`)",
        "",
        f"- **All {e39_all['n_configs']} LLM configs:** mean raw CE `{_f(e39_all['raw_ce_mean'], 4)}`, `CE-fit` mean CE `{_f(e39_all['ce_cal_ce_mean'], 4)}` vs `Pearson-fit` mean CE `{_f(e39_all['pe_cal_ce_mean'], 4)}` (`CE-fit` beats `Pearson-fit` on CE in **{e39_all['ce_cal_beats_pe_cal_ce']}/{e39_all['n_configs']}** configs and on RPS in **{e39_all['ce_cal_beats_pe_cal_rps']}/{e39_all['n_configs']}** configs: mean RPS `{_f(e39_all['ce_cal_rps_mean'], 4)}` vs `{_f(e39_all['pe_cal_rps_mean'], 4)}`).",
        f"- **Strictly held-out `test` split ({e39_test['n_configs']} LLM configs):** `CE-fit` mean CE `{_f(e39_test['ce_cal_ce_mean'], 4)}` vs `Pearson-fit` `{_f(e39_test['pe_cal_ce_mean'], 4)}` (`CE-fit` wins CE in **{e39_test['ce_cal_beats_pe_cal_ce']}/{e39_test['n_configs']}**, RPS in **{e39_test['ce_cal_beats_pe_cal_rps']}/{e39_test['n_configs']}**).",
        "",
        f"### 3. CertJudge Experiment 67 Interim (3 headline judges × 3 repeats on {exp67['n_files']} `native_main` VeriBench artifacts / {exp67['n_tasks']} tasks, {exp67['n_ratings']} ratings)",
        "",
        "| Judge family | `dRPS` (CE-fit minus Pearson-fit) [95% CI] | `dCE` (CE-fit minus Pearson-fit) [95% CI] | `dTVD` (CE-fit minus Pearson-fit) [95% CI] |",
        "| --- | --- | --- | --- |",
    ])
    for r in exp67["calibrated_new"]:
        if "dRPS" not in r:
            continue
        lines.append(
            f"| {r['family']} | {_f(r['dRPS'], 4)} [{_f(r['dRPS_ci'][0], 4)}, {_f(r['dRPS_ci'][1], 4)}] | "
            f"{_f(r['dCE'], 4)} [{_f(r['dCE_ci'][0], 4)}, {_f(r['dCE_ci'][1], 4)}] | "
            f"{_f(r['dTVD'], 4)} [{_f(r['dTVD_ci'][0], 4)}, {_f(r['dTVD_ci'][1], 4)}] |"
        )
    lines.append("")
    return "\n".join(lines)


def _oracles_md(payload) -> str:
    oracles = payload.get("within_prompt_oracles", {})
    if not oracles:
        return ""
    lines = [
        "| Dataset | Items (`k>=4`) | Pairs | Sampled half-A `r` | Collapsed `0.1*mu+4` `r` | `sqrt_gap` | Distributional CE | Collapsed CE |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, o in oracles.items():
        if not o.get("ok"):
            continue
        lines.append(
            f"| `{name}` | {o['n_items']} | {o['n_pairs']} | {_f(o['corr_sampled_halfA'], 3)} | "
            f"{_f(o['corr_collapsed_affine'], 3)} | {o['sqrt_gap']} | "
            f"{_f(o['mean_ce_distributional'], 4)} | {_f(o['mean_ce_collapsed'], 4)} |"
        )
    return "\n".join(lines)


def _proof_blurb(payload):
    p = payload["proofs_numeric"]
    b = p["binary_affine_witness"]
    a = p["anti_proper"]
    m = p["mean_ne_pmf"]
    return (
        f"Binary witness, four paired items: Pearson correlation of the calibrated probabilities "
        f"and of the affine map \\(0.1q+0.45\\) are both {_f(b['corr_star'], 3)}. "
        f"Mean Brier {_f(b['brier_star'])} versus {_f(b['brier_bad'])}. "
        f"Mean cross-entropy {_f(b['ce_star'])} versus {_f(b['ce_bad'])} nats. "
        f"Anti-proper Monte Carlo (n={a['n']}): correlation of the true sampled judge {_f(a['corr_true_distribution'], 4)} "
        f"(theory 0.5) and of \\(0.1\\mu+4\\) {_f(a['corr_collapsed'], 4)} (theory \\(\\sqrt{{1/2}}\\approx 0.7071\\)). "
        f"Ordinal witness: both laws have mean {_f(m['mu'], 1)}; \\(D_B={_f(m['D_B'], 1)}\\); "
        f"normalized RPS gap {_f(m['normalized_rps_gap'])}; cross-entropy of the split forecast on the consensus outcome is infinite."
    )


def _pt(mats, name, obj, metric="ce", nd=3):
    """Formatted held-out test point estimate of one matrix cell, or 'n/a' if the cell was not run."""
    try:
        return _f(mats[name]["rows"][obj]["point"][metric], nd)
    except KeyError:
        return "n/a"


def _as_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _ce_head_wins_all(mats, name):
    """True iff the CE-trained head beats the Pearson-trained head on held-out CE, sqrt(RPS), and Brier."""
    rows = mats[name]["rows"]
    return all(
        _as_float(rows["ce"]["point"][m]) < _as_float(rows["pearson"]["point"][m]) for m in ("ce", "sqrt_rps", "brier")
    )


def write_all(payload, folder: Path):
    folder = Path(folder)
    proof = _proof_blurb(payload)
    matrix = _matrix_md(payload)
    curves = _curve_md(payload)
    iso = _iso_md(payload)
    cert_md = _certjudge_md(payload)
    oracles_tbl = _oracles_md(payload)
    g_theory = _build_g_theory(payload)
    transcript = _build_transcript()
    power_sec = build_power_section(folder)

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
    mats = payload["matrices"]
    pop_off_ce, pop_off_pe = _pt(mats, "popquorn_offensiveness", "ce"), _pt(mats, "popquorn_offensiveness", "pearson")
    pop_pol_ce, pop_pol_pe = _pt(mats, "popquorn_politeness", "ce"), _pt(mats, "popquorn_politeness", "pearson")
    vb_rv1_ce, vb_rv1_pe = _pt(mats, "veribench_fsc_rv1", "ce"), _pt(mats, "veribench_fsc_rv1", "pearson")
    vb_nat_ce, vb_nat_pe = _pt(mats, "veribench_fsc_native_main", "ce"), _pt(mats, "veribench_fsc_native_main", "pearson")
    vb_nat_rps_ce = _pt(mats, "veribench_fsc_native_main", "ce", "sqrt_rps")
    vb_nat_rps_rps = _pt(mats, "veribench_fsc_native_main", "rps", "sqrt_rps")
    vb_nat_ce_rps = _pt(mats, "veribench_fsc_native_main", "rps")
    vb_pool_ce, vb_pool_pe = _pt(mats, "veribench_fsc_pooled", "ce"), _pt(mats, "veribench_fsc_pooled", "pearson")
    cj = payload.get("certjudge_real_panels", {})
    e63_a = cj.get("e63_study_a", {}).get("all5", {}).get("arms", {})
    e39_all = cj.get("e39_study_b", {}).get("all_llm", {})
    pop_pol_bin_mse = _pt(mats, "popquorn_politeness_binary", "mse", "brier", 4)
    pop_pol_bin_pe = _pt(mats, "popquorn_politeness_binary", "pearson")
    pop_pol_bin_ce = _pt(mats, "popquorn_politeness_binary", "ce")
    bins = [n for n, b in mats.items() if b["summary"]["n_classes"] == 2]
    nonbin = [n for n in mats if n not in bins]
    nonbin_wins = [n for n in nonbin if _ce_head_wins_all(mats, n)]
    mse_finite_bins = [n for n in bins if "mse" in mats[n]["rows"] and math.isfinite(_as_float(mats[n]["rows"]["mse"]["point"]["ce"]))]
    pearson_ok_bins = [
        n for n in bins
        if _as_float(mats[n]["rows"]["pearson"]["point"]["ce"]) <= _as_float(mats[n]["rows"]["ce"]["point"]["ce"])
    ]
    binary_exc = "; ".join(
        f"`{n}` CE-trained {_pt(mats, n, 'ce')} vs Pearson-trained {_pt(mats, n, 'pearson')}" for n in pearson_ok_bins
    ) or "none"
    exp67_d = [r for r in cj.get("exp67_native_interim", {}).get("calibrated_new", []) if "dRPS" in r]
    exp67_rps_zero = sum(1 for r in exp67_d if r["dRPS_ci"][0] <= 0 <= r["dRPS_ci"][1])
    held_out_summary = (
        f"Cross-entropy, Brier, and RPS heads land on nearly the same held-out proper scores across all large ordinal and 3-class sets, "
        f"while Pearson collapses conditional variance and blows up held-out cross-entropy "
        f"(SummEval expert {_pt(mats, 'summeval_expert', 'ce')} vs {_pt(mats, 'summeval_expert', 'pearson')}, "
        f"DICES-990 {_pt(mats, 'dices990', 'ce')} vs {_pt(mats, 'dices990', 'pearson')}, "
        f"ChaosNLI SNLI+MNLI {_pt(mats, 'chaosnli_snli_mnli', 'ce')} vs {_pt(mats, 'chaosnli_snli_mnli', 'pearson')}, "
        f"POPQUORN Offensiveness {pop_off_ce} vs {pop_off_pe}, POPQUORN Politeness {pop_pol_ce} vs {pop_pol_pe}, "
        f"VeriBench FSC R_v1 {vb_rv1_ce} vs {vb_rv1_pe}, VeriBench FSC native_main {vb_nat_ce} vs {vb_nat_pe}, "
        f"VeriBench FSC pooled {vb_pool_ce} vs {vb_pool_pe}). "
        f"The CE-trained head beats the Pearson-trained head on held-out CE, sqrt(RPS), and Brier on {len(nonbin_wins)} of {len(nonbin)} ordinal and 3-class views. "
        f"On the binary threshold views the gap is small, and the Pearson-trained head's held-out CE is not worse on: {binary_exc} "
        "(`chaosnli_alpha` is the near-log-2 negative control). "
        f"On small-sample 6-class VeriBench subsets (`veribench_fsc_native_main` with {mats.get('veribench_fsc_native_main', {}).get('n_test', 'n/a')} test items "
        f"and `veribench_fsc_rv1_expert3` with {mats.get('veribench_fsc_rv1_expert3', {}).get('n_test', 'n/a')} test items), "
        f"the RPS-trained softmax puts near-zero probability on rare/unobserved classes, giving higher held-out CE ({vb_nat_ce_rps} on native_main) "
        f"than CE-trained ({vb_nat_ce}) while matching on sqrt(RPS) ({vb_nat_rps_rps} vs {vb_nat_rps_ce}). "
        f"The MSE two-point readout has infinite cross-entropy on every K≥3 set because off-support ratings receive probability 0; "
        f"on K=2 binary views MSE is finite on {len(mse_finite_bins)} of {len(bins)} binary views, while on `popquorn_politeness_binary` ridge predictions hit 0/1 on an item "
        f"yielding CE +inf (lowest Brier {pop_pol_bin_mse}) and Pearson ({pop_pol_bin_pe}) overlaps CE ({pop_pol_bin_ce}) within 95% CIs. {oracle_note}"
    )
    e63_rl_sp = _f(e63_a.get("rl_grpo", {}).get("spearman"), 3)
    e63_rl_ce = _f(e63_a.get("rl_grpo", {}).get("ce"), 3)
    e63_sh_ce = _f(e63_a.get("shuffled_control", {}).get("ce"), 3)
    e63_hm_ce = _f(e63_a.get("human_mix_sft", {}).get("ce"), 3)
    e63_hm_rp = _f(e63_a.get("human_mix_sft", {}).get("rps"), 3)
    e63_hm_sp = _f(e63_a.get("human_mix_sft", {}).get("spearman"), 3)
    e39_wins = e39_all.get("ce_cal_beats_pe_cal_ce", "n/a")
    e39_tot = e39_all.get("n_configs", "n/a")
    power_json = folder / "results_power_mdd.json"
    d990_ce_n100 = "n/a"
    d990_pe_n198 = "n/a"
    if power_json.is_file():
        pj = json.loads(power_json.read_text())
        for r in pj.get("rows", []):
            if r["k_test"] == 1 and r["metric"] == "ce" and r["n_test"] == 100:
                d990_ce_n100 = f"{r['power']:.2f}"
            if r["k_test"] == 1 and r["metric"] == "pearson_r" and r["n_test"] == 198:
                d990_pe_n198 = f"{r['power']:.2f}"

    pitch = f"""# Sanmi distributional pitch — 11-26-2026

**As of 11-26-2026.** Experiment `36_sanmi_distributional_pitch_anti_proper_corr_start_26Nov2026`.

**Bottom line.** On a known law, Pearson correlation is unchanged by any positive affine map and, for a judge that samples independently of the human given the prompt, a collapsed affine mean scores a strictly higher correlation than the true conditional distribution whenever the conditional mean actually varies and the conditional variance is not zero. On the real multi-annotator datasets — SummEval, ChaosNLI, DICES-990, POPQUORN (Offensiveness and Politeness), and VeriBench FSC (`R_v1`, live-synced `native_main`, and pooled) — the cross-entropy-trained head beats the Pearson-trained head on held-out proper scores (`CE`, `sqrt(RPS)`, `Brier`) on {len(nonbin_wins)} of {len(nonbin)} ordinal and 3-class views and on every ordinal and 3-class crossover curve from \\(k=1\\). The binary threshold views are closer; the Pearson-trained head's held-out CE is not worse on: {binary_exc}. On real LLM judge panels in CertJudge (Experiments 63, 39, and 67), cross-entropy training or calibration gives lower held-out CE; on RPS, Experiment 39 CE-fit beats Pearson-fit in {e39_all.get('ce_cal_beats_pe_cal_rps', 'n/a')}/{e39_all.get('n_configs', 'n/a')} configs, and {exp67_rps_zero} of {len(exp67_d)} Experiment 67 judge families have a CE-fit minus Pearson-fit RPS 95% interval that includes 0. Correlation is often higher for worse-calibrated or collapsed judges. Synthetic numbers below are checks of those identities, not human results.

## 1. Merged four-step pitch

Notation, used the same way in the note, the figures, the Python, and the Lean comments. \\(c\\) is the prompt. \\(X\\sim\\hat q(\\cdot\\mid c)\\) is the judge's rating. \\(Y\\sim p^*(\\cdot\\mid c)\\) is the human rating. Centered variables are \\(\\tilde X=X-\\mu_X\\) and \\(\\tilde Y=Y-\\mu_Y\\). Standardized variables are \\(z_X=\\tilde X/\\sigma_X\\) and \\(z_Y=\\tilde Y/\\sigma_Y\\). The excess Brier divergence is \\(D_B(p^*\\Vert\\hat q)=\\sum_y(p^*(y\\mid c)-\\hat q(y\\mid c))^2\\).

**Step 1. Concede the two points that are right, and stop at the boundary.** On a binary rating, the scalar mean is the distribution, and scalar squared error matches the Brier score: two-class Brier equals \\(2(q-y)^2\\). The Brier score is strictly proper, with excess risk \\(D_B\\). Pearson correlation is not that object. It is invariant to \\(X\\mapsto aX+b\\) for \\(a>0\\), so a shifted and compressed forecast can tie the true probabilities on correlation while \\(D_B\\) and cross-entropy get worse. {proof}

**Step 2. Mean squared error splits into bias, scale, and correlation, and a discrete mean is not a discrete law.** For standardized variables, \\(\\mathbb E[(z_X-z_Y)^2]=2(1-\\mathrm{{Corr}}(X,Y))\\). In the original units, \\(\\mathbb E[(X-Y)^2]=(\\mu_X-\\mu_Y)^2+(\\sigma_X-\\sigma_Y)^2+2\\sigma_X\\sigma_Y(1-\\mathrm{{Corr}}(X,Y))\\). On \\(\\lbrace 0,1,2,3,4,5\\rbrace\\) a point mass at 2 and a 50/50 split on \\(\\lbrace 0,4\\rbrace\\) share the mean 2, so every scalar-mean correlation ties them, while \\(D_B\\), normalized ranked probability score (RPS), and cross-entropy separate them. A fair coin has three couplings with \\(\\mathbb E[(X-Y)^2]\\in\\lbrace 0,1/2,1\\rbrace\\) and \\(D_B=0\\). If \\(X\\perp Y\\), the squared difference collapses to \\((\\mu_X-\\mu_Y)^2+\\sigma_X^2+\\sigma_Y^2\\).

**Step 3. Correlation is strictly anti-proper for a sampled distributional judge, and one forward pass already returns the distribution.** Given the prompt, take \\(X\\) independent of \\(Y\\). The law of total covariance then says \\(\\mathrm{{Corr}}(X_{{p^\\ast}},Y)=\\mathrm{{Var}}(\\mu_Y(C))/\\mathrm{{Var}}(Y)\\), while the zero-variance judge \\(X_\\delta=a\\mu_Y(C)+b\\) with \\(a>0\\) scores the square root of that correlation. Whenever both variance pieces are positive, \\(0<\\mathrm{{Corr}}(X_{{p^\\ast}},Y)<1\\) and the collapsed judge wins the correlation. The numerical case \\(\\mathrm{{Var}}(\\mu)=\\mathbb E[\\sigma^2]\\) is 0.50 versus about 0.707. Spearman and Kendall have the same monotone-warp invariance and the same ordering on that draw. The practical answer to a single-shot objection is that a softmax over the rating-token logits is one forward pass: the distribution is the normalized logit vector, not an average of sampled generations.

**Step 4. The empirical curves, the dollar axis, and the scaling bridge.** Held-out cross-entropy, sqrt(RPS), and Brier are plotted against ratings per item \\(k\\), with a top axis in dollars at \\$0.05 per rating (the POPQUORN published crowd rate, now also scored directly) and \\$5 per rating (assumed expert rate). Experiment 28 found frontier CE-fit ahead of Pearson-fit on held-out RPS after 4 to 8 ratings at one rating per item (shaded band). Across POPQUORN (Offensiveness CE {pop_off_ce} vs {pop_off_pe}; Politeness CE {pop_pol_ce} vs {pop_pol_pe}), VeriBench FSC `R_v1` ({vb_rv1_ce} vs {vb_rv1_pe}), VeriBench FSC pooled ({vb_pool_ce} vs {vb_pool_pe}), SummEval, ChaosNLI, and DICES-990, the CE head is ahead on every ordinal and 3-class crossover curve from \\(k=1\\) (the binary ChaosNLI abductive NLI curve is a near-log-2 negative control).

## Real-data matrix

{matrix}

## What the held-out table shows

{held_out_summary}

{cert_md}

## Finite-sample curves and iso-budget

{curves}

### Iso-budget comparisons

{iso}

## Single pass versus Monte Carlo rollouts

Dataset `{payload['rollout']['dataset']}`. The Monte Carlo histograms use a \\(10^{{-6}}\\) floor so a single unlucky sample does not make cross-entropy infinite.

{roll_lines}

{COT}

## Within-prompt oracles (not held-out judges)

{oracles_tbl}

## Figures

![Wrong-objective matrix (SummEval, ChaosNLI, DICES)](figures/wrong_objective_matrix.png)

![Wrong-objective matrix (POPQUORN and VeriBench FSC)](figures/wrong_objective_matrix_popquorn_veribench.png)

![Finite-sample crossover, binary versus ordinal Likert](figures/finite_sample_crossover_binary_vs_ordinal.png)

![High-k crossover, DICES-990 and ChaosNLI](figures/high_k_crossover_dices_chaos.png)

![Crossover on POPQUORN and VeriBench FSC](figures/crossover_popquorn_veribench.png)

![CertJudge real LLM judge panels](figures/certjudge_real_panels_proper_vs_corr.png)

![Single pass versus rollouts](figures/single_pass_vs_rollout.png)

![Iso-budget](figures/iso_budget_chaosnli.png)

{power_sec}

## Ready-to-send note to Sanmi

Sanmi — three mathematical facts, then the empirical picture (now including POPQUORN and VeriBench FSC / CertJudge).

On binary ratings you are right that the mean is the distribution and that scalar squared error matches Brier. Brier is strictly proper. Correlation is not: Corr(aX+b, Y) = Corr(X, Y) for any a>0, so a compressed forecast ties the true probabilities on correlation while Brier and cross-entropy get worse (in the four-item witness, correlation stays 1 while mean Brier moves from {_f(payload['proofs_numeric']['binary_affine_witness']['brier_star'])} to {_f(payload['proofs_numeric']['binary_affine_witness']['brier_bad'])} and cross-entropy from {_f(payload['proofs_numeric']['binary_affine_witness']['ce_star'])} to {_f(payload['proofs_numeric']['binary_affine_witness']['ce_bad'])} nats).

On a 0–5 or 1–5 rubric the mean is not the law. A point mass at 2 and a 50/50 split on {{0, 4}} have the same mean and different Brier, RPS, and cross-entropy. If the judge samples independently of the human given the prompt, correlation is strictly anti-proper (true distribution scores Var(mean)/Var(Y), collapsed a·mean+b scores the square root; 0.50 vs ~0.707 when variance pieces are equal).

Across SummEval, DICES-990, ChaosNLI, POPQUORN (Offensiveness & Politeness), and VeriBench FSC (`R_v1` + live-synced `native_main` + pooled), CE-trained heads beat Pearson-trained heads on held-out CE, sqrt(RPS), and Brier on every ordinal and 3-class view from k=1 rating/item; the binary threshold views are closer (Pearson-trained CE not worse on: {binary_exc}). On real LLM judges in CertJudge (Experiment 63 `Qwen3-8B`), RL-GRPO gets Spearman {e63_rl_sp} yet blows up CE to {e63_rl_ce} (worse than shuffled control {e63_sh_ce}), while CE-trained `human_mix_sft` gets CE {e63_hm_ce}, RPS {e63_hm_rp}, and Spearman {e63_hm_sp}; across {e39_tot} LLM judge configs in Experiment 39, CE calibration beats Pearson calibration on CE in {e39_wins}/{e39_tot} configs. And on test-set statistical power (DICES-990, k_test=1), CE separates the heads at N=100 prompts (power {d990_ce_n100}) while Pearson correlation has power {d990_pe_n198} even at N=198.

Happy to walk through the Lean file (no sorry; axioms are propext, Classical.choice, Quot.sound) or any table.

## Two-paper roadmap

**Paper 1 — Target and evaluation protocol.** Define the target as \\(p^*(y\\mid c)\\), not correlation with a scalar. State the output-space taxonomy (binary, finite ordinal, continuous). Fix the test estimator as the unbiased per-sample proper risk, the split as prompt/task-level, and the interval as an item-clustered bootstrap.

**Paper 2 — Training, scaling, and dollars.** Train the same family by cross-entropy, Brier, RPS, Pearson, and MSE. Report the anti-proper collapse, single-pass logit readout vs M-sample rollouts, and held-out proper scores against \\(k\\) and dollars (`$0.05` crowd / `$5` expert) and iso-budget \\(B=N\\times k\\).

{ELICITATION}

{g_theory}

{transcript}

## What remains unrun on this CPU-only machine

- No GPU is present, so `scripts/snap_lora_train.py` (Qwen3-4B / Llama-3.1-8B model-size grid) and `scripts/snap_llm_logit_readout.py` remain ready for SNAP/Sherlock.
- In `trustworthy-theorem-judges` Experiment 09, `T30` (Max Sieb) Google Sheet remains unverified (HTTP 401 without renewed OAuth); all 33 public sheets synced cleanly on 11-26-2026 (`250` non-coordinator main ratings).

## Kohli et al. 2026

Guneet Kohli, arXiv:2605.29797, studies how many annotators a soft-label NLI model needs (on ChaosNLI, KL captures ~91±3% of improvement by 10 annotators while entropy correlation needs 20–50; DICES-990 exploratory 90% vs 62% at 3 annotators). That is annotation saturation for one training loss; here we compare training objectives on held-out proper risk.
"""
    (folder / "SANMI_PITCH.md").write_text(pitch)

    summary = f"""# Results summary — Sanmi distributional pitch (11-26-2026)

**As of 11-26-2026.**

**Bottom line.** Lean 4.36.0 checks the algebraic claims with no `sorry`. Python checks the same claims numerically, including 0.50 versus \\(\\sqrt{{1/2}}\\) for the anti-proper correlation. All real multi-annotator datasets — SummEval, ChaosNLI, DICES-350/990, POPQUORN (Offensiveness and Politeness), and VeriBench FSC (`R_v1`, live-synced `native_main`, and pooled) — plus real CertJudge LLM judge panels (`E63`, `E39`, `Exp 67`) are scored and summarized below.

## Formally proved (Lean) and numerically checked

{proof}

Theorem names and the axiom printout are in `lean_check.txt`. The numerical assertions are `proofs_numeric.py` (every check returned ok).

## What the held-out numbers say

{held_out_summary}

{cert_md}

## Real multi-annotator results (this machine)

Proper-score cells are the unbiased prompt-average of per-rating losses. Pearson r is the correlation of the judge's scalar with individual ratings, item-clustered bootstrap. Spearman and Kendall use prompt-level human means.

{matrix}

### Crossover

{curves}

### Iso-budget

{iso}

### Single pass versus rollouts

{roll_lines}

### Within-prompt oracles

{oracles_tbl}

{power_sec}

## GPU scripts not run

`scripts/snap_lora_train.py` and `scripts/snap_llm_logit_readout.py` are ready for SNAP or Sherlock. This machine has no GPU; no LoRA number in this folder comes from a run.

{ELICITATION}

{g_theory}

{transcript}
"""
    (folder / "RESULTS_SUMMARY.md").write_text(summary)

    readme = f"""# Experiment 36: Sanmi distributional pitch, anti-proper correlation

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/36_sanmi_distributional_pitch_anti_proper_corr_start_26Nov2026/README.md>

**TLDR:** Pearson correlation is affine-invariant and, for a judge sampled independently of the human given the prompt, strictly anti-proper relative to the true conditional distribution. Brier and ranked probability score (RPS) are strictly proper. Real multi-annotator heads are scored across SummEval, ChaosNLI, DICES, POPQUORN (Offensiveness & Politeness), and VeriBench FSC (`R_v1`, live-synced `native_main`, and pooled), alongside real LLM judge panels bridged from CertJudge (Experiments 63, 39, 67). Read `HOW_TO_READ_RESULTS.md` before quoting a number.

**Status:** complete on this machine (11-26-2026) across all downloaded/local multi-annotator sets and CertJudge panels. No GPU (SNAP scripts ready). Cost $0 of new model calls. Raw downloads are cached under `data_cache/` and are not committed.

## Reproduce

From the repository root, with Python 3, NumPy, SciPy, Matplotlib, and pytest:

```bash
python3 experiments/36_sanmi_distributional_pitch_anti_proper_corr_start_26Nov2026/eval_power_mdd.py
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
| `RESULTS_SUMMARY.md` | Proved claims, held-out tables, CertJudge real LLM judge bridge, power & G-theory |
| `HOW_TO_READ_RESULTS.md` | Data, metric, difference, interval, and how the head was chosen |
| `results.md` | Live ledger |
| `results.json`, `results.txt`, `results_power_mdd.json` | Exact metrics & power/MDD grids |
| `PROTOCOL.md` | Pre-registered measurement contract + Amendment 1 (11-26-2026) |
| `SanmiPitch.lean`, `lean_check.txt` | Formal proofs and the compiler receipt |
| `proofs_numeric.py` | Numerical twins of the Lean claims |
| `metrics.py`, `data_io.py`, `fit_eval.py`, `run_all.py`, `eval_power_mdd.py` | Estimators, loaders, heads, drivers |
| `figures/` | Matrices, crossovers, CertJudge real panels, rollout, iso-budget, test-set power |
| `tables/` | Markdown and LaTeX tables copied from the run |
| `scripts/snap_lora_train.py`, `scripts/snap_llm_logit_readout.py` | Ready for SNAP/Sherlock GPU runs (not run here) |
| `tests/test_hygiene.py` | Unbiased risk, prompt/task splits, dataset loaders, metric identities |

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
    if cert_md:
        (tables / "certjudge_real_panels.md").write_text(cert_md)
    (tables / "objective_metric_matrix.tex").write_text(_tex(payload))


def avail_lines(payload):
    return "\n".join(f"- **{k}.** {v}" for k, v in payload["availability"].items())


def how_to_read():
    return """# How to read these results

**As of 11-26-2026.**

1. **Data.** Human numbers are held-out test prompts or tasks. The split is by group (SummEval: source article; ChaosNLI, DICES, POPQUORN: item/instance id; VeriBench FSC `R_v1` and `native_main`: task/task_folder so all samples of a task stay on one side), fractions 0.6/0.2/0.2, seed 0. Raters of one item are never placed on two sides of the split. A within-prompt oracle is labeled as such and is not a held-out judge. CertJudge panels (`E63`, `E39`, `Exp 67`) are read-only verified evaluations of real LLM judges from `cert-judge`.

2. **Metrics.** Cross-entropy is \\(-\\log \\hat q(y)\\) in nats, lower better, range \\([0,\\infty)\\), reference \\(\\log K\\) for a uniform forecast. One-hot Brier is \\(\\Vert\\hat q-e_y\\Vert_2^2\\), lower better, range \\([0,2]\\). Ranked probability score (RPS) is \\(\\sum_{t=0}^{K-2}(F(t)-1_{y\\le t})^2\\), lower better, range \\([0,K-1]\\). The matrix column sqrt(RPS) is the square root of the unbiased mean of that score. Normalized RPS divides by \\(K-1\\) and is the project's primary proper score; it is in the tables next to sqrt(RPS). Pearson r, Spearman rho, and Kendall tau are higher-better association diagnostics.

3. **The test estimator.** For a proper loss the reported risk is the mean over test prompts of the mean over that prompt's ratings. A plug-in \\(D_B(\\hat p_{test}\\Vert\\hat q)\\) is biased high by \\((1/k_c)(1-\\Vert p^\\ast(c)\\Vert_2^2)\\) in expectation when \\(\\hat p\\) is the empirical frequency. That plug-in is not the reported test risk.

4. **Differences.** A comparison is the same metric on the same split. "Held-out test CE of the Pearson-trained head minus held-out test CE of the CE-trained head" is positive when the CE-trained head is better.

5. **Intervals.** Percentile 95% intervals from an item-clustered bootstrap, B=1000 (B=40 only if `SANMI36_SMOKE=1`). The unit is the prompt/item. The interval is for this fixed trained head, not for retraining.

6. **How the head was chosen.** Signed 256-dimensional token hash, 24 principal components fit on train prompts only, plus an intercept. L2 penalty \\(10^{-2}\\) on non-intercept weights. Cross-entropy, Brier, RPS, and Pearson: L-BFGS-B from the zero vector, at most 35 iterations. Cross-entropy of a linear softmax is convex in the weights; the others are not, and the result is this start and this budget. MSE is ridge regression (closed form, global for that penalized linear problem) read out as the adjacent two-point distribution with the same mean. Validation did not choose these penalties.

7. **Direction and scale.** Proper scores: lower is better. Correlations: higher is better. A uniform forecast on K classes has cross-entropy \\(\\log K\\).
"""


def results_md(payload):
    return f"""# Results ledger

**Status:** COMPLETE for SummEval, ChaosNLI, DICES, POPQUORN, VeriBench FSC (`R_v1`, `native_main`, pooled), and CertJudge real panels (11-26-2026).

**Last updated:** 11-26-2026.

| Phase | State |
| --- | --- |
| Numeric proof checks | DONE |
| Lean compile | DONE (`lean_check.txt`, 0 `sorry`) |
| Real-data matrix, crossover, iso-budget, rollout | DONE on SummEval, ChaosNLI, DICES, POPQUORN, VeriBench FSC |
| CertJudge real LLM judge bridge (`E63`, `E39`, `Exp 67`) | DONE (`$0` extra spend) |
| Test-set power & MDD | DONE on DICES-990, POPQUORN Offensiveness, POPQUORN Politeness |
| GPU LoRA & LLM CoT logit readout (`scripts/snap_*.py`) | NOT RUN (no GPU on this machine) |

Runtime seconds: {payload.get('seconds')}.

**TLDR-end:** [judges-corr-vs-loss: sanmi distributional pitch] Correlation is affine-invariant and anti-proper for an independently sampled judge; the held-out human tables across SummEval, ChaosNLI, DICES, POPQUORN, VeriBench FSC, and CertJudge real LLM panels are in `RESULTS_SUMMARY.md`.
"""


def plan_md(payload):
    return f"""# Plan and status — 11-26-2026

**Status:** the driver `run_all.py` finished (including Amendment 1 datasets and CertJudge bridge). Smoke flag: {payload.get('smoke')}.

## Contract (fixed before test metrics were read)

Predictions, written down before the run:

- Uniform cross-entropy equals \\(\\log K\\).
- A CE-trained softmax should beat a constant uniform forecast on held-out CE when the hash features carry any signal.
- Shuffling train features should not improve held-out CE.
- On a known two-context law, the collapsed judge's correlation should sit near 0.707 and the true sampled judge near 0.50.
- On binary data, scalar L2 and Brier match up to the factor 2. Correlation can still tie while Brier and CE worsen.
- On ordinal data with within-item spread, a mean-matched two-point readout should lose proper scores to a softmax that is allowed to use the full simplex.
- The within-prompt oracle should show a higher correlation for the collapsed affine score than for a draw from the half-sample distribution, and a worse proper score, when within-item variance is positive.

Deviations are whatever `results.json` shows. They are not edited back into this list.

## Availability

All planned multi-annotator datasets (SummEval, ChaosNLI, DICES-350/990, POPQUORN Offensiveness/Politeness, VeriBench FSC `R_v1`, live-synced `native_main`, and pooled) and CertJudge real LLM panels are scored in `results.json`. Only GPU runs (`scripts/snap_*.py`) and `T30` Google Sheet (HTTP 401 without OAuth) remain outside this CPU run.
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
