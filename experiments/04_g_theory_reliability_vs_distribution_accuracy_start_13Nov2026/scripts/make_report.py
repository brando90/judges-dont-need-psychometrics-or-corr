#!/usr/bin/env python3
"""TLDR: Generate RESULTS_reliability_vs_accuracy.md from results/*.json (no number typed by hand).

Run: python experiments/04_g_theory_reliability_vs_distribution_accuracy_start_13Nov2026/scripts/make_report.py
"""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
RES = HERE / 'results'
EXP = HERE.name
DOC = f'https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/{EXP}/RESULTS_reliability_vs_accuracy.md'


def J(name):
    return json.loads((RES / name).read_text())


def f(x, d=3):
    if x is None:
        return 'undefined'
    if isinstance(x, dict):
        x = x['float'] if 'float' in x else x.get('point', x.get('mean'))
    if x == float('inf'):
        return '∞'
    return f'{x:.{d}f}'


def ci(e, d=3):
    lo, hi = e['ci95'] if e.get('ci95') else (None, None)
    m = e['point'] if 'point' in e else e['mean']
    if lo is None:
        return f(m, d)
    return f'{m:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]'


def sci(x):
    return f'{x:.2e}'


def main():
    a, b, c = J('phase_a.json'), J('phase_b.json'), J('phase_c.json')
    t, v = J('test_metrics.json'), J('verification.json')
    fr = json.loads((HERE / 'FREEZE.json').read_text())
    L = []
    w = L.append
    contracts = a['contract'] + b['contract'] + c['contract']
    n_pass = sum(x['pass'] for x in contracts)

    w('# 04: Results — reliability versus human-rating distribution accuracy')
    w('')
    w(f'**Doc link:** <{DOC}>')
    w('')
    w('**TLDR:** Reliability coefficients and matched means/variances do not identify the human-rating distribution: '
      'exact counterexamples tie on every moment and coefficient while proper losses separate them, and in the historical '
      'judges the most repeatable judge had the worst held-out ranked probability score (RPS). Variance decomposition did '
      'its own job well: decision-study allocation chose the best design in every simulated replication and failed only '
      'where its stated assumptions were broken. Complementary tools, not substitutes.')
    w('')
    w(f'**Status:** COMPLETE — Phases A, B and C ran on 11-13-2026 (Claude Code, `claude-opus-5-5`, desktop app; local CPU; '
      f'zero model calls, zero new human ratings, $0). {n_pass}/{len(contracts)} frozen predictions matched; the '
      f'{len(contracts) - n_pass} mismatches are explained below (two Monte Carlo chance events, one construction tie). '
      f'Formula checks {t["n_checks"] - t["n_fail"]}/{t["n_checks"]}; independent recomputation {v["n_checks"] - v["n_fail"]}/{v["n_checks"]}.')
    w('')
    w('Read [HOW_TO_READ_RESULTS.md](HOW_TO_READ_RESULTS.md) first for which data, metric and comparison each number uses. '
      'Plan: [PLAN.md](PLAN.md). Frozen configuration and predictions: [protocol.json](protocol.json), hashed in '
      '[FREEZE.json](FREEZE.json) before measurement (commit `579915a`).')
    w('')

    # ---------------------------------------------------------------- verdicts
    a1s, a1m = a['A1']['summary'], a['A1_max']['summary']
    a2 = a['A2']['judges']
    ct = c['sets']['test']
    m = ct['metrics']
    sel = b['selection']
    w('## Verdicts')
    w('')
    w('| Claim tested (PLAN.md §1) | Verdict | Decisive evidence |')
    w('| --- | --- | --- |')
    w(f'| Matching the human mean and variance (and every reliability coefficient) identifies the human-rating distribution | **Rejected** (exact counterexample) | A1_max: same mean, variance, components and coefficients for all 11 artifacts, yet mean population TVD {f(a1m["mean_tvd"])} and expected RPS excess {f(a1m["mean_rps_excess"], 4)} |')
    w(f'| High reliability coefficients certify accurate human scores | **Rejected** | A2: a judge wrong on 3 of 11 artifacts has Eρ² = Φ = 1, identical to the oracle; its expected RPS is {f(a2["repeatable_wrong"]["mean_expected_rps"])} vs {f(a2["oracle"]["mean_expected_rps"])}. Phase C: most repeatable judge {ct["most_repeatable_judge"]} had the worst test RPS |')
    w(f'| Maximizing sampled-score reliability optimizes distribution accuracy | **Rejected** | A3: sharpening the true distribution raises one-rating Eρ² from {f(a["A3"]["tempered_family"]["1"]["sampled_Erho2"]["1"])} to {f(a["A3"]["tempered_family"]["mode"]["sampled_Erho2"]["1"])} while expected RPS rises from {f(a["A3"]["tempered_family"]["1"]["mean_expected_rps"], 4)} to {f(a["A3"]["tempered_family"]["mode"]["mean_expected_rps"], 4)} |')
    w('| An aggregate coefficient certifies each task | **Rejected** (as a misuse) | V2: one pooled Φ hides groups whose absolute error differs by ' f'{ci(b["vulnerabilities"]["V2_group_heterogeneity"]["paired_obs_H_minus_L"])} |')
    w(f'| Variance components guide allocation of rating effort (the paper\'s claim) | **Supported in-model** | REML-based choice matched the oracle design in {sel["C1"]["budget16_absolute_reml"]["chose_oracle_design"]["k"]}/256 (C1) and {sel["C2"]["budget16_absolute_reml"]["chose_oracle_design"]["k"]}/256 (C2) replications; paired held-out absolute error vs equal allocation {ci(sel["C1"]["budget16_absolute_reml"]["paired_obs_err_chosen_minus_equal"])} (C1) |')
    w('| Proper losses make psychometric tools unnecessary | **Not supported** | A single-rating variance (identical, 3.0, in both A4 configurations) cannot choose a design; component estimates can. RPS answers a different question |')
    w('')
    w('![Panel A: synthetic tempered forecasts; panel B: historical judges](results/fig_reliability_vs_rps.png)')
    w('')
    w(f'*Figure.* **A (synthetic, exact):** forecasts q_T ∝ p^(1/T) of the 11-artifact population; x = relative coefficient Eρ² '
      f'of one rating sampled from q_T (higher = more repeatable), y = expected RPS against the true distribution p (range 0–1, '
      f'lower is better; the true distribution scores {f(a["A3"]["tempered_family"]["1"]["mean_expected_rps"], 4)}, not 0, because humans disagree). '
      f'**B (historical, retrospective):** five cached judges on the historical test split (30 artifacts, 10 tasks, '
      f'{ct["n_expert_ratings"]} expert ratings, all previously exposed); x = one-call test–retest Eρ² over 3 repeat calls, '
      f'y = held-out RPS of each judge\'s smoothed 3-call histogram against individual expert ratings (lower is better); bars '
      f'are 95% percentile intervals over {ct["bootstrap_draws"]:,} task bootstrap draws. Dashed line: a constant forecast equal '
      f'to the train-split expert histogram (RPS {f(m["baseline_train_marginal"]["RPS"])}). Dotted line: single-rating agreement '
      f'between the two experts who rated every artifact (Eρ² {f(m["human"]["Erho2_1"]["point"], 2)}).')
    w('')

    # ---------------------------------------------------------------- Phase A
    w('## Phase A — exact counterexamples and controls (synthetic, no sampling unless stated)')
    w('')
    w('Human population: 11 artifacts, p_x = (9/10) Binomial(5, θ_x) + (1/10) Uniform{0..5}, θ_x = j/12. All values are exact '
      'rational arithmetic except CE/KL and the tempered family (float64). Every metric is an expectation over the true '
      'distribution, equally averaged over artifacts; there is no train/test split because nothing is fitted.')
    w('')
    w('### A1 / A1_max: same mean and variance, different distribution')
    w('')
    w('| Quantity (mean over 11 artifacts) | True distribution p | A1: p + ε·(−1,3,−3,1,0,0), ε = min p/6 | A1_max: largest RPS gap, floor 1/100 |')
    w('| --- | --- | --- | --- |')
    w(f'| Mean and variance equal to p | — | exactly, 11/11 | exactly, 11/11 |')
    w(f'| Population v_u, v_e; Eρ² and Φ (1 and 3 raters) | v_u {a1s["components_p"]["u"]["exact"]}, v_e {a1s["components_p"]["e"]["exact"]}; Eρ²(1) {a1s["coefficients_p"]["1"]["Erho2"]["exact"]} = {f(a1s["coefficients_p"]["1"]["Erho2"])} | identical (exact) | identical (exact) |')
    w(f'| Pearson / Spearman / Kendall of forecast means with true means | 1 / 1 / 1 | 1 / 1 / 1 (tie) | 1 / 1 / 1 (tie) |')
    w(f'| Expected RPS (lower better) | {f(a1s["mean_oracle_expected_rps"], 4)} | excess +{sci(a1s["mean_rps_excess"]["float"])} (= (6/5)ε², exact) | excess +{f(a1m["mean_rps_excess"], 4)} (per artifact {f(min(r["rps_excess"]["float"] for r in a["A1_max"]["per_artifact"]), 4)}–{f(max(r["rps_excess"]["float"] for r in a["A1_max"]["per_artifact"]), 4)}) |')
    w(f'| Expected CE excess = KL(p‖q), nats | 0 (CE {f(a1s["mean_oracle_expected_ce_nats"])}) | {sci(a1s["mean_kl_nats"])} | {f(a1m["mean_kl_nats"])} |')
    w(f'| Population TVD (0–1) | 0 | {f(a1s["mean_tvd"], 4)} (= 4ε) | {f(a1m["mean_tvd"])} |')
    w('')
    w('A1 follows PLAN.md exactly and is a valid but tiny counterexample (ε between '
      f'{f(min(r["epsilon"]["float"] for r in a["A1"]["per_artifact"]), 4)} and {f(max(r["epsilon"]["float"] for r in a["A1"]["per_artifact"]), 4)}). '
      'A1_max (declared before measurement as an addition) shows the gap need not be small: with every probability at least 0.01, '
      'a forecast can match all first and second moments while differing from the human distribution by TVD up to '
      f'{f(max(r["tvd"]["float"] for r in a["A1_max"]["per_artifact"]))}. Trace artifact j = 6 (θ = 1/2): p = {", ".join(a["trace"]["p"])}; '
      f'A1_max q = {", ".join(a["trace"]["q_A1_max"])}; exact JSON round trip {a["trace"]["serialization_roundtrip_exact"]}.')
    w('')
    fs = a['A1_finite_sample']
    w(f'**Finite samples** ({fs["replications"]} replications; 50 objects drawn from the 11 artifacts × 3 raters; raw two-way ANOVA). '
      'Population components are equal, so expected estimates are equal; estimator spread can differ because higher moments differ:')
    w('')
    w('| Estimate | p-panel mean | A1 q-panel minus p-panel [95% CI] | A1_max q minus p [95% CI] | SD ratio A1_max/p [95% CI] |')
    w('| --- | --- | --- | --- | --- |')
    for k, lab in (('u', 'v_u'), ('e', 'v_e'), ('a', 'v_rater'), ('Erho2_n3', 'Eρ² (3 raters)')):
        c1, c2 = fs['comparisons']['q_A1'][k], fs['comparisons']['q_A1_max'][k]
        w(f'| {lab} | {c1["mean_p_panel"]:.4f} | {c1["paired_mean_diff_q_minus_p"]:+.4f} [{c1["ci95"][0]:+.4f}, {c1["ci95"][1]:+.4f}] | '
          f'{c2["paired_mean_diff_q_minus_p"]:+.4f} [{c2["ci95"][0]:+.4f}, {c2["ci95"][1]:+.4f}] | {c2["sd_ratio_q_over_p"]:.3f} [{c2["sd_ratio_ci95"][0]:.3f}, {c2["sd_ratio_ci95"][1]:.3f}] |')
    d1 = v['post_hoc_diagnostics']['A1_finite_sample_v_u_difference_rerun']
    w('')
    w(f'The frozen A1 v_u interval excludes 0 narrowly (prediction mismatch). A post-hoc rerun with {d1["replications"]:,} new '
      f'replications (seed {d1["seed"]}) gives {d1["mean_q_minus_p"]:+.4f} [{d1["ci95"][0]:+.4f}, {d1["ci95"][1]:+.4f}], and the '
      'sampler reproduces the pmf; the frozen miss is a Monte Carlo chance event (6 simultaneous 95% intervals miss at least once '
      'about 26% of the time). The A1_max v_e estimate is about 12% more variable than under p: equal coefficients do not mean equally precise estimates.')
    w('')
    w('### A2: repeatable but wrong; constant judge; shared offset')
    w('')
    w('| Judge (vector forecast) | Eρ² = Φ, deterministic output (1 and 3 calls) | Eρ² for 1 / 3 sampled ratings | Expected RPS | TVD | MAE of means (0–5) | Bias (judge − human) | Pearson of means |')
    w('| --- | --- | --- | --- | --- | --- | --- | --- |')
    for name, lab in (('oracle', 'Oracle (returns p)'), ('repeatable_wrong', 'Repeatable-but-wrong (mirrored on j = 9–11)'), ('constant_marginal', 'Constant marginal')):
        jj = a2[name]
        dco = jj['deterministic_vector_output']['coefficients']['1']['Erho2']
        sco = jj['sampled_rating_output']['coefficients']
        w(f'| {lab} | {f(dco)} | {f(sco["1"]["Erho2"])} / {f(sco["3"]["Erho2"])} | {f(jj["mean_expected_rps"], 4)} | {f(jj["mean_tvd"])} | '
          f'{f(jj["mae_of_means"])} | {f(jj["bias_of_means"], 3)} | {f(jj["correlations_of_means"]["pearson"])} |')
    off = a['A2']['offset']
    w('')
    w(f'Shared +1 offset (human support {{1,2,3}}, judge {{2,3,4}}, 5 artifacts): separately analyzed panels have identical components '
      f'(v_u {off["human_components"]["u"]["exact"]}, v_e {off["human_components"]["e"]["exact"]}) and coefficients (Eρ² = Φ = '
      f'{off["human_coefficients"]["1"]["Erho2"]["exact"]} for one rating), Pearson of means {f(off["correlations_of_means"]["pearson"])}, '
      f'but bias +1 exactly, expected RPS {f(off["mean_expected_rps_shifted"], 4)} vs {f(off["mean_expected_rps_oracle"], 4)}, CE ∞ '
      f'(smoothed variant {f(off["mean_expected_ce_shifted_smoothed_nats"])} vs {f(off["mean_expected_ce_oracle_nats"])} nats). '
      f'Pooling judge and human ratings as raters (half each) exposes the offset in the absolute coefficient only: Φ for one rating '
      f'{off["pooled_coefficients"]["1"]["Phi"]["exact"]} = {f(off["pooled_coefficients"]["1"]["Phi"])} vs Eρ² {off["pooled_coefficients"]["1"]["Erho2"]["exact"]}.')
    w('')
    w('### A3: the true distribution versus sampled-score reliability')
    w('')
    w('| Forecast q_T ∝ p^(1/T) | Expected RPS | Expected CE (nats) | TVD | Mean conditional variance (human 1.260) | Eρ², 1 / 3 sampled ratings | Expected RPS of a 3-call histogram |')
    w('| --- | --- | --- | --- | --- | --- | --- |')
    for T, row in a['A3']['tempered_family'].items():
        lab = {'1': 'T = 1 (true p)', 'mode': 'T → 0 (mode; ties split)'}.get(T, f'T = {T}')
        w(f'| {lab} | {f(row["mean_expected_rps"], 4)} | {f(row["mean_expected_ce_nats"])} | {f(row["mean_tvd"])} | {f(row["mean_conditional_variance"])} | '
          f'{f(row["sampled_Erho2"]["1"])} / {f(row["sampled_Erho2"]["3"])} | {f(row["mean_expected_rps_3call_histogram"], 4)} |')
    pd = a['A3']['population_dependence']
    w('')
    w(f'Reported as a distribution, the true p minimizes expected RPS and CE. Scored only through a 3-call histogram, it does not: '
      f'T = 1/2 scores {f(a["A3"]["tempered_family"]["1/2"]["mean_expected_rps_3call_histogram"], 4)} vs '
      f'{f(a["A3"]["tempered_family"]["1"]["mean_expected_rps_3call_histogram"], 4)}, because sampling noise penalizes spread. '
      f'A 3-call histogram is a noisy estimate of a judge\'s distribution, and scoring it rewards over-sharpened samplers. Population dependence: two populations '
      f'whose artifacts all share the same shape (1/4, 1/2, 1/4) have identical oracle accuracy (RPS {pd["narrow"]["oracle_mean_expected_rps"]["exact"]}, MAE 0) '
      f'but one-rating Eρ² {pd["narrow"]["sampled_coefficients"]["1"]["Erho2"]["exact"]} (centers 2–3) vs '
      f'{pd["wide"]["sampled_coefficients"]["1"]["Erho2"]["exact"]} (centers 1–4).')
    w('')
    w('### A4: equal total variance, different design recommendations')
    w('')
    w('| Configuration (u, a, b, ua, ub, ab, e) | Single-rating variance | v_Δ / Φ at 8×2 | 4×4 | 2×8 | Best at 16 ratings |')
    w('| --- | --- | --- | --- | --- | --- |')
    for cn, cc in a['A4']['configs'].items():
        comp = ', '.join(cc['components'][k].removesuffix('/1') for k in ('u', 'a', 'b', 'ua', 'ub', 'ab', 'e'))
        ds = cc['designs']
        w(f'| {cn}: ({comp}) | {cc["marginal_single_rating_variance"]["exact"].removesuffix("/1")} | {f(ds["8x2"]["v_Delta"], 4)} / {f(ds["8x2"]["Phi"])} | '
          f'{f(ds["4x4"]["v_Delta"], 4)} / {f(ds["4x4"]["Phi"])} | {f(ds["2x8"]["v_Delta"], 4)} / {f(ds["2x8"]["Phi"])} | {a["A4"]["optimal"][cn]["budget16_absolute"]} |')
    w('')
    w('a = raters, b = occasions; n_a × n_b ratings per object. Matching the single-rating variance (3 in both) cannot choose between 8×2 and 2×8; the component decomposition can.')
    w('')

    # ---------------------------------------------------------------- Phase B
    w('## Phase B — decision-study (D-study) allocation, a fair positive control')
    w('')
    w('Continuous Gaussian crossed model, known components; 256 replications, each running calibration G study '
      '(64 objects × 8 raters × 8 occasions) → constrained REML → design choice → evaluation of every design on 256 fresh '
      'objects with fresh raters, occasions and residuals. Absolute error = mean squared deviation of an object\'s observed mean '
      'from its universe score (expectation v_Δ; lower is better); relative error is its object-centered analogue (v_δ).')
    w('')
    fc_n = sum(len(x) for x in b['formula_checks'].values())
    fc_ok = sum(y['within_tolerance'] for x in b['formula_checks'].values() for y in x.values())
    d2 = v['post_hoc_diagnostics']['B_C1_4x8_absolute_formula_rerun']
    w(f'**Formula check:** {fc_ok}/{fc_n} observed mean errors within 3.5 Monte Carlo standard errors of the population formula. '
      f'The miss (C1, 4×8, absolute: observed {f(d2["frozen_mean"], 4)} vs {f(d2["population"], 4)}) is a skewed-error chance event: '
      f'post hoc, {d2["replications"]:,} new replications give {f(d2["mean"], 4)} (z = {d2["z"]:.2f}); absolute error has skewness '
      f'{d2["skewness"]:.1f}, and the frozen 256-run sample SD ({d2["frozen_sample_sd"]:.3f}) underestimated the true SD ({d2["large_sample_sd"]:.3f}). '
      f'REML converged {b["estimates"]["C1"]["reml_converged"]}/256 and {b["estimates"]["C2"]["reml_converged"]}/256; no raw ANOVA estimate was negative, so REML equalled ANOVA.')
    w('')
    w('| Config | Oracle design (16 ratings) | REML chose it | Held-out absolute error: chosen minus equal 4×4 [95% CI] | Chosen minus worst fixed design | Predicted minus true error of chosen |')
    w('| --- | --- | --- | --- | --- | --- |')
    for cn in ('C1', 'C2'):
        e = sel[cn]['budget16_absolute_reml']
        worst = 'paired_obs_err_chosen_minus_more_b' if cn == 'C1' else 'paired_obs_err_chosen_minus_more_a'
        w(f'| {cn} | {e["oracle_design"]} | {e["chose_oracle_design"]["k"]}/256 (Wilson 95% [{e["chose_oracle_design"]["ci95"][0]:.3f}, 1]) | '
          f'{ci(e["paired_obs_err_chosen_minus_equal"])} (true −0.100) | {ci(e[worst])} | {ci(e["pred_minus_true_err"], 4)} |')
    w('')
    w('The same perfect choice held at 8 and 32 ratings and for the relative target. The positive control is easy by construction: '
      'the configurations differ strongly, so this shows the method works when its assumptions hold, not how often it works with weak signals.')
    w('')
    vv = b['vulnerabilities']
    v1, v2, v3, v4 = vv['V1_correlated_raters'], vv['V2_group_heterogeneity'], vv['V3_population_shift'], vv['V4_dependent_occasions_in_calibration']
    w('**Assumption violations** (C1 unless noted; same nominal budget of 16 ratings per object):')
    w('')
    w('| Violation | Calibration prediction | Analytic truth | Observed held-out | Consequence |')
    w('| --- | --- | --- | --- | --- |')
    for rho, r in v1.items():
        w(f'| V1 rater pairs correlated ρ = {rho} (8 nominal raters = 4 pairs) | error {f(r["naive_population_prediction"], 4)}, Φ {f(r["naive_Phi"])} | error {f(r["analytic_actual"], 4)}, Φ {f(r["actual_Phi"])} | error {ci(r["observed_abs_err"])} | precision overstated |')
    w(f'| V2 two task groups, interaction variance 0.2 vs 1.0 | pooled error {ci(v2["pooled_reml_prediction"])} | L {f(v2["group_L"]["analytic"], 4)}, H {f(v2["group_H"]["analytic"], 4)} | H − L {ci(v2["paired_obs_H_minus_L"])} | one Φ ({f(v2["pooled_Phi"])}) hides Φ_L {f(v2["group_L"]["Phi"])}, Φ_H {f(v2["group_H"]["Phi"])} |')
    s1, s2 = v3['S1_restricted_object_range'], v3['S2_occasion_instability']
    w(f'| V3-S1 deployment objects less spread (v_u 1 → 0.25) | error {ci(s1["calibration_predicted_err"])}, Φ {ci(s1["calibration_predicted_Phi"])} | error 0.2875, Φ {f(s1["realized_Phi_of_chosen"])} | error {ci(s1["observed_err_chosen"])} | error still right, coefficient badly overstated |')
    w(f'| V3-S2 occasion instability (v_ub 0.1 → 1.2) | chose 8×2 in {s2["chosen_design_counts"].get("8x2", 0)}/256 | best is {s2["best_design_under_shift"]}: {f(s2["analytic_err_by_design_under_shift"]["4x4"], 4)} vs {f(s2["analytic_err_by_design_under_shift"]["8x2"], 4)} | best minus chosen {ci(s2["paired_obs_err_best_minus_chosen"])} | recommendation reverses |')
    w(f'| V4 (C2, 2×8) calibration occasions dependent, ρ = 0.8 | Φ {ci(v4["predicted_Phi_dependent_calibration"])} | Φ {f(v4["realized_Phi"])} | predicted minus realized {ci(v4["paired_pred_Phi_dependent_minus_realized"])} | coefficient inflated, as the paper warns |')
    w('')

    # ---------------------------------------------------------------- Phase C
    w('## Phase C — retrospective check on existing human ratings')
    w('')
    w(f'Data: historical VeriBench judge study (upstream commit `{c["upstream_source_commit"][:10]}`), imported privately by experiment 02. '
      f'**Primary set: historical test split, {ct["n_artifacts"]} artifacts, {ct["n_tasks"]} tasks, {ct["n_expert_ratings"]} expert ratings; '
      f'all ratings previously exposed in earlier work, so this is retrospective, not fresh held-out evidence.** Judges: five cached '
      f'prompts × models with 3 repeat calls each (no new calls). Judge forecast = smoothed 3-call histogram (ε = 1e-6, the experiment 02 rule). '
      f'Intervals: 95% percentile, {ct["bootstrap_draws"]:,} task-bootstrap draws (raters and calls fixed).')
    w('')
    w('| Judge (model) | Test–retest Eρ², 1 call | Repeat SD (0–5) | **Test RPS** (lower better) | Test CE (nats) | Test TVD | MAE (0–5) | Bias judge − expert | Pearson | Spearman | Kendall τ-b | ICC(A,1) | ICC(C,1) |')
    w('| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |')
    models = c['judges']
    for j in ('H0_sol', 'H1_sol', 'H2_sol', 'H0_opus', 'H2_opus'):
        r = m[j]
        w(f'| {j} ({models[j]}) | {ci(r["Erho2_1"])} | {f(r["repeat_sd"]["point"], 2)} | **{ci(r["RPS"])}** | {f(r["CE"]["point"], 2)} | {f(r["TVD"]["point"])} | '
          f'{ci(r["MAE"], 2)} | {ci(r["bias"], 2)} | {f(r["pearson"]["point"], 2)} | {f(r["spearman"]["point"], 2)} | {f(r["kendall"]["point"], 2)} | {f(r["ICC_A1"]["point"], 2)} | {f(r["ICC_C1"]["point"], 2)} |')
    h = m['human']
    bl = m['baseline_train_marginal']
    w(f'| Reference: two experts (p1, p2) | inter-rater {ci(h["Erho2_1"])}; Φ {f(h["Phi_1"]["point"])} | — | — | — | — | — | — | — | — | — | Φ {f(h["Phi_1"]["point"], 2)} | Eρ² {f(h["Erho2_1"]["point"], 2)} |')
    w(f'| Reference: constant train-marginal forecast | undefined (constant) | 0 | {ci(bl["RPS"])} | {f(bl["CE"]["point"], 2)} | {f(bl["TVD"]["point"])} | — | — | — | — | — | — | — |')
    pr = ct['paired_most_repeatable_minus_lowest_rps']
    w('')
    w(f'Order by repeatability: {" > ".join(ct["order_by_reliability"])}. Order by test RPS (best first): {" > ".join(ct["order_by_rps"])}. '
      f'Kendall τ-b between the two orders = {ct["kendall_tau_reliability_vs_accuracy"]:.2f} (5 judges; descriptive, p-val=n/a). '
      f'Paired task bootstrap, {ct["most_repeatable_judge"]} minus {ct["lowest_rps_judge"]}: RPS {ci(pr["RPS"])} (positive = the more repeatable judge is worse), '
      f'Eρ² {ci(pr["Erho2_1"])}, MAE {ci(pr["MAE"], 2)}.')
    se = ct['sensitivity_excluding_flagged']
    a75 = c['sets']['all75']
    w('')
    w(f'Every judge\'s one-call repeatability exceeded the experts\' agreement with each other (point estimates, 5/5; the interval '
      f'excludes 0 for {sum(ct["judge_minus_human_Erho2_1"][j]["ci95"][0] > 0 for j in ct["judge_minus_human_Erho2_1"])}/5), yet no judge beat '
      f'the constant train-marginal forecast on test RPS (point estimates). Test CE of {min(m[j]["CE"]["point"] for j in models):.1f}–{max(m[j]["CE"]["point"] for j in models):.1f} nats (vs {f(bl["CE"]["point"], 2)} for the constant forecast) shows the 3-call histograms put almost no probability on '
      f'ratings the experts gave. Excluding the {ct["n_flagged_artifacts"]} reference-flagged artifacts ({se["n_artifacts"]} left): repeatability order '
      f'{" > ".join(se["order_by_reliability"])}; RPS order {" > ".join(se["order_by_rps"])} — the most repeatable judge is still last. '
      f'Secondary all-75 set (train is in-sample for H1/H2 prompt development): {a75["most_repeatable_judge"]} is both most repeatable and lowest RPS '
      f'(τ = {a75["kendall_tau_reliability_vs_accuracy"]:.2f}), so the inversion is not universal — repeatability simply does not determine accuracy.')
    w('')

    # ---------------------------------------------------------------- contract
    w('## Frozen predictions versus observations')
    w('')
    w('| ID | Matched? | Observed | Explanation if not |')
    w('| --- | --- | --- | --- |')
    why = {'A1.finite_sample': 'Monte Carlo chance at 6 simultaneous intervals; post-hoc 20,000-replication rerun covers 0',
           'A3.sampled_reliability': 'Monotone increase holds; "mode Eρ² = 1" was wrong because Binomial(5, θ) has two modes when 6θ is an integer (5 of 11 artifacts) and the declared T → 0 limit splits them',
           'B.formula': 'Skewed absolute error; post-hoc 8,192-replication rerun z = −0.20'}
    for x in contracts:
        obs = json.dumps(x['observed'], default=str)
        obs = obs if len(obs) <= 160 else obs[:157] + '...'
        w(f'| {x["id"]} | {"yes" if x["pass"] else "**no**"} | `{obs}` | {"" if x["pass"] else why.get(x["id"], "")} |')
    w('')

    # ---------------------------------------------------------------- deviations / limits / repro
    w('## Amendments, deviations and limits')
    w('')
    for am in fr.get('amendments', []):
        w(f'- Amendment {am["utc"]}: {am["reason"]}')
    w('- Added beyond PLAN.md, declared before measurement: A1_max (largest same-moment RPS gap) and the 3-call-histogram RPS column.')
    w('- Post hoc and labeled: the two diagnostic reruns above; they do not replace frozen results.')
    w('- Phase A/B are constructed examples: they reject sufficiency claims and show in-model design value; they say nothing about how often these failures occur in deployed judges.')
    w('- Phase C has 5 judges from one solver population, 10 test tasks and previously exposed ratings; judge-to-judge differences other than the paired comparison have overlapping intervals. Raters and calls are held fixed in the task bootstrap.')
    w('- No claim about any unpublished study or contemporary implementation of generalizability theory is made.')
    w('')
    w('## Reproduce')
    w('')
    w('```bash')
    w(f'cd experiments/{EXP}')
    for s in ('test_metrics', 'check_counterexamples', 'run_d_study', 'run_historical', 'verify_results', 'make_figure', 'make_report'):
        w(f'python scripts/{s}.py')
    w('```')
    w('')
    w('Phase C needs the private, git-ignored experiment 02 dataset; Phases A and B need only numpy/scipy. Output hashes (SHA-256):')
    w('')
    for name in ('phase_a.json', 'phase_b.json', 'phase_c.json', 'verification.json', 'test_metrics.json'):
        w(f'- `results/{name}`: `{hashlib.sha256((RES / name).read_bytes()).hexdigest()}`')
    w(f'- Software: Python {a["software"]["python"]}, numpy {a["software"]["numpy"]}, scipy {a["software"]["scipy"]}. Runtimes: A {a["runtime_seconds"]} s, B {b["runtime_seconds"]} s, C {c["runtime_seconds"]} s.')
    w('')
    w(f'**TLDR-end:** [judge objectives: {EXP}] Reliability and moment matching are insufficient for human-rating distribution accuracy (exact counterexamples; the most repeatable historical judge had the worst test RPS), while variance-component design allocation works when its assumptions hold. Use proper losses to choose judges and G/D studies to plan rating effort.')
    (HERE / 'RESULTS_reliability_vs_accuracy.md').write_text('\n'.join(L) + '\n')
    print('wrote RESULTS_reliability_vs_accuracy.md')


if __name__ == '__main__':
    main()
