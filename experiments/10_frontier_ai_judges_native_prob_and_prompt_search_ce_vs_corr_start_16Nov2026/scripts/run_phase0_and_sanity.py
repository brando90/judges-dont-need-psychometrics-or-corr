#!/usr/bin/env python3
"""Phase 1 (Karpathy Steps 1 & 2.1–2.7):
1. Byte-level 225-file input hash check (zero calls).
2. Human leave-one-rater-out (LORO) ceiling, empirical disagreement noise floor, and dumb theorem-count baseline (zero calls).
3. Live 5-task smoke controls on gpt-6.1-sol (oracle, vacuous, blank, shuffled, label-in-prompt reproduction).
4. Live k=3 repeat stability check on the 15-artifact smoke subset.
"""
from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

from common import (
    RESULTS_DIR,
    call_llm,
    ledger_summary,
    load_private_dataset_and_inputs,
    utc_now_iso,
    write_public_json,
)
from metrics_and_solvers import (
    GRID_01,
    artifact_disagreement_diagnostics,
    icc_pair,
    kendall_safe,
    pearson_safe,
    rating_histogram,
    spearman_safe,
    summarize_panel,
)
from prompts import (
    VACUOUS_LEAN,
    build_ast_obligation_scaffold,
    get_base_rubric_text,
    parse_probability_response,
    render_label_in_prompt_check,
    render_single_call_prompt,
)

SMOKE_TASKS = (
    "0_bisect_right",
    "13_MyUppercase",
    "4_json_decoder",
    "humaneval_13__gcd",
    "0_unsafeCopy_safe",
)


def compute_human_ceiling_and_dumb_baselines(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute aggregate leave-one-expert-out ceiling, empirical RPS noise floor, and dumb baselines (zero private values exported)."""
    # Leave-one-expert-out using p1, p2, p3 (identical rater definitions to VeriBench exp 100)
    loo_by_rater = {}
    for slot_idx, rater in enumerate(("p1", "p2", "p3"), start=1):
        others = [o for o in ("p1", "p2", "p3") if o != rater]
        xs, ys = [], []
        for r in rows:
            pcols = r["_private_p_cols"]
            if pcols[rater] is not None and any(pcols[o] is not None for o in others):
                xs.append(float(pcols[rater]) / 5.0)
                ys.append(float(np.mean([pcols[o] for o in others if pcols[o] is not None])) / 5.0)
        x_arr, y_arr = np.asarray(xs), np.asarray(ys)
        loo_by_rater[f"expert_slot_{slot_idx}"] = {
            "n_items": len(xs),
            "spearman": round(spearman_safe(x_arr, y_arr), 6),
            "pearson": round(pearson_safe(x_arr, y_arr), 6),
            "kendall": round(kendall_safe(x_arr, y_arr), 6),
            "icc_A1": round(icc_pair(x_arr, y_arr, "A1"), 6),
        }

    disag = artifact_disagreement_diagnostics([r["_private_ratings"] for r in rows])
    sd_pos = int(np.sum(disag["empirical_sd_0_5"] > 0))
    sd_zero = int(np.sum(disag["empirical_sd_0_5"] == 0))

    # Dumb baselines
    p_emp = np.array([rating_histogram(r["_private_ratings"]) for r in rows])
    y_01 = p_emp @ GRID_01
    ord_mask = np.array([r["ordinary"] for r in rows], dtype=bool)

    thm_ratios = []
    cand_thm_counts = []
    for r in rows:
        ast_sc = build_ast_obligation_scaffold(r["py_text"], r["ref_text"], r["cand_text"])
        thm_ratios.append(float(ast_sc["theorem_count_ratio_capped"]))
        cand_thm_counts.append(float(ast_sc["n_candidate_public_theorems"]))
    tr_arr = np.asarray(thm_ratios)
    ct_arr = np.asarray(cand_thm_counts)

    return {
        "leave_one_expert_out": {
            "mean_spearman": round(float(np.nanmean([v["spearman"] for v in loo_by_rater.values()])), 6),
            "mean_pearson": round(float(np.nanmean([v["pearson"] for v in loo_by_rater.values()])), 6),
            "mean_kendall": round(float(np.nanmean([v["kendall"] for v in loo_by_rater.values()])), 6),
            "mean_icc_A1": round(float(np.nanmean([v["icc_A1"] for v in loo_by_rater.values()])), 6),
            "per_anonymous_slot": loo_by_rater,
        },
        "rater_disagreement_summary": {
            "total_artifacts": len(rows),
            "artifacts_with_positive_sd": sd_pos,
            "artifacts_with_unanimous_sd_zero": sd_zero,
            "ordinary_artifacts": int(np.sum(ord_mask)),
            "placeholder_artifacts": int(np.sum(~ord_mask)),
            "mean_empirical_rps_disagreement_floor_all75": round(float(np.mean(disag["empirical_rps_floor"])), 6),
            "mean_empirical_rps_disagreement_floor_sd_pos": round(
                float(np.mean(disag["empirical_rps_floor"][disag["empirical_sd_0_5"] > 0])), 6
            ),
        },
        "dumb_baselines": {
            "theorem_count_ratio_capped": {
                "all_75": {
                    "spearman": round(spearman_safe(tr_arr, y_01), 6),
                    "pearson": round(pearson_safe(tr_arr, y_01), 6),
                    "kendall": round(kendall_safe(tr_arr, y_01), 6),
                },
                "ordinary_60": {
                    "spearman": round(spearman_safe(tr_arr[ord_mask], y_01[ord_mask]), 6),
                    "pearson": round(pearson_safe(tr_arr[ord_mask], y_01[ord_mask]), 6),
                    "kendall": round(kendall_safe(tr_arr[ord_mask], y_01[ord_mask]), 6),
                },
            },
            "raw_candidate_theorem_count": {
                "all_75_spearman": round(spearman_safe(ct_arr, y_01), 6),
                "ordinary_60_spearman": round(spearman_safe(ct_arr[ord_mask], y_01[ord_mask]), 6),
            },
        },
    }


def main() -> None:
    data = load_private_dataset_and_inputs()
    rows = data["rows"]
    offline = compute_human_ceiling_and_dumb_baselines(rows)
    print("Phase 0 offline checks complete:", offline["leave_one_expert_out"]["mean_spearman"])

    smoke_rows = [r for r in rows if r["task"] in SMOKE_TASKS]
    first_by_task = {t: next(r for r in smoke_rows if r["task"] == t) for t in SMOKE_TASKS}
    rubric_p0 = get_base_rubric_text("P0_human_mirror")

    # 1. Build control jobs (oracle, vacuous, blank, shuffled) + label_in_prompt (15 items)
    jobs: list[dict[str, Any]] = []
    for i, t in enumerate(SMOKE_TASKS):
        base_r = first_by_task[t]
        next_r = first_by_task[SMOKE_TASKS[(i + 1) % len(SMOKE_TASKS)]]
        # Oracle: gold reference judged against itself
        jobs.append(
            {
                "kind": "oracle",
                "id": f"oracle__{t}",
                "ordinary": base_r["ordinary"],
                "prompt": render_single_call_prompt(
                    py_name=base_r["py_file_name"],
                    py_text=base_r["py_text"],
                    ref_name=base_r["gold_file_name"],
                    ref_text=base_r["ref_text"],
                    cand_name=f"{t}__oracle.lean",
                    cand_text=base_r["ref_text"],
                    rubric_text=rubric_p0,
                ),
                "cache_key": f"exp10-ctl-oracle-{t}",
                "repeat_slot": 1,
            }
        )
        # Blank: empty candidate
        jobs.append(
            {
                "kind": "blank",
                "id": f"blank__{t}",
                "ordinary": base_r["ordinary"],
                "prompt": render_single_call_prompt(
                    py_name=base_r["py_file_name"],
                    py_text=base_r["py_text"],
                    ref_name=base_r["gold_file_name"],
                    ref_text=base_r["ref_text"],
                    cand_name=f"{t}__blank.lean",
                    cand_text="",
                    rubric_text=rubric_p0,
                ),
                "cache_key": f"exp10-ctl-blank-{t}",
                "repeat_slot": 1,
            }
        )
        # Shuffled: cross-task candidate
        jobs.append(
            {
                "kind": "shuffled",
                "id": f"shuffled__{t}",
                "ordinary": base_r["ordinary"],
                "prompt": render_single_call_prompt(
                    py_name=base_r["py_file_name"],
                    py_text=base_r["py_text"],
                    ref_name=base_r["gold_file_name"],
                    ref_text=base_r["ref_text"],
                    cand_name=f"{t}__shuffled.lean",
                    cand_text=next_r["cand_text"],
                    rubric_text=rubric_p0,
                ),
                "cache_key": f"exp10-ctl-shuffled-{t}",
                "repeat_slot": 1,
            }
        )
        if base_r["ordinary"]:
            jobs.append(
                {
                    "kind": "vacuous",
                    "id": f"vacuous__{t}",
                    "ordinary": True,
                    "prompt": render_single_call_prompt(
                        py_name=base_r["py_file_name"],
                        py_text=base_r["py_text"],
                        ref_name=base_r["gold_file_name"],
                        ref_text=base_r["ref_text"],
                        cand_name=f"{t}__vacuous.lean",
                        cand_text=VACUOUS_LEAN,
                        rubric_text=rubric_p0,
                    ),
                    "cache_key": f"exp10-ctl-vacuous-{t}",
                    "repeat_slot": 1,
                }
            )

    # Label-in-prompt reproduction (15 smoke artifacts)
    for r in smoke_rows:
        emp_p = rating_histogram(r["_private_ratings"]).tolist()
        jobs.append(
            {
                "kind": "label_in_prompt",
                "id": f"lip__{r['item']}",
                "item": r["item"],
                "ordinary": r["ordinary"],
                "prompt": render_label_in_prompt_check(r, emp_p),
                "cache_key": f"exp10-item-{r['item']}",
                "repeat_slot": 1,
            }
        )

    # k=3 repeat stability on 15 smoke artifacts (slot 1 is standard cache key so Phase 2 reuses it at $0!)
    for r in smoke_rows:
        p_tup = render_single_call_prompt(
            py_name=r["py_file_name"],
            py_text=r["py_text"],
            ref_name=r["gold_file_name"],
            ref_text=r["ref_text"],
            cand_name=r["candidate_file"],
            cand_text=r["cand_text"],
            rubric_text=rubric_p0,
        )
        for rep in (1, 2, 3):
            jobs.append(
                {
                    "kind": "repeat_stability",
                    "id": f"rep{rep}__{r['item']}",
                    "item": r["item"],
                    "repeat": rep,
                    "ordinary": r["ordinary"],
                    "prompt": p_tup,
                    "cache_key": f"exp10-item-{r['item']}",
                    "repeat_slot": rep,
                }
            )

    def _run_one(job: dict[str, Any]) -> dict[str, Any]:
        sys_p, usr_p = job["prompt"]
        resp = call_llm(
            model="gpt-6.1-sol",
            system=sys_p,
            user=usr_p,
            phase="phase1_sanity",
            effort="low",
            max_output_tokens=500,
            cache_key=job["cache_key"],
            repeat_slot=job["repeat_slot"],
        )
        parsed = parse_probability_response(resp.get("text", ""))
        return {**job, "resp": resp, "parsed": parsed}

    with ThreadPoolExecutor(max_workers=10) as pool:
        done = list(pool.map(_run_one, jobs))

    # Summarize controls (ordinary tasks vs all)
    control_summary: dict[str, Any] = {}
    for kind in ("oracle", "vacuous", "blank", "shuffled"):
        sub_ord = [d for d in done if d["kind"] == kind and d["ordinary"] and d["parsed"]["ok"]]
        sub_all = [d for d in done if d["kind"] == kind and d["parsed"]["ok"]]
        q_ord = np.array([d["parsed"]["probabilities"] for d in sub_ord])
        q_all = np.array([d["parsed"]["probabilities"] for d in sub_all])
        target_ord = np.tile([0, 0, 0, 0, 0, 1.0] if kind == "oracle" else [1.0, 0, 0, 0, 0, 0], (len(q_ord), 1))
        pan_ord = summarize_panel(target_ord, q_ord) if len(q_ord) else {}
        control_summary[kind] = {
            "n_scheduled": sum(1 for d in done if d["kind"] == kind),
            "n_parsed_ok": len(sub_all),
            "ordinary_n": len(sub_ord),
            "ordinary_mean_expected_0_1": round(float((q_ord @ GRID_01).mean()), 6) if len(q_ord) else None,
            "ordinary_mean_expected_0_5": round(float((q_ord @ (GRID_01 * 5.0)).mean()), 6) if len(q_ord) else None,
            "ordinary_rps_vs_ideal": round(float(pan_ord["rps"]), 6) if pan_ord else None,
            "all_mean_expected_0_1": round(float((q_all @ GRID_01).mean()), 6) if len(q_all) else None,
        }

    # Summarize label_in_prompt reproduction
    lip_by_item = {d["item"]: d for d in done if d["kind"] == "label_in_prompt" and d["parsed"]["ok"]}
    lip_p_emp = np.array([rating_histogram(r["_private_ratings"]) for r in smoke_rows if r["item"] in lip_by_item])
    lip_q = np.array([lip_by_item[r["item"]]["parsed"]["probabilities"] for r in smoke_rows if r["item"] in lip_by_item])
    lip_panel = summarize_panel(lip_p_emp, lip_q)

    # Summarize k=3 repeat stability
    rep_map: dict[str, list[list[float]]] = {r["item"]: [] for r in smoke_rows}
    for d in sorted([x for x in done if x["kind"] == "repeat_stability"], key=lambda z: z["repeat"]):
        if d["parsed"]["ok"]:
            rep_map[d["item"]].append(d["parsed"]["probabilities"])
    exp_sds, bin_sds = [], []
    complete3_matrix = []
    for item_id, q_list in rep_map.items():
        if len(q_list) >= 2:
            arr = np.array(q_list)  # (k, 6)
            mus = arr @ GRID_01
            exp_sds.append(float(np.std(mus, ddof=1)))
            bin_sds.append(float(np.mean(np.std(arr, axis=0, ddof=1))))
        if len(q_list) == 3:
            complete3_matrix.append([float(np.asarray(q) @ GRID_01) for q in q_list])
    # One-way test-retest ICC(1,1) across the 3 repeats
    m3 = np.array(complete3_matrix)
    n3, k3 = m3.shape
    grand = float(m3.mean())
    msr = k3 * float(((m3.mean(axis=1) - grand) ** 2).sum()) / (n3 - 1)
    msw = float(((m3 - m3.mean(axis=1, keepdims=True)) ** 2).sum()) / (n3 * (k3 - 1))
    retest_icc = float((msr - msw) / (msr + (k3 - 1) * msw)) if (msr + (k3 - 1) * msw) > 0 else float("nan")

    parse_ok_total = sum(1 for d in done if d["parsed"]["ok"])
    parse_rate = parse_ok_total / len(done)

    # Verify pre-scaling smoke gates!
    oracle_gap = min(
        control_summary["oracle"]["ordinary_mean_expected_0_1"] - control_summary[neg]["ordinary_mean_expected_0_1"]
        for neg in ("blank", "vacuous", "shuffled")
    )
    gates_passed = (
        lip_panel["pearson"] >= 0.95
        and lip_panel["tvd"] <= 0.08
        and oracle_gap >= 0.50
        and parse_rate >= 0.95
    )

    out = {
        "created_utc": utc_now_iso(),
        "hash_audit_summary": data["hash_audit_summary"],
        "offline_baselines_and_ceiling": offline,
        "smoke_controls_gpt61_sol": control_summary,
        "label_in_prompt_reproduction": {
            "n_scheduled": len(smoke_rows),
            "n_parsed_ok": len(lip_q),
            "pearson": round(lip_panel["pearson"], 6),
            "spearman": round(lip_panel["spearman"], 6),
            "tvd": round(lip_panel["tvd"], 6),
            "rps": round(lip_panel["rps"], 6),
            "ce": round(lip_panel["ce"], 6),
        },
        "repeat_stability_k3_15items": {
            "items_scheduled": len(smoke_rows),
            "items_with_all_3_repeats": len(complete3_matrix),
            "mean_expected_score_repeat_sd_0_1": round(float(np.mean(exp_sds)), 6),
            "mean_expected_score_repeat_sd_0_5": round(float(np.mean(exp_sds) * 5.0), 6),
            "p90_expected_score_repeat_sd_0_1": round(float(np.percentile(exp_sds, 90)), 6),
            "mean_per_bin_prob_repeat_sd": round(float(np.mean(bin_sds)), 6),
            "test_retest_icc_1_1": round(retest_icc, 6),
        },
        "smoke_gate_evaluation": {
            "total_calls": len(done),
            "valid_parse_rate": round(parse_rate, 6),
            "min_oracle_minus_degenerate_0_1": round(oracle_gap, 6),
            "all_smoke_gates_passed": bool(gates_passed),
        },
        "ledger_after_phase1": ledger_summary(),
    }
    write_public_json(RESULTS_DIR / "phase1_sanity_and_ceilings.json", out)
    print(
        f"Phase 1 gates_passed={gates_passed} | parse_rate={parse_rate:.3f} | "
        f"LIP r={lip_panel['pearson']:.4f}, tvd={lip_panel['tvd']:.4f} | "
        f"oracle_gap={oracle_gap:.3f} | repeat_sd_01={np.mean(exp_sds):.4f} | "
        f"spent=${out['ledger_after_phase1']['settled_actual_usd']:.4f}"
    )
    if not gates_passed:
        raise RuntimeError("Phase 1 Karpathy smoke gate FAILED — stopping before Phase 2!")


if __name__ == "__main__":
    main()
