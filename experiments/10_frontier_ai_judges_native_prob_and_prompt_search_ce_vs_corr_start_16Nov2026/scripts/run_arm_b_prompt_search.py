#!/usr/bin/env python3
"""Phase 3 (Axis iii.a): Blind Agentic Prompt Optimization (Correlation Track vs Proper-Loss CE/RPS Track).

Locked grouped 3-way split of the 19 task families:
- Train: 9 task families (27 artifacts, 69 ratings)
- Validation: 5 task families (18 artifacts, 45 ratings)
- Held-out Test: 5 task families (30 artifacts, 81 ratings)

Runs 4 iterations x 2 candidates per iteration = 8 proposed prompt candidates per track on gpt-6.1-sol,
enforcing a strict training-only feedback firewall, then selects Formulations ①, ②, ③, and ④ and
scores all finalists across all 75 artifacts (with and without 2-parameter post-hoc calibration).
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
from scipy.stats import rankdata

from common import (
    PROBABILITY_FLOOR,
    RESULTS_DIR,
    call_llm,
    ledger_summary,
    load_private_dataset_and_inputs,
    utc_now_iso,
    write_public_json,
)
from metrics_and_solvers import (
    GRID_05,
    apply_temp_tilt,
    cluster_bootstrap_ci,
    fit_temp_tilt_head,
    per_artifact_metrics,
    rating_histogram,
    summarize_panel,
)
from prompts import (
    build_ast_obligation_scaffold,
    build_proposer_prompt_ce,
    build_proposer_prompt_corr,
    enforce_rubric_word_cap,
    extract_json_objects,
    get_base_rubric_text,
    parse_probability_response,
    render_single_call_prompt,
)

LOCKED_TRAIN_FAMILIES = {
    "0_unsafeCopy_safe",
    "13_MyUppercase",
    "18_MyFibonacci",
    "1_MyAdd",
    "33_MyMinimum",
    "43_replace_spaces",
    "4_json_decoder",
    "4_unsafeCounter_safe",
    "humaneval_8_sum_product",
}
LOCKED_VAL_FAMILIES = {
    "1_unsafeMultiply_safe",
    "9_pathTraversal_safe",
    "heap_family",
    "humaneval_13__gcd",
    "humaneval_31_is_prime",
}
LOCKED_TEST_FAMILIES = {
    "5_unsafeFormatString_safe",
    "binary_search_family",
    "humaneval_3_below_zero",
    "humaneval_47_median",
    "sorting_family",
}


def eval_rubric_on_indices(
    rubric_text: str, rows: list[dict[str, Any]], indices: np.ndarray, phase_tag: str
) -> tuple[np.ndarray, list[str]]:
    """Evaluate a rubric on gpt-5.4-mini (effort=low) for specified row indices; returns (len(indices), 6) probs and rationales."""
    sub_rows = [rows[int(i)] for i in indices]

    def _one(r: dict[str, Any]) -> tuple[list[float], str]:
        sys_p, usr_p = render_single_call_prompt(
            py_name=r["py_file_name"],
            py_text=r["py_text"],
            ref_name=r["gold_file_name"],
            ref_text=r["ref_text"],
            cand_name=r["candidate_file"],
            cand_text=r["cand_text"],
            rubric_text=rubric_text,
        )
        resp = call_llm(
            model="gpt-5.4-mini",
            system=sys_p,
            user=usr_p,
            phase=phase_tag,
            effort="low",
            max_output_tokens=450,
            cache_key=f"exp10-item-{r['item']}",
            repeat_slot=1,
        )
        parsed = parse_probability_response(resp.get("text", ""))
        if not parsed["ok"] and resp.get("status") == "OK":
            resp2 = call_llm(
                model="gpt-5.4-mini",
                system=sys_p,
                user=usr_p,
                phase=phase_tag,
                effort="low",
                max_output_tokens=900,
                cache_key=f"exp10-item-{r['item']}",
                repeat_slot=99,
            )
            parsed2 = parse_probability_response(resp2.get("text", ""))
            if parsed2["ok"]:
                parsed = parsed2
        if parsed["ok"]:
            return parsed["probabilities"], parsed["rationale"]
        return [1.0 / 6.0] * 6, ""

    with ThreadPoolExecutor(max_workers=12) as pool:
        res = list(pool.map(_one, sub_rows))
    q_mat = np.array([x[0] for x in res], dtype=float)
    rats = [x[1] for x in res]
    return q_mat, rats


def build_corr_feedback_items(
    rows_tr: list[dict[str, Any]], p_tr: np.ndarray, q_tr: np.ndarray, rats_tr: list[str]
) -> list[dict[str, Any]]:
    """Build strictly standardized z-score / rank residuals for Correlation proposer (ZERO raw ratings, ZERO bias/histograms!)."""
    m_j = q_tr @ GRID_05
    m_h = p_tr @ GRID_05
    zj = (m_j - m_j.mean()) / (m_j.std() + 1e-9)
    zh = (m_h - m_h.mean()) / (m_h.std() + 1e-9)
    z_res = zj - zh
    rj = rankdata(m_j) / len(m_j)
    rh = rankdata(m_h) / len(m_h)
    order = np.argsort(-np.abs(z_res))[:6]
    out = []
    for idx in order:
        r = rows_tr[int(idx)]
        ast_s = build_ast_obligation_scaffold(r["py_text"], r["ref_text"], r["cand_text"])
        out.append(
            {
                "anonymous_train_item": f"train_item_{int(idx):02d}",
                "is_placeholder_gold": ast_s["is_placeholder_gold_reference"],
                "n_gold_theorems": ast_s["n_gold_public_theorems"],
                "n_candidate_theorems": ast_s["n_candidate_public_theorems"],
                "candidate_sorry_count": ast_s["candidate_sorry_count"],
                "z_judge_minus_z_target": round(float(z_res[int(idx)]), 3),
                "percentile_rank_gap": round(float(rj[int(idx)] - rh[int(idx)]), 3),
                "judge_rationale": rats_tr[int(idx)][:140],
            }
        )
    return out


def build_ce_feedback_items(
    rows_tr: list[dict[str, Any]], p_tr: np.ndarray, q_tr: np.ndarray, rats_tr: list[str]
) -> list[dict[str, Any]]:
    """Build proper-loss CE/RPS training diagnostics (ZERO correlation numbers!)."""
    pa = per_artifact_metrics(p_tr, q_tr, eps=PROBABILITY_FLOOR)
    order = np.argsort(-pa["ce"])[:6]
    out = []
    for idx in order:
        r = rows_tr[int(idx)]
        ast_s = build_ast_obligation_scaffold(r["py_text"], r["ref_text"], r["cand_text"])
        # Only anonymized training-item empirical probabilities (derived strictly from training split!)
        out.append(
            {
                "anonymous_train_item": f"train_item_{int(idx):02d}",
                "is_placeholder_gold": ast_s["is_placeholder_gold_reference"],
                "n_gold_theorems": ast_s["n_gold_public_theorems"],
                "n_candidate_theorems": ast_s["n_candidate_public_theorems"],
                "predicted_q_0_to_5": [round(float(x), 3) for x in q_tr[int(idx)]],
                "target_empirical_dist_0_to_5": [round(float(x), 3) for x in p_tr[int(idx)]],
                "item_ce_nats": round(float(pa["ce"][int(idx)]), 3),
                "item_rps": round(float(pa["rps"][int(idx)]), 3),
                "judge_rationale": rats_tr[int(idx)][:140],
            }
        )
    return out


def propose_two_candidates(sys_p: str, usr_p: str, fallback_parent: str, phase_tag: str) -> tuple[str, str, str]:
    resp = call_llm(
        model="gpt-6.1-sol",
        system=sys_p,
        user=usr_p,
        phase=phase_tag,
        effort="medium",
        max_output_tokens=1800,
    )
    objs = extract_json_objects(resp.get("text", ""))
    for obj in reversed(objs):
        c1 = obj.get("candidate_1")
        c2 = obj.get("candidate_2")
        if isinstance(c1, str) and len(c1.strip()) > 40 and isinstance(c2, str) and len(c2.strip()) > 40:
            return (
                enforce_rubric_word_cap(c1),
                enforce_rubric_word_cap(c2),
                str(obj.get("diagnosis", ""))[:200],
            )
    raise RuntimeError(f"Proposer failed to return valid JSON candidates: {resp.get('text', '')[:300]}")


def main() -> None:
    data = load_private_dataset_and_inputs()
    rows = data["rows"]
    assert LOCKED_TRAIN_FAMILIES | LOCKED_VAL_FAMILIES | LOCKED_TEST_FAMILIES == {r["family"] for r in rows}
    assert not (LOCKED_TRAIN_FAMILIES & LOCKED_VAL_FAMILIES)
    assert not (LOCKED_TRAIN_FAMILIES & LOCKED_TEST_FAMILIES)
    assert not (LOCKED_VAL_FAMILIES & LOCKED_TEST_FAMILIES)

    tr_idx = np.array([i for i, r in enumerate(rows) if r["family"] in LOCKED_TRAIN_FAMILIES], dtype=int)
    va_idx = np.array([i for i, r in enumerate(rows) if r["family"] in LOCKED_VAL_FAMILIES], dtype=int)
    te_idx = np.array([i for i, r in enumerate(rows) if r["family"] in LOCKED_TEST_FAMILIES], dtype=int)
    tr_va_idx = np.concatenate([tr_idx, va_idx])

    forbidden_tasks = {r["task"] for r in rows if r["family"] in (LOCKED_VAL_FAMILIES | LOCKED_TEST_FAMILIES)}
    p_all = np.array([rating_histogram(r["_private_ratings"]) for r in rows])
    rows_tr = [rows[int(i)] for i in tr_idx]

    # Iteration 0: untuned P0_human_mirror (0 extra API calls — cached from Phase 2 gpt61_sol__P0!)
    p0_text = enforce_rubric_word_cap(get_base_rubric_text("P0_human_mirror"))
    q_p0_all, rats_p0_all = eval_rubric_on_indices(p0_text, rows, np.arange(len(rows)), "phase3_iter0")
    pan_p0_tr = summarize_panel(p_all, q_p0_all, tr_idx)
    pan_p0_va = summarize_panel(p_all, q_p0_all, va_idx)
    pan_p0_te = summarize_panel(p_all, q_p0_all, te_idx)
    print(
        f"[Phase 3 Iter 0 P0] Train: r={pan_p0_tr['pearson']:.4f} CE={pan_p0_tr['ce']:.4f} RPS={pan_p0_tr['rps']:.4f} | "
        f"Test: RPS={pan_p0_te['rps']:.4f} CE={pan_p0_te['ce']:.4f}"
    )

    tracks: dict[str, list[dict[str, Any]]] = {
        "corr": [
            {
                "cand_id": "iter0_P0",
                "iteration": 0,
                "rubric": p0_text,
                "diagnosis": "Untuned verbatim human rater instructions",
                "q_tr": q_p0_all[tr_idx],
                "q_va": q_p0_all[va_idx],
                "rats_tr": [rats_p0_all[int(i)] for i in tr_idx],
                "train_metrics": pan_p0_tr,
                "val_metrics": pan_p0_va,
            }
        ],
        "ce": [
            {
                "cand_id": "iter0_P0",
                "iteration": 0,
                "rubric": p0_text,
                "diagnosis": "Untuned verbatim human rater instructions",
                "q_tr": q_p0_all[tr_idx],
                "q_va": q_p0_all[va_idx],
                "rats_tr": [rats_p0_all[int(i)] for i in tr_idx],
                "train_metrics": pan_p0_tr,
                "val_metrics": pan_p0_va,
            }
        ],
    }

    # Run 4 blind iterations x 2 candidates per track
    for track_name in ("corr", "ce"):
        parent = tracks[track_name][0]
        for it in range(1, 5):
            if track_name == "corr":
                hist = [
                    {
                        "candidate": c["cand_id"],
                        "train_pearson": round(c["train_metrics"]["pearson"], 4),
                        "train_spearman": round(c["train_metrics"]["spearman"], 4),
                        "train_kendall": round(c["train_metrics"]["kendall"], 4),
                    }
                    for c in tracks["corr"]
                ]
                worst_items = build_corr_feedback_items(
                    rows_tr, p_all[tr_idx], parent["q_tr"], parent["rats_tr"]
                )
                sys_p, usr_p = build_proposer_prompt_corr(
                    iteration=it,
                    parent_rubric=parent["rubric"],
                    history_summary=hist,
                    worst_standardized_items=worst_items,
                    forbidden_task_names=forbidden_tasks,
                )
            else:
                hist = [
                    {
                        "candidate": c["cand_id"],
                        "train_ce_nats": round(c["train_metrics"]["ce"], 4),
                        "train_rps": round(c["train_metrics"]["rps"], 4),
                    }
                    for c in tracks["ce"]
                ]
                worst_items = build_ce_feedback_items(
                    rows_tr, p_all[tr_idx], parent["q_tr"], parent["rats_tr"]
                )
                sys_p, usr_p = build_proposer_prompt_ce(
                    iteration=it,
                    parent_rubric=parent["rubric"],
                    history_summary=hist,
                    worst_ce_items=worst_items,
                    forbidden_task_names=forbidden_tasks,
                )

            r1, r2, diag = propose_two_candidates(sys_p, usr_p, parent["rubric"], f"phase3_propose_{track_name}")
            iter_cands = []
            for c_idx, rub in enumerate((r1, r2), start=1):
                cid = f"{track_name}_iter{it}_c{c_idx}"
                # Evaluate on train + val (45 items); proposer NEVER sees val_metrics!
                q_tv, rats_tv = eval_rubric_on_indices(rub, rows, tr_va_idx, f"phase3_eval_{track_name}")
                q_tr_c = q_tv[: len(tr_idx)]
                q_va_c = q_tv[len(tr_idx) :]
                rats_tr_c = rats_tv[: len(tr_idx)]
                m_tr = summarize_panel(p_all[tr_idx], q_tr_c)
                m_va = summarize_panel(p_all[va_idx], q_va_c)
                rec = {
                    "cand_id": cid,
                    "iteration": it,
                    "rubric": rub,
                    "diagnosis": diag,
                    "q_tr": q_tr_c,
                    "q_va": q_va_c,
                    "rats_tr": rats_tr_c,
                    "train_metrics": m_tr,
                    "val_metrics": m_va,
                }
                tracks[track_name].append(rec)
                iter_cands.append(rec)
                print(
                    f"[Phase 3 {cid}] Train: r={m_tr['pearson']:.4f} sp={m_tr['spearman']:.4f} "
                    f"RPS={m_tr['rps']:.4f} CE={m_tr['ce']:.4f} | Val: r={m_va['pearson']:.4f} RPS={m_va['rps']:.4f}"
                )

            # Advance parent strictly using TRAINING objective only!
            all_so_far = tracks[track_name]
            if track_name == "corr":
                parent = max(
                    all_so_far,
                    key=lambda x: (
                        x["train_metrics"]["pearson"] if np.isfinite(x["train_metrics"]["pearson"]) else -2.0,
                        x["train_metrics"]["spearman"] if np.isfinite(x["train_metrics"]["spearman"]) else -2.0,
                    ),
                )
            else:
                parent = min(all_so_far, key=lambda x: (x["train_metrics"]["ce"], x["train_metrics"]["rps"]))

    # Select Formulations ①, ②, ③, ④ strictly before held-out test evaluation!
    sel_f1_corr_val = max(tracks["corr"], key=lambda x: (x["val_metrics"]["pearson"], x["val_metrics"]["spearman"]))
    sel_f2_corr_tr = max(tracks["corr"], key=lambda x: (x["train_metrics"]["pearson"], x["train_metrics"]["spearman"]))
    sel_f3_ce_val = min(tracks["ce"], key=lambda x: (x["val_metrics"]["rps"], x["val_metrics"]["ce"]))
    sel_f4_ce_tr = min(tracks["ce"], key=lambda x: (x["train_metrics"]["ce"], x["train_metrics"]["rps"]))

    selected_map = {
        "F0_untuned_P0": tracks["corr"][0],
        "F1_corr_val_selected": sel_f1_corr_val,
        "F2_corr_train_selected": sel_f2_corr_tr,
        "F3_ce_val_rps_selected": sel_f3_ce_val,
        "F4_ce_train_ce_selected": sel_f4_ce_tr,
    }

    # Evaluate all unique selected finalists across ALL 75 artifacts (only the 30 test items aren't cached yet!)
    full75_by_cand_id: dict[str, np.ndarray] = {"iter0_P0": q_p0_all}
    for label, cand in selected_map.items():
        cid = cand["cand_id"]
        if cid not in full75_by_cand_id:
            q_te, _ = eval_rubric_on_indices(cand["rubric"], rows, te_idx, "phase3_heldout_test")
            q_full = np.zeros((len(rows), 6), dtype=float)
            q_full[tr_idx] = cand["q_tr"]
            q_full[va_idx] = cand["q_va"]
            q_full[te_idx] = q_te
            full75_by_cand_id[cid] = q_full

    # 2x3 Factorial "Prompts vs Post-Hoc Calibration Weights" Decomposition on the 30 Held-Out Test Artifacts!
    test_fams = [rows[int(i)]["family"] for i in te_idx]
    fam_to_local_te = {
        g: np.array([k for k, gi in enumerate(test_fams) if gi == g], dtype=int) for g in sorted(set(test_fams))
    }

    factorial_results: dict[str, Any] = {}
    heldout_q_by_condition: dict[str, np.ndarray] = {}

    for label, cand in selected_map.items():
        q_full = full75_by_cand_id[cand["cand_id"]]
        q_tr_f, q_te_f = q_full[tr_idx], q_full[te_idx]

        # (a) Raw prompt forecast
        heldout_q_by_condition[f"{label}__raw"] = q_te_f
        # (b) Post-hoc 2-param calibration fitted by Pearson on train
        (a_corr, b_corr), cal_corr_meta = fit_temp_tilt_head(q_tr_f, p_all[tr_idx], objective="pearson")
        q_te_cal_corr = apply_temp_tilt(q_te_f, a_corr, b_corr)
        heldout_q_by_condition[f"{label}__cal_pearson"] = q_te_cal_corr
        # (c) Post-hoc 2-param calibration fitted by convex CE on train
        (a_ce, b_ce), cal_ce_meta = fit_temp_tilt_head(q_tr_f, p_all[tr_idx], objective="ce")
        q_te_cal_ce = apply_temp_tilt(q_te_f, a_ce, b_ce)
        heldout_q_by_condition[f"{label}__cal_ce"] = q_te_cal_ce

        factorial_results[label] = {
            "selected_candidate_id": cand["cand_id"],
            "selected_iteration": cand["iteration"],
            "word_count": len(cand["rubric"].split()),
            "train_in_sample": summarize_panel(p_all[tr_idx], q_tr_f),
            "val_selection_split": summarize_panel(p_all[va_idx], q_full[va_idx]),
            "heldout_test_raw": summarize_panel(p_all[te_idx], q_te_f),
            "heldout_test_cal_pearson": {
                **summarize_panel(p_all[te_idx], q_te_cal_corr),
                "calibration_params": cal_corr_meta,
            },
            "heldout_test_cal_ce": {
                **summarize_panel(p_all[te_idx], q_te_cal_ce),
                "calibration_params": cal_ce_meta,
            },
        }

    # Compute 10,000-draw task-family bootstrap intervals on held-out test contrasts
    p_te = p_all[te_idx]

    def _contrast(cond_corr: str, cond_ce: str) -> dict[str, Any]:
        pa_corr = per_artifact_metrics(p_te, heldout_q_by_condition[cond_corr])
        pa_ce = per_artifact_metrics(p_te, heldout_q_by_condition[cond_ce])
        out_c = {}
        for met in ("rps", "ce", "tvd"):
            est = float(pa_corr[met].mean() - pa_ce[met].mean())
            ci = cluster_bootstrap_ci(
                lambda sampled, m=met: float(
                    np.mean(
                        np.concatenate([pa_corr[m][fam_to_local_te[g]] - pa_ce[m][fam_to_local_te[g]] for g in sampled])
                    )
                ),
                test_fams,
                draws=10000,
                seed=20261116,
            )
            out_c[f"corr_minus_ce_{met}"] = {"estimate": round(est, 6), **ci}
        return out_c

    contrasts = {
        "F1_val_corr_raw_minus_F3_val_ce_raw": _contrast("F1_corr_val_selected__raw", "F3_ce_val_rps_selected__raw"),
        "F2_train_corr_raw_minus_F4_train_ce_raw": _contrast("F2_corr_train_selected__raw", "F4_ce_train_ce_selected__raw"),
        "F1_val_corr_calCE_minus_F3_val_ce_calCE": _contrast("F1_corr_val_selected__cal_ce", "F3_ce_val_rps_selected__cal_ce"),
        "F2_train_corr_calCorr_minus_F4_train_ce_calCE": _contrast("F2_corr_train_selected__cal_pearson", "F4_ce_train_ce_selected__cal_ce"),
        "F0_untuned_raw_minus_F3_val_ce_raw": _contrast("F0_untuned_P0__raw", "F3_ce_val_rps_selected__raw"),
        "F4_train_ce_raw_minus_F3_val_ce_raw_3vs4": _contrast("F4_ce_train_ce_selected__raw", "F3_ce_val_rps_selected__raw"),
    }

    # Save public outputs (prompts, trajectories, full-75 forecasts of finalists, and held-out tables — ZERO human ratings!)
    public_trajectories = {
        tname: [
            {
                "cand_id": c["cand_id"],
                "iteration": c["iteration"],
                "word_count": len(c["rubric"].split()),
                "diagnosis": c["diagnosis"],
                "rubric_text": c["rubric"],
                "train_metrics": c["train_metrics"],
                "val_metrics": c["val_metrics"],
            }
            for c in c_list
        ]
        for tname, c_list in tracks.items()
    }

    write_public_json(
        RESULTS_DIR / "phase3_arm_b_blind_prompt_search.json",
        {
            "created_utc": utc_now_iso(),
            "split_counts": {
                "train_families": len(LOCKED_TRAIN_FAMILIES),
                "train_artifacts": int(len(tr_idx)),
                "val_families": len(LOCKED_VAL_FAMILIES),
                "val_artifacts": int(len(va_idx)),
                "test_families": len(LOCKED_TEST_FAMILIES),
                "test_artifacts": int(len(te_idx)),
            },
            "locked_families": {
                "train": sorted(LOCKED_TRAIN_FAMILIES),
                "val": sorted(LOCKED_VAL_FAMILIES),
                "test": sorted(LOCKED_TEST_FAMILIES),
            },
            "trajectories": public_trajectories,
            "selected_finalists_factorial": factorial_results,
            "heldout_test_contrasts": contrasts,
            "finalist_full75_forecasts": {
                cid: [[round(float(v), 8) for v in row_q] for row_q in q_mat]
                for cid, q_mat in full75_by_cand_id.items()
            },
            "ledger_after_phase3": ledger_summary(),
        },
    )
    print(
        "[Phase 3 Complete] Held-out Test RPS (raw): "
        f"P0={factorial_results['F0_untuned_P0']['heldout_test_raw']['rps']:.4f}, "
        f"① Corr-Val={factorial_results['F1_corr_val_selected']['heldout_test_raw']['rps']:.4f}, "
        f"② Corr-Tr={factorial_results['F2_corr_train_selected']['heldout_test_raw']['rps']:.4f}, "
        f"③ CE-Val={factorial_results['F3_ce_val_rps_selected']['heldout_test_raw']['rps']:.4f}, "
        f"④ CE-Tr={factorial_results['F4_ce_train_ce_selected']['heldout_test_raw']['rps']:.4f} | "
        f"spent=${ledger_summary()['settled_actual_usd']:.3f}"
    )


if __name__ == "__main__":
    main()
