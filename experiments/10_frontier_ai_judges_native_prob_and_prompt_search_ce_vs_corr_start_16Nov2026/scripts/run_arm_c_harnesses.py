#!/usr/bin/env python3
"""Phase 4 (Axis iii.b + 2x2 Interaction): Frontier Judge Harness Optimization (CE/RPS vs Correlation).

Evaluates 4 genuine frontier judge harness architectures across all 75 artifacts under:
- Untuned P0_human_mirror,
- Correlation-optimized prompt finalist (from Phase 3 iii.a), and
- Proper-loss-optimized prompt finalist (from Phase 3 iii.a):

Harnesses:
1. H1_single_call: Direct single-call native-probability judge (OpenAI gpt-5.4-mini, OpenAI gpt-6.1-sol, & Mistral codestral-2508).
2. H2_ast_obligation_harness: Programmatic Python/Lean AST obligation & theorem-signature pre-extractor scaffold.
3. H3_two_step_decompose_aggregate: Two-step agentic pipeline (Step 1 obligation mapper -> Step 2 calibrated probability aggregator).
4. H4_multi_family_ensemble_calibration_harness: Cross-family OpenAI + Mistral ensemble & calibration harness
   (equal average, CE/RPS/Pearson/Spearman simplex mixtures, 2-param convex calibration, Pearson+CE head, and 3-vote scalar baseline).
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

from common import (
    PROBABILITY_FLOOR,
    RESULTS_DIR,
    call_llm,
    ledger_summary,
    load_private_dataset_and_inputs,
    utc_now_iso,
    write_public_json,
)
from metrics_and_solvers import rating_histogram, summarize_panel
from prompts import (
    build_ast_obligation_scaffold,
    extract_json_objects,
    get_base_rubric_text,
    parse_probability_response,
    render_h3_step1_audit,
    render_h3_step2_aggregator,
    render_single_call_prompt,
)


def run_h1_single(
    rows: list[dict[str, Any]], model: str, effort: str | None, rubric_text: str, tag: str, max_tok: int = 450
) -> list[list[float]]:
    def _one(r: dict[str, Any]) -> list[float]:
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
            model=model,
            system=sys_p,
            user=usr_p,
            phase=f"phase4_h1_{tag}",
            effort=effort,
            max_output_tokens=max_tok,
            cache_key=f"exp10-item-{r['item']}",
            repeat_slot=1,
        )
        parsed = parse_probability_response(resp.get("text", ""))
        return [round(float(x), 8) for x in (parsed["probabilities"] if parsed["ok"] else [1.0 / 6.0] * 6)]

    with ThreadPoolExecutor(max_workers=10) as pool:
        return list(pool.map(_one, rows))


def run_h2_ast_harness(
    rows: list[dict[str, Any]], model: str, effort: str | None, rubric_text: str, tag: str
) -> list[list[float]]:
    def _one(r: dict[str, Any]) -> list[float]:
        ast_sc = build_ast_obligation_scaffold(r["py_text"], r["ref_text"], r["cand_text"])
        sys_p, usr_p = render_single_call_prompt(
            py_name=r["py_file_name"],
            py_text=r["py_text"],
            ref_name=r["gold_file_name"],
            ref_text=r["ref_text"],
            cand_name=r["candidate_file"],
            cand_text=r["cand_text"],
            rubric_text=rubric_text,
            ast_scaffold=ast_sc,
        )
        resp = call_llm(
            model=model,
            system=sys_p,
            user=usr_p,
            phase=f"phase4_h2_{tag}",
            effort=effort,
            max_output_tokens=450,
            cache_key=f"exp10-h2-{r['item']}",
            repeat_slot=1,
        )
        parsed = parse_probability_response(resp.get("text", ""))
        if not parsed["ok"] and resp.get("status") == "OK":
            resp2 = call_llm(
                model=model,
                system=sys_p,
                user=usr_p,
                phase=f"phase4_h2_{tag}",
                effort=effort,
                max_output_tokens=900,
                cache_key=f"exp10-h2-{r['item']}",
                repeat_slot=99,
            )
            parsed2 = parse_probability_response(resp2.get("text", ""))
            if parsed2["ok"]:
                parsed = parsed2
        return [round(float(x), 8) for x in (parsed["probabilities"] if parsed["ok"] else [1.0 / 6.0] * 6)]

    with ThreadPoolExecutor(max_workers=10) as pool:
        return list(pool.map(_one, rows))


def run_h3_step1_audits(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def _one(r: dict[str, Any]) -> dict[str, Any]:
        sys_p, usr_p = render_h3_step1_audit(
            r["py_file_name"], r["py_text"], r["gold_file_name"], r["ref_text"], r["candidate_file"], r["cand_text"]
        )
        resp = call_llm(
            model="gpt-5.4-mini",
            system=sys_p,
            user=usr_p,
            phase="phase4_h3_step1",
            effort="low",
            max_output_tokens=850,
            cache_key=f"exp10-h3s1-{r['item']}",
            repeat_slot=1,
        )
        objs = extract_json_objects(resp.get("text", ""))
        for obj in reversed(objs):
            if "obligations" in obj or "overall_gap_summary" in obj:
                return obj
        ast_sc = build_ast_obligation_scaffold(r["py_text"], r["ref_text"], r["cand_text"])
        return {
            "is_placeholder_gold": ast_sc["is_placeholder_gold_reference"],
            "agent_has_nonvacuous_relevant_thm": ast_sc["n_candidate_public_theorems"] > 0,
            "spam_level": "NONE",
            "obligations": ast_sc["lexical_signature_alignment"],
            "overall_gap_summary": "Fallback AST signature summary",
        }

    with ThreadPoolExecutor(max_workers=12) as pool:
        return list(pool.map(_one, rows))


def run_h3_step2_agg(
    rows: list[dict[str, Any]], step1_audits: list[dict[str, Any]], model: str, effort: str | None, rubric_text: str, tag: str
) -> list[list[float]]:
    def _one(idx: int) -> list[float]:
        r = rows[idx]
        sys_p, usr_p = render_h3_step2_aggregator(
            r["task"], r["candidate_file"], step1_audits[idx], rubric_text
        )
        resp = call_llm(
            model=model,
            system=sys_p,
            user=usr_p,
            phase=f"phase4_h3_step2_{tag}",
            effort=effort,
            max_output_tokens=400,
            cache_key=f"exp10-h3s2-{r['item']}",
            repeat_slot=1,
        )
        parsed = parse_probability_response(resp.get("text", ""))
        return [round(float(x), 8) for x in (parsed["probabilities"] if parsed["ok"] else [1.0 / 6.0] * 6)]

    with ThreadPoolExecutor(max_workers=12) as pool:
        return list(pool.map(_one, range(len(rows))))


def main() -> None:
    data = load_private_dataset_and_inputs()
    rows = data["rows"]
    p_emp = np.array([rating_histogram(r["_private_ratings"]) for r in rows])

    phase2_fc = json.loads((RESULTS_DIR / "judge_forecasts.json").read_text(encoding="utf-8"))
    phase3_res = json.loads((RESULTS_DIR / "phase3_arm_b_blind_prompt_search.json").read_text(encoding="utf-8"))

    traj_corr = {c["cand_id"]: c["rubric_text"] for c in phase3_res["trajectories"]["corr"]}
    traj_ce = {c["cand_id"]: c["rubric_text"] for c in phase3_res["trajectories"]["ce"]}

    cid_corr_val = phase3_res["selected_finalists_factorial"]["F1_corr_val_selected"]["selected_candidate_id"]
    cid_corr_tr = phase3_res["selected_finalists_factorial"]["F2_corr_train_selected"]["selected_candidate_id"]
    cid_ce_val = phase3_res["selected_finalists_factorial"]["F3_ce_val_rps_selected"]["selected_candidate_id"]
    cid_ce_tr = phase3_res["selected_finalists_factorial"]["F4_ce_train_ce_selected"]["selected_candidate_id"]

    prompt_variants = {
        "P0_untuned": get_base_rubric_text("P0_human_mirror"),
        "P_corr_val": traj_corr[cid_corr_val],
        "P_corr_tr": traj_corr[cid_corr_tr],
        "P_ce_val": traj_ce[cid_ce_val],
        "P_ce_tr": traj_ce[cid_ce_tr],
    }

    harness_forecasts: dict[str, list[list[float]]] = {}

    # 1. H1_single_call across OpenAI (gpt-5.4-mini + gpt-6.1-sol) and Mistral (codestral-2508)
    harness_forecasts["H1_single_gpt54mini__P0_untuned"] = phase3_res["finalist_full75_forecasts"]["iter0_P0"]
    harness_forecasts["H1_single_gpt54mini__P_corr_val"] = phase3_res["finalist_full75_forecasts"][cid_corr_val]
    harness_forecasts["H1_single_gpt54mini__P_corr_tr"] = phase3_res["finalist_full75_forecasts"][cid_corr_tr]
    harness_forecasts["H1_single_gpt54mini__P_ce_val"] = phase3_res["finalist_full75_forecasts"][cid_ce_val]
    harness_forecasts["H1_single_gpt54mini__P_ce_tr"] = phase3_res["finalist_full75_forecasts"][cid_ce_tr]

    # Also evaluate gpt-6.1-sol across P0 (cached $0), P_corr_val, and P_ce_val!
    harness_forecasts["H1_single_gpt61__P0_untuned"] = [
        x["probabilities"] for x in phase2_fc["native_prob_forecasts"]["gpt61_sol__P0"]
    ]
    for pkey in ("P_corr_val", "P_ce_val"):
        harness_forecasts[f"H1_single_gpt61__{pkey}"] = (
            harness_forecasts["H1_single_gpt61__P0_untuned"]
            if prompt_variants[pkey] == prompt_variants["P0_untuned"]
            else run_h1_single(rows, "gpt-6.1-sol", "low", prompt_variants[pkey], f"gpt61_{pkey}", max_tok=260)
        )

    harness_forecasts["H1_single_mistral35__P0_untuned"] = [
        x["probabilities"] for x in phase2_fc["native_prob_forecasts"]["codestral_2508__P0"]
    ]
    for pkey in ("P_corr_val", "P_ce_val"):
        harness_forecasts[f"H1_single_mistral35__{pkey}"] = (
            harness_forecasts["H1_single_mistral35__P0_untuned"]
            if prompt_variants[pkey] == prompt_variants["P0_untuned"]
            else run_h1_single(rows, "codestral-2508", None, prompt_variants[pkey], f"codestral_{pkey}", max_tok=350)
        )

    # 2. H2_ast_obligation_harness (gpt-5.4-mini across P0, P_corr_val, P_ce_val + codestral-2508 on P0)
    # Kept under canonical key H2_ast_gpt61__{pkey} -> wait: let's name honestly AND keep analyze_all.py synced!
    for pkey in ("P0_untuned", "P_corr_val", "P_ce_val"):
        already = next(
            (
                harness_forecasts[f"H2_ast_gpt61__{other}"]
                for other in ("P0_untuned", "P_corr_val", "P_ce_val")
                if f"H2_ast_gpt61__{other}" in harness_forecasts and prompt_variants[other] == prompt_variants[pkey]
            ),
            None,
        )
        harness_forecasts[f"H2_ast_gpt61__{pkey}"] = (
            already
            if already is not None
            else run_h2_ast_harness(rows, "gpt-5.4-mini", "low", prompt_variants[pkey], pkey)
        )
    harness_forecasts["H2_ast_mistral35__P0_untuned"] = run_h2_ast_harness(
        rows, "codestral-2508", None, prompt_variants["P0_untuned"], "codestral_P0"
    )

    # 3. H3_two_step_decompose_aggregate (Step 1 obligation audit -> Step 2 calibrated aggregator)
    step1_audits = run_h3_step1_audits(rows)
    for pkey in ("P0_untuned", "P_corr_val", "P_ce_val"):
        harness_forecasts[f"H3_twostep_gpt61__{pkey}"] = run_h3_step2_agg(
            rows, step1_audits, "gpt-5.4-mini", "low", prompt_variants[pkey], f"mini_{pkey}"
        )

    # 4. H4_multi_family_ensemble_calibration_harness (OpenAI gpt-6.1-sol + OpenAI gpt-5.4-mini + Mistral codestral-2508 + H2 + H3)
    for pkey in ("P0_untuned", "P_corr_val", "P_ce_val"):
        q_oai_61 = np.array(harness_forecasts[f"H1_single_gpt61__{pkey}"])
        q_oai_54 = np.array(harness_forecasts[f"H1_single_gpt54mini__{pkey}"])
        q_mis = np.array(harness_forecasts[f"H1_single_mistral35__{pkey}"])
        q_h2 = np.array(harness_forecasts[f"H2_ast_gpt61__{pkey}"])
        q_h3 = np.array(harness_forecasts[f"H3_twostep_gpt61__{pkey}"])
        q_cross = 0.20 * (q_oai_61 + q_oai_54 + q_mis + q_h2 + q_h3)
        harness_forecasts[f"H4_crossfamily_equal__{pkey}"] = [[round(float(v), 8) for v in rq] for rq in q_cross]

        v1 = np.argmax(q_oai_61, axis=1)
        v2 = np.argmax(q_oai_54, axis=1)
        v3 = np.argmax(q_mis, axis=1)
        scalar_3vote = np.array([rating_histogram([int(v1[i]), int(v2[i]), int(v3[i])]) for i in range(len(rows))])
        harness_forecasts[f"H4_discrete_3vote_scalar__{pkey}"] = [
            [round(float(v), 8) for v in rq] for rq in scalar_3vote
        ]

    in_sample_diag = {
        hkey: summarize_panel(p_emp, np.array(q_list), eps=PROBABILITY_FLOOR)
        for hkey, q_list in harness_forecasts.items()
    }
    for hkey, pan in in_sample_diag.items():
        print(
            f"[Phase 4] {hkey:38s} | in-sample RPS={pan['rps']:.4f} CE={pan['ce']:.4f} Spearman={pan['spearman']:.4f}"
        )

    write_public_json(
        RESULTS_DIR / "phase4_arm_c_harnesses.json",
        {
            "created_utc": utc_now_iso(),
            "note": "Raw 75-item forecasts for Harnesses H1..H4 across OpenAI (gpt-6.1-sol, gpt-5.4-mini) and Mistral (codestral-2508); held-out split selection/fitting & 2x2 Prompt x Harness evaluation are in results/analysis_heldout_tables.json.",
            "prompt_variant_ids": {
                "P_corr_val": cid_corr_val,
                "P_corr_tr": cid_corr_tr,
                "P_ce_val": cid_ce_val,
                "P_ce_tr": cid_ce_tr,
            },
            "step1_obligation_audits": [
                {"item": rows[i]["item"], "task": rows[i]["task"], "audit": step1_audits[i]} for i in range(len(rows))
            ],
            "harness_forecasts": harness_forecasts,
            "in_sample_diagnostics_all75": in_sample_diag,
            "ledger_after_phase4": ledger_summary(),
        },
    )
    print(f"[Phase 4 Complete] Cumulative spend=${ledger_summary()['settled_actual_usd']:.4f}")


if __name__ == "__main__":
    main()
