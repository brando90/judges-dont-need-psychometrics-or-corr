#!/usr/bin/env python3
"""Phase 2: Multi-family frontier native-probability & token-logprob judge pool across all 75 artifacts.
Evaluates 15 zero-exposure frontier judges across 7 models in 2 families (OpenAI & Mistral),
saves public `results/judge_forecasts.json` (zero human ratings!), and reports full-pool diagnostics.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

from common import (
    HISTORICAL_SCALAR_JUDGES,
    PROBABILITY_FLOOR,
    RESULTS_DIR,
    call_llm,
    call_logprobs_6cat,
    ledger_summary,
    load_private_dataset_and_inputs,
    utc_now_iso,
    write_public_json,
)
from metrics_and_solvers import rating_histogram, smooth_probs, summarize_panel
from prompts import get_base_rubric_text, parse_probability_response, render_single_call_prompt

ARM_A_JUDGE_SPECS: list[dict[str, Any]] = [
    {"judge_id": "gpt6_astra__P0", "model": "gpt-6-astra", "family": "OpenAI", "prompt": "P0_human_mirror", "effort": "low", "mode": "json_prob", "max_tokens": 500},
    {"judge_id": "gpt61_sol__P0", "model": "gpt-6.1-sol", "family": "OpenAI", "prompt": "P0_human_mirror", "effort": "low", "mode": "json_prob", "max_tokens": 500},
    {"judge_id": "gpt61_sol__P1", "model": "gpt-6.1-sol", "family": "OpenAI", "prompt": "P1_conventions", "effort": "low", "mode": "json_prob", "max_tokens": 500},
    {"judge_id": "gpt61_sol__P2", "model": "gpt-6.1-sol", "family": "OpenAI", "prompt": "P2_disagreement_aware", "effort": "low", "mode": "json_prob", "max_tokens": 500},
    {"judge_id": "gpt56_sol__P0", "model": "gpt-5.6-sol", "family": "OpenAI", "prompt": "P0_human_mirror", "effort": "none", "mode": "json_prob", "max_tokens": 350},
    {"judge_id": "gpt54_mini__P0", "model": "gpt-5.4-mini", "family": "OpenAI", "prompt": "P0_human_mirror", "effort": "low", "mode": "json_prob", "max_tokens": 450},
    {"judge_id": "gpt54_mini__P1", "model": "gpt-5.4-mini", "family": "OpenAI", "prompt": "P1_conventions", "effort": "low", "mode": "json_prob", "max_tokens": 450},
    {"judge_id": "gpt54_mini__P2", "model": "gpt-5.4-mini", "family": "OpenAI", "prompt": "P2_disagreement_aware", "effort": "low", "mode": "json_prob", "max_tokens": 450},
    {"judge_id": "gpt54_mini__L0_logprobs", "model": "gpt-5.4-mini", "family": "OpenAI", "prompt": "P0_human_mirror", "effort": "none", "mode": "logprobs", "max_tokens": 4},
    {"judge_id": "gpt61_terra__P0", "model": "gpt-6.1-terra", "family": "OpenAI", "prompt": "P0_human_mirror", "effort": "low", "mode": "json_prob", "max_tokens": 500},
    {"judge_id": "gpt61_terra__P2", "model": "gpt-6.1-terra", "family": "OpenAI", "prompt": "P2_disagreement_aware", "effort": "low", "mode": "json_prob", "max_tokens": 500},
    {"judge_id": "gpt56_terra__P0", "model": "gpt-5.6-terra", "family": "OpenAI", "prompt": "P0_human_mirror", "effort": "none", "mode": "json_prob", "max_tokens": 350},
    {"judge_id": "codestral_2508__P0", "model": "codestral-2508", "family": "Mistral", "prompt": "P0_human_mirror", "effort": None, "mode": "json_prob", "max_tokens": 350},
    {"judge_id": "codestral_2508__P1", "model": "codestral-2508", "family": "Mistral", "prompt": "P1_conventions", "effort": None, "mode": "json_prob", "max_tokens": 350},
    {"judge_id": "codestral_2508__P2", "model": "codestral-2508", "family": "Mistral", "prompt": "P2_disagreement_aware", "effort": None, "mode": "json_prob", "max_tokens": 350},
]


def evaluate_one_judge_item(spec: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    rubric = get_base_rubric_text(spec["prompt"])
    sys_p, usr_p = render_single_call_prompt(
        py_name=row["py_file_name"],
        py_text=row["py_text"],
        ref_name=row["gold_file_name"],
        ref_text=row["ref_text"],
        cand_name=row["candidate_file"],
        cand_text=row["cand_text"],
        rubric_text=rubric,
        logprob_mode=(spec["mode"] == "logprobs"),
    )
    ckey = f"exp10-item-{row['item']}"
    if spec["mode"] == "logprobs":
        resp = call_logprobs_6cat(
            model=spec["model"], system=sys_p, user=usr_p, phase="phase2_pool", cache_key=ckey
        )
        ok = resp.get("status") == "OK" and resp.get("probabilities") is not None
        probs = resp["probabilities"] if ok else [1.0 / 6.0] * 6
        return {
            "judge_id": spec["judge_id"],
            "item": row["item"],
            "status": "OK" if ok else resp.get("status", "PARSE_ERROR"),
            "probabilities": [round(float(x), 8) for x in probs],
            "fallback_uniform_used": not ok,
            "rationale": f"top_logprobs digit_mass={resp.get('total_digit_mass')}",
        }

    resp = call_llm(
        model=spec["model"],
        system=sys_p,
        user=usr_p,
        phase="phase2_pool",
        effort=spec["effort"],
        max_output_tokens=spec["max_tokens"],
        cache_key=ckey,
        repeat_slot=1,
    )
    parsed = parse_probability_response(resp.get("text", ""))
    # Single deterministic repair retry with increased token budget if JSON was truncated/malformed
    if not parsed["ok"] and resp.get("status") == "OK":
        resp2 = call_llm(
            model=spec["model"],
            system=sys_p,
            user=usr_p,
            phase="phase2_pool_repair",
            effort=spec["effort"],
            max_output_tokens=min(2200, spec["max_tokens"] * 2),
            cache_key=ckey,
            repeat_slot=99,
        )
        parsed2 = parse_probability_response(resp2.get("text", ""))
        if parsed2["ok"]:
            resp, parsed = resp2, parsed2

    ok = resp.get("status") == "OK" and parsed["ok"]
    probs = parsed["probabilities"] if ok else [1.0 / 6.0] * 6
    return {
        "judge_id": spec["judge_id"],
        "item": row["item"],
        "status": "OK" if ok else (resp.get("status") if resp.get("status") != "OK" else "PARSE_ERROR"),
        "probabilities": [round(float(x), 8) for x in probs],
        "exact_unit_sum_3dec": bool(parsed.get("exact_unit_sum_3dec", False)) if ok else False,
        "fallback_uniform_used": not ok,
        "rationale": parsed.get("rationale", "") if ok else "",
    }


def main() -> None:
    data = load_private_dataset_and_inputs()
    rows = data["rows"]
    p_emp = np.array([rating_histogram(r["_private_ratings"]) for r in rows])

    # Run gpt61_sol__P0 first to warm OpenAI prefix caches cleanly for gpt-6.1-sol P1/P2!
    forecasts_by_judge: dict[str, list[dict[str, Any]]] = {}
    in_sample_summary: dict[str, Any] = {}

    for spec in ARM_A_JUDGE_SPECS:
        jid = spec["judge_id"]
        with ThreadPoolExecutor(max_workers=12) as pool:
            recs = list(pool.map(lambda r: evaluate_one_judge_item(spec, r), rows))
        forecasts_by_judge[jid] = recs
        q_mat = np.array([x["probabilities"] for x in recs])
        ok_cnt = sum(1 for x in recs if x["status"] == "OK")
        pan = summarize_panel(p_emp, q_mat, eps=PROBABILITY_FLOOR)
        in_sample_summary[jid] = {
            "model": spec["model"],
            "family": spec["family"],
            "prompt": spec["prompt"],
            "mode": spec["mode"],
            "scheduled_artifacts": len(rows),
            "clean_ok_artifacts": ok_cnt,
            "fallback_count": len(rows) - ok_cnt,
            "in_sample_all75": {k: round(float(v), 6) if isinstance(v, float) and np.isfinite(v) else v for k, v in pan.items()},
        }
        tot = ledger_summary()
        print(
            f"[Phase 2] {jid:24s} ok={ok_cnt}/75 | in-sample RPS={pan['rps']:.4f} CE={pan['ce']:.4f} "
            f"Spearman={pan['spearman']:.4f} | cumulative=${tot['settled_actual_usd']:.3f}"
        )

    # Also compute historical 5 scalar judges' 3-vote histograms (public forecasts only, zero human ratings!)
    historical_forecasts: dict[str, list[list[float]]] = {}
    for hname in HISTORICAL_SCALAR_JUDGES:
        votes_list = data["historical_outputs"][hname]
        h_mat = np.array([rating_histogram(v) for v in votes_list])
        historical_forecasts[hname] = [[round(float(x), 8) for x in row_q] for row_q in h_mat]
        pan_h = summarize_panel(p_emp, h_mat, eps=PROBABILITY_FLOOR)
        in_sample_summary[hname] = {
            "model": hname,
            "family": "Historical_3Sample_Scalar",
            "prompt": hname.split("_")[0],
            "mode": "scalar_3vote_histogram",
            "scheduled_artifacts": 75,
            "clean_ok_artifacts": 75,
            "fallback_count": 0,
            "in_sample_all75": {k: round(float(v), 6) if isinstance(v, float) and np.isfinite(v) else v for k, v in pan_h.items()},
        }

    # Export public judge_forecasts.json (strictly sanitized — zero human ratings!)
    artifact_meta = [
        {
            "index": r["index"],
            "item": r["item"],
            "task": r["task"],
            "family": r["family"],
            "ordinary": r["ordinary"],
            "ambiguous": r["ambiguous"],
            "ref_placeholder_at_annotation": r["ref_placeholder_at_annotation"],
            "cand_sha256": r["cand_sha256"],
            "ref_disk_sha256": r["ref_disk_sha256"],
            "py_sha256": r["py_sha256"],
        }
        for r in rows
    ]
    write_public_json(
        RESULTS_DIR / "judge_forecasts.json",
        {
            "created_utc": utc_now_iso(),
            "dataset_sha256": data["dataset_sha256"],
            "probability_floor": PROBABILITY_FLOOR,
            "artifacts": artifact_meta,
            "arm_a_specs": ARM_A_JUDGE_SPECS,
            "native_prob_forecasts": {
                jid: [
                    {
                        "item": x["item"],
                        "status": x["status"],
                        "probabilities": x["probabilities"],
                        "fallback_uniform_used": x["fallback_uniform_used"],
                    }
                    for x in recs
                ]
                for jid, recs in forecasts_by_judge.items()
            },
            "historical_3sample_forecasts": historical_forecasts,
        },
    )
    write_public_json(
        RESULTS_DIR / "phase2_pool_in_sample_summary.json",
        {
            "created_utc": utc_now_iso(),
            "note": "All numbers in this file are IN-SAMPLE across all 75 artifacts; held-out grouped task-family evaluations are in results/analysis_heldout_tables.json.",
            "judges": in_sample_summary,
            "ledger_after_phase2": ledger_summary(),
        },
    )


if __name__ == "__main__":
    main()
