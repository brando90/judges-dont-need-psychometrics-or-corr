#!/usr/bin/env python3
"""Recover omitted forecast failure metadata from existing caches; makes no model calls."""
from __future__ import annotations

import hashlib
import json
from typing import Any

from common import API_CACHE_DIR, RESULTS_DIR, load_private_dataset_and_inputs, summarize_forecast_missingness, write_public_json
from prompts import (
    build_ast_obligation_scaffold, extract_json_objects, get_base_rubric_text,
    parse_probability_response, render_h3_step1_audit, render_h3_step2_aggregator,
    render_single_call_prompt,
)
from run_arm_b_prompt_search import LOCKED_TRAIN_FAMILIES, LOCKED_VAL_FAMILIES


def cached_status(item: str, prompt: tuple[str, str], model: str, effort: str | None,
                  tokens: int, repair: bool = False, audit: bool = False) -> dict[str, Any]:
    """Use exact historical call fingerprints; absent caches are explicitly unresolved."""
    for tok, slot in ((tokens, 1), (900, 99)) if repair else ((tokens, 1),):
        request = dict(provider="mistral" if model == "codestral-2508" else "koyejolab-openai-bm",
                       model=model, effort=effort, max_output_tokens=tok, temperature=None,
                       system=prompt[0], user=prompt[1], repeat_slot=slot, mode="text")
        fingerprint = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
        path = API_CACHE_DIR / f"{fingerprint}.json"
        if not path.exists():
            return {"item": item, "status": "PROVENANCE_UNAVAILABLE", "fallback_uniform_used": None,
                    "fallback_scaffold_used": None}
        response = json.loads(path.read_text())
        ok = (any("obligations" in obj or "overall_gap_summary" in obj
                  for obj in extract_json_objects(response.get("text", ""))) if audit
              else parse_probability_response(response.get("text", ""))["ok"])
        if ok:
            return {"item": item, "status": "OK", "fallback_uniform_used": False,
                    "fallback_scaffold_used": False}
    return {"item": item, "status": "PARSE_ERROR", "fallback_uniform_used": not audit,
            "fallback_scaffold_used": audit}


def single_prompt(row: dict[str, Any], rubric: str, scaffold: bool = False) -> tuple[str, str]:
    return render_single_call_prompt(
        py_name=row["py_file_name"], py_text=row["py_text"], ref_name=row["gold_file_name"],
        ref_text=row["ref_text"], cand_name=row["candidate_file"], cand_text=row["cand_text"],
        rubric_text=rubric,
        ast_scaffold=build_ast_obligation_scaffold(row["py_text"], row["ref_text"], row["cand_text"]) if scaffold else None,
    )



def main() -> None:
    rows = load_private_dataset_and_inputs()["rows"]
    phase3_path = RESULTS_DIR / "phase3_arm_b_blind_prompt_search.json"
    phase4_path = RESULTS_DIR / "phase4_arm_c_harnesses.json"
    p2 = json.loads((RESULTS_DIR / "judge_forecasts.json").read_text())
    p3 = json.loads(phase3_path.read_text())
    p4 = json.loads(phase4_path.read_text())
    rubrics = {c["cand_id"]: c["rubric_text"] for track in p3["trajectories"].values() for c in track}
    status3 = {}
    for cid, rubric in rubrics.items():
        subset = rows if cid in p3["finalist_full75_forecasts"] else [r for r in rows if r["family"] in LOCKED_TRAIN_FAMILIES | LOCKED_VAL_FAMILIES]
        status3[cid] = [cached_status(r["item"], single_prompt(r, rubric), "gpt-5.4-mini", "low", 450, repair=True) for r in subset]
    p3["finalist_forecast_status"] = {cid: status3[cid] for cid in p3["finalist_full75_forecasts"]}
    for track in p3["trajectories"].values():
        for candidate in track:
            records = {r["item"]: r for r in status3[candidate["cand_id"]]}
            for name, families in (("train", LOCKED_TRAIN_FAMILIES), ("validation", LOCKED_VAL_FAMILIES)):
                candidate[f"{name}_forecast_status"] = [records[r["item"]] for r in rows if r["family"] in families]
    p3["forecast_missingness_summary"] = {cid: summarize_forecast_missingness(recs, rows, LOCKED_TRAIN_FAMILIES, LOCKED_VAL_FAMILIES) for cid, recs in p3["finalist_forecast_status"].items()}

    variants = {"P0_untuned": get_base_rubric_text("P0_human_mirror")}
    variants.update({key: rubrics[cid] for key, cid in p4["prompt_variant_ids"].items()})
    statuses = {}
    for key in p4["harness_forecasts"]:
        harness, prompt_key = key.split("__", 1)
        rubric = variants[prompt_key]
        if harness == "H1_single_gpt54mini":
            cid = "iter0_P0" if prompt_key == "P0_untuned" else p4["prompt_variant_ids"][prompt_key]
            statuses[key] = status3[cid]
        elif harness.startswith("H1_single"):
            model = "gpt-6.1-sol" if harness == "H1_single_gpt61" else "codestral-2508"
            if rubric == variants["P0_untuned"]:
                jid = "gpt61_sol__P0" if model == "gpt-6.1-sol" else "codestral_2508__P0"
                statuses[key] = [{k: rec[k] for k in ("item", "status", "fallback_uniform_used")} for rec in p2["native_prob_forecasts"][jid]]
            else:
                statuses[key] = [cached_status(r["item"], single_prompt(r, rubric), model, "low" if model == "gpt-6.1-sol" else None, 260 if model == "gpt-6.1-sol" else 350) for r in rows]
        elif harness.startswith("H2_ast"):
            model = "gpt-5.4-mini" if harness == "H2_ast_gpt61" else "codestral-2508"
            statuses[key] = [cached_status(r["item"], single_prompt(r, rubric, True), model, "low" if model == "gpt-5.4-mini" else None, 450, repair=True) for r in rows]
        elif harness == "H3_twostep_gpt61":
            statuses[key] = [cached_status(r["item"], render_h3_step2_aggregator(r["task"], r["candidate_file"], p4["step1_obligation_audits"][i]["audit"], rubric), "gpt-5.4-mini", "low", 400) for i, r in enumerate(rows)]
    statuses["H3_step1_audits"] = [cached_status(r["item"], render_h3_step1_audit(r["py_file_name"], r["py_text"], r["gold_file_name"], r["ref_text"], r["candidate_file"], r["cand_text"]), "gpt-5.4-mini", "low", 850, audit=True) for r in rows]
    for key, recs in statuses.items():
        if key.startswith("H3_twostep"):
            for i, rec in enumerate(recs):
                rec["upstream_fallback_used"] = statuses["H3_step1_audits"][i]["fallback_scaffold_used"]
    for key in p4["harness_forecasts"]:
        harness, prompt_key = key.split("__", 1)
        if not harness.startswith("H4"):
            continue
        members = ["H1_single_gpt61", "H1_single_gpt54mini", "H1_single_mistral35"]
        if harness == "H4_crossfamily_equal":
            members += ["H2_ast_gpt61", "H3_twostep_gpt61"]
        records = []
        for i, row in enumerate(rows):
            component_records = [statuses[f"{member}__{prompt_key}"][i] for member in members]
            unavailable = any(r["status"] == "PROVENANCE_UNAVAILABLE" or r.get("upstream_fallback_used", False) is None for r in component_records)
            records.append({"item": row["item"], "status": "PROVENANCE_UNAVAILABLE" if unavailable else "DERIVED", "fallback_uniform_used": False,
                            "component_fallback_used": None if unavailable else any(r.get("fallback_uniform_used", False) or r.get("upstream_fallback_used", False) for r in component_records)})
        statuses[key] = records
    p4["harness_forecast_status"] = statuses
    p4["forecast_missingness_summary"] = {key: summarize_forecast_missingness(recs, rows, LOCKED_TRAIN_FAMILIES, LOCKED_VAL_FAMILIES) for key, recs in statuses.items()}
    for obj in (p3, p4):
        obj["forecast_status_provenance"] = "Post-run QA reconstruction from exact request fingerprints and cached responses; no model calls, forecasts and losses unchanged. Missing caches are PROVENANCE_UNAVAILABLE, never inferred from uniform probabilities."
    write_public_json(phase3_path, p3)
    write_public_json(phase4_path, p4)
    print(json.dumps({"phase3": p3["forecast_missingness_summary"], "phase4": p4["forecast_missingness_summary"]}, indent=2))


if __name__ == "__main__":
    main()
