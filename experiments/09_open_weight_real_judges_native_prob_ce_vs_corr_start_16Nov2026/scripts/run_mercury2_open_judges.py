#!/usr/bin/env python3
"""Run Experiment 09 ii.a (Prompt Optimization) and ii.b (Harness Optimization) on mercury2 (vLLM 0.11.0).

Supports:
  1. Six zero-leakage prompt designs (P0..P5) + two native probability representations (q_verbal, q_logprob)
  2. Harness H3 (Lean AST scaffold), Harness H4 (Two-call Critic -> Aggregator), Harness H5 (Thinking traces),
     Karpathy sanity controls (oracle, vacuous, blank, shuffled, label_in_prompt), and k=3 stochastic repeats
  3. Blind automated LLM Meta-Prompt Optimization (--run-meta-prompt-opt on q3_30b_a3b): 3-fold outer grouped
     task-family cross-validation comparing Correlation Prompt Search (①/②) vs Proper-Loss CE Prompt Search (③/④).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import glob
import json
import math
import os
import re
import signal
import subprocess
import time
import urllib.request
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

PY_BIN = "/lfs/mercury2/0/brando9/at979/venv128/bin"
HF_MAIN = "/dfs/scratch0/brando9/at1025_hf"
HF_ALT = "/dfs/scratch0/brando9/at1131_hf"

# key -> (hf_root, repo_dir, tp, mode, max_tokens_step1, full_suite, do_controls, do_repeats)
MODELS = {
    "q3_30b_a3b": (HF_ALT, "cpatonn--Qwen3-30B-A3B-Instruct-2507-AWQ-4bit", 2, "instruct", 500, True, True, True),
    "q3_14b_think": (HF_MAIN, "Qwen--Qwen3-14B", 4, "qwen3_think", 1450, False, True, False),
    "q3_14b_nothink": (HF_MAIN, "Qwen--Qwen3-14B", 4, "qwen3_nothink", 500, False, True, False),
    "q25c_14b": (HF_MAIN, "Qwen--Qwen2.5-Coder-14B-Instruct", 4, "instruct", 500, True, True, True),
    "q25_14b_awq": (HF_ALT, "Qwen--Qwen2.5-14B-Instruct-AWQ", 2, "instruct", 500, False, False, False),
    "ds_r1_qwen_7b": (HF_MAIN, "deepseek-ai--DeepSeek-R1-Distill-Qwen-7B", 2, "r1_think", 1450, False, True, False),
    "q3_4b_2507": (HF_MAIN, "Qwen--Qwen3-4B-Instruct-2507", 2, "instruct", 500, True, True, False),
    "llama31_8b": (HF_MAIN, "unsloth--Llama-3.1-8B-Instruct", 2, "instruct", 500, False, False, False),
    "hermes3_8b": (HF_MAIN, "NousResearch--Hermes-3-Llama-3.1-8B", 2, "instruct", 500, False, False, False),
    "olmo3_7b": (HF_MAIN, "allenai--Olmo-3-7B-Instruct", 2, "instruct", 500, False, False, False),
}


def resolve_snapshot(hf_root: str, repo_dir: str) -> str:
    snaps = sorted(glob.glob(f"{hf_root}/hub/models--{repo_dir}/snapshots/*"))
    if not snaps:
        raise FileNotFoundError(f"No snapshot under {hf_root}/hub/models--{repo_dir}")
    return snaps[-1]


def verify_digit_token_ids(snap_path: str) -> dict[str, int]:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(snap_path, local_files_only=True)
    digit_ids: dict[str, int] = {}
    for d in range(6):
        enc = tok.encode(str(d), add_special_tokens=False)
        assert len(enc) == 1, f"Digit {d} is not single-token in {snap_path}: {enc}"
        assert tok.decode(enc).strip() == str(d)
        digit_ids[str(d)] = int(enc[0])
    assert len(set(digit_ids.values())) == 6
    return digit_ids


def extract_json_objects(text: str) -> list[dict]:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>")[-1].strip()
    out, start, depth, in_str, esc = [], None, 0, False, False
    for i, ch in enumerate(cleaned):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    obj = json.loads(cleaned[start : i + 1])
                    if isinstance(obj, dict):
                        out.append(obj)
                except Exception:
                    pass
                start = None
    return out


def parse_verbal_response(text: str) -> dict:
    objs = extract_json_objects(text)
    for obj in reversed(objs):
        probs = obj.get("probabilities")
        score = obj.get("final_score_0_5")
        rat = obj.get("rationale", "")
        valid_probs = None
        if isinstance(probs, list) and len(probs) == 6:
            try:
                vals = [float(x) for x in probs]
                s = sum(vals)
                if all(math.isfinite(v) and v >= 0.0 for v in vals) and 0.95 <= s <= 1.05 and s > 0:
                    valid_probs = [v / s for v in vals]
            except Exception:
                valid_probs = None
        valid_score = None
        if not isinstance(score, bool) and isinstance(score, (int, float)):
            if 0 <= float(score) <= 5 and float(score).is_integer():
                valid_score = int(score)
        if valid_probs is not None or valid_score is not None:
            return {
                "status": "ok" if valid_probs is not None else "partial_scalar_only",
                "q_verbal": valid_probs,
                "scalar_score": valid_score,
                "rationale": str(rat)[:600] if isinstance(rat, str) else "",
                "checklist": {k: obj.get(k) for k in ("A", "B", "C", "D") if k in obj},
            }
    return {"status": "parse_failed", "q_verbal": None, "scalar_score": None, "rationale": "", "checklist": {}}


def post_json(port: int, payload: dict, timeout: int = 600) -> dict:
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        json.dumps(payload).encode("utf-8"),
        {"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def score_one_cell(
    port: int,
    model_key: str,
    mode: str,
    max_toks: int,
    digit_ids: dict[str, int],
    row: dict,
    temperature: float = 0.0,
) -> dict:
    t0 = time.time()
    user_content = row["user"]

    # If Two-Call Critic -> Aggregator Harness (H4), run Stage 1 Critic first!
    critic_summary = None
    if row["kind"] == "harness_H4_two_call":
        c_body: dict = {
            "model": model_key,
            "messages": [
                {"role": "system", "content": row["critic_system"]},
                {"role": "user", "content": row["user"]},
            ],
            "temperature": 0.0,
            "max_tokens": 350,
        }
        if mode in ("qwen3_nothink", "qwen3_think"):
            c_body["chat_template_kwargs"] = {"enable_thinking": False}
        try:
            rc = post_json(port, c_body, timeout=300)
            critic_summary = (rc["choices"][0]["message"].get("content") or "").strip()[:1200]
            user_content = (
                f"<stage1_theorem_obligation_audit>\n{critic_summary}\n</stage1_theorem_obligation_audit>\n\n"
                + row["user"]
            )
        except Exception as exc:
            critic_summary = f"CRITIC_ERROR: {exc!r}"

    msgs = [
        {"role": "system", "content": row["system"]},
        {"role": "user", "content": user_content},
    ]
    body1: dict = {
        "model": model_key,
        "messages": msgs,
        "temperature": temperature,
        "max_tokens": max_toks,
    }
    if temperature > 0:
        body1["top_p"] = 0.95
    if mode == "qwen3_nothink":
        body1["chat_template_kwargs"] = {"enable_thinking": False}
    elif mode == "qwen3_think":
        body1["chat_template_kwargs"] = {"enable_thinking": True}

    err1 = None
    raw_text = ""
    usage1 = None
    for _ in range(2):
        try:
            r1 = post_json(port, body1)
            raw_text = r1["choices"][0]["message"].get("content") or ""
            usage1 = r1.get("usage")
            err1 = None
            break
        except Exception as exc:
            err1 = repr(exc)
            time.sleep(1)

    parsed = (
        parse_verbal_response(raw_text)
        if err1 is None
        else {"status": "http_error", "q_verbal": None, "scalar_score": None, "rationale": "", "checklist": {}}
    )

    rationale_clean = parsed["rationale"]
    if not rationale_clean and raw_text:
        tail = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()[-350:]
        rationale_clean = re.sub(r'[\r\n"\\]+', " ", tail).strip()
    assistant_prefix = json.dumps({"rationale": rationale_clean})[:-1] + ', "final_score_0_5": '

    logit_bias = {str(tid): 100.0 for tid in digit_ids.values()}
    msgs2 = msgs + [{"role": "assistant", "content": assistant_prefix}]
    body2: dict = {
        "model": model_key,
        "messages": msgs2,
        "continue_final_message": True,
        "add_generation_prompt": False,
        "max_tokens": 1,
        "temperature": 0.0,
        "logprobs": True,
        "top_logprobs": 20,
        "logit_bias": logit_bias,
    }
    if mode in ("qwen3_nothink", "qwen3_think"):
        body2["chat_template_kwargs"] = {"enable_thinking": False}

    q_logprob = None
    raw_digit_logprobs = None
    err2 = None
    for _ in range(2):
        try:
            r2 = post_json(port, body2, timeout=180)
            tlp = r2["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
            lp_by_digit: dict[str, float] = {}
            for entry in tlp:
                tok_str = entry["token"].strip()
                if tok_str in ("0", "1", "2", "3", "4", "5") and tok_str not in lp_by_digit:
                    lp_by_digit[tok_str] = float(entry["logprob"])
            if len(lp_by_digit) >= 1:
                min_seen = min(lp_by_digit.values())
                # Any digit outside top-20 has strictly smaller logprob than rank-20 token
                rank20_lp = min(float(entry["logprob"]) for entry in tlp) if tlp else (min_seen - 10.0)
                tail_lp = min(min_seen, rank20_lp) - 2.0
                full_lps = [lp_by_digit.get(str(d), tail_lp) for d in range(6)]
                max_lp = max(full_lps)
                exps = [math.exp(lp - max_lp) for lp in full_lps]
                s_exp = sum(exps)
                q_logprob = [v / s_exp for v in exps]
                raw_digit_logprobs = full_lps
                err2 = None
                break
            err2 = f"no_digits_in_top20:{[e['token'] for e in tlp[:5]]}"
        except Exception as exc:
            err2 = repr(exc)
            time.sleep(1)

    return {
        "cell_id": row["cell_id"],
        "kind": row["kind"],
        "prompt_id": row["prompt_id"],
        "harness_id": row.get("harness_id", ""),
        "item": row["item"],
        "task": row["task"],
        "family": row["family"],
        "expected_synthetic_label": row.get("expected_synthetic_label"),
        "model": model_key,
        "temperature": temperature,
        "verbal_status": parsed["status"],
        "q_verbal": parsed["q_verbal"],
        "scalar_score": parsed["scalar_score"],
        "checklist": parsed["checklist"],
        "rationale": parsed["rationale"],
        "critic_summary": critic_summary[:400] if critic_summary else None,
        "logprob_status": "ok" if q_logprob is not None else "failed",
        "q_logprob": q_logprob,
        "raw_digit_logprobs": raw_digit_logprobs,
        "err_step1": err1,
        "err_step2": err2,
        "usage_step1": usage1,
        "elapsed_s": round(time.time() - t0, 3),
    }


def eval_prompt_metrics(q_mat: np.ndarray, p_mat: np.ndarray) -> dict[str, float]:
    floor = 1e-6
    qf = np.clip(q_mat, floor, None)
    qf = qf / qf.sum(axis=1, keepdims=True)
    grid = np.arange(6, dtype=float)
    mu = qf @ grid
    hm = p_mat @ grid
    ce_val = float(-(p_mat * np.log(qf)).sum(axis=1).mean())
    cq = np.cumsum(qf, axis=1)[:, :5]
    cp = np.cumsum(p_mat, axis=1)[:, :5]
    rps_val = float((((cq - cp) ** 2).mean(axis=1) + (cp * (1.0 - cp)).mean(axis=1)).mean())
    pear = float(np.corrcoef(mu, hm)[0, 1]) if mu.std() > 1e-12 and hm.std() > 1e-12 else 0.0
    spear = float(spearmanr(mu, hm).statistic) if mu.std() > 1e-12 and hm.std() > 1e-12 else 0.0
    bias = float((mu - hm).mean())
    return {"rps": rps_val, "ce": ce_val, "pearson": pear, "spearman": spear, "bias_0_5": bias, "judge_mean": float(mu.mean())}


def run_blind_meta_prompt_opt(
    port: int,
    model_key: str,
    digit_ids: dict[str, int],
    lora_bundle_path: str,
    out_dir: Path,
    conc: int = 8,
) -> None:
    """3-fold grouped task-family blind iterative Meta-Prompt Optimization (ii.a):
    Compares Track Corr (①/②: sees ONLY train Pearson/Spearman & z-scored rank errors)
    against Track CE (③/④: sees ONLY train individual-rating CE/RPS, bias & spread errors).
    Saves ONLY prompt texts, training/val summaries, and held-out test model probabilities (zero human ratings!)."""
    meta_out = out_dir / "meta_prompt_opt_q3_30b.json"
    if meta_out.exists():
        print("[meta_prompt_opt] already exists, skipping.", flush=True)
        return

    bundle = json.loads(Path(lora_bundle_path).read_text(encoding="utf-8"))
    items = bundle["items"]
    instructions = bundle["rater_instructions"]
    json_spec = bundle["json_output_spec"]

    p_mat = np.array([[c / it["n_ratings"] for c in it["counts_0_5"]] for it in items], dtype=float)
    fams = [it["family"] for it in items]
    unique_fams = sorted(set(fams))
    rng = np.random.default_rng(20260901)
    perm = [unique_fams[i] for i in rng.permutation(len(unique_fams))]
    # 3 outer grouped task-family folds covering all 19 families (~6, 6, 7 test families -> all 75 artifacts tested held-out once!)
    outer_3 = [sorted(perm[0:6]), sorted(perm[6:12]), sorted(perm[12:19])]

    def make_sys(addendum: str) -> str:
        extra = f"<optimization_rubric_guidance>\n{addendum.strip()}\n</optimization_rubric_guidance>\n\n" if addendum.strip() else ""
        return (
            "You are an expert rater in a human-judgment study of Lean 4 formal specification coverage. "
            "You receive the exact same three files the human raters received.\n\n"
            f"<rater_instructions>\n{instructions}\n</rater_instructions>\n\n"
            + extra
            + json_spec
        )

    def score_subset(addendum: str, indices: list[int], tag: str) -> tuple[np.ndarray, list[str]]:
        sys_p = make_sys(addendum)
        work = [
            {
                "cell_id": f"{tag}__{idx}",
                "kind": "meta_prompt",
                "prompt_id": tag,
                "item": items[idx]["item"],
                "task": items[idx]["task"],
                "family": items[idx]["family"],
                "system": sys_p,
                "user": items[idx]["user_plain"],
            }
            for idx in indices
        ]
        with cf.ThreadPoolExecutor(max_workers=conc) as ex:
            res = list(ex.map(lambda r: score_one_cell(port, model_key, "instruct", 480, digit_ids, r, 0.0), work))
        # Use q_verbal when valid (since prompt instructions directly shape verbalized probabilities!), fallback q_logprob or uniform
        qs = []
        rats = []
        for r in res:
            q = r["q_verbal"] if r["q_verbal"] is not None else (r["q_logprob"] if r["q_logprob"] is not None else [1 / 6] * 6)
            qs.append(q)
            rats.append(r["rationale"])
        return np.array(qs, dtype=float), rats

    folds_out = []
    for f_idx, test_fams in enumerate(outer_3):
        rem_fams = sorted(set(unique_fams) - set(test_fams))
        val_fams = rem_fams[:4]
        tr_fams = rem_fams[4:]
        tr_idx = [i for i, it in enumerate(items) if it["family"] in tr_fams]
        val_idx = [i for i, it in enumerate(items) if it["family"] in val_fams]
        te_idx = [i for i, it in enumerate(items) if it["family"] in test_fams]
        print(f"[meta_prompt_opt] Fold {f_idx}: tr={len(tr_idx)}, val={len(val_idx)}, te={len(te_idx)}", flush=True)

        # Evaluate P0 seed (empty addendum) on train + val
        q_tr0, rat_tr0 = score_subset("", tr_idx, f"f{f_idx}_p0_tr")
        q_val0, _ = score_subset("", val_idx, f"f{f_idx}_p0_val")
        m_tr0 = eval_prompt_metrics(q_tr0, p_mat[tr_idx])
        m_val0 = eval_prompt_metrics(q_val0, p_mat[val_idx])

        tracks_record = {}
        for track in ("corr", "ce"):
            cands = [{"id": "cand_0_seed_P0", "addendum": "", "train_metrics": m_tr0, "val_metrics": m_val0, "q_tr": q_tr0, "rats": rat_tr0}]
            for rnd in (1, 2):
                best_so_far = (
                    max(cands, key=lambda c: c["train_metrics"]["pearson"])
                    if track == "corr"
                    else min(cands, key=lambda c: c["train_metrics"]["ce"])
                )
                q_cur = best_so_far["q_tr"]
                mu_cur = q_cur @ np.arange(6, dtype=float)
                hm_tr = p_mat[tr_idx] @ np.arange(6, dtype=float)

                if track == "corr":
                    # ONLY standardized z-score ordering feedback — blind to absolute bias, scale, and spread!
                    z_pred = (mu_cur - mu_cur.mean()) / (mu_cur.std() + 1e-8)
                    z_hum = (hm_tr - hm_tr.mean()) / (hm_tr.std() + 1e-8)
                    z_err = np.abs(z_pred - z_hum)
                    worst_local = np.argsort(-z_err)[:4]
                    err_lines = [
                        f"- Task `{items[tr_idx[k]]['task']}`: judge standardized z-score={z_pred[k]:+.2f} vs target standardized z-score={z_hum[k]:+.2f} (rationale: {best_so_far['rats'][k][:140]})"
                        for k in worst_local
                    ]
                    meta_prompt = (
                        "You are optimizing a rubric addendum for an LLM Lean-4 specification coverage judge. "
                        "Your SOLE objective is to MAXIMIZE Pearson and Spearman correlation (relative ordering across items).\n"
                        f"Current addendum:\n{best_so_far['addendum'] or '(none — base instructions only)'}\n"
                        f"Current training Pearson={best_so_far['train_metrics']['pearson']:.3f}, Spearman={best_so_far['train_metrics']['spearman']:.3f}.\n"
                        "Largest standardized ordering mismatches on training items:\n" + "\n".join(err_lines) + "\n\n"
                        "Propose TWO distinct improved rubric addenda (each 4-7 concise bullet sentences, no item-specific names) "
                        "that sharpen relative ranking between strong, partial, implementation-restatement, and placeholder items. "
                        'Respond ONLY in JSON: {"candidate_A": "...", "candidate_B": "..."}'
                    )
                else:
                    # Proper-loss (CE + RPS) feedback — sees overconfident zero/low-mass bins, leniency bias, and rater spread!
                    qf = np.clip(q_cur, 1e-6, None)
                    qf /= qf.sum(axis=1, keepdims=True)
                    item_ce = -(p_mat[tr_idx] * np.log(qf)).sum(axis=1)
                    worst_local = np.argsort(-item_ce)[:4]
                    err_lines = [
                        f"- Task `{items[tr_idx[k]]['task']}`: judge forecast={[round(x,2) for x in q_cur[k]]} (mean {mu_cur[k]:.2f}), item log-loss={item_ce[k]:.2f} nats (rationale: {best_so_far['rats'][k][:140]})"
                        for k in worst_local
                    ]
                    meta_prompt = (
                        "You are optimizing a rubric addendum for an LLM Lean-4 specification coverage judge. "
                        "Your SOLE objective is to MINIMIZE individual-rating cross-entropy (log-loss) and ranked probability score (RPS) over ratings 0..5.\n"
                        f"Current addendum:\n{best_so_far['addendum'] or '(none — base instructions only)'}\n"
                        f"Current training CE={best_so_far['train_metrics']['ce']:.3f} nats, RPS={best_so_far['train_metrics']['rps']:.3f}, "
                        f"signed leniency bias (judge minus human)={best_so_far['train_metrics']['bias_0_5']:+.2f} rating points.\n"
                        "Highest cross-entropy loss training items (note overconfident extreme probabilities and leniency bias):\n"
                        + "\n".join(err_lines)
                        + "\n\nPropose TWO distinct improved rubric addenda (each 4-7 concise bullet sentences, no item-specific names) "
                        "that fix probability calibration, eliminate leniency bias, keep non-zero uncertainty mass across plausible ratings when coverage is partial, and penalize implementation restatements. "
                        'Respond ONLY in JSON: {"candidate_A": "...", "candidate_B": "..."}'
                    )

                proposals = []
                try:
                    prop_payload = {
                        "model": model_key,
                        "messages": [{"role": "user", "content": meta_prompt}],
                        "temperature": 0.4,
                        "max_tokens": 900,
                    }
                    if "q3_" in model_key or "qwen3" in model_key:
                        prop_payload["chat_template_kwargs"] = {"enable_thinking": False}
                    rm = post_json(port, prop_payload, timeout=300)
                    objs = extract_json_objects(rm["choices"][0]["message"].get("content") or "")
                    if objs:
                        for k_ab in ("candidate_A", "candidate_B"):
                            v_ab = objs[-1].get(k_ab)
                            if isinstance(v_ab, list):
                                v_ab = "\n".join(str(x) for x in v_ab if str(x).strip())
                            elif isinstance(v_ab, dict):
                                v_ab = "\n".join(f"- {k}: {v}" for k, v in v_ab.items())
                            if isinstance(v_ab, str) and len(v_ab.strip()) > 20:
                                proposals.append(v_ab.strip())
                    else:
                        print(f"[meta_prompt_opt] unparsed proposal content: {(rm['choices'][0]['message'].get('content') or '')[:200]!r}", flush=True)
                except Exception as exc:
                    print(f"[meta_prompt_opt] proposal err: {exc!r}", flush=True)

                for p_i, add_txt in enumerate(proposals[:2]):
                    cid = f"cand_r{rnd}_{p_i}"
                    q_tr_c, rat_tr_c = score_subset(add_txt, tr_idx, f"f{f_idx}_{track}_{cid}_tr")
                    q_val_c, _ = score_subset(add_txt, val_idx, f"f{f_idx}_{track}_{cid}_val")
                    cands.append(
                        {
                            "id": cid,
                            "addendum": add_txt,
                            "train_metrics": eval_prompt_metrics(q_tr_c, p_mat[tr_idx]),
                            "val_metrics": eval_prompt_metrics(q_val_c, p_mat[val_idx]),
                            "q_tr": q_tr_c,
                            "rats": rat_tr_c,
                        }
                    )

            # Select winning prompt under train-only (② or ④) and inner-validation (① or ③)
            if track == "corr":
                sel_train = max(cands, key=lambda c: c["train_metrics"]["pearson"])
                sel_val = max(cands, key=lambda c: c["val_metrics"]["pearson"])
            else:
                sel_train = min(cands, key=lambda c: c["train_metrics"]["ce"])
                sel_val = min(cands, key=lambda c: c["val_metrics"]["rps"])

            # Score unique selected prompts + seed on sealed held-out test fold ONCE
            needed = {sel_train["id"]: sel_train["addendum"], sel_val["id"]: sel_val["addendum"], "cand_0_seed_P0": ""}
            test_forecasts = {}
            for cid, add_txt in needed.items():
                q_te_c, _ = score_subset(add_txt, te_idx, f"f{f_idx}_{track}_{cid}_te")
                test_forecasts[cid] = q_te_c.tolist()

            tracks_record[track] = {
                "candidates": [
                    {"id": c["id"], "addendum": c["addendum"], "train_metrics": c["train_metrics"], "val_metrics": c["val_metrics"]}
                    for c in cands
                ],
                "selected_train_id": sel_train["id"],
                "selected_val_id": sel_val["id"],
                "test_forecasts_by_id": test_forecasts,
            }

        folds_out.append(
            {
                "fold": f_idx,
                "test_families": test_fams,
                "val_families": val_fams,
                "train_families": tr_fams,
                "test_indices": te_idx,
                "tracks": tracks_record,
            }
        )

    meta_out.write_text(json.dumps({"model": model_key, "folds": folds_out}, indent=2) + "\n", encoding="utf-8")
    print("[meta_prompt_opt] COMPLETE.", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--lora-bundle", default="")
    ap.add_argument("--run-meta-prompt-opt", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--conc", type=int, default=8)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows = [json.loads(line) for line in Path(args.prompts).read_text(encoding="utf-8").splitlines() if line.strip()]

    for key in [m.strip() for m in args.models.split(",") if m.strip()]:
        hf_root, repo_dir, tp, mode, max_toks, full_suite, do_ctl, do_rep = MODELS[key]
        done_flag = out_dir / f"{key}.done"
        out_jsonl = out_dir / f"{key}.jsonl"
        need_meta = args.run_meta_prompt_opt and key == "q3_30b_a3b" and not (out_dir / "meta_prompt_opt_q3_30b.json").exists()
        if done_flag.exists() and out_jsonl.exists() and not need_meta:
            print(f"[{key}] already done, skipping.", flush=True)
            continue

        snap = resolve_snapshot(hf_root, repo_dir)
        digit_ids = verify_digit_token_ids(snap)
        gpu_list = args.gpus.split(",")
        assert len(gpu_list) >= tp
        vis_gpus = ",".join(gpu_list[:tp])

        env = dict(
            os.environ,
            CUDA_VISIBLE_DEVICES=vis_gpus,
            HF_HUB_OFFLINE="1",
            VLLM_USE_FLASHINFER_SAMPLER="0",
            HF_HOME=hf_root,
            PATH=PY_BIN + ":" + os.environ.get("PATH", ""),
        )
        serve_log = (out_dir / f"{key}.serve.log").open("a", encoding="utf-8")
        cmd = [
            f"{PY_BIN}/vllm",
            "serve",
            snap,
            "--served-model-name",
            key,
            "--host",
            "127.0.0.1",
            "--port",
            str(args.port),
            "--max-model-len",
            "24576",
            "--tensor-parallel-size",
            str(tp),
            "--gpu-memory-utilization",
            "0.88",
            "--max-num-seqs",
            "10",
            "--dtype",
            "bfloat16",
            "--enforce-eager",
        ]
        t_start = time.time()
        print(f"[{key}] starting vLLM on GPUs {vis_gpus} port {args.port}...", flush=True)
        proc = subprocess.Popen(cmd, env=env, stdout=serve_log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            ready = False
            for _ in range(240):
                if proc.poll() is not None:
                    break
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{args.port}/v1/models", timeout=3) as r:
                        if key in r.read().decode("utf-8"):
                            ready = True
                            break
                except Exception:
                    time.sleep(3)
            if not ready:
                print(f"[{key}] FAILED to start vLLM!", flush=True)
                (out_dir / f"{key}.failed").write_text("vllm_start_failed\n", encoding="utf-8")
                continue

            if not (done_flag.exists() and out_jsonl.exists()):
                if full_suite:
                    work = [r for r in all_rows if not r["kind"].startswith("control_")]
                else:
                    work = [
                        r
                        for r in all_rows
                        if (r["kind"] == "main_prompt" and r["prompt_id"] in ("P0_human_mirror", "P1_conventions", "P2_disagreement_calibrated"))
                        or r["kind"] == "harness_H3_ast"
                    ]
                if do_ctl:
                    work.extend([r for r in all_rows if r["kind"].startswith("control_")])

                print(f"[{key}] running {len(work)} cells...", flush=True)
                results = []
                with cf.ThreadPoolExecutor(max_workers=args.conc) as ex:
                    futs = [ex.submit(score_one_cell, args.port, key, mode, max_toks, digit_ids, r, 0.0) for r in work]
                    for idx, fut in enumerate(futs, 1):
                        results.append(fut.result())
                        if idx % 75 == 0 or idx == len(futs):
                            ok_lp = sum(1 for x in results if x["logprob_status"] == "ok")
                            ok_vb = sum(1 for x in results if x["verbal_status"] == "ok")
                            print(f"[{key}] {idx}/{len(futs)} done (logprob_ok={ok_lp}, verbal_ok={ok_vb})", flush=True)

                if do_rep:
                    rep_items = [r for r in all_rows if r["kind"] == "main_prompt" and r["prompt_id"] == "P2_disagreement_calibrated"]
                    rep_work = []
                    for rep_idx in (1, 2, 3):
                        for r in rep_items:
                            rep_work.append(dict(r, cell_id=f"repeat{rep_idx}__{r['item']}", kind=f"repeat_{rep_idx}", harness_id="H6_ensemble_repeat"))
                    print(f"[{key}] running {len(rep_work)} stochastic repeat cells (T=0.7) for H6 ensemble & repeat SD...", flush=True)
                    with cf.ThreadPoolExecutor(max_workers=args.conc) as ex:
                        for res in ex.map(lambda r: score_one_cell(args.port, key, mode, max_toks, digit_ids, r, 0.7), rep_work):
                            results.append(res)

                with out_jsonl.open("w", encoding="utf-8") as fh:
                    for rec in results:
                        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                done_flag.write_text(f"{time.time() - t_start:.2f}\n", encoding="utf-8")
                print(f"[{key}] main+controls+repeats COMPLETE ({len(results)} records).", flush=True)

            if need_meta and args.lora_bundle:
                run_blind_meta_prompt_opt(args.port, key, digit_ids, args.lora_bundle, out_dir, conc=args.conc)

        finally:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                time.sleep(3)
                if proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                pass
            serve_log.close()
            time.sleep(4)


if __name__ == "__main__":
    main()
