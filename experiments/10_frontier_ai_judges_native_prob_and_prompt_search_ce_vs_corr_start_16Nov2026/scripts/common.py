#!/usr/bin/env python3
"""Shared paths, SHA-256 input verification, atomic $18.00 spend ledger, and OpenAI/Mistral API clients for Experiment 10.

Privacy & safety invariants:
- API keys are read transiently from ~/keys/ inside HTTP call helpers and NEVER printed, logged, or serialized.
- Raw human ratings live only in memory or inside gitignored private/; sanitize_for_public() blocks accidental leaks.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import subprocess
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

EXPT_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = EXPT_DIR.parents[1]
PRIVATE_DIR = EXPT_DIR / "private"
API_CACHE_DIR = PRIVATE_DIR / "api_cache"
RESULTS_DIR = EXPT_DIR / "results"
FIGURES_DIR = EXPT_DIR / "figures"

DATASET_PATH = Path(
    "/Users/brandomiranda/judges-dont-need-psychometrics-or-corr/experiments/"
    "02_exp06_ce_vs_pearson_matched_budget_start_09Nov2026/private/historical_exp06/cached_v1/dataset.json"
)
DATASET_SHA256 = "b3a05699a26d3ffd9f28e793178811219885df78319b3aa57f9d393c68724ba0"

VB_REPO_DIR = Path("/Users/brandomiranda/veribench")
VB_EXP100_DIR = VB_REPO_DIR / "experiments/100_karpathy_recipe_tc_judge_search_human_train_val_test"

OPENAI_KEY_PATH = Path.home() / "keys/openai_bm_key_koyejolab.txt"
MISTRAL_KEY_PATH = Path.home() / "keys/mistral_api_key.txt"

HARD_CAP_USD = 18.00
LEDGER_PATH = PRIVATE_DIR / "cost_ledger.jsonl"
LEDGER_LOCK_PATH = PRIVATE_DIR / "cost_ledger.lock"

HISTORICAL_SCALAR_JUDGES = ("H0_opus", "H0_sol", "H1_sol", "H2_opus", "H2_sol")
PROBABILITY_FLOOR = 1e-6

# Verified local catalog pricing (USD per 1,000,000 tokens)
MODEL_CATALOG: dict[str, dict[str, Any]] = {
    "gpt-6-astra": {
        "provider": "koyejolab-openai-bm",
        "family": "OpenAI",
        "in_per_m": 10.00,
        "out_per_m": 50.00,
        "cache_read_per_m": 1.00,
        "supports_reasoning": True,
    },
    "gpt-6.1-sol": {
        "provider": "koyejolab-openai-bm",
        "family": "OpenAI",
        "in_per_m": 2.00,
        "out_per_m": 10.00,
        "cache_read_per_m": 0.10,
        "supports_reasoning": True,
    },
    "gpt-6.1-terra": {
        "provider": "koyejolab-openai-bm",
        "family": "OpenAI",
        "in_per_m": 2.00,
        "out_per_m": 10.00,
        "cache_read_per_m": 0.10,
        "supports_reasoning": True,
    },
    "gpt-5.6-sol": {
        "provider": "koyejolab-openai-bm",
        "family": "OpenAI",
        "in_per_m": 4.00,
        "out_per_m": 20.00,
        "cache_read_per_m": 0.40,
        "supports_reasoning": True,
    },
    "gpt-5.6-terra": {
        "provider": "koyejolab-openai-bm",
        "family": "OpenAI",
        "in_per_m": 2.00,
        "out_per_m": 12.00,
        "cache_read_per_m": 0.20,
        "supports_reasoning": True,
    },
    "gpt-5.4-mini": {
        "provider": "koyejolab-openai-bm",
        "family": "OpenAI",
        "in_per_m": 0.75,
        "out_per_m": 4.50,
        "cache_read_per_m": 0.075,
        "supports_reasoning": True,
    },
    "mistral-medium-3.5": {
        "provider": "mistral",
        "family": "Mistral",
        "in_per_m": 1.50,
        "out_per_m": 7.50,
        "cache_read_per_m": 0.00,
        "supports_reasoning": True,
    },
    "magistral-medium-latest": {
        "provider": "mistral",
        "family": "Mistral",
        "in_per_m": 2.00,
        "out_per_m": 5.00,
        "cache_read_per_m": 0.20,
        "supports_reasoning": True,
    },
    "codestral-2508": {
        "provider": "mistral",
        "family": "Mistral",
        "in_per_m": 0.30,
        "out_per_m": 0.90,
        "cache_read_per_m": 0.03,
        "supports_reasoning": False,
    },
}

FORBIDDEN_PUBLIC_KEYS = {
    "ratings",
    "ratings_by_expert",
    "p1",
    "p2",
    "p3",
    "p4",
    "p5",
    "raw_ratings",
    "human_ratings",
    "api_key",
    "authorization",
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def pst_now_str() -> str:
    return subprocess.run(
        ["date", "+%m-%d-%Y %H:%M %Z"], check=True, capture_output=True, text=True
    ).stdout.strip()


def sanitize_for_public(obj: Any, path: str = "root") -> Any:
    """Recursively verify zero private human-rating arrays or secret keys exist before writing public JSON."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            kl = str(k).lower()
            if kl in FORBIDDEN_PUBLIC_KEYS:
                raise ValueError(f"Privacy guard blocked forbidden key {k!r} at {path}")
            out[k] = sanitize_for_public(v, f"{path}.{k}")
        return out
    if isinstance(obj, list):
        return [sanitize_for_public(x, f"{path}[{i}]") for i, x in enumerate(obj)]
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if isinstance(obj, float) and math.isinf(obj):
        return "Infinity" if obj > 0 else "-Infinity"
    return obj


def write_public_json(path: Path, obj: Any) -> None:
    clean = sanitize_for_public(obj)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(clean, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_private_json(path: Path, obj: Any) -> None:
    path = Path(path)
    if not path.resolve().is_relative_to(PRIVATE_DIR.resolve()):
        raise ValueError(f"write_private_json target {path} must stay inside {PRIVATE_DIR}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


@contextmanager
def _ledger_lock():
    PRIVATE_DIR.mkdir(parents=True, exist_ok=True)
    with LEDGER_LOCK_PATH.open("a", encoding="utf-8") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def _ledger_totals_unlocked() -> dict[str, Any]:
    reserved: dict[str, dict[str, Any]] = {}
    settled: dict[str, dict[str, Any]] = {}
    if LEDGER_PATH.exists():
        for line in LEDGER_PATH.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            ev = json.loads(line)
            cid = ev["call_id"]
            if ev["event"] == "reserve":
                reserved[cid] = ev
            elif ev["event"] == "settle":
                settled[cid] = ev
    effective_usd = 0.0
    settled_usd = 0.0
    by_model: dict[str, dict[str, Any]] = {}
    by_phase: dict[str, float] = {}
    for cid, r in reserved.items():
        s = settled.get(cid)
        if s is not None and s.get("actual_usd") is not None:
            cost = float(s["actual_usd"])
            settled_usd += cost
        else:
            cost = float(r["worst_usd"])
        effective_usd += cost
        m = r["model"]
        rec = by_model.setdefault(
            m,
            {
                "provider": MODEL_CATALOG[m]["provider"],
                "calls_reserved": 0,
                "calls_settled_ok": 0,
                "calls_failed": 0,
                "input_tokens": 0,
                "cached_tokens": 0,
                "output_tokens": 0,
                "reasoning_tokens": 0,
                "actual_usd": 0.0,
            },
        )
        rec["calls_reserved"] += 1
        if s is not None:
            if s.get("status") == "OK":
                rec["calls_settled_ok"] += 1
            else:
                rec["calls_failed"] += 1
            u = s.get("usage") or {}
            rec["input_tokens"] += int(u.get("input_tokens") or 0)
            rec["cached_tokens"] += int(u.get("cached_tokens") or 0)
            rec["output_tokens"] += int(u.get("output_tokens") or 0)
            rec["reasoning_tokens"] += int(u.get("reasoning_tokens") or 0)
            rec["actual_usd"] += float(s.get("actual_usd") or 0.0)
        ph = r.get("phase", "unspecified")
        by_phase[ph] = by_phase.get(ph, 0.0) + (float(s.get("actual_usd") or 0.0) if s is not None and s.get("actual_usd") is not None else cost)
    return {
        "hard_cap_usd": HARD_CAP_USD,
        "effective_committed_usd": round(effective_usd, 6),
        "settled_actual_usd": round(settled_usd, 6),
        "remaining_usd": round(max(0.0, HARD_CAP_USD - effective_usd), 6),
        "total_reserved_calls": len(reserved),
        "total_settled_calls": len(settled),
        "by_phase_usd": {k: round(v, 6) for k, v in sorted(by_phase.items())},
        "by_model": {
            k: {**v, "actual_usd": round(v["actual_usd"], 6)} for k, v in sorted(by_model.items())
        },
    }


def ledger_summary() -> dict[str, Any]:
    with _ledger_lock():
        return _ledger_totals_unlocked()


def worst_case_usd(model: str, prompt_chars: int, max_output_tokens: int) -> float:
    spec = MODEL_CATALOG[model]
    est_in_tokens = max(64, int(math.ceil(prompt_chars / 3.2)) + 128)
    return (est_in_tokens * spec["in_per_m"] + max_output_tokens * spec["out_per_m"]) / 1e6


def compute_actual_usd(model: str, input_tokens: int, cached_tokens: int, output_tokens: int) -> float:
    spec = MODEL_CATALOG[model]
    uncached_in = max(0, int(input_tokens) - int(cached_tokens))
    cached_in = max(0, int(cached_tokens))
    out_tok = max(0, int(output_tokens))
    return (
        uncached_in * spec["in_per_m"]
        + cached_in * spec["cache_read_per_m"]
        + out_tok * spec["out_per_m"]
    ) / 1e6


def _reserve_call(call_id: str, phase: str, model: str, worst_usd: float) -> bool:
    with _ledger_lock():
        tot = _ledger_totals_unlocked()
        if tot["effective_committed_usd"] + worst_usd > HARD_CAP_USD:
            return False
        with LEDGER_PATH.open("a", encoding="utf-8") as fh:
            fh.write(
                json.dumps(
                    {
                        "event": "reserve",
                        "call_id": call_id,
                        "phase": phase,
                        "model": model,
                        "worst_usd": round(worst_usd, 8),
                        "ts": utc_now_iso(),
                    }
                )
                + "\n"
            )
        return True


def _settle_call(
    call_id: str, status: str, actual_usd: float | None, usage: dict[str, int] | None = None
) -> None:
    with _ledger_lock():
        with LEDGER_PATH.open("a", encoding="utf-8") as fh:
            fh.write(
                json.dumps(
                    {
                        "event": "settle",
                        "call_id": call_id,
                        "status": status,
                        "actual_usd": round(actual_usd, 8) if actual_usd is not None else None,
                        "usage": usage or {},
                        "ts": utc_now_iso(),
                    }
                )
                + "\n"
            )


def _read_key_transient(provider: str) -> str:
    if provider == "koyejolab-openai-bm":
        key = OPENAI_KEY_PATH.read_text(encoding="utf-8").strip()
    elif provider == "mistral":
        key = MISTRAL_KEY_PATH.read_text(encoding="utf-8").strip()
    else:
        raise ValueError(f"Unsupported or out-of-scope provider: {provider!r}")
    if not key:
        raise RuntimeError(f"Empty key file for provider {provider}")
    return key


def _extract_mistral_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for chunk in content:
            if isinstance(chunk, dict):
                if chunk.get("type") == "text" and isinstance(chunk.get("text"), str):
                    parts.append(chunk["text"])
            elif isinstance(chunk, str):
                parts.append(chunk)
        return "\n".join(parts)
    return ""


def call_llm(
    *,
    model: str,
    system: str,
    user: str,
    phase: str,
    effort: str | None = None,
    max_output_tokens: int = 600,
    temperature: float | None = None,
    cache_key: str | None = None,
    repeat_slot: int = 1,
    timeout_sec: int = 180,
) -> dict[str, Any]:
    """Execute one cached/ledgered LLM call on OpenAI Responses or Mistral Chat Completions."""
    spec = MODEL_CATALOG[model]
    provider = spec["provider"]
    req_fingerprint = sha256_text(
        json.dumps(
            {
                "provider": provider,
                "model": model,
                "effort": effort,
                "max_output_tokens": max_output_tokens,
                "temperature": temperature,
                "system": system,
                "user": user,
                "repeat_slot": repeat_slot,
                "mode": "text",
            },
            sort_keys=True,
        )
    )
    API_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = API_CACHE_DIR / f"{req_fingerprint}.json"
    if cache_file.exists():
        cached = json.loads(cache_file.read_text(encoding="utf-8"))
        if cached.get("status") == "OK":
            return {**cached, "from_local_cache": True}

    worst = worst_case_usd(model, len(system) + len(user), max_output_tokens)
    for attempt in range(1, 4):
        call_id = f"{phase}:{model}:{req_fingerprint[:16]}:a{attempt}"
        if not _reserve_call(call_id, phase, model, worst):
            return {
                "status": "BUDGET_CAP_REACHED",
                "model": model,
                "provider": provider,
                "text": "",
                "usage": {},
                "actual_usd": 0.0,
                "error": "HARD_CAP_USD reached before admission",
            }
        t0 = time.monotonic()
        try:
            if provider == "koyejolab-openai-bm":
                url = "https://api.openai.com/v1/responses"
                payload: dict[str, Any] = {
                    "model": model,
                    "input": [
                        {"role": "developer", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "max_output_tokens": max(32, max_output_tokens),
                    "store": False,
                }
                if cache_key:
                    payload["prompt_cache_key"] = cache_key[:64]
                if spec["supports_reasoning"] and effort is not None:
                    payload["reasoning"] = {"effort": effort}
                if temperature is not None and (not spec["supports_reasoning"] or effort == "none"):
                    payload["temperature"] = temperature
                headers = {
                    "Authorization": f"Bearer {_read_key_transient(provider)}",
                    "Content-Type": "application/json",
                }
                req = urllib.request.Request(
                    url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST"
                )
                with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                out_texts = []
                if isinstance(body.get("output_text"), str) and body["output_text"]:
                    out_texts.append(body["output_text"])
                else:
                    for item in body.get("output") or []:
                        if item.get("type") == "message":
                            for c in item.get("content") or []:
                                if c.get("type") in ("output_text", "text") and c.get("text"):
                                    out_texts.append(c["text"])
                text = "\n".join(out_texts).strip()
                u_raw = body.get("usage") or {}
                in_tok = int(u_raw.get("input_tokens") or 0)
                cached_tok = int((u_raw.get("input_tokens_details") or {}).get("cached_tokens") or 0)
                out_tok = int(u_raw.get("output_tokens") or 0)
                reas_tok = int((u_raw.get("output_tokens_details") or {}).get("reasoning_tokens") or 0)
                finish_status = body.get("status", "completed")
            else:
                url = (
                    "https://codestral.mistral.ai/v1/chat/completions"
                    if model.startswith("codestral")
                    else "https://api.mistral.ai/v1/chat/completions"
                )
                payload = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "max_tokens": max_output_tokens,
                    "temperature": 0.0 if temperature is None else temperature,
                }
                if spec["supports_reasoning"] and effort in ("none", "high"):
                    payload["reasoning_effort"] = effort
                headers = {
                    "Authorization": f"Bearer {_read_key_transient(provider)}",
                    "Content-Type": "application/json",
                }
                req = urllib.request.Request(
                    url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST"
                )
                with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                choice0 = (body.get("choices") or [{}])[0]
                msg = choice0.get("message") or {}
                text = _extract_mistral_text(msg.get("content")).strip()
                u_raw = body.get("usage") or {}
                in_tok = int(u_raw.get("prompt_tokens") or 0)
                cached_tok = int(
                    (u_raw.get("prompt_tokens_details") or {}).get("cached_tokens")
                    or u_raw.get("num_cached_tokens")
                    or 0
                )
                out_tok = int(u_raw.get("completion_tokens") or 0)
                reas_tok = 0
                finish_status = choice0.get("finish_reason") or "stop"

            actual = compute_actual_usd(model, in_tok, cached_tok, out_tok)
            usage = {
                "input_tokens": in_tok,
                "cached_tokens": cached_tok,
                "output_tokens": out_tok,
                "reasoning_tokens": reas_tok,
            }
            _settle_call(call_id, "OK", actual, usage)
            result = {
                "status": "OK",
                "model": model,
                "provider": provider,
                "effort": effort,
                "finish_status": finish_status,
                "text": text,
                "usage": usage,
                "actual_usd": round(actual, 8),
                "wall_seconds": round(time.monotonic() - t0, 3),
                "fingerprint": req_fingerprint,
                "from_local_cache": False,
            }
            write_private_json(cache_file, result)
            return result
        except urllib.error.HTTPError as exc:
            err_body = ""
            try:
                err_body = exc.read().decode("utf-8", errors="replace")[:400]
            except Exception:
                pass
            billed_zero = exc.code in (400, 401, 403, 404, 422, 429)
            _settle_call(call_id, f"HTTP_{exc.code}", 0.0 if billed_zero else worst, {})
            if exc.code in (429, 500, 502, 503, 504) and attempt < 3:
                time.sleep(2.5 * attempt)
                continue
            return {
                "status": f"HTTP_{exc.code}",
                "model": model,
                "provider": provider,
                "text": "",
                "usage": {},
                "actual_usd": 0.0 if billed_zero else worst,
                "error": f"HTTP {exc.code}: {err_body}",
            }
        except Exception as exc:  # noqa: BLE001
            _settle_call(call_id, "ERROR", None, {})
            if attempt < 3:
                time.sleep(2.5 * attempt)
                continue
            return {
                "status": "ERROR",
                "model": model,
                "provider": provider,
                "text": "",
                "usage": {},
                "actual_usd": worst,
                "error": f"{type(exc).__name__}: {str(exc)[:250]}",
            }
    return {"status": "ERROR", "model": model, "provider": provider, "text": "", "usage": {}, "actual_usd": 0.0}


def call_logprobs_6cat(
    *,
    model: str,
    system: str,
    user: str,
    phase: str,
    cache_key: str | None = None,
    timeout_sec: int = 120,
) -> dict[str, Any]:
    """Elicit exact single-token next-token logprobs over rating digits '0'..'5' on Chat Completions."""
    spec = MODEL_CATALOG[model]
    provider = spec["provider"]
    if provider != "koyejolab-openai-bm":
        raise ValueError("Exact top-20 token logprobs configured for OpenAI chat/completions")
    req_fingerprint = sha256_text(
        json.dumps(
            {"provider": provider, "model": model, "system": system, "user": user, "mode": "logprobs_6cat"},
            sort_keys=True,
        )
    )
    API_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = API_CACHE_DIR / f"{req_fingerprint}.json"
    if cache_file.exists():
        cached = json.loads(cache_file.read_text(encoding="utf-8"))
        if cached.get("status") == "OK":
            return {**cached, "from_local_cache": True}

    worst = worst_case_usd(model, len(system) + len(user), 16)
    for attempt in range(1, 4):
        call_id = f"{phase}:{model}:lp:{req_fingerprint[:16]}:a{attempt}"
        if not _reserve_call(call_id, phase, model, worst):
            return {"status": "BUDGET_CAP_REACHED", "model": model, "probabilities": None, "usage": {}}
        t0 = time.monotonic()
        try:
            payload: dict[str, Any] = {
                "model": model,
                "messages": [
                    {"role": "developer", "content": system},
                    {"role": "user", "content": user},
                ],
                "reasoning_effort": "none",
                "max_completion_tokens": 4,
                "temperature": 0.0,
                "logprobs": True,
                "top_logprobs": 5,
            }
            if cache_key:
                payload["prompt_cache_key"] = cache_key[:64]
            headers = {
                "Authorization": f"Bearer {_read_key_transient(provider)}",
                "Content-Type": "application/json",
            }
            req = urllib.request.Request(
                "https://api.openai.com/v1/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            choice0 = (body.get("choices") or [{}])[0]
            lp_content = ((choice0.get("logprobs") or {}).get("content")) or []
            # Find first token position that contains any digit 0..5 among top_logprobs
            digit_mass = [0.0] * 6
            matched_pos = None
            for pos_idx, tok_entry in enumerate(lp_content):
                tops = tok_entry.get("top_logprobs") or []
                temp_mass = [0.0] * 6
                for cand in tops:
                    tok_str = str(cand.get("token", "")).strip()
                    if tok_str in ("0", "1", "2", "3", "4", "5"):
                        temp_mass[int(tok_str)] += math.exp(float(cand["logprob"]))
                if sum(temp_mass) > 1e-6:
                    digit_mass = temp_mass
                    matched_pos = pos_idx
                    break
            total_digit_mass = sum(digit_mass)
            probs = [x / total_digit_mass for x in digit_mass] if total_digit_mass > 1e-6 else None
            u_raw = body.get("usage") or {}
            in_tok = int(u_raw.get("prompt_tokens") or 0)
            cached_tok = int((u_raw.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
            out_tok = int(u_raw.get("completion_tokens") or 0)
            actual = compute_actual_usd(model, in_tok, cached_tok, out_tok)
            usage = {"input_tokens": in_tok, "cached_tokens": cached_tok, "output_tokens": out_tok, "reasoning_tokens": 0}
            _settle_call(call_id, "OK", actual, usage)
            result = {
                "status": "OK" if probs is not None else "PARSE_ERROR",
                "model": model,
                "provider": provider,
                "text": ((choice0.get("message") or {}).get("content") or "").strip(),
                "probabilities": probs,
                "total_digit_mass": round(total_digit_mass, 6),
                "matched_token_pos": matched_pos,
                "usage": usage,
                "actual_usd": round(actual, 8),
                "wall_seconds": round(time.monotonic() - t0, 3),
                "fingerprint": req_fingerprint,
                "from_local_cache": False,
            }
            if result["status"] == "OK":
                write_private_json(cache_file, result)
            return result
        except urllib.error.HTTPError as exc:
            err_body = ""
            try:
                err_body = exc.read().decode("utf-8", errors="replace")[:400]
            except Exception:
                pass
            billed_zero = exc.code in (400, 401, 403, 404, 422, 429)
            _settle_call(call_id, f"HTTP_{exc.code}", 0.0 if billed_zero else worst, {})
            if exc.code in (429, 500, 502, 503, 504) and attempt < 3:
                time.sleep(2.5 * attempt)
                continue
            return {"status": f"HTTP_{exc.code}", "model": model, "probabilities": None, "error": f"HTTP {exc.code}: {err_body}"}
        except Exception as exc:  # noqa: BLE001
            _settle_call(call_id, "ERROR", None, {})
            if attempt < 3:
                time.sleep(2.5 * attempt)
                continue
            return {"status": "ERROR", "model": model, "probabilities": None, "error": str(exc)[:250]}
    return {"status": "ERROR", "model": model, "probabilities": None}


def load_private_dataset_and_inputs() -> dict[str, Any]:
    """Load private human dataset + on-disk VeriBench exp-100 annotation-era files and verify SHA-256 provenance."""
    raw_ds = DATASET_PATH.read_bytes()
    actual_ds_sha = sha256_bytes(raw_ds)
    if actual_ds_sha != DATASET_SHA256:
        raise RuntimeError(f"dataset.json SHA-256 mismatch: {actual_ds_sha} != {DATASET_SHA256}")
    ds = json.loads(raw_ds)

    items_path = VB_EXP100_DIR / "inputs/items.jsonl"
    items_list = [json.loads(line) for line in items_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    items_by_id = {r["item"]: r for r in items_list}
    if len(ds["rows"]) != 75 or len(items_by_id) != 75:
        raise RuntimeError("Expected 75 artifacts in both dataset.json and items.jsonl")

    merged_rows = []
    cand_exact = py_exact = ref_disk_matches_recorded = 0
    for idx, r in enumerate(ds["rows"]):
        item_id = r["item"]
        it = items_by_id[item_id]
        cand_file = VB_EXP100_DIR / it["candidate_path"]
        ref_file = VB_EXP100_DIR / it["ref_ann_path"]
        py_file = VB_EXP100_DIR / it["py_ann_path"]

        cand_bytes = cand_file.read_bytes()
        ref_bytes = ref_file.read_bytes()
        py_bytes = py_file.read_bytes()

        cand_sha = sha256_bytes(cand_bytes)
        ref_disk_sha = sha256_bytes(ref_bytes)
        py_sha = sha256_bytes(py_bytes)

        if cand_sha == it["candidate_sha256"]:
            cand_exact += 1
        if py_sha == it["py_ann_sha256"]:
            py_exact += 1
        if ref_disk_sha == it["ref_ann_sha256"]:
            ref_disk_matches_recorded += 1

        merged_rows.append(
            {
                "index": idx,
                "item": item_id,
                "candidate_file": it["candidate_file"],
                "gold_file_name": it["gold_file_name"],
                "py_file_name": Path(it["py_ann_path"]).name,
                "task": r["task"],
                "family": r["family"],
                "domain": it["domain"],
                "ordinary": bool(r["ordinary"]),
                "ambiguous": bool(r["ambiguous"]),
                "ref_placeholder_at_annotation": bool(it["ref_placeholder_at_annotation"]),
                "n_ref_theorems_ann": int(it["n_ref_theorems_ann"]),
                "n_cand_theorems": int(it["n_cand_theorems"]),
                "cand_sorry_count": int(it["cand_sorry_count"]),
                "cand_lines": int(it["cand_lines"]),
                "historical_split": r["split"],
                "cand_text": cand_bytes.decode("utf-8", errors="replace"),
                "ref_text": ref_bytes.decode("utf-8", errors="replace"),
                "py_text": py_bytes.decode("utf-8", errors="replace"),
                "cand_sha256": cand_sha,
                "ref_disk_sha256": ref_disk_sha,
                "ref_recorded_items_jsonl_sha256": it["ref_ann_sha256"],
                "py_sha256": py_sha,
                # Private fields (kept strictly in memory / private/ only!):
                "_private_ratings": [int(x) for x in r["ratings"]],
                "_private_ratings_by_expert": r["ratings_by_expert"],
                "_private_p_cols": {k: it[k] for k in ("p1", "p2", "p3", "p4", "p5")},
            }
        )

    if cand_exact != 75 or py_exact != 75:
        raise RuntimeError(f"Annotation file hash check failed: cand={cand_exact}/75, py={py_exact}/75")

    return {
        "dataset_sha256": actual_ds_sha,
        "source_commit": ds.get("source_commit"),
        "rows": merged_rows,
        "historical_outputs": ds["outputs"],
        "historical_models": ds["models"],
        "hash_audit_summary": {
            "total_artifacts": 75,
            "candidate_exact_sha256_matches": cand_exact,
            "python_exact_sha256_matches": py_exact,
            "reference_disk_matches_items_jsonl_sha256": ref_disk_matches_recorded,
            "note": (
                "All 75 candidate_lean and 75 python_source files match items.jsonl SHA-256 75/75; "
                "on-disk inputs/references/annotation/*.lean are hashed directly so Exp 09 and Exp 10 "
                "score identical reference bytes to exp 100/12/14."
            ),
        },
    }
