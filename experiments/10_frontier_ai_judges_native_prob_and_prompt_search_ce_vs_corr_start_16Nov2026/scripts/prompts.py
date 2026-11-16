#!/usr/bin/env python3
"""Zero-exposure native-probability prompts, AST obligation scaffold (H2), two-step audit/aggregator (H3),
blind proposer prompts (iii.a), and strict JSON probability parser for Experiment 10."""
from __future__ import annotations

import difflib
import json
import math
import re
from typing import Any

from common import VB_REPO_DIR

HUMAN_MD_PATH = VB_REPO_DIR / "py_src/veribench_prompts/human_judgements_prompt.md"

VACUOUS_LEAN = (
    "import Mathlib\n\nnamespace Candidate\n\n/-- Main claim. -/\n"
    "theorem main_claim : True := trivial\n\nend Candidate\n"
)

SHARED_JUDGE_SYSTEM = (
    "You are an expert formal-verification rater in a study of Lean 4 formalizations. "
    "Read the three task files carefully, follow the evaluation rubric at the end, and answer with "
    "ONLY one valid JSON object (no markdown fences, no commentary outside JSON)."
)

JSON_PROB_CONTRACT = (
    "Output format requirement:\n"
    "Return ONLY one JSON object (no markdown code fences) with exact keys:\n"
    '{"probabilities": [q_0, q_1, q_2, q_3, q_4, q_5], "rationale": "1 concise sentence (<=25 words)"}\n'
    "where q_0, q_1, q_2, q_3, q_4, q_5 are six nonnegative numbers in [0, 1] summing to 1.000 "
    "representing Pr[rating = 0], Pr[rating = 1], Pr[rating = 2], Pr[rating = 3], Pr[rating = 4], Pr[rating = 5] "
    "for an expert human rater scoring statement coverage on the 0-5 integer scale."
)

LOGPROB_CONTRACT = (
    "Output format requirement:\n"
    "Respond with ONLY a single integer character from 0, 1, 2, 3, 4, or 5 representing the score (no words, no JSON, no spaces)."
)


def rater_instructions_verbatim() -> str:
    """Exact human rater instructions from VeriBench minus the Google-Sheet submission paragraph."""
    text = HUMAN_MD_PATH.read_text(encoding="utf-8")
    head, _, tail = text.partition("## Output format (required)")
    tie = tail[tail.index("## Tie-break rule") :] if "## Tie-break rule" in tail else ""
    return (head.rstrip() + "\n\n" + tie.strip()).strip()


P0_RUBRIC = (
    "Rate exactly as a careful expert human rater following these instructions would:\n\n"
    "<rater_instructions>\n"
    "{rater_instructions}\n"
    "</rater_instructions>"
)

# Zero-exposure specification-coverage conventions (written strictly from general Lean 4 specification principles, zero split exposure)
P1_CONVENTIONS_ADDENDUM = """
<specification_conventions>
Apply these general Lean 4 specification-coverage conventions alongside the instructions above:
1. Implementation-independent coverage: Count a gold theorem as covered when an agent theorem expresses the same mathematical or behavioral guarantee about the agent's function, even if the agent uses different container or numeric types (e.g., List vs Array, Int vs Nat) or helper definitions.
2. Trivial or tautological claims do not cover behavioral properties: Theorems that merely restate a function body, assert `rfl` on concrete unit tests, or prove `: True` do not cover non-trivial gold specification theorems.
3. Partial preconditions: Omitting a minor precondition (such as bounds or sortedness) while capturing the core postcondition lowers the rating by about 1 point rather than zeroing coverage.
4. Placeholder gold files: If the gold Lean file itself contains only a trivial placeholder postcondition (such as `Post := True` with no substantive property), evaluate whether the candidate provides meaningful task specifications without penalizing it for missing nonexistent gold obligations.
</specification_conventions>""".strip()

P2_DISAGREEMENT_ADDENDUM = """
<probability_and_disagreement_calibration>
Calibrating your 6-category human-rating probability distribution q = (q_0, ..., q_5):
- Multiple human experts rate each file independently and frequently disagree by 1-2 points on partial coverage, bundled theorems, missing preconditions, or placeholder gold references.
- Reflect genuine inter-rater uncertainty across all plausible scores 0..5 rather than placing 1.00 on a single integer or exact 0.000 on plausible neighboring categories.
- Assign small non-zero probability (e.g. >= 0.01) to any score an attentive stricter or more lenient expert could reasonably assign, concentrating mass sharply only when coverage is unambiguously complete or unambiguously vacuous/empty.
</probability_and_disagreement_calibration>""".strip()


def get_base_rubric_text(prompt_id: str) -> str:
    ri = rater_instructions_verbatim()
    base = P0_RUBRIC.format(rater_instructions=ri)
    if prompt_id == "P0_human_mirror":
        return base
    if prompt_id == "P1_conventions":
        return base + "\n\n" + P1_CONVENTIONS_ADDENDUM
    if prompt_id == "P2_disagreement_aware":
        return base + "\n\n" + P1_CONVENTIONS_ADDENDUM + "\n\n" + P2_DISAGREEMENT_ADDENDUM
    raise KeyError(f"Unknown base prompt_id: {prompt_id}")


# --- AST / Static Obligation Pre-Extractor for Harness H2 ---

_COMMENT_BLOCK_RE = re.compile(r"/-.*?-/", flags=re.S)
_COMMENT_LINE_RE = re.compile(r"--[^\n]*")
_DECL_RE = re.compile(
    r"(?m)^[ \t]*(?:@\[[^\]\n]*\][ \t]*\n?[ \t]*)?(?:(?:private|protected|noncomputable)\s+)*(theorem|lemma|def)\s+([^\s:(\[{]+)"
)


def strip_lean_comments(src: str) -> str:
    return _COMMENT_LINE_RE.sub(" ", _COMMENT_BLOCK_RE.sub(" ", src))


def extract_lean_declarations(src: str, kinds: tuple[str, ...] = ("theorem", "lemma")) -> list[dict[str, str]]:
    clean = strip_lean_comments(src)
    out = []
    for m in _DECL_RE.finditer(clean):
        kind, name = m.group(1), m.group(2)
        if kind not in kinds:
            continue
        is_priv = "private " in clean[m.start() : m.end()]
        j, depth = m.start(), 0
        while j < len(clean):
            ch = clean[j]
            if ch in "([{⟨":
                depth += 1
            elif ch in ")]}⟩":
                depth = max(0, depth - 1)
            elif depth == 0 and clean.startswith(":=", j):
                break
            elif depth == 0 and ch == "\n" and re.match(r"\n[ \t]*\|", clean[j:]):
                break
            j += 1
        sig = " ".join(clean[m.start() : j].split())
        out.append({"kind": kind, "name": name, "private": "yes" if is_priv else "no", "signature": sig[:260]})
    return out


def build_ast_obligation_scaffold(py_text: str, ref_text: str, cand_text: str) -> dict[str, Any]:
    """Deterministic Python/Lean AST & signature pre-extractor for Harness H2."""
    gold_thms = [d for d in extract_lean_declarations(ref_text, ("theorem", "lemma")) if d["private"] == "no"]
    cand_thms = [d for d in extract_lean_declarations(cand_text, ("theorem", "lemma")) if d["private"] == "no"]
    cand_defs = extract_lean_declarations(cand_text, ("def",))
    clean_ref = strip_lean_comments(ref_text)
    clean_cand = strip_lean_comments(cand_text)

    is_placeholder_gold = bool(re.search(r"def\s+Post\s*:\s*Prop\s*:=\s*True", clean_ref)) and len(gold_thms) <= 1
    cand_sorry = len(re.findall(r"\bsorry\b", clean_cand))
    cand_trivial_true = len(re.findall(r":\s*True\s*:=", clean_cand))

    alignments = []
    for g in gold_thms:
        best_name, best_sim = "NONE", 0.0
        for c in cand_thms:
            sim = difflib.SequenceMatcher(None, g["signature"], c["signature"], autojunk=False).ratio()
            if sim > best_sim:
                best_sim, best_name = sim, c["name"]
        alignments.append({"gold": g["name"], "best_cand_match": best_name, "sig_similarity": round(best_sim, 3)})

    py_defs = re.findall(r"(?m)^\s*def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", py_text)
    ratio = min(1.0, len(cand_thms) / max(1, len(gold_thms)))
    return {
        "python_functions": py_defs[:8],
        "is_placeholder_gold_reference": is_placeholder_gold,
        "n_gold_public_theorems": len(gold_thms),
        "n_candidate_public_theorems": len(cand_thms),
        "n_candidate_definitions": len(cand_defs),
        "theorem_count_ratio_capped": round(ratio, 3),
        "candidate_sorry_count": cand_sorry,
        "candidate_vacuous_true_theorems": cand_trivial_true,
        "gold_theorem_signatures": [g["signature"] for g in gold_thms[:12]],
        "candidate_theorem_signatures": [c["signature"] for c in cand_thms[:16]],
        "lexical_signature_alignment": alignments[:12],
    }


def render_task_files_block(
    py_name: str, py_text: str, ref_name: str, ref_text: str, cand_name: str, cand_text: str
) -> str:
    return (
        f"## 1. Python source: {py_name}\n```python\n{py_text.rstrip()}\n```\n\n"
        f"## 2. Gold Lean file: {ref_name}\n```lean\n{ref_text.rstrip()}\n```\n\n"
        f"## 3. Agent Lean file: {cand_name}\n```lean\n{cand_text.rstrip()}\n```"
    )


def render_single_call_prompt(
    *,
    py_name: str,
    py_text: str,
    ref_name: str,
    ref_text: str,
    cand_name: str,
    cand_text: str,
    rubric_text: str,
    ast_scaffold: dict[str, Any] | None = None,
    logprob_mode: bool = False,
) -> tuple[str, str]:
    """Prefix-aligned prompt builder: files first (for prefix caching & zero layout confound), rubric at end."""
    files_block = render_task_files_block(py_name, py_text, ref_name, ref_text, cand_name, cand_text)
    parts = [files_block]
    if ast_scaffold is not None:
        parts.append(
            "## 4. Programmatic AST & Obligation Pre-Extraction Scaffold\n"
            "```json\n" + json.dumps(ast_scaffold, indent=2) + "\n```"
        )
    contract = LOGPROB_CONTRACT if logprob_mode else JSON_PROB_CONTRACT
    parts.append(f"## Evaluation Rubric\n{rubric_text.strip()}\n\n{contract}")
    return SHARED_JUDGE_SYSTEM, "\n\n".join(parts)


# --- Harness H3: Two-Step Decompose -> Calibrated Aggregator ---

H3_STEP1_SYSTEM = (
    "You are a formal-verification static auditor for Lean 4 theorem statements. "
    "Audit how each gold theorem statement is covered by the agent Lean file without assigning a numeric score. "
    "Return ONLY one valid JSON object (no markdown fences)."
)

H3_STEP1_INSTRUCTIONS = """## Structured Obligation Audit Instructions
Inspect the Python source, Gold Lean file, and Agent Lean file above.
For every public theorem/lemma in the Gold Lean file, classify its statement coverage in the Agent Lean file as one of:
- "COVERED": equivalent or stronger mathematical guarantee about the agent's implementation (allowing benign type/name differences).
- "PARTIAL_PRECOND": right core conclusion, but omits or weakens a precondition or edge case.
- "IMPL_RESTATEMENT": agent theorem merely restates the code/implementation or unit tests rather than an independent property.
- "MISSING": no matching guarantee in the agent file.

Also record whether the Gold Lean file is a `True`-only placeholder specification, whether the Agent file has theorem spam ("NONE", "HANDFUL", or "DOMINATES"), and whether the Agent file states at least one non-vacuous task-relevant theorem.

Return ONLY one JSON object with exact keys:
{"is_placeholder_gold": true or false, "agent_has_nonvacuous_relevant_thm": true or false, "spam_level": "NONE" or "HANDFUL" or "DOMINATES", "obligations": [{"gold_theorem": "name", "status": "COVERED"|"PARTIAL_PRECOND"|"IMPL_RESTATEMENT"|"MISSING", "note": "<=15 words"}], "overall_gap_summary": "1 concise sentence (<=25 words)"}"""


def render_h3_step1_audit(
    py_name: str, py_text: str, ref_name: str, ref_text: str, cand_name: str, cand_text: str
) -> tuple[str, str]:
    files_block = render_task_files_block(py_name, py_text, ref_name, ref_text, cand_name, cand_text)
    ast_info = build_ast_obligation_scaffold(py_text, ref_text, cand_text)
    brief_ast = {
        k: ast_info[k]
        for k in (
            "is_placeholder_gold_reference",
            "n_gold_public_theorems",
            "n_candidate_public_theorems",
            "candidate_vacuous_true_theorems",
            "gold_theorem_signatures",
            "candidate_theorem_signatures",
        )
    }
    user = (
        files_block
        + "\n\n## Extracted Theorem Signatures\n```json\n"
        + json.dumps(brief_ast, indent=2)
        + "\n```\n\n"
        + H3_STEP1_INSTRUCTIONS
    )
    return H3_STEP1_SYSTEM, user


def render_h3_step2_aggregator(
    task_name: str, cand_name: str, audit_obj: dict[str, Any], rubric_text: str
) -> tuple[str, str]:
    user = (
        f"## Task `{task_name}` — Candidate `{cand_name}`\n"
        "### Step-1 Structured Obligation Coverage Audit\n"
        "```json\n"
        + json.dumps(audit_obj, indent=2)
        + "\n```\n\n"
        f"## Evaluation Rubric\n{rubric_text.strip()}\n\n{JSON_PROB_CONTRACT}"
    )
    return SHARED_JUDGE_SYSTEM, user


# --- Label-in-Prompt Sanity Reproduction Check (Karpathy Step 2.3) ---

def render_label_in_prompt_check(row: dict[str, Any], empirical_probs: list[float]) -> tuple[str, str]:
    probs_fmt = [round(x, 4) for x in empirical_probs]
    # Re-adjust last nonzero entry so rounded decimals sum to exact 1.0
    diff = round(1.0 - sum(probs_fmt), 4)
    for k in range(5, -1, -1):
        if probs_fmt[k] + diff >= 0:
            probs_fmt[k] = round(probs_fmt[k] + diff, 4)
            break
    rubric = (
        get_base_rubric_text("P0_human_mirror")
        + "\n\n<oracle_verification_override>\n"
        + "PIPELINE REPRODUCTION TEST: The verified target probability distribution over 0..5 for this exact item is "
        + json.dumps(probs_fmt)
        + ". Reproduce this exact 6-element probability vector in `probabilities`.\n"
        + "</oracle_verification_override>"
    )
    return render_single_call_prompt(
        py_name=row["py_file_name"],
        py_text=row["py_text"],
        ref_name=row["gold_file_name"],
        ref_text=row["ref_text"],
        cand_name=row["candidate_file"],
        cand_text=row["cand_text"],
        rubric_text=rubric,
    )


# --- Axis iii.a Blind Proposer Prompts & Strict Leakage Firewall ---

MAX_PROPOSED_RUBRIC_WORDS = 420

PROPOSER_SYSTEM = (
    "You are an automated machine-learning judge-prompt researcher optimizing a scoring rubric for a "
    "Lean 4 specification-coverage judge. Propose TWO distinct improved candidate rubrics within the "
    "strict word limit. Never include item-specific filenames or task names in the rubric; write general, "
    "reusable evaluation principles only. Return ONLY one valid JSON object."
)


def assert_blind_firewall(
    text_sent_to_proposer: str,
    forbidden_task_names: set[str],
    track: str,
) -> None:
    """Deterministic firewall check: zero held-out/val task names, and zero cross-track leakage."""
    for tname in forbidden_task_names:
        if tname in text_sent_to_proposer:
            raise AssertionError(f"FIREWALL VIOLATION: held-out/val task {tname!r} appeared in proposer input!")
    if track == "corr":
        # Correlation proposer must NEVER see CE, RPS, rating histograms, or raw rating means/biases!
        for banned_tok in ("cross_entropy", "rps_loss", "rating_counts", "human_mean", "signed_bias"):
            if banned_tok in text_sent_to_proposer.lower():
                raise AssertionError(f"FIREWALL VIOLATION: {banned_tok!r} leaked into correlation proposer!")
    elif track == "ce":
        for banned_tok in ("pearson_r", "spearman_rho", "kendall_tau", "z_residual"):
            if banned_tok in text_sent_to_proposer.lower():
                raise AssertionError(f"FIREWALL VIOLATION: {banned_tok!r} leaked into proper-loss proposer!")


def build_proposer_prompt_corr(
    *,
    iteration: int,
    parent_rubric: str,
    history_summary: list[dict[str, Any]],
    worst_standardized_items: list[dict[str, Any]],
    forbidden_task_names: set[str],
) -> tuple[str, str]:
    """Correlation Track (①/②): sees ONLY training correlations + standardized z-score / rank residuals."""
    user = (
        f"## Prompt Optimization Track: CORRELATION MAXIMIZATION (Iteration {iteration} of 4)\n"
        "Your SOLE objective is to maximize Pearson and Spearman correlation between the judge's expected "
        "score E[R] = sum_{r=0}^5 r * q_r and expert ratings on the training tasks.\n"
        f"Hard constraint: each candidate rubric MUST be <= {MAX_PROPOSED_RUBRIC_WORDS} words, preserve the 0-5 "
        "rating scale meaning, and apply generally to unseen tasks (do NOT mention task names).\n\n"
        "### Current Parent Rubric\n```text\n"
        + parent_rubric.strip()
        + "\n```\n\n"
        "### Training-Split Correlation History So Far\n```json\n"
        + json.dumps(history_summary, indent=2)
        + "\n```\n\n"
        "### Top-6 Largest Standardized Z-Score / Rank Residuals on Training Artifacts\n"
        "(Positive `z_judge_minus_z_target` means the judge ranked this item too high relative to other items; "
        "negative means ranked too low.)\n```json\n"
        + json.dumps(worst_standardized_items, indent=2)
        + "\n```\n\n"
        "Propose TWO distinct candidate rubrics (`candidate_1` and `candidate_2`) that fix these ordering errors. "
        "Return ONLY one JSON object:\n"
        '{"diagnosis": "<=40 words", "candidate_1": "full rubric text (<=420 words)", "candidate_2": "full rubric text (<=420 words)"}'
    )
    assert_blind_firewall(user, forbidden_task_names, track="corr")
    return PROPOSER_SYSTEM, user


def build_proposer_prompt_ce(
    *,
    iteration: int,
    parent_rubric: str,
    history_summary: list[dict[str, Any]],
    worst_ce_items: list[dict[str, Any]],
    forbidden_task_names: set[str],
) -> tuple[str, str]:
    """Proper-Loss Track (③/④): sees ONLY training individual-rating CE & RPS losses + per-item q vs rating counts."""
    user = (
        f"## Prompt Optimization Track: PROPER-LOSS (CROSS-ENTROPY & RPS) MINIMIZATION (Iteration {iteration} of 4)\n"
        "Your SOLE objective is to minimize individual-rating cross-entropy (CE, -ln q(r)) and ranked probability "
        "score (RPS) of the judge's 6-category distribution q = (q_0..q_5) against individual human expert ratings on the training tasks.\n"
        "Note: Proper losses penalize (1) systematic leniency/harshness bias, (2) overconfident near-zero probabilities on "
        "scores experts actually assign, and (3) ignoring genuine inter-rater spread on partial-coverage or ambiguous items.\n"
        f"Hard constraint: each candidate rubric MUST be <= {MAX_PROPOSED_RUBRIC_WORDS} words, preserve the 0-5 "
        "rating scale meaning, and apply generally to unseen tasks (do NOT mention task names).\n\n"
        "### Current Parent Rubric\n```text\n"
        + parent_rubric.strip()
        + "\n```\n\n"
        "### Training-Split Proper-Loss History So Far\n```json\n"
        + json.dumps(history_summary, indent=2)
        + "\n```\n\n"
        "### Top-6 Highest Individual-Rating CE Loss Artifacts on Training Split\n```json\n"
        + json.dumps(worst_ce_items, indent=2)
        + "\n```\n\n"
        "Propose TWO distinct candidate rubrics (`candidate_1` and `candidate_2`) that reduce CE and RPS on unseen tasks. "
        "Return ONLY one JSON object:\n"
        '{"diagnosis": "<=40 words", "candidate_1": "full rubric text (<=420 words)", "candidate_2": "full rubric text (<=420 words)"}'
    )
    assert_blind_firewall(user, forbidden_task_names, track="ce")
    return PROPOSER_SYSTEM, user


def enforce_rubric_word_cap(text: str, max_words: int = MAX_PROPOSED_RUBRIC_WORDS) -> str:
    words = text.strip().split()
    if len(words) <= max_words:
        return text.strip()
    return " ".join(words[:max_words])


# --- Strict JSON Object & Probability Parser ---

def extract_json_objects(text: str) -> list[dict[str, Any]]:
    out, start, depth, in_str, esc = [], None, 0, False, False
    for i, ch in enumerate(text):
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
                snippet = text[start : i + 1]
                try:
                    obj = json.loads(snippet)
                    if isinstance(obj, dict):
                        out.append(obj)
                except json.JSONDecodeError:
                    pass
                start = None
    return out


def parse_probability_response(raw_text: str) -> dict[str, Any]:
    """Parse {"probabilities": [q0..q5], "rationale": ...} strictly."""
    objs = extract_json_objects(raw_text)
    for obj in reversed(objs):
        if "probabilities" not in obj:
            continue
        q_raw = obj["probabilities"]
        if not isinstance(q_raw, list) or len(q_raw) != 6:
            continue
        vals = []
        valid = True
        for x in q_raw:
            if isinstance(x, bool) or not isinstance(x, (int, float)):
                valid = False
                break
            fx = float(x)
            if not math.isfinite(fx) or fx < -1e-6 or fx > 1.0 + 1e-3:
                valid = False
                break
            vals.append(max(0.0, fx))
        if not valid:
            continue
        s = sum(vals)
        if not (0.98 <= s <= 1.02) or s <= 0:
            continue
        q_norm = [v / s for v in vals]
        return {
            "ok": True,
            "probabilities": q_norm,
            "raw_sum_deviation": round(abs(s - 1.0), 8),
            "exact_unit_sum_3dec": abs(s - 1.0) <= 1e-6,
            "rationale": str(obj.get("rationale", ""))[:300],
        }
    return {"ok": False, "probabilities": None, "error": "no_valid_6cat_probabilities_json"}
