#!/usr/bin/env python3
"""Build byte-verified prompt manifests, Lean AST harness scaffolds, Karpathy controls,
and deterministic 5-fold grouped task-family splits for Experiment 09 (ii.a, ii.b, ii.c).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import numpy as np
from pathlib import Path

EXPT_DIR = Path(__file__).resolve().parents[1]
VB100_DIR = Path("/Users/brandomiranda/veribench/experiments/100_karpathy_recipe_tc_judge_search_human_train_val_test")
DATASET_PATH = Path(
    "/Users/brandomiranda/judges-dont-need-psychometrics-or-corr/experiments/"
    "02_exp06_ce_vs_pearson_matched_budget_start_09Nov2026/private/historical_exp06/cached_v1/dataset.json"
)
EXPECTED_DATASET_SHA256 = "b3a05699a26d3ffd9f28e793178811219885df78319b3aa57f9d393c68724ba0"
EXPECTED_ITEMS_JSONL_SHA256 = "ad9ed552fc2ffac6328e6010bf6be82215dd74f1eddd791622d433126877a96c"

VACUOUS_LEAN = (
    "import Mathlib\n\nnamespace Candidate\n\n"
    "/-- Main claim. -/\ntheorem main_claim : True := trivial\n\nend Candidate\n"
)

CONVENTIONS_TEXT = """Rating conventions expert Lean 4 raters apply (use together with the instructions above):
1. The candidate's theorems describe the candidate's own implementation of the Python task. Count a gold theorem as covered when some candidate theorem states the same semantic property about the candidate's implementation, even with different types (List vs Array, Int vs Nat), different helper names, or minor elaboration issues. Do not require a formal bridge theorem between candidate and gold definitions.
2. A theorem that merely restates the implementation (for example asserting the function equals its own body, a library call it wraps, or unit-test examples) states no general specification property; if those are all the candidate states, the score is 0 or 1.
3. Missing or slightly narrower preconditions deduct at most 1 rating point when the core property conclusion is present.
4. An internal equivalence theorem between functional and imperative gold helpers is a minor obligation whose absence alone deducts at most 1 point."""

DISAGREEMENT_RUBRIC_TEXT = """Distributional calibration and expert disagreement guide (for the six probabilities over 0, 1, 2, 3, 4, 5):
- Report your calibrated probability distribution [q0, q1, q2, q3, q4, q5] (six non-negative numbers summing to 1.0) that a randomly chosen qualified Lean 4 expert assigns rating r in {0, 1, 2, 3, 4, 5} to this candidate.
- Do NOT put >0.85 mass on rating 4 or 5 unless every non-trivial gold property theorem is clearly stated without implementation-restatement shortcuts; raw LLM judges systematically over-rate partial Lean files by +1.2 to +2.0 points.
- Across typical candidates, expert ratings center near 2.6 out of 5.0 and spread across 0..5. When coverage is partial, missing preconditions, or ambiguous, spread probability mass realistically across plausible categories (for example [0.05, 0.25, 0.35, 0.25, 0.08, 0.02]) rather than putting 0.95 on one integer.
- When the gold reference has only `True` placeholders and the candidate cleanly formalizes the Python code, high ratings (4 or 5) carry most mass; when the candidate is empty, unrelated, or purely vacuous (`True`) against real gold obligations, rating 0 carries most mass."""

STRICT_ANTI_LENIENCY_TEXT = """Strict anti-leniency calibration protocol:
- Default baseline score is 2 (partial/weak specification), NOT 5. Require explicit theorem-by-theorem evidence to award 4 or 5.
- Deduct heavily (score 0 or 1) whenever candidate theorems only restate the function definition, prove `True`, or check concrete examples instead of universal `forall` properties.
- Never assign zero probability (`0.0`) to every intermediate rating: maintain honest uncertainty mass across at least 3 adjacent rating bins whenever human experts could reasonably differ by 1 point."""

RANK_CONTRASTIVE_TEXT = """Fine-grained rank-ordering protocol:
- Your primary goal is to place candidates in the exact relative order of specification completeness from weakest (0) to strongest (5).
- Distinguish sharply between: (a) 0 = empty/unrelated/vacuous; (b) 1 = implementation restatement or trivial lemma only; (c) 2 = covers <50% of core properties; (d) 3 = covers >=50% of core properties with gaps; (e) 4 = covers all core properties with minor precondition gaps; (f) 5 = complete coverage of all gold properties."""

CDF_THRESHOLD_TEXT = """Cumulative threshold probability protocol (aligned with ranked probability score and cross-entropy):
- Before writing `probabilities`, ask five cumulative threshold questions:
  Q0: Pr(expert rating <= 0 | files)?  Q1: Pr(expert rating <= 1 | files)?  Q2: Pr(expert rating <= 2 | files)?
  Q3: Pr(expert rating <= 3 | files)?  Q4: Pr(expert rating <= 4 | files)?
- Ensure 0 < Q0 < Q1 < Q2 < Q3 < Q4 < 1, then set q0=Q0, q1=Q1-Q0, q2=Q2-Q1, q3=Q3-Q2, q4=Q4-Q3, q5=1-Q4 so every rating retains non-zero probability and reflects realistic inter-rater disagreement."""

JSON_OUTPUT_SPEC = """Answer with ONLY one valid JSON object (no markdown fences, no extra prose outside JSON) with keys in this exact order:
{"A": "Y" or "N", "B": "Y" or "N", "C": "Y" or "N", "D": "handful" or "dominates" or "N/A", "rationale": "2-3 concise sentences naming covered and uncovered gold theorems", "probabilities": [q0, q1, q2, q3, q4, q5], "final_score_0_5": integer 0-5}
Ensure `probabilities` has 6 non-negative numbers summing to 1.0 over ratings 0, 1, 2, 3, 4, 5, and `final_score_0_5` is the integer rating in 0..5."""

CRITIC_SYSTEM_PROMPT = """You are Stage 1 (Theorem-Obligation Auditor) of a two-stage Lean 4 formal specification judge.
Given the Python task, Gold Lean reference, and Candidate Lean file:
1. List every public `theorem` or `lemma` in the Gold Lean file (or note `PLACEHOLDER_ONLY` if Gold only proves `True`).
2. For each Gold obligation, classify it as `COVERED`, `PARTIAL_PRECONDITION_GAP`, `IMPLEMENTATION_RESTATEMENT_ONLY`, or `MISSING`, naming the matching Candidate theorem if any.
3. Flag any spam/trivial theorems in the Candidate file.
Respond with concise bullet points only (under 220 words). Do NOT output a numeric 0-5 score yet."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def strip_lean_comments(text: str) -> str:
    text = re.sub(r"/-.*?-/", " ", text, flags=re.DOTALL)
    text = re.sub(r"--[^\n]*", "", text)
    return text


def extract_lean_ast_summary(ref_text: str, cand_text: str) -> str:
    """Programmatic Lean 4 AST / declaration pre-extractor for Harness H3_ast_obligation_scaffold."""
    def decls(src: str, kinds: tuple[str, ...]) -> list[str]:
        clean = strip_lean_comments(src)
        pattern = re.compile(r"^\s*(?:private\s+|protected\s+)?(" + "|".join(kinds) + r")\s+([^\s:({\[=]+[^\n:=]*)", re.M)
        out = []
        for m in pattern.finditer(clean):
            sig = " ".join((m.group(1) + " " + m.group(2)).split())
            out.append(sig[:140])
        return out

    gold_thms = decls(ref_text, ("theorem", "lemma"))
    gold_defs = decls(ref_text, ("def", "abbrev", "structure"))
    cand_thms = decls(cand_text, ("theorem", "lemma"))
    cand_defs = decls(cand_text, ("def", "abbrev", "structure"))
    cand_ex = len(re.findall(r"^\s*example\b", strip_lean_comments(cand_text), flags=re.M))
    cand_sorry = len(re.findall(r"\bsorry\b", strip_lean_comments(cand_text)))
    gold_true_only = len(gold_thms) == 0 or all(": True" in t or ": True :=" in ref_text for t in gold_thms)
    ratio = 1.0 if len(gold_thms) == 0 else min(1.0, len(cand_thms) / max(1, len(gold_thms)))

    lines = [
        "<lean_ast_scaffold>",
        f"- Gold theorem/lemma count: {len(gold_thms)} | Gold def count: {len(gold_defs)} | Gold True-placeholder-only heuristic: {gold_true_only}",
        f"- Candidate theorem/lemma count: {len(cand_thms)} | Candidate def count: {len(cand_defs)} | Candidate `example` count: {cand_ex} | Candidate `sorry` count: {cand_sorry}",
        f"- Syntactic candidate/gold theorem-count ratio (clipped to [0,1]): {ratio:.3f}",
        "- Gold theorem signatures:",
    ]
    lines.extend([f"    * {t}" for t in gold_thms[:12]] or ["    * (none)"])
    lines.append("- Candidate theorem signatures:")
    lines.extend([f"    * {t}" for t in cand_thms[:12]] or ["    * (none)"])
    lines.append("</lean_ast_scaffold>")
    return "\n".join(lines)


def load_rater_instructions() -> str:
    human_md = VB100_DIR.parents[1] / "py_src/veribench_prompts/human_judgements_prompt.md"
    text = human_md.read_text(encoding="utf-8")
    head, _, tail = text.partition("## Output format (required)")
    tie = tail[tail.index("## Tie-break rule") :] if "## Tie-break rule" in tail else ""
    return (head.rstrip() + "\n\n" + tie.strip()).strip()


def build_system_prompt(prompt_id: str, instructions: str) -> str:
    base = (
        "You are an expert rater in a human-judgment study of Lean 4 formal specification coverage. "
        "You receive the exact same three files the human raters received.\n\n"
        f"<rater_instructions>\n{instructions}\n</rater_instructions>\n\n"
    )
    if prompt_id == "P0_human_mirror":
        return base + JSON_OUTPUT_SPEC
    if prompt_id == "P1_conventions":
        return base + f"<rater_conventions>\n{CONVENTIONS_TEXT}\n</rater_conventions>\n\n" + JSON_OUTPUT_SPEC
    if prompt_id == "P2_disagreement_calibrated":
        return (
            base
            + f"<rater_conventions>\n{CONVENTIONS_TEXT}\n</rater_conventions>\n\n"
            + f"<calibration_guide>\n{DISAGREEMENT_RUBRIC_TEXT}\n</calibration_guide>\n\n"
            + JSON_OUTPUT_SPEC
        )
    if prompt_id == "P3_strict_anti_leniency":
        return (
            base
            + f"<rater_conventions>\n{CONVENTIONS_TEXT}\n</rater_conventions>\n\n"
            + f"<anti_leniency_protocol>\n{STRICT_ANTI_LENIENCY_TEXT}\n</anti_leniency_protocol>\n\n"
            + JSON_OUTPUT_SPEC
        )
    if prompt_id == "P4_rank_contrastive":
        return (
            base
            + f"<rater_conventions>\n{CONVENTIONS_TEXT}\n</rater_conventions>\n\n"
            + f"<rank_ordering_protocol>\n{RANK_CONTRASTIVE_TEXT}\n</rank_ordering_protocol>\n\n"
            + JSON_OUTPUT_SPEC
        )
    if prompt_id == "P5_cdf_threshold":
        return (
            base
            + f"<rater_conventions>\n{CONVENTIONS_TEXT}\n</rater_conventions>\n\n"
            + f"<cdf_threshold_protocol>\n{CDF_THRESHOLD_TEXT}\n</cdf_threshold_protocol>\n\n"
            + JSON_OUTPUT_SPEC
        )
    raise ValueError(f"Unknown prompt_id: {prompt_id}")


def render_user(py_name: str, py_text: str, ref_name: str, ref_text: str, cand_name: str, cand_text: str, extra_prefix: str = "") -> str:
    prefix = (extra_prefix + "\n\n") if extra_prefix else ""
    return (
        prefix
        + f"## 1. Python source: {py_name}\n```python\n{py_text.rstrip()}\n```\n\n"
        f"## 2. Gold Lean file: {ref_name}\n```lean\n{ref_text.rstrip()}\n```\n\n"
        f"## 3. Agent Lean file: {cand_name}\n```lean\n{cand_text.rstrip()}\n```\n"
    )


def build_deterministic_family_folds(families: list[str], seed: int = 20261116) -> dict:
    """Partition the 19 sorted unique task families into 5 outer folds (4,4,4,4,3)
    and assign 4 inner-validation families per outer fold for Formulations 1 and 3."""
    unique_fams = sorted(set(families))
    assert len(unique_fams) == 19
    rng = np.random.default_rng(seed)
    perm = [unique_fams[i] for i in rng.permutation(len(unique_fams))]
    # Balanced round-robin across 5 folds -> sizes 4, 4, 4, 4, 3
    outer_folds = [sorted(perm[f::5]) for f in range(5)]
    folds_spec = []
    for f in range(5):
        test_fams = outer_folds[f]
        train_pool = sorted(set(unique_fams) - set(test_fams))
        rng_f = np.random.default_rng(seed + 100 + f)
        p_in = [train_pool[i] for i in rng_f.permutation(len(train_pool))]
        val_fams = sorted(p_in[:4])
        inner_train_fams = sorted(p_in[4:])
        folds_spec.append(
            {
                "fold": f,
                "outer_test_families": test_fams,
                "inner_val_families": val_fams,
                "inner_train_families": inner_train_fams,
            }
        )
    return {"seed": seed, "n_families": 19, "folds": folds_spec}


def main() -> None:
    ds_bytes = DATASET_PATH.read_bytes()
    ds_sha = sha256_bytes(ds_bytes)
    assert ds_sha == EXPECTED_DATASET_SHA256, f"dataset.json SHA mismatch: {ds_sha}"
    ds = json.loads(ds_bytes.decode("utf-8"))

    items_jsonl_path = VB100_DIR / "inputs/items.jsonl"
    items_jsonl_sha = sha256_bytes(items_jsonl_path.read_bytes())
    assert items_jsonl_sha == EXPECTED_ITEMS_JSONL_SHA256, f"items.jsonl SHA mismatch: {items_jsonl_sha}"
    vb_items = {
        row["item"]: row
        for row in (json.loads(line) for line in items_jsonl_path.read_text().splitlines() if line.strip())
    }
    assert len(ds["rows"]) == 75 and len(vb_items) == 75

    instructions = load_rater_instructions()
    prompt_ids = [
        "P0_human_mirror",
        "P1_conventions",
        "P2_disagreement_calibrated",
        "P3_strict_anti_leniency",
        "P4_rank_contrastive",
        "P5_cdf_threshold",
    ]
    system_prompts = {pid: build_system_prompt(pid, instructions) for pid in prompt_ids}

    public_artifacts = []
    bundle_rows = []
    lora_items = []

    for idx_row, row in enumerate(ds["rows"]):
        item_id = row["item"]
        it = vb_items[item_id]
        py_path = VB100_DIR / it["py_ann_path"]
        ref_path = VB100_DIR / it["ref_ann_path"]
        cand_path = VB100_DIR / it["candidate_path"]

        py_bytes, ref_bytes, cand_bytes = py_path.read_bytes(), ref_path.read_bytes(), cand_path.read_bytes()
        assert sha256_bytes(py_bytes) == it["py_ann_sha256"], f"py hash mismatch on {item_id}"
        assert sha256_bytes(cand_bytes) == it["candidate_sha256"], f"cand hash mismatch on {item_id}"
        ref_disk_sha = sha256_bytes(ref_bytes)

        py_txt = py_bytes.decode("utf-8", errors="replace")
        ref_txt = ref_bytes.decode("utf-8", errors="replace")
        cand_txt = cand_bytes.decode("utf-8", errors="replace")
        ast_scaffold = extract_lean_ast_summary(ref_txt, cand_txt)

        n_ref = int(it["n_ref_theorems_ann"])
        n_cand = int(it["n_cand_theorems"])
        thm_ratio = 1.0 if n_ref == 0 else min(1.0, n_cand / n_ref)

        public_artifacts.append(
            {
                "index": idx_row,
                "item": item_id,
                "task": row["task"],
                "family": row["family"],
                "historical_split": row["split"],
                "ordinary": bool(row["ordinary"]),
                "ref_placeholder_at_annotation": bool(it["ref_placeholder_at_annotation"]),
                "n_ref_theorems_ann": n_ref,
                "n_cand_theorems": n_cand,
                "cand_sorry_count": int(it["cand_sorry_count"]),
                "theorem_count_ratio_0_1": thm_ratio,
                "py_ann_sha256": it["py_ann_sha256"],
                "ref_ann_disk_sha256": ref_disk_sha,
                "ref_ann_items_jsonl_sha256": it["ref_ann_sha256"],
                "candidate_sha256": it["candidate_sha256"],
            }
        )

        py_name = os.path.basename(it["py_ann_path"])
        ref_name = os.path.basename(it["gold_file_name"])
        cand_name = it["candidate_file"]

        user_plain = render_user(py_name, py_txt, ref_name, ref_txt, cand_name, cand_txt)
        user_ast = render_user(py_name, py_txt, ref_name, ref_txt, cand_name, cand_txt, extra_prefix=ast_scaffold)

        for pid in prompt_ids:
            bundle_rows.append(
                {
                    "cell_id": f"main__{pid}__{item_id}",
                    "kind": "main_prompt",
                    "prompt_id": pid,
                    "harness_id": "H1_H2_direct",
                    "item": item_id,
                    "task": row["task"],
                    "family": row["family"],
                    "system": system_prompts[pid],
                    "user": user_plain,
                }
            )

        # Harness H3: Programmatic Lean AST Obligation Pre-Extractor Scaffold
        bundle_rows.append(
            {
                "cell_id": f"harness__H3_ast__{item_id}",
                "kind": "harness_H3_ast",
                "prompt_id": "P1_conventions",
                "harness_id": "H3_ast_obligation_scaffold",
                "item": item_id,
                "task": row["task"],
                "family": row["family"],
                "system": system_prompts["P1_conventions"]
                + "\nUse the programmatic `<lean_ast_scaffold>` declaration table at the top of the user message to cross-check every Gold obligation against Candidate theorems.",
                "user": user_ast,
            }
        )

        # Harness H4: Two-call Critic -> Aggregator (Stage 1 Critic prompt + Stage 2 Aggregator system prompt)
        bundle_rows.append(
            {
                "cell_id": f"harness__H4_two_call__{item_id}",
                "kind": "harness_H4_two_call",
                "prompt_id": "P2_disagreement_calibrated",
                "harness_id": "H4_two_call_critic_aggregator",
                "item": item_id,
                "task": row["task"],
                "family": row["family"],
                "critic_system": CRITIC_SYSTEM_PROMPT,
                "system": system_prompts["P2_disagreement_calibrated"],
                "user": user_ast,
            }
        )

        # Anonymous rating counts for private LoRA / meta-prompt training on mercury2 (NO rater names or IDs!)
        ratings = [int(r) for r in row["ratings"]]
        counts = [ratings.count(c) for c in range(6)]
        lora_items.append(
            {
                "index": idx_row,
                "item": item_id,
                "task": row["task"],
                "family": row["family"],
                "ordinary": bool(row["ordinary"]),
                "counts_0_5": counts,
                "n_ratings": len(ratings),
                "system_p0": system_prompts["P0_human_mirror"],
                "system_p1": system_prompts["P1_conventions"],
                "user_ast": user_ast,
                "user_plain": user_plain,
            }
        )

    # 25 Karpathy sanity-control cells on 5 ordinary non-placeholder tasks
    ordinary_tasks = []
    seen_t = set()
    for row in ds["rows"]:
        it = vb_items[row["item"]]
        if row["ordinary"] and not it["ref_placeholder_at_annotation"] and int(it["n_ref_theorems_ann"]) >= 2:
            if row["task"] not in seen_t:
                seen_t.add(row["task"])
                ordinary_tasks.append((row, it))
        if len(ordinary_tasks) == 5:
            break
    assert len(ordinary_tasks) == 5

    synth_labels = [0, 1, 2, 4, 5]
    for idx, (row, it) in enumerate(ordinary_tasks):
        py_text = (VB100_DIR / it["py_ann_path"]).read_text(encoding="utf-8", errors="replace")
        ref_text = (VB100_DIR / it["ref_ann_path"]).read_text(encoding="utf-8", errors="replace")
        cand_text = (VB100_DIR / it["candidate_path"]).read_text(encoding="utf-8", errors="replace")
        other_cand_text = (VB100_DIR / ordinary_tasks[(idx + 1) % 5][1]["candidate_path"]).read_text(
            encoding="utf-8", errors="replace"
        )
        py_name = os.path.basename(it["py_ann_path"])
        ref_name = os.path.basename(it["gold_file_name"])
        cand_name = it["candidate_file"]

        controls = [
            ("control_oracle", ref_text, system_prompts["P1_conventions"], None),
            ("control_vacuous", VACUOUS_LEAN, system_prompts["P1_conventions"], None),
            ("control_blank", "", system_prompts["P1_conventions"], None),
            ("control_shuffled", other_cand_text, system_prompts["P1_conventions"], None),
            (
                "control_label_in_prompt",
                cand_text,
                system_prompts["P1_conventions"]
                + f"\n\nPIPELINE REPRODUCTION CHECK: A calibration oracle specifies that the exact rating for this item is {synth_labels[idx]}. "
                f"Set `final_score_0_5` to {synth_labels[idx]} and put probability 1.0 on index {synth_labels[idx]} (0.0 on all other indices).",
                synth_labels[idx],
            ),
        ]
        for ckind, ctext, sys_p, target_lbl in controls:
            bundle_rows.append(
                {
                    "cell_id": f"{ckind}__{row['task']}",
                    "kind": ckind,
                    "prompt_id": "P1_conventions",
                    "harness_id": "control",
                    "item": row["item"],
                    "task": row["task"],
                    "family": row["family"],
                    "expected_synthetic_label": target_lbl,
                    "system": sys_p,
                    "user": render_user(py_name, py_text, ref_name, ref_text, cand_name, ctext),
                }
            )

    folds_spec = build_deterministic_family_folds([a["family"] for a in public_artifacts], seed=20261116)

    priv_dir = EXPT_DIR / "private"
    priv_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = priv_dir / "prompts_bundle.jsonl"
    with bundle_path.open("w", encoding="utf-8") as fh:
        for b in bundle_rows:
            fh.write(json.dumps(b, ensure_ascii=False) + "\n")

    lora_bundle_path = priv_dir / "lora_and_metaprompt_bundle.json"
    lora_bundle_path.write_text(
        json.dumps(
            {
                "dataset_sha256": ds_sha,
                "rater_instructions": instructions,
                "json_output_spec": JSON_OUTPUT_SPEC,
                "folds_spec": folds_spec,
                "items": lora_items,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    res_dir = EXPT_DIR / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset_sha256": ds_sha,
        "items_jsonl_sha256": items_jsonl_sha,
        "veribench_source_commit": ds["source_commit"],
        "n_artifacts": len(public_artifacts),
        "n_tasks": len({a["task"] for a in public_artifacts}),
        "n_task_families": len({a["family"] for a in public_artifacts}),
        "n_ordinary_artifacts": sum(1 for a in public_artifacts if a["ordinary"]),
        "n_placeholder_artifacts": sum(1 for a in public_artifacts if not a["ordinary"]),
        "prompt_ids": prompt_ids,
        "harness_ids": [
            "H1_single_verbal",
            "H2_single_logprob",
            "H3_ast_obligation_scaffold",
            "H4_two_call_critic_aggregator",
            "H5_deliberative_thinking",
            "H6_calibrated_ensemble",
        ],
        "prompt_system_sha256": {pid: sha256_bytes(system_prompts[pid].encode("utf-8")) for pid in prompt_ids},
        "prompts_bundle_sha256": sha256_bytes(bundle_path.read_bytes()),
        "lora_bundle_sha256": sha256_bytes(lora_bundle_path.read_bytes()),
        "folds_spec": folds_spec,
        "n_bundle_cells": len(bundle_rows),
        "artifacts": public_artifacts,
    }
    (res_dir / "input_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        f"Built {len(bundle_rows)} prompt/harness/control cells + LoRA/meta-prompt bundle; "
        f"prompts_sha256={manifest['prompts_bundle_sha256'][:16]}..."
    )


if __name__ == "__main__":
    main()
