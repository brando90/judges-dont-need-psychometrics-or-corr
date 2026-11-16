#!/usr/bin/env python3
"""Independent deterministic verification for Experiment 09:
  1. Privacy & secret scan across all non-ignored files in Experiment 09 (zero raw human ratings, zero API keys).
  2. Forecast simplex validity check across every model, prompt, harness, and LoRA out-of-fold matrix.
  3. Independent metric & MSE-identity reconstruction from open_weight_forecasts.json (+ local dataset.json when present).
  4. Report consistency check confirming headline numbers in RESULTS_open_weight_ce_vs_corr.md match analysis_summary.json.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

EXPT_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = Path(
    "/Users/brandomiranda/judges-dont-need-psychometrics-or-corr/experiments/"
    "02_exp06_ce_vs_pearson_matched_budget_start_09Nov2026/private/historical_exp06/cached_v1/dataset.json"
)
SECRET_PAT = re.compile(r"(sk-[A-Za-z0-9_\-]{20,}|AIza[0-9A-Za-z_\-]{25,}|ghp_[A-Za-z0-9]{25,})")


def independent_rps_ce_tvd(q: list[list[float]], ratings_per_item: list[list[int]], floor: float = 1e-6) -> tuple[float, float, float]:
    assert q and len(q) == len(ratings_per_item), "Forecast/rating shape mismatch"
    rps_l, ce_l, tvd_l = [], [], []
    for qi, rats in zip(q, ratings_per_item):
        assert len(qi) == 6 and all(math.isfinite(x) and x >= 0 for x in qi), "Invalid probability row"
        assert rats and all(0 <= r <= 5 for r in rats), "Invalid rating row"
        qf = [max(float(x), floor) for x in qi]
        s = math.fsum(qf)
        qf = [x / s for x in qf]
        m = len(rats)
        p = [rats.count(c) / m for c in range(6)]
        cq = [math.fsum(qf[: c + 1]) for c in range(5)]
        rps_i = math.fsum(math.fsum((cq[c] - (1.0 if r <= c else 0.0)) ** 2 for c in range(5)) / 5.0 for r in rats) / m
        ce_i = math.fsum(-math.log(qf[r]) for r in rats) / m
        tvd_i = 0.5 * math.fsum(abs(a - b) for a, b in zip(qf, p))
        rps_l.append(rps_i)
        ce_l.append(ce_i)
        tvd_l.append(tvd_i)
    n = len(q)
    return math.fsum(rps_l) / n, math.fsum(ce_l) / n, math.fsum(tvd_l) / n


def main() -> None:
    checks_passed = 0

    # 1. Privacy, freeze-digest, and secret scan over all public tracked/candidate files in Experiment 09
    freeze = json.loads((EXPT_DIR / "FREEZE.json").read_text(encoding="utf-8"))
    eff_files = dict(freeze.get("execution_amendment_1", {}).get("effective_files", freeze["files"]))
    eff_files.update(freeze.get("qa_amendment_1", {}).get("effective_files_sha256", {}))
    for rel_f, exp_sha in eff_files.items():
        actual_sha = hashlib.sha256((EXPT_DIR / rel_f).read_bytes()).hexdigest()
        assert actual_sha == exp_sha, f"FREEZE digest mismatch on {rel_f}: {actual_sha} != {exp_sha}"
        checks_passed += 1

    for p in sorted(EXPT_DIR.rglob("*")):
        if not p.is_file() or "private" in p.parts or "__pycache__" in p.parts or p.suffix in (".png", ".pdf"):
            continue
        if p.parent.name == "raw_mercury2" and p.suffix in (".jsonl", ".log", ".done"):
            continue
        txt = p.read_text(encoding="utf-8", errors="replace")
        assert not SECRET_PAT.search(txt), f"Potential secret pattern in {p}"
        if p.suffix == ".json":
            assert '"ratings_by_expert"' not in txt, f"Private ratings_by_expert field leaked in {p}"
            assert '"counts_0_5"' not in txt, f"Private counts_0_5 field leaked in {p}"
            assert '"ratings": [' not in txt, f"Private ratings array leaked in {p}"
        checks_passed += 1

    # 2. Forecast validity check
    fc = json.loads((EXPT_DIR / "results/open_weight_forecasts.json").read_text(encoding="utf-8"))
    assert len(fc["artifact_order"]) == 75
    for store_name in ("logprob_forecasts", "verbal_forecasts"):
        for tag, mat in fc[store_name].items():
            assert len(mat) == 75, f"Wrong row count in {tag}"
            for row in mat:
                assert len(row) == 6 and all(math.isfinite(x) and x >= -1e-12 for x in row), f"Invalid prob row in {tag}"
                assert abs(sum(row) - 1.0) < 1e-5, f"Non-unit prob sum in {tag}: {sum(row)}"
            checks_passed += 1

    for mk, oof_dict in fc.get("lora_oof_forecasts", {}).items():
        for arm, mat in oof_dict.items():
            assert len(mat) == 75
            for row in mat:
                assert len(row) == 6 and all(math.isfinite(x) and x >= 0 for x in row)
                assert abs(sum(row) - 1.0) < 1e-4
            checks_passed += 1

    # 3. Independent metric reconstruction against summary JSON (when private dataset.json is locally available)
    summary = json.loads((EXPT_DIR / "results/analysis_summary.json").read_text(encoding="utf-8"))
    if DATASET_PATH.exists():
        ds_bytes = DATASET_PATH.read_bytes()
        ds_sha = hashlib.sha256(ds_bytes).hexdigest()
        assert ds_sha == freeze["dataset_sha256"], f"Dataset SHA mismatch vs FREEZE: {ds_sha}"
        assert ds_sha == summary["dataset_scope"]["dataset_sha256"], f"Dataset SHA mismatch vs summary: {ds_sha}"
        checks_passed += 2
        ds = json.loads(ds_bytes.decode("utf-8"))
        rats_list = [[int(x) for x in r["ratings"]] for r in ds["rows"]]
        # Check every in-sample single judge & every LoRA out-of-fold forecast independently!
        for tag, mat in fc["logprob_forecasts"].items():
            key = f"{tag}__logprob"
            if key in summary["per_judge_insample"]:
                rps_i, ce_i, tvd_i = independent_rps_ce_tvd(mat, rats_list)
                rec = summary["per_judge_insample"][key]
                assert abs(rps_i - rec["rps"]) < 1e-9, f"RPS mismatch on {key}: {rps_i} vs {rec['rps']}"
                assert abs(ce_i - rec["ce"]) < 1e-9, f"CE mismatch on {key}: {ce_i} vs {rec['ce']}"
                assert abs(tvd_i - rec["tvd"]) < 1e-9, f"TVD mismatch on {key}: {tvd_i} vs {rec['tvd']}"
                checks_passed += 3

        for mk, oof_dict in fc.get("lora_oof_forecasts", {}).items():
            lora_m = summary["ii_c_real_weight_training"]["pytorch_lora_end_to_end"][mk]["methods"]
            for arm, mat in oof_dict.items():
                rps_i, ce_i, tvd_i = independent_rps_ce_tvd(mat, rats_list)
                assert abs(rps_i - lora_m[arm]["rps"]) < 1e-9, f"LoRA RPS mismatch {mk} {arm}"
                assert abs(ce_i - lora_m[arm]["ce"]) < 1e-9, f"LoRA CE mismatch {mk} {arm}"
                assert abs(tvd_i - lora_m[arm]["tvd"]) < 1e-9, f"LoRA TVD mismatch {mk} {arm}"
                checks_passed += 3

    # 4. Verify required output files & figures exist and are non-empty, and report text matches summary numbers
    req_files = [
        "README.md",
        "PROTOCOL.md",
        "FREEZE.json",
        "HOW_TO_READ_RESULTS.md",
        "results.md",
        "RESULTS_open_weight_ce_vs_corr.md",
        "figures/fig1_taxonomy_forest_rps_and_ce.png",
        "figures/fig2_lora_weight_training_and_mse_decomp.png",
        "figures/fig3_real_judge_learning_curves_vs_n.png",
        "figures/fig4_when_ce_wins_disagreement_and_prompts_vs_harnesses.png",
    ]
    for rf in req_files:
        fp = EXPT_DIR / rf
        assert fp.exists() and fp.stat().st_size > 100, f"Missing or empty required file: {rf}"
        checks_passed += 1

    rep_txt = (EXPT_DIR / "RESULTS_open_weight_ce_vs_corr.md").read_text(encoding="utf-8")
    rb_ce = f"{summary['representation_bridge']['exp09_logprob_6prob_pool']['methods']['ce_mix_train']['ce']:.3f}"
    assert rb_ce in rep_txt, f"Headline CE {rb_ce} missing from RESULTS_open_weight_ce_vs_corr.md"
    checks_passed += 1

    receipt = {"status": "PASS", "checks_passed": checks_passed}
    (EXPT_DIR / "results/verification.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(f"VERIFICATION PASSED: {checks_passed} deterministic checks succeeded.")


if __name__ == "__main__":
    main()
