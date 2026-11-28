#!/usr/bin/env python3
"""Build the frozen local-judge subset (PROTOCOL.md, 'Local open-weight panel') and the label-free SummEval prompt table.
Usage: python make_subset.py SUMMEVAL_CACHE_DIR   (the cache written by 23_*/summeval/prepare.py; holds gpt4_*_detailed.json)
"""
import json, sys
from pathlib import Path
import numpy as np

EXP = Path(__file__).resolve().parent.parent
E28 = EXP.parent / "28_public_rating_sets_chaosnli_dices_frontier_judges_ce_vs_pearson_start_23Nov2026"
SETS = ["chaosnli_snli", "chaosnli_mnli", "chaosnli_alphanli", "dices350", "dices990", "summeval_consistency", "summeval_relevance"]


def main(cache):
    ids = {}
    for line in (E28 / "requests.jsonl").read_text().splitlines():
        r = json.loads(line); ids.setdefault(r["id"].split("/")[0], []).append(r["id"])
    subset = {}
    for k, s in enumerate(SETS[:5]):
        perm = np.random.default_rng(np.random.SeedSequence([20261127, 46, k])).permutation(len(ids[s]))
        subset[s] = [ids[s][j] for j in perm[:150]]
    prompts = {}
    for k, (s, short) in enumerate([("summeval_consistency", "con"), ("summeval_relevance", "rel")], start=5):
        rows = json.loads((Path(cache) / f"gpt4_{short}_detailed.json").read_text())
        by = {(r["doc_id"], r["system_id"]): r for r in rows}
        docs = sorted({d for d, _ in by})
        perm = np.random.default_rng(np.random.SeedSequence([20261127, 46, k])).permutation(len(docs))
        chosen = [docs[j] for j in perm[:15]]
        items = sorted(key for key in by if key[0] in chosen)
        assert len(items) == 240
        subset[s] = [f"{d}::{m}" for d, m in items]
        for d, m in items:
            prompts[f"{d}::{m}"] = [by[d, m]["source"], by[d, m]["system_output"]]
    (EXP / "subset_ids.json").write_text(json.dumps(subset, indent=0))
    (EXP / "summeval_prompts.json").write_text(json.dumps(prompts))
    print({s: len(v) for s, v in subset.items()})


if __name__ == "__main__":
    main(sys.argv[1])
