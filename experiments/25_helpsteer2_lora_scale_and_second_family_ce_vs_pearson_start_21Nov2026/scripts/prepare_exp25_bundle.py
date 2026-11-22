#!/usr/bin/env python3
"""Prepare NVIDIA HelpSteer2 prompt-disjoint master bundle for Experiment 25 (Scale + Second Family)."""
from __future__ import annotations

import gzip
import hashlib
import json
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

COMMIT_SHA = "990b2711a36180dd19d9c94b8627844866f8982a"
BASE_URL = f"https://huggingface.co/datasets/nvidia/HelpSteer2/resolve/{COMMIT_SHA}"
FILES = {
    "disagreements": "disagreements/disagreements.jsonl.gz",
    "train": "train.jsonl.gz",
    "validation": "validation.jsonl.gz",
}

EXP_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = Path("/tmp/exp25_helpsteer2_raw")
MANIFEST_PATH = EXP_DIR / "results" / "data_manifest.json"
BUNDLE_PATH = RAW_DIR / "exp25_bundle.json"

N_TRAIN_A = 4096
N_VAL_A = 2048
N_CAL_A = 2048
DRAWS_A = 20
BUDGETS_A = [512, 1024, 2048, 4096, 8192]

N_TRAIN_B = 1600
N_VAL_B = 800
N_CAL_B = 800
DRAWS_B = 10
BUDGETS_B = [16, 32, 64, 128, 256, 512, 1024, 2048]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_str(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def download_or_load(rel_path: str) -> tuple[Path, str]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    local_path = RAW_DIR / Path(rel_path).name
    if not local_path.exists():
        req = urllib.request.Request(f"{BASE_URL}/{rel_path}", headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            local_path.write_bytes(resp.read())
    return local_path, sha256_bytes(local_path.read_bytes())


def load_jsonl_gz(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def leave_one_vote_out_ceiling(vote_lists: list[list[int]]) -> dict[str, float]:
    rps_items, ce_items, exact_items = [], [], []
    for votes in vote_lists:
        k = len(votes)
        assert k >= 2
        counts = np.bincount(votes, minlength=5).astype(np.float64)
        item_rps = item_ce = item_acc = 0.0
        for r in range(5):
            if counts[r] == 0:
                continue
            rem = counts.copy()
            rem[r] -= 1.0
            prob = (rem + 0.5) / (rem.sum() + 2.5)
            obs = np.zeros(5, dtype=np.float64)
            obs[r] = 1.0
            rps_r = float(np.mean((np.cumsum(prob)[:4] - np.cumsum(obs)[:4]) ** 2))
            ce_r = float(-np.log(prob[r]))
            modes = prob >= prob.max() - 1e-12
            acc_r = float(modes[r] / modes.sum())
            w = counts[r] / k
            item_rps += w * rps_r
            item_ce += w * ce_r
            item_acc += w * acc_r
        rps_items.append(item_rps)
        ce_items.append(item_ce)
        exact_items.append(item_acc)
    return {
        "rps": float(np.mean(rps_items)),
        "ce_nats": float(np.mean(ce_items)),
        "map_accuracy": float(np.mean(exact_items)),
        "n_items": len(vote_lists),
        "n_votes": int(sum(len(v) for v in vote_lists)),
    }


def make_draws(items: list[dict], train_ids: list[int], val_ids: list[int], cal_ids: list[int], n_draws: int, seed_tag: int) -> list[dict]:
    draws = []
    for draw_id in range(n_draws):
        ss = np.random.SeedSequence([20261121, seed_tag, draw_id])
        rng_tr, rng_va, rng_ca, rng_v = [np.random.default_rng(c) for c in ss.spawn(4)]
        tr_perm = rng_tr.permutation(train_ids).tolist()
        va_perm = rng_va.permutation(val_ids).tolist()
        ca_perm = rng_ca.permutation(cal_ids).tolist()
        sampled_pair: dict[str, list[int]] = {}
        for iid in tr_perm + va_perm + ca_perm:
            votes = items[iid]["votes"]
            chosen_pos = rng_v.choice(len(votes), size=2, replace=False)
            sampled_pair[str(iid)] = [int(votes[chosen_pos[0]]), int(votes[chosen_pos[1]])]
        draws.append({
            "draw_id": draw_id,
            "train_order": tr_perm,
            "val_order": va_perm,
            "cal_order": ca_perm,
            "sampled_vote_pairs": sampled_pair,
        })
    return draws


def assert_disjoint(items: list[dict], splits: dict[str, list[int]]) -> None:
    psets = [{items[i]["prompt"] for i in splits[k]} for k in ("train_ids", "val_ids", "cal_ids", "test_ids")]
    for a in range(len(psets)):
        for b in range(a + 1, len(psets)):
            assert len(psets[a] & psets[b]) == 0, f"Prompt overlap between split {a} and {b}"


def pool_stats(pool: list[dict]) -> dict:
    v_lens = Counter(len(r["votes"]) for r in pool)
    v_vals = Counter(v for r in pool for v in r["votes"])
    all_v = [v for r in pool for v in r["votes"]]
    return {
        "n_responses": len(pool),
        "n_unique_prompts": len({r["prompt"] for r in pool}),
        "total_votes": len(all_v),
        "vote_length_counts": dict(sorted((str(k), v) for k, v in v_lens.items())),
        "vote_value_counts": dict(sorted((str(k), v) for k, v in v_vals.items())),
        "mean_vote": float(np.mean(all_v)),
        "leave_one_vote_out_ceiling": leave_one_vote_out_ceiling([r["votes"] for r in pool]),
    }


def main() -> None:
    file_hashes: dict[str, str] = {}
    loaded: dict[str, list[dict]] = {}
    for name, rel in FILES.items():
        path, digest = download_or_load(rel)
        file_hashes[rel] = digest
        loaded[name] = load_jsonl_gz(path)

    dis_raw, train_raw, val_raw = loaded["disagreements"], loaded["train"], loaded["validation"]
    dedup_dis: dict[tuple[str, str], dict] = {}
    for row in dis_raw:
        key = (row["prompt"], row["response"])
        if key not in dedup_dis:
            dedup_dis[key] = row
    assert len(dis_raw) == 23652 and len(dedup_dis) == 23651

    train_by_prompt: dict[str, list[dict]] = defaultdict(list)
    for row in train_raw:
        key = (row["prompt"], row["response"])
        votes = [int(v) for v in dedup_dis[key]["helpfulness"]]
        assert len(votes) >= 2 and all(0 <= v <= 4 for v in votes)
        train_by_prompt[row["prompt"]].append({
            "prompt": row["prompt"],
            "response": row["response"],
            "votes": votes,
            "official_helpfulness": int(row["helpfulness"]),
        })

    test_items: list[dict] = []
    test_prompts_set: set[str] = set()
    for row in val_raw:
        key = (row["prompt"], row["response"])
        votes = [int(v) for v in dedup_dis[key]["helpfulness"]]
        assert len(votes) >= 2 and all(0 <= v <= 4 for v in votes)
        test_prompts_set.add(row["prompt"])
        test_items.append({
            "prompt": row["prompt"],
            "response": row["response"],
            "votes": votes,
            "official_helpfulness": int(row["helpfulness"]),
        })

    train_prompts_set = set(train_by_prompt.keys())
    assert len(train_prompts_set & test_prompts_set) == 0
    ranked_prompts = sorted(train_prompts_set, key=lambda p: sha256_str(f"exp24-prompt|{p}"))
    assert len(ranked_prompts) >= N_TRAIN_A + N_VAL_A + N_CAL_A

    def pick_one(prompts: list[str]) -> list[dict]:
        out = []
        for p in prompts:
            cands = sorted(train_by_prompt[p], key=lambda r: sha256_str(f"exp24-resp|{r['prompt']}\0{r['response']}"))
            out.append(dict(cands[0]))
        return out

    # Master list: [0:8192] ranked official train prompts, [8192:9230] official validation test items
    dev_pool = pick_one(ranked_prompts[: N_TRAIN_A + N_VAL_A + N_CAL_A])
    items = dev_pool + test_items
    for idx, it in enumerate(items):
        it["item_id"] = idx
        it["prompt_sha256"] = sha256_str(it["prompt"])
        it["pair_sha256"] = sha256_str(it["prompt"] + "\0" + it["response"])

    test_ids = list(range(len(dev_pool), len(items)))
    splits_a = {
        "train_ids": list(range(0, N_TRAIN_A)),
        "val_ids": list(range(N_TRAIN_A, N_TRAIN_A + N_VAL_A)),
        "cal_ids": list(range(N_TRAIN_A + N_VAL_A, N_TRAIN_A + N_VAL_A + N_CAL_A)),
        "test_ids": test_ids,
    }
    splits_b = {
        "train_ids": list(range(0, N_TRAIN_B)),
        "val_ids": list(range(N_TRAIN_B, N_TRAIN_B + N_VAL_B)),
        "cal_ids": list(range(N_TRAIN_B + N_VAL_B, N_TRAIN_B + N_VAL_B + N_CAL_B)),
        "test_ids": test_ids,
    }
    assert_disjoint(items, splits_a)
    assert_disjoint(items, splits_b)

    draws_a = make_draws(items, splits_a["train_ids"], splits_a["val_ids"], splits_a["cal_ids"], DRAWS_A, seed_tag=25)
    draws_b = make_draws(items, splits_b["train_ids"], splits_b["val_ids"], splits_b["cal_ids"], DRAWS_B, seed_tag=24)

    bundle = {
        "dataset": "nvidia/HelpSteer2",
        "commit_sha": COMMIT_SHA,
        "criterion": "helpfulness",
        "scale": [0, 1, 2, 3, 4],
        "items": items,
        "studies": {
            "qwen3_4b": {
                "model_key": "qwen3_4b",
                "n_draws": DRAWS_A,
                "budgets": BUDGETS_A,
                "max_steps_cap": 256,
                "splits": splits_a,
                "draws": draws_a,
            },
            "llama31_8b": {
                "model_key": "llama31_8b",
                "n_draws": DRAWS_B,
                "budgets": BUDGETS_B,
                "max_steps_cap": 160,
                "splits": splits_b,
                "draws": draws_b,
            },
        },
    }
    BUNDLE_PATH.write_text(json.dumps(bundle), encoding="utf-8")
    bundle_sha = sha256_bytes(BUNDLE_PATH.read_bytes())

    est_tokens_a = sum((len(it["prompt"]) + len(it["response"])) / 3.8 + 135 for it in items)
    b_ids_set = sorted(set(splits_b["train_ids"] + splits_b["val_ids"] + splits_b["cal_ids"] + splits_b["test_ids"]))
    est_tokens_b = sum((len(items[i]["prompt"]) + len(items[i]["response"])) / 3.8 + 135 for i in b_ids_set)

    manifest = {
        "dataset": "nvidia/HelpSteer2",
        "commit_sha": COMMIT_SHA,
        "raw_file_sha256": file_hashes,
        "prepared_bundle_path": str(BUNDLE_PATH),
        "prepared_bundle_sha256": bundle_sha,
        "n_master_items": len(items),
        "studies": {
            "qwen3_4b": {
                "n_draws": DRAWS_A,
                "budgets": BUDGETS_A,
                "est_cache_gb": round(est_tokens_a * 2560 * 2 / 1e9, 2),
                "train_pool": pool_stats([items[i] for i in splits_a["train_ids"]]),
                "val_pool": pool_stats([items[i] for i in splits_a["val_ids"]]),
                "cal_pool": pool_stats([items[i] for i in splits_a["cal_ids"]]),
                "test_set": pool_stats([items[i] for i in splits_a["test_ids"]]),
            },
            "llama31_8b": {
                "n_draws": DRAWS_B,
                "budgets": BUDGETS_B,
                "est_cache_gb": round(est_tokens_b * 4096 * 2 / 1e9, 2),
                "train_pool": pool_stats([items[i] for i in splits_b["train_ids"]]),
                "val_pool": pool_stats([items[i] for i in splits_b["val_ids"]]),
                "cal_pool": pool_stats([items[i] for i in splits_b["cal_ids"]]),
                "test_set": pool_stats([items[i] for i in splits_b["test_ids"]]),
            },
        },
    }
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared bundle: {BUNDLE_PATH} ({BUNDLE_PATH.stat().st_size/1e6:.2f} MB, sha256={bundle_sha[:16]}...)")
    print("Est cache GB — Qwen3-4B (9,230 items):", manifest["studies"]["qwen3_4b"]["est_cache_gb"],
          "Llama-3.1-8B (4,238 items):", manifest["studies"]["llama31_8b"]["est_cache_gb"])


if __name__ == "__main__":
    main()
