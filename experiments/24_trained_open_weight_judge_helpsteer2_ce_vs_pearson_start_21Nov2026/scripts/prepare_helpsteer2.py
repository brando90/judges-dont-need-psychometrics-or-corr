#!/usr/bin/env python3
"""Prepare NVIDIA HelpSteer2 helpfulness votes and prompt-disjoint nested draws for Experiment 24."""
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
RAW_DIR = Path("/tmp/exp24_helpsteer2_raw")
MANIFEST_PATH = EXP_DIR / "results" / "data_manifest.json"
BUNDLE_PATH = RAW_DIR / "prepared_bundle.json"

N_TRAIN_POOL = 1600
N_VAL_POOL = 800
N_CAL_POOL = 800
N_DRAWS = 10
BUDGETS_D1 = [16, 32, 64, 128, 256, 512, 1024, 2048]
BUDGETS_D2 = [64, 256, 1024]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_str(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def download_or_load(name: str, rel_path: str) -> tuple[Path, str]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    local_path = RAW_DIR / Path(rel_path).name
    if not local_path.exists():
        req = urllib.request.Request(f"{BASE_URL}/{rel_path}", headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            local_path.write_bytes(resp.read())
    digest = sha256_bytes(local_path.read_bytes())
    return local_path, digest


def load_jsonl_gz(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def leave_one_vote_out_ceiling(vote_lists: list[list[int]]) -> dict[str, float]:
    """Predict one held-out human vote from remaining votes + Jeffreys 0.5 prior per class."""
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
            cdf_diff = np.cumsum(prob)[:4] - np.cumsum(obs)[:4]
            rps_r = float(np.mean(cdf_diff ** 2))
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


def _load_raw_tables() -> tuple[dict[str, str], dict[str, list[dict]]]:
    file_hashes: dict[str, str] = {}
    loaded: dict[str, list[dict]] = {}
    for name, rel in FILES.items():
        path, digest = download_or_load(name, rel)
        file_hashes[rel] = digest
        loaded[name] = load_jsonl_gz(path)
    return file_hashes, loaded


def _dedup_disagreements(dis_raw: list[dict]) -> tuple[dict[tuple[str, str], dict], list[dict]]:
    dedup_dis: dict[tuple[str, str], dict] = {}
    dup_details: list[dict] = []
    for idx, row in enumerate(dis_raw):
        key = (row["prompt"], row["response"])
        if key in dedup_dis:
            dup_details.append({
                "duplicate_row_index": idx,
                "first_votes": dedup_dis[key]["helpfulness"],
                "duplicate_votes": row["helpfulness"],
            })
            continue
        dedup_dis[key] = row
    assert len(dis_raw) == 23652 and len(dedup_dis) == 23651, (len(dis_raw), len(dedup_dis))
    assert len(dup_details) == 1 and dup_details[0]["first_votes"] == dup_details[0]["duplicate_votes"]
    return dedup_dis, dup_details


def _extract_item(row: dict, dedup_dis: dict[tuple[str, str], dict], split_label: str) -> dict:
    key = (row["prompt"], row["response"])
    assert key in dedup_dis, f"Missing {split_label} key in disagreements"
    votes = [int(v) for v in dedup_dis[key]["helpfulness"]]
    assert len(votes) >= 2 and all(0 <= v <= 4 for v in votes)
    return {
        "prompt": row["prompt"],
        "response": row["response"],
        "votes": votes,
        "official_helpfulness": int(row["helpfulness"]),
        "split": split_label,
    }


def _pick_one_per_prompt(train_by_prompt: dict[str, list[dict]], prompts: list[str], split_name: str) -> list[dict]:
    out = []
    for p in prompts:
        cands = sorted(train_by_prompt[p], key=lambda r: sha256_str(f"exp24-resp|{r['prompt']}\0{r['response']}"))
        chosen = dict(cands[0])
        chosen["split"] = split_name
        out.append(chosen)
    return out


def _assert_disjoint_pools(pools: list[list[dict]]) -> None:
    prompt_sets = [{r["prompt"] for r in pool} for pool in pools]
    for i, p_i in enumerate(prompt_sets):
        for p_j in prompt_sets[i + 1:]:
            assert not (p_i & p_j)


def _make_draws(all_items: list[dict], train_ids: list[int], val_ids: list[int], cal_ids: list[int]) -> list[dict]:
    draws = []
    for draw_id in range(N_DRAWS):
        ss = np.random.SeedSequence([20261121, 24, draw_id])
        rng_tr, rng_va, rng_ca, rng_v = [np.random.default_rng(c) for c in ss.spawn(4)]
        tr_perm = rng_tr.permutation(train_ids).tolist()
        va_perm = rng_va.permutation(val_ids).tolist()
        ca_perm = rng_ca.permutation(cal_ids).tolist()
        sampled_pair: dict[str, list[int]] = {}
        for iid in tr_perm + va_perm + ca_perm:
            votes = all_items[iid]["votes"]
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


def _pool_stats(pool: list[dict]) -> dict:
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
    file_hashes, loaded = _load_raw_tables()
    dis_raw, train_raw, val_raw = loaded["disagreements"], loaded["train"], loaded["validation"]
    dedup_dis, dup_details = _dedup_disagreements(dis_raw)

    train_by_prompt: dict[str, list[dict]] = defaultdict(list)
    for row in train_raw:
        train_by_prompt[row["prompt"]].append(_extract_item(row, dedup_dis, "official_train"))
    test_items = [_extract_item(row, dedup_dis, "test") for row in val_raw]

    train_prompts_set = set(train_by_prompt.keys())
    test_prompts_set = {it["prompt"] for it in test_items}
    assert not (train_prompts_set & test_prompts_set), "Prompt overlap between official train and val!"

    ranked = sorted(train_prompts_set, key=lambda p: sha256_str(f"exp24-prompt|{p}"))
    assert len(ranked) >= N_TRAIN_POOL + N_VAL_POOL + N_CAL_POOL
    train_pool = _pick_one_per_prompt(train_by_prompt, ranked[:N_TRAIN_POOL], "train_pool")
    val_pool = _pick_one_per_prompt(train_by_prompt, ranked[N_TRAIN_POOL:N_TRAIN_POOL + N_VAL_POOL], "val_pool")
    cal_pool = _pick_one_per_prompt(
        train_by_prompt, ranked[N_TRAIN_POOL + N_VAL_POOL:N_TRAIN_POOL + N_VAL_POOL + N_CAL_POOL], "cal_pool"
    )
    _assert_disjoint_pools([train_pool, val_pool, cal_pool, test_items])

    all_items = train_pool + val_pool + cal_pool + test_items
    for idx, it in enumerate(all_items):
        it["item_id"] = idx
        it["prompt_sha256"] = sha256_str(it["prompt"])
        it["pair_sha256"] = sha256_str(it["prompt"] + "\0" + it["response"])

    splits = {
        "train_ids": [it["item_id"] for it in train_pool],
        "val_ids": [it["item_id"] for it in val_pool],
        "cal_ids": [it["item_id"] for it in cal_pool],
        "test_ids": [it["item_id"] for it in test_items],
    }
    draws = _make_draws(all_items, splits["train_ids"], splits["val_ids"], splits["cal_ids"])

    bundle = {
        "dataset": "nvidia/HelpSteer2",
        "commit_sha": COMMIT_SHA,
        "criterion": "helpfulness",
        "scale": [0, 1, 2, 3, 4],
        "budgets_d1": BUDGETS_D1,
        "budgets_d2": BUDGETS_D2,
        "n_draws": N_DRAWS,
        "splits": splits,
        "items": all_items,
        "draws": draws,
    }
    BUNDLE_PATH.write_text(json.dumps(bundle), encoding="utf-8")
    bundle_sha = sha256_bytes(BUNDLE_PATH.read_bytes())

    manifest = {
        "dataset": "nvidia/HelpSteer2",
        "commit_sha": COMMIT_SHA,
        "raw_file_sha256": file_hashes,
        "raw_counts": {
            "disagreements_rows": len(dis_raw),
            "disagreements_unique_pairs": len(dedup_dis),
            "duplicate_pair_details": dup_details,
            "official_train_rows": len(train_raw),
            "official_train_unique_prompts": len(train_prompts_set),
            "official_validation_rows": len(val_raw),
            "official_validation_unique_prompts": len(test_prompts_set),
            "train_val_prompt_overlap": len(train_prompts_set & test_prompts_set),
        },
        "prepared_bundle_path": str(BUNDLE_PATH),
        "prepared_bundle_sha256": bundle_sha,
        "n_draws": N_DRAWS,
        "budgets_d1": BUDGETS_D1,
        "budgets_d2": BUDGETS_D2,
        "pool_statistics": {
            "train_pool": _pool_stats(train_pool),
            "val_pool": _pool_stats(val_pool),
            "cal_pool": _pool_stats(cal_pool),
            "test_set": _pool_stats(test_items),
        },
    }
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared bundle: {BUNDLE_PATH} (sha256={bundle_sha[:16]}...)")
    print(f"Saved manifest: {MANIFEST_PATH}")
    print(json.dumps(manifest["pool_statistics"], indent=2))


if __name__ == "__main__":
    main()
