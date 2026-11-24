#!/usr/bin/env python3
"""Prepare group-disjoint multi-rater public datasets (ChaosNLI, DICES, HelpSteer2) for Experiment 29."""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

EXP_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = Path("/tmp/exp29_public_raw")
BUNDLE_PATH = RAW_DIR / "exp29_bundle.json"
MANIFEST_PATH = EXP_DIR / "results" / "data_manifest.json"

SOURCES = {
    "chaosNLI_snli.jsonl": (
        "https://huggingface.co/datasets/earino/chaosnli/resolve/eb7cc18dbfa0f161d93a2b22c7856ca0faafcf2b/raw/chaosNLI_snli.jsonl",
        "99f9015ddda7d85f66a087452bc30d53974314fe27e7d589e2f41ad44bd509c1",
    ),
    "chaosNLI_mnli_m.jsonl": (
        "https://huggingface.co/datasets/earino/chaosnli/resolve/eb7cc18dbfa0f161d93a2b22c7856ca0faafcf2b/raw/chaosNLI_mnli_m.jsonl",
        "8eb49b589488e7b1a9ec95fb8a864bd50f0509b4f808eed0aed3f8167617664d",
    ),
    "chaosNLI_alphanli.jsonl": (
        "https://huggingface.co/datasets/earino/chaosnli/resolve/eb7cc18dbfa0f161d93a2b22c7856ca0faafcf2b/raw/chaosNLI_alphanli.jsonl",
        "f55817ecc11a02ba3030cd7d2c567d23843456809020723abad53f4fde024c9e",
    ),
    "diverse_safety_adversarial_dialog_350.csv": (
        "https://raw.githubusercontent.com/google-research-datasets/dices-dataset/main/350/diverse_safety_adversarial_dialog_350.csv",
        "63cb8620eb04e64957f9e45b1f7cb3d079c4eed8e8b28909f46e0b7038ce4301",
    ),
    "diverse_safety_adversarial_dialog_990.csv": (
        "https://raw.githubusercontent.com/google-research-datasets/dices-dataset/main/990/diverse_safety_adversarial_dialog_990.csv",
        "14332b6391cdee80383b765c734de3bebf6d8553141ee7d2f47ef0910a12fcf7",
    ),
    "disagreements.jsonl.gz": (
        "https://huggingface.co/datasets/nvidia/HelpSteer2/resolve/990b2711a36180dd19d9c94b8627844866f8982a/disagreements/disagreements.jsonl.gz",
        "53435e0e2926620bf08ce91f278d7368ee6903f416a47764fe8839f3b71b3ab4",
    ),
    "train.jsonl.gz": (
        "https://huggingface.co/datasets/nvidia/HelpSteer2/resolve/990b2711a36180dd19d9c94b8627844866f8982a/train.jsonl.gz",
        "c0d7e91d738d42e8a08070db26c4c09a9c7631308e1f0fd380ff43d130c9f713",
    ),
    "validation.jsonl.gz": (
        "https://huggingface.co/datasets/nvidia/HelpSteer2/resolve/990b2711a36180dd19d9c94b8627844866f8982a/validation.jsonl.gz",
        "610eeb5289494d613c4c0f70aade2df8df0b499f3a24e76d232f74e6909d010a",
    ),
}
EXP24_BUNDLE_SHA256 = "c916357850510076f17c2974c45ff128873987eb9c4a4f8f0c9dae1ce47a475d"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_str(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def ensure_sources() -> dict[str, str]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}
    for fname, (url, expected_sha) in SOURCES.items():
        path = RAW_DIR / fname
        if not path.exists():
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                path.write_bytes(resp.read())
        digest = sha256_bytes(path.read_bytes())
        assert digest == expected_sha, f"Checksum mismatch for {fname}: {digest} != {expected_sha}"
        hashes[fname] = digest
    return hashes


def count_variance(counts: list[int]) -> float:
    p = np.asarray(counts, dtype=np.float64)
    p = p / p.sum()
    grid = np.arange(len(counts), dtype=np.float64)
    mean = float(np.dot(p, grid))
    return float(np.dot(p, (grid - mean) ** 2))


def leave_one_vote_out_ceiling(counts_list: list[list[int]], n_cat: int) -> dict[str, float]:
    rps_items, ce_items, acc_items = [], [], []
    for raw_c in counts_list:
        counts = np.asarray(raw_c, dtype=np.float64)
        total = float(counts.sum())
        assert total >= 2.0 and len(counts) == n_cat
        item_rps = item_ce = item_acc = 0.0
        for r in range(n_cat):
            if counts[r] == 0:
                continue
            rem = counts.copy()
            rem[r] -= 1.0
            prob = (rem + 0.5) / (rem.sum() + 0.5 * n_cat)
            obs = np.zeros(n_cat, dtype=np.float64)
            obs[r] = 1.0
            w = float(counts[r] / total)
            item_rps += w * float(np.mean((np.cumsum(prob)[: n_cat - 1] - np.cumsum(obs)[: n_cat - 1]) ** 2))
            item_ce += w * float(-np.log(prob[r]))
            modes = prob >= prob.max() - 1e-12
            item_acc += w * float(modes[r] / modes.sum())
        rps_items.append(item_rps)
        ce_items.append(item_ce)
        acc_items.append(item_acc)
    return {
        "rps": float(np.mean(rps_items)),
        "ce_nats": float(np.mean(ce_items)),
        "map_accuracy": float(np.mean(acc_items)),
        "n_items": len(counts_list),
        "n_votes": int(sum(sum(c) for c in counts_list)),
    }


def _assign_group_splits(items: list[dict], salt: str, targets: tuple[int, int, int]) -> dict[str, list[dict]]:
    by_group: dict[str, list[dict]] = defaultdict(list)
    for it in items:
        by_group[it["group_key"]].append(it)
    ordered_groups = sorted(by_group.keys(), key=lambda g: sha256_str(f"{salt}|{g}"))
    buckets: dict[str, list[dict]] = {"train_pool": [], "val_pool": [], "cal_pool": [], "test_set": []}
    for g in ordered_groups:
        if len(buckets["train_pool"]) < targets[0]:
            name = "train_pool"
        elif len(buckets["val_pool"]) < targets[1]:
            name = "val_pool"
        elif len(buckets["cal_pool"]) < targets[2]:
            name = "cal_pool"
        else:
            name = "test_set"
        for it in by_group[g]:
            it["split"] = name
            buckets[name].append(it)
    return buckets


def _assert_disjoint_groups(buckets: dict[str, list[dict]], also_key: str | None = None) -> None:
    keys = ["train_pool", "val_pool", "cal_pool", "test_set"]
    g_sets = [{it["group_key"] for it in buckets[k]} for k in keys]
    for i, gi in enumerate(g_sets):
        for gj in g_sets[i + 1:]:
            assert not (gi & gj), "Group overlap across splits"
    if also_key:
        a_sets = [{it[also_key] for it in buckets[k]} for k in keys]
        for i, ai in enumerate(a_sets):
            for aj in a_sets[i + 1:]:
                assert not (ai & aj), f"{also_key} overlap across splits"


def _make_count_draws(items: list[dict], splits: dict[str, list[int]], seed_tag: int, n_draws: int = 10) -> list[dict]:
    draws = []
    for draw_id in range(n_draws):
        ss = np.random.SeedSequence([20261123, seed_tag, draw_id])
        rng_tr, rng_va, rng_ca, rng_v = [np.random.default_rng(c) for c in ss.spawn(4)]
        tr_perm = rng_tr.permutation(splits["train_ids"]).tolist()
        va_perm = rng_va.permutation(splits["val_ids"]).tolist()
        ca_perm = rng_ca.permutation(splits["cal_ids"]).tolist()
        sampled: dict[str, list[int]] = {}
        for iid in tr_perm + va_perm + ca_perm:
            cnt = np.asarray(items[iid]["counts"], dtype=np.int64)
            pool = np.repeat(np.arange(len(cnt), dtype=np.int64), cnt)
            chosen = rng_v.choice(pool, size=5, replace=False)
            sampled[str(iid)] = [int(x) for x in chosen]
        draws.append({
            "draw_id": draw_id,
            "train_order": tr_perm,
            "val_order": va_perm,
            "cal_order": ca_perm,
            "sampled_votes_5": sampled,
        })
    return draws


def _pool_stats(buckets: dict[str, list[dict]], n_cat: int, med_var: float) -> dict:
    stats = {}
    for k in ("train_pool", "val_pool", "cal_pool", "test_set"):
        stats[k] = {
            "n_items": len(buckets[k]),
            "n_unique_groups": len({it["group_key"] for it in buckets[k]}),
            "total_votes": int(sum(sum(it["counts"]) for it in buckets[k])),
            "lovo": leave_one_vote_out_ceiling([it["counts"] for it in buckets[k]], n_cat),
        }
    stats["test_median_variance"] = med_var
    return stats


def _finalize_suite(
    name: str,
    n_cat: int,
    buckets: dict[str, list[dict]],
    seed_tag: int,
    budgets_d1: list[int],
    budgets_d5: list[int] | None = None,
) -> tuple[dict, dict]:
    all_items = buckets["train_pool"] + buckets["val_pool"] + buckets["cal_pool"] + buckets["test_set"]
    for idx, it in enumerate(all_items):
        it["item_id"] = idx
        it["variance"] = count_variance(it["counts"])
    med_var = float(np.median([it["variance"] for it in buckets["test_set"]]))
    for it in buckets["test_set"]:
        it["is_high_dis"] = bool(it["variance"] >= med_var)
    splits = {
        "train_ids": [it["item_id"] for it in buckets["train_pool"]],
        "val_ids": [it["item_id"] for it in buckets["val_pool"]],
        "cal_ids": [it["item_id"] for it in buckets["cal_pool"]],
        "test_ids": [it["item_id"] for it in buckets["test_set"]],
    }
    suite = {
        "name": name,
        "K": n_cat,
        "budgets_d1": budgets_d1,
        "budgets_d5": budgets_d5 or [],
        "splits": splits,
        "items": all_items,
        "draws": _make_count_draws(all_items, splits, seed_tag=seed_tag),
        "test_median_variance": med_var,
    }
    return suite, _pool_stats(buckets, n_cat, med_var)


def build_chaos3() -> tuple[dict, dict]:
    items = []
    for part, fn in (("snli", "chaosNLI_snli.jsonl"), ("mnli", "chaosNLI_mnli_m.jsonl")):
        for line in (RAW_DIR / fn).read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            ex = row["example"]
            lc = row["label_count"]
            counts = [int(lc[2]), int(lc[1]), int(lc[0])]
            assert sum(counts) == 100
            gkey = f"{part}|{row['uid'].split('#')[0] if part == 'snli' else row['uid']}"
            items.append({
                "task_kind": "chaos3",
                "subset": part,
                "uid": row["uid"],
                "group_key": gkey,
                "text_sha256": sha256_str(ex["premise"].strip() + "\0" + ex["hypothesis"].strip()),
                "premise": ex["premise"],
                "hypothesis": ex["hypothesis"],
                "counts": counts,
            })
    buckets = _assign_group_splits(items, "exp29-chaos3", (1200, 600, 600))
    _assert_disjoint_groups(buckets, also_key="text_sha256")
    return _finalize_suite("chaos3", 3, buckets, 2901, [16, 32, 64, 128, 256, 512, 1024, 2048], [80, 320, 1280])


def build_alpha2() -> tuple[dict, dict]:
    items = []
    for line in (RAW_DIR / "chaosNLI_alphanli.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        ex = row["example"]
        counts = [int(row["label_count"][0]), int(row["label_count"][1])]
        assert sum(counts) == 100
        thash = sha256_str("\0".join([ex["obs1"].strip(), ex["obs2"].strip(), ex["hyp1"].strip(), ex["hyp2"].strip()]))
        items.append({
            "task_kind": "alpha2",
            "subset": "alphanli",
            "uid": row["uid"],
            "group_key": f"alpha|{row['uid']}",
            "text_sha256": thash,
            "obs1": ex["obs1"],
            "obs2": ex["obs2"],
            "hyp1": ex["hyp1"],
            "hyp2": ex["hyp2"],
            "counts": counts,
        })
    buckets = _assign_group_splits(items, "exp29-alpha2", (560, 280, 280))
    _assert_disjoint_groups(buckets, also_key="text_sha256")
    return _finalize_suite("alpha2", 2, buckets, 2902, [16, 64, 256, 1024])


def build_dices3() -> tuple[dict, dict]:
    seen: dict[str, dict] = {}
    label_idx = {"No": 0, "Unsure": 1, "Yes": 2}
    for subset, fn in (("dices350", "diverse_safety_adversarial_dialog_350.csv"), ("dices990", "diverse_safety_adversarial_dialog_990.csv")):
        by_id: dict[str, list[dict]] = defaultdict(list)
        with (RAW_DIR / fn).open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                by_id[row["item_id"]].append(row)
        for iid in sorted(by_id, key=int):
            rs = by_id[iid]
            ctx = rs[0]["context"].replace("\\n", "\n").strip()
            resp = rs[0]["response"].strip()
            chash = sha256_str(ctx + "\0" + resp)
            if chash in seen:
                continue
            counts = [0, 0, 0]
            for r in rs:
                counts[label_idx[r["Q_overall"]]] += 1
            seen[chash] = {
                "task_kind": "dices3",
                "subset": subset,
                "uid": f"{subset}/{iid}",
                "group_key": chash,
                "context": ctx,
                "response": resp,
                "counts": counts,
            }
    buckets = _assign_group_splits(list(seen.values()), "exp29-dices3", (520, 260, 260))
    _assert_disjoint_groups(buckets)
    return _finalize_suite("dices3", 3, buckets, 2903, [16, 64, 256, 1024])


def _load_jsonl_gz(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def _pick_hs2_split(by_prompt: dict[str, list[dict]], prompts: list[str], split_name: str) -> list[dict]:
    out = []
    for p in prompts:
        cands = sorted(by_prompt[p], key=lambda r: sha256_str(f"exp24-resp|{r['prompt']}\0{r['response']}"))
        ch = dict(cands[0])
        ch["split"] = split_name
        out.append(ch)
    return out


def _make_hs2_draws(all_items: list[dict], splits: dict[str, list[int]]) -> list[dict]:
    draws = []
    for draw_id in range(10):
        ss = np.random.SeedSequence([20261121, 24, draw_id])
        rng_tr, rng_va, rng_ca, rng_v = [np.random.default_rng(c) for c in ss.spawn(4)]
        tr_perm = rng_tr.permutation(splits["train_ids"]).tolist()
        va_perm = rng_va.permutation(splits["val_ids"]).tolist()
        ca_perm = rng_ca.permutation(splits["cal_ids"]).tolist()
        sp = {}
        for iid in tr_perm + va_perm + ca_perm:
            picked = rng_v.choice(len(all_items[iid]["votes"]), size=2, replace=False)
            sp[str(iid)] = [int(all_items[iid]["votes"][i]) for i in picked]
        draws.append({"draw_id": draw_id, "train_order": tr_perm, "val_order": va_perm, "cal_order": ca_perm, "sampled_vote_pairs": sp})
    return draws


def _hs2_items_and_draws() -> tuple[list[dict], list[dict], dict[str, list[int]]]:
    dedup_dis = {
        (row["prompt"], row["response"]): row
        for row in reversed(_load_jsonl_gz(RAW_DIR / "disagreements.jsonl.gz"))
    }
    by_prompt: dict[str, list[dict]] = defaultdict(list)
    for row in _load_jsonl_gz(RAW_DIR / "train.jsonl.gz"):
        votes = [int(v) for v in dedup_dis[(row["prompt"], row["response"])]["helpfulness"]]
        by_prompt[row["prompt"]].append({"prompt": row["prompt"], "response": row["response"], "votes": votes, "official_helpfulness": int(row["helpfulness"]), "split": "official_train"})
    test_items = [
        {"prompt": row["prompt"], "response": row["response"], "votes": [int(v) for v in dedup_dis[(row["prompt"], row["response"])]["helpfulness"]], "official_helpfulness": int(row["helpfulness"]), "split": "test"}
        for row in _load_jsonl_gz(RAW_DIR / "validation.jsonl.gz")
    ]
    ranked = sorted(by_prompt.keys(), key=lambda p: sha256_str(f"exp24-prompt|{p}"))
    all_items = (
        _pick_hs2_split(by_prompt, ranked[:1600], "train_pool")
        + _pick_hs2_split(by_prompt, ranked[1600:2400], "val_pool")
        + _pick_hs2_split(by_prompt, ranked[2400:3200], "cal_pool")
        + test_items
    )
    for idx, it in enumerate(all_items):
        it["item_id"] = idx
        it["prompt_sha256"] = sha256_str(it["prompt"])
        it["pair_sha256"] = sha256_str(it["prompt"] + "\0" + it["response"])
    splits = {"train_ids": list(range(1600)), "val_ids": list(range(1600, 2400)), "cal_ids": list(range(2400, 3200)), "test_ids": list(range(3200, len(all_items)))}
    return all_items, _make_hs2_draws(all_items, splits), splits


def build_hs2() -> tuple[dict, dict]:
    all_items, draws, splits = _hs2_items_and_draws()
    exp24_blob = json.dumps({"dataset": "nvidia/HelpSteer2", "commit_sha": "990b2711a36180dd19d9c94b8627844866f8982a", "criterion": "helpfulness", "scale": [0, 1, 2, 3, 4], "budgets_d1": [16, 32, 64, 128, 256, 512, 1024, 2048], "budgets_d2": [64, 256, 1024], "n_draws": 10, "splits": splits, "items": all_items, "draws": draws}).encode("utf-8")
    assert sha256_bytes(exp24_blob) == EXP24_BUNDLE_SHA256, "HelpSteer2 reproduction mismatch vs Exp 24!"
    for it in all_items:
        it["task_kind"], it["group_key"] = "hs2", it["prompt_sha256"]
        it["counts"] = [int(c) for c in np.bincount(it["votes"], minlength=5)]
        it["variance"] = count_variance(it["counts"])
    med_var = float(np.median([all_items[i]["variance"] for i in splits["test_ids"]]))
    for i in splits["test_ids"]:
        all_items[i]["is_high_dis"] = bool(all_items[i]["variance"] >= med_var)
    suite = {"name": "hs2", "K": 5, "budgets_d1": [16, 32, 64, 128, 256, 512, 1024, 2048], "budgets_d5": [], "splits": splits, "items": all_items, "draws": draws, "test_median_variance": med_var}
    stats = {"exp24_bundle_sha256_verified": EXP24_BUNDLE_SHA256, "n_test": len(splits["test_ids"]), "lovo_test": leave_one_vote_out_ceiling([all_items[i]["counts"] for i in splits["test_ids"]], 5), "test_median_variance": med_var}
    return suite, stats


def main() -> None:
    file_hashes = ensure_sources()
    chaos3_s, chaos3_st = build_chaos3()
    alpha2_s, alpha2_st = build_alpha2()
    dices3_s, dices3_st = build_dices3()
    hs2_s, hs2_st = build_hs2()
    bundle = {"suites": {"chaos3": chaos3_s, "alpha2": alpha2_s, "dices3": dices3_s, "hs2": hs2_s}}
    BUNDLE_PATH.write_text(json.dumps(bundle), encoding="utf-8")
    bundle_sha = sha256_bytes(BUNDLE_PATH.read_bytes())
    manifest = {
        "raw_file_sha256": file_hashes,
        "prepared_bundle_path": str(BUNDLE_PATH),
        "prepared_bundle_sha256": bundle_sha,
        "suites": {"chaos3": chaos3_st, "alpha2": alpha2_st, "dices3": dices3_st, "hs2": hs2_st},
    }
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared {BUNDLE_PATH} sha256={bundle_sha}")
    print(json.dumps(manifest["suites"], indent=2))


if __name__ == "__main__":
    main()
