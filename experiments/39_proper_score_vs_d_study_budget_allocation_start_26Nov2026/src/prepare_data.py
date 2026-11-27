#!/usr/bin/env python3
"""Reconstruct Experiment 28 item subsets and attach frozen judge forecasts.

Writes data/processed/study_a_bundle.npz and data/processed/manifest.json.
Does not call any model API.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"

DATASETS = {
    "chaosnli_snli": {"k": 0, "K": 3, "kind": "chaos", "file": "chaosNLI_snli.jsonl", "n": 600},
    "chaosnli_mnli": {"k": 1, "K": 3, "kind": "chaos", "file": "chaosNLI_mnli_m.jsonl", "n": 600},
    "chaosnli_alphanli": {"k": 2, "K": 2, "kind": "chaos", "file": "chaosNLI_alphanli.jsonl", "n": 400},
    "dices350": {"k": 3, "K": 3, "kind": "dices", "file": "diverse_safety_adversarial_dialog_350.csv", "n": 0},
    "dices990": {"k": 4, "K": 3, "kind": "dices", "file": "diverse_safety_adversarial_dialog_990.csv", "n": 400},
}
JUDGES = ["gpt-6-astra", "claude-opus-5-5", "kimi-k3", "glm-5.3", "deepseek-v4.1-flash"]
DICES_MAP = {"No": 0, "Unsure": 1, "Yes": 2}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sub(n_total: int, n: int, k: int):
    rng = np.random.default_rng(np.random.SeedSequence([20261123, k]))
    return sorted(rng.choice(n_total, size=n, replace=False).tolist())


def load_forecasts():
    by = defaultdict(dict)  # id -> judge -> (valid, p)
    with (RAW / "forecasts_11-23-2026.csv").open() as f:
        for row in csv.DictReader(f):
            K_hint = 2 if row["id"].startswith("chaosnli_alphanli/") else 3
            raw = [row["p0"], row["p1"], row["p2"]]
            p = []
            for j in range(K_hint):
                v = raw[j]
                p.append(float(v) if v not in ("", None) else np.nan)
            valid = row["status"] == "valid" and np.all(np.isfinite(p)) and np.sum(p) > 0
            if valid:
                p = np.asarray(p, float)
                p = p / p.sum()
            else:
                p = np.full(K_hint, np.nan)
                valid = False
            by[row["id"]][row["judge"]] = (valid, p)
    return by


def mix_judges(per_judge, K):
    vecs = []
    valid = np.zeros(len(JUDGES), dtype=bool)
    P = np.full((len(JUDGES), K), np.nan)
    for j, name in enumerate(JUDGES):
        if name not in per_judge:
            continue
        ok, p = per_judge[name]
        if ok and len(p) == K:
            valid[j] = True
            P[j] = p
            vecs.append(p)
    if vecs:
        m = np.mean(np.stack(vecs, 0), 0)
        m = m / m.sum()
    else:
        m = np.full(K, 1.0 / K)
    return m, P, valid


def load_chaos(name, meta, forecasts):
    rows = [json_loads_line(l) for l in (RAW / "chaosNLI_v1.0" / meta["file"] if False else RAW / meta["file"]).read_text().splitlines()]
    idx = sub(len(rows), meta["n"], meta["k"])
    K = meta["K"]
    ids, counts, ratings = [], [], []
    p_mix, p_j, valid = [], [], []
    for j in idx:
        r = rows[j]
        ident = f"{name}/{r['uid']}"
        lab = list(r["label_count"])
        if name != "chaosnli_alphanli":
            lab = [lab[2], lab[1], lab[0]]  # [e,n,c] -> [c,n,e]
        lab = np.asarray(lab, int)
        assert lab.sum() == 100 and lab.shape == (K,)
        # expand ratings 0..K-1
        exp = np.repeat(np.arange(K), lab)
        ids.append(ident)
        counts.append(lab)
        ratings.append(exp)
        m, P, v = mix_judges(forecasts[ident], K)
        p_mix.append(m)
        p_j.append(P)
        valid.append(v)
    return pack(name, K, ids, counts, ratings, None, p_mix, p_j, valid)


def json_loads_line(l):
    import json
    return json.loads(l)


def load_dices(name, meta, forecasts):
    by = defaultdict(list)
    with (RAW / meta["file"]).open(newline="") as f:
        for row in csv.DictReader(f):
            by[row["item_id"]].append(row)
    ids_all = sorted(by, key=int)
    pick = range(len(ids_all)) if meta["n"] == 0 else sub(len(ids_all), meta["n"], meta["k"])
    K = 3
    ids, counts, ratings, rater_ids = [], [], [], []
    p_mix, p_j, valid = [], [], []
    for j in pick:
        item = ids_all[j]
        rs = by[item]
        ident = f"{name}/{item}"
        c = np.zeros(K, dtype=int)
        ys, rs_ids = [], []
        for r in rs:
            y = DICES_MAP[r["Q_overall"]]
            c[y] += 1
            ys.append(y)
            rs_ids.append(r["rater_id"])
        ids.append(ident)
        counts.append(c)
        ratings.append(np.asarray(ys, int))
        rater_ids.append(np.asarray(rs_ids))
        m, P, v = mix_judges(forecasts[ident], K)
        p_mix.append(m)
        p_j.append(P)
        valid.append(v)
    return pack(name, K, ids, counts, ratings, rater_ids, p_mix, p_j, valid)


def pack(name, K, ids, counts, ratings, rater_ids, p_mix, p_j, valid):
    N = len(ids)
    Rmax = max(len(r) for r in ratings)
    Y = np.full((N, Rmax), -1, dtype=int)
    n_avail = np.zeros(N, dtype=int)
    Rater = None
    if rater_ids is not None:
        Rater = np.full((N, Rmax), "", dtype=object)
    for i, r in enumerate(ratings):
        n = len(r)
        n_avail[i] = n
        Y[i, :n] = r
        if Rater is not None:
            Rater[i, :n] = rater_ids[i]
    return {
        "name": name,
        "K": K,
        "ids": np.asarray(ids),
        "counts": np.stack(counts, 0).astype(int),
        "ratings": Y,
        "n_avail": n_avail,
        "rater_ids": Rater,
        "p_mix": np.stack(p_mix, 0).astype(float),
        "p_judges": np.stack(p_j, 0).astype(float),
        "judge_valid": np.stack(valid, 0),
    }


def frozen_split(n, k, frac=0.30):
    rng = np.random.default_rng(np.random.SeedSequence([20261126, 1, k]))
    order = rng.permutation(n)
    n_test = int(round(frac * n))
    test = np.sort(order[:n_test])
    dev = np.sort(order[n_test:])
    return test, dev


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    forecasts = load_forecasts()
    bundle = {}
    manifest = {"datasets": {}, "judges": JUDGES, "source_sha256": {}}
    for path in sorted(RAW.iterdir()):
        if path.name == "SHA256SUMS.txt":
            continue
        manifest["source_sha256"][path.name] = sha256(path)
    for name, meta in DATASETS.items():
        if meta["kind"] == "chaos":
            d = load_chaos(name, meta, forecasts)
        else:
            d = load_dices(name, meta, forecasts)
        test, dev = frozen_split(len(d["ids"]), meta["k"])
        d["test_idx"] = test
        d["dev_idx"] = dev
        # sanity: forecasts exist
        missing = [i for i in d["ids"] if i not in forecasts]
        if missing:
            raise SystemExit(f"missing forecasts for {name}: {missing[:3]} ({len(missing)})")
        bundle[name] = d
        manifest["datasets"][name] = {
            "k": meta["k"],
            "K": int(d["K"]),
            "n_items": int(len(d["ids"])),
            "n_test": int(len(test)),
            "n_dev": int(len(dev)),
            "ratings_per_item_min": int(d["n_avail"].min()),
            "ratings_per_item_max": int(d["n_avail"].max()),
            "has_rater_ids": d["rater_ids"] is not None,
            "n_valid_judge_item": int(d["judge_valid"].sum()),
            "n_items_with_any_valid_judge": int(d["judge_valid"].any(1).sum()),
            "count_sum_mean": float(d["counts"].sum(1).mean()),
        }
        print(name, manifest["datasets"][name])
    np.savez_compressed(OUT / "study_a_bundle.npz", bundle=np.array(bundle, dtype=object))
    # also save a json index of ids/splits for inspection without numpy object arrays
    index = {}
    for name, d in bundle.items():
        index[name] = {
            "ids": d["ids"].tolist(),
            "test_idx": d["test_idx"].tolist(),
            "dev_idx": d["dev_idx"].tolist(),
            "K": int(d["K"]),
        }
    (OUT / "splits.json").write_text(json.dumps(index))
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("wrote", OUT / "study_a_bundle.npz")


if __name__ == "__main__":
    main()
