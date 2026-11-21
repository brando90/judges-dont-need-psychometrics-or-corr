"""Freeze May-trained distribution forecasts before opening any Set N outcomes."""
import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from prepare import ARCHIVE_SHA, MAY, MAY_REV, git_bytes, json_bytes, sha

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "short_answers"))
from analyze import select
from metrics import COEFFICIENTS, GRADE, calibrations

PIN = "cf049a6242b78c783eefcc0bb1c45a7be988bcb0b92d7b0d966ee4516e537d1d"
REQUESTS_SHA = "8fb986600708fb323fea37f17daf24825d5295d9ef2a022ab526ba014f39d51d"
MANIFEST_SHA = "3b73702d1b67fabd9d693529976f808d60b58ae95b501b75d715f66e6bb4b76c"
SPLIT_FILE = "experiments/75_J1_validating_formal_spec_judge_train_val_test/results/splits.json"
SPLIT_SHA = "6405e9b2af86056e72d8963438ce1ec58ba446a456c879a69579cf6f197e8a52"
MAY_EXPOSURE_SHA = "ad9ed552fc2ffac6328e6010bf6be82215dd74f1eddd791622d433126877a96c"
PANEL_POOLS = ("strict", "broader")
OVERLAPS = {"heap_family", "sorting_family", "humaneval_31_is_prime", "humaneval_47_median",
            "18_MyFibonacci", "33_MyMinimum", "humaneval_8_sum_product", "1_unsafeMultiply_safe"}
JUDGES = ("qwen","claude")
MODELS = {"qwen":"qwen3.5:9b-q4_K_M","claude":"claude-sonnet-5"}
BUDGETS = (4,8,16,32,64,128)
DRAWS = 200


def checked_content(path, digest):
    data = path.read_bytes()
    if sha(data) != digest:
        raise ValueError(f"{path.name} differs from its protocol commitment")
    return data


def historical_eligibility(legacy):
    partition = git_bytes(legacy, MAY_REV, SPLIT_FILE)
    assert sha(partition) == SPLIT_SHA
    partitions = json.loads(partition)
    allowed = set(partitions["train_tasks"]) | set(partitions["val_tasks"])
    assert len(allowed) == 15 and allowed.isdisjoint(partitions["test_tasks"])
    archive, = HERE.parents[1].glob("10_frontier*/results/judge_forecasts.json")
    rows = json.loads(checked_content(archive, ARCHIVE_SHA))["artifacts"]
    return {r["item"] for r in rows if r["task"] in allowed and r["ordinary"]}


def inputs(directory, forecast_paths):
    manifest = list(map(json.loads, checked_content(directory/"manifest.jsonl",MANIFEST_SHA).splitlines()))
    prompts = list(map(json.loads, checked_content(directory/"requests.jsonl",REQUESTS_SHA).splitlines()))
    assert len(manifest) == len(prompts) == 175
    hashes = {p["id"]:sha(json_bytes([{"role": k,"content": p[k]} for k in ("system","user")])) for p in prompts}
    lookup, statuses = {}, Counter()
    for path in forecast_paths:
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                key = (row["id"],row["judge"])
                assert key not in lookup and hashes[row["id"]] == row["messages_sha256"] and row["judge"] in JUDGES
                assert row["model"] == MODELS[row["judge"]]
                assert row["status"] in {"valid","invalid_uniform","failure_uniform"}
                q = np.asarray([float(row[f"p{i}"]) for i in range(6)])
                assert q.shape == (6,) and np.isfinite(q).all() and q.min() >= 0 and abs(q.sum()-1) < 1e-12
                if row["status"] != "valid":
                    np.testing.assert_allclose(q, np.full(6,1/6), atol=1e-12, rtol=0)
                lookup[key] = q
                statuses[row["judge"]+"/"+row["status"]] += 1
    assert len(lookup) == 350
    return manifest,lookup,statuses


def load_eligible_labels(legacy, eligibility):
    # Historical importer uses the available p1/p2/p3 fields in order. Test rows are
    # bypassed by identity before JSON decoding; no new evaluation labels are read.
    raw = git_bytes(legacy, MAY_REV, MAY + "/items.jsonl")
    assert sha(raw) == MAY_EXPOSURE_SHA
    labels = {}
    for line in raw.splitlines():
        item = re.search(rb'"item"\s*:\s*"([^"]+)"',line).group(1).decode()
        if item not in eligibility:
            continue
        r = json.loads(line)
        assert r["split"] in ("train","val") and not r["ref_placeholder_at_annotation"]
        vals = [r[k] for k in ("p1","p2","p3") if r[k] is not None]
        assert len(vals) in (2,3) and all(v in range(6) for v in vals)
        labels[item] = np.array(vals,int)
    assert set(labels) == eligibility
    return labels


def draws(pool, base, labels, index):
    rows = sorted(pool,key=lambda r:r["id"])
    q = calibrations(np.stack([base[r["id"],j] for r in rows for j in JUDGES]).reshape(len(rows),2,6))
    y = [labels[r["source_item"]] for r in rows]
    families = np.asarray([r["may_family"] for r in rows])
    family_sets = [np.flatnonzero(families==f) for f in sorted(set(families))]
    for draw in range(DRAWS):
        rng = np.random.default_rng(np.random.SeedSequence([20261121,index,draw]))
        groups = [rng.permutation(ids) for ids in family_sets]
        order = [groups[g][p] for p in range(max(map(len,groups)))
                 for g in rng.permutation(len(groups)) if p < len(groups[g])]
        assert sorted(order) == list(range(len(rows)))
        slots = [rng.permutation(values) for values in y]
        for k in (1,2):
            for budget in BUDGETS:
                n = budget // k
                if n > len(rows):
                    continue
                train = order[:n]
                ratings = np.stack([slots[i][:k] for i in train])
                hist = (ratings[:,:,None]==GRADE).mean(axis=1)
                ce,pearson,undefined = select(q[train],hist,ratings.mean(axis=1))
                prior = np.bincount(ratings.ravel(),minlength=6) + 0.5
                yield {"pool":PANEL_POOLS[index],"draw":draw,"budget":budget,"ratings_per_artifact":k,
                       "ce":ce.tolist(),"pearson":pearson.tolist(),"undefined":undefined.tolist(),
                       "histogram":(prior/prior.sum()).tolist()}


def save_exact(path,data):
    if path.exists() and path.read_bytes()!=data:
        raise ValueError(f"Existing {path.name} differs; choose a fresh output directory")
    if not path.exists():
        path.write_bytes(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory",type=Path)
    parser.add_argument("output",type=Path)
    parser.add_argument("forecasts",type=Path,nargs="+",help="Receipt-verified CSVs covering each request and both judges once")
    parser.add_argument("--legacy",type=Path,default=Path.home()/"veribench")
    args = parser.parse_args()
    checked_content(HERE/"CALIBRATION_PROTOCOL_11-21-2026.md",PIN)
    manifest,base,statuses = inputs(args.directory,args.forecasts)
    eligibility = historical_eligibility(args.legacy)
    broad = [r for r in manifest if r["split"]=="may_development" and r["source_item"] in eligibility]
    strict = [r for r in broad if r["may_family"] not in OVERLAPS]
    assert len(strict)==15 and len(broad)==36
    labels = load_eligible_labels(args.legacy,eligibility)
    target = [r for r in manifest if r["split"]=="set_n_test"]
    with (HERE/"family_audit.csv").open(newline="") as f:
        audit = {r["set_n_task"]:r for r in csv.DictReader(f)}
    assert len(target)==100 and sum(audit[r["task"]]["may_relation"]=="distinct_operation" for r in target)==56
    transforms = calibrations(np.stack([[base[r["id"],j] for j in JUDGES] for r in target]))
    assert transforms.shape==(100,2,27,6) and np.all(transforms>0)
    np.testing.assert_allclose(transforms.sum(axis=-1),1,rtol=0,atol=1e-12)
    fits = []
    for i,p in enumerate((strict,broad)):
        fits.extend(draws(p,base,labels,i))
    assert len(fits) == DRAWS*(5+9)
    out = args.output
    out.mkdir(parents=True,exist_ok=True,mode=0o700)
    selected = b"".join(json_bytes(r)+b"\n" for r in fits)
    save_exact(out/"training_draws.jsonl",selected)
    tests = []
    for i,r in enumerate(target):
        group = audit[r["task"]]
        tests.append({"id":r["id"],"source_item":r["source_item"],"candidate_sha256":r["file_sha256"]["candidate_lean"],
                      "task":r["task"],"operation":group["operation"],"distinct_operation":group["may_relation"]=="distinct_operation",
                      "distributions":transforms[i].tolist()})
    target_data = b"".join(json_bytes(r)+b"\n" for r in tests)
    save_exact(out/"test_distributions.jsonl",target_data)
    receipt = {"date":"11-21-2026","protocol_sha256":PIN,"combined_requests_sha256":REQUESTS_SHA,
               "source_forecasts_sha256":[sha(p.read_bytes()) for p in args.forecasts],
               "grid":COEFFICIENTS,"judges":JUDGES,"models":MODELS,"draws_per_budget":DRAWS,"scheduled_test_cases":len(tests),
               "distinct_operation_cases":56,"training_conditions":len(fits),"parse_statuses":dict(statuses),
               "training_draws_sha256":sha(selected),"test_distributions_sha256":sha(target_data),
               "fitted_source_sha256":sha(Path(__file__).read_bytes()),
               "selection_source_sha256":sha((HERE.parent/"short_answers/analyze.py").read_bytes()),
               "metrics_source_sha256":sha((HERE.parent/"short_answers/metrics.py").read_bytes())}
    encoded = json.dumps(receipt,indent=2,sort_keys=True).encode()+b"\n"
    save_exact(out/"commitments.json",encoded)
    print("Calibrated Set N cases:",len(tests),"draw conditions:",len(fits),"commitments SHA-256:",sha(encoded))


if __name__ == "__main__":
    main()
