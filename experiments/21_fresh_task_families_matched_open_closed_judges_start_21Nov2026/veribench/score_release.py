"""Score registered Set N forecasts only against the collection owner's released primary ratings."""
import argparse
import csv
import io
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from calibrate import DRAWS, JUDGES, PIN, checked_content, save_exact
from metrics import GRADE, METRICS, score
from prepare import sha

METHODS = ("ce", "pearson", "identity", "uniform", "training_histogram")
PANELS = (("strict_all","strict",False),("broader_all","broader",False),("distinct_operations","broader",True))


def records(directory):
    receipt_bytes = (directory/"commitments.json").read_bytes()
    public_receipt = Path(__file__).resolve().with_name("REGISTRATION_11-21-2026.json").read_bytes()
    if receipt_bytes != public_receipt:
        raise ValueError("Registration differs from the public forecast commitment")
    receipt = json.loads(receipt_bytes)
    assert receipt["protocol_sha256"] == PIN and tuple(receipt["judges"]) == JUDGES
    target = list(map(json.loads,checked_content(directory/"test_distributions.jsonl",receipt["test_distributions_sha256"]).splitlines()))
    fits = list(map(json.loads,checked_content(directory/"training_draws.jsonl",receipt["training_draws_sha256"]).splitlines()))
    assert len(target)==100 and len({r["source_item"] for r in target})==100 and len(fits)==2800
    return target,fits,sha(receipt_bytes)


def observations(path,target):
    """Input JSONL columns: source_item, candidate_sha256, rater_id, rating, eligible_primary."""
    ids = {x["source_item"]: x["candidate_sha256"] for x in target}
    obs = defaultdict(dict)
    if path is None:
        return obs
    for line in path.read_bytes().splitlines():
        r = json.loads(line)
        item,person = r["source_item"],r["rater_id"]
        if item not in ids or r["candidate_sha256"] != ids[item]:
            raise ValueError("Released observation differs from the registered input")
        if type(r["eligible_primary"]) is not bool:
            raise ValueError("Collection-owner primary eligibility must be explicit")
        if not r["eligible_primary"]:
            continue
        if not isinstance(person,str) or not person or person=="A1":
            raise ValueError("Invalid primary rater or author-only sensitivity rating")
        if type(r["rating"]) is not int or r["rating"] not in range(6):
            raise ValueError("Primary data require an observed integer rating in 0–5")
        if person in obs[item]:
            raise ValueError("Repeated primary rater/item observation")
        obs[item][person] = r["rating"]
    return obs


def bootstrap_weights(rows):
    groups, codes = np.unique([r["operation"] for r in rows],return_inverse=True)
    counts = np.random.default_rng(20261122).multinomial(len(groups),np.ones(len(groups))/len(groups),size=10000)
    w = counts[:,codes]
    return w/w.sum(axis=1,keepdims=True)


def correlation(x,y):
    if len(x)<2 or np.ptp(x)<=1e-12 or np.ptp(y)<=1e-12:
        return np.nan
    return float(np.corrcoef(x,y)[0,1])


def diag(means,y):
    return np.array([correlation(means,y),correlation(rankdata(means),rankdata(y))])


def evaluate(rows,scheduled,observed,fits,judge,pool,weights):
    h = np.stack([np.bincount(list(observed[r["source_item"]].values()),minlength=6)/
                  len(observed[r["source_item"]]) for r in rows])
    truth_mean = h @ GRADE
    j = JUDGES.index(judge)
    q = np.stack([r["distributions"][j] for r in rows])
    losses = score(q,h[:,None,:])
    diags = np.stack([diag(q[:,k,:] @ GRADE,truth_mean) for k in range(27)])
    chosen = defaultdict(list)
    for r in fits:
        if r["pool"]==pool:
            chosen[r["ratings_per_artifact"],r["budget"]].append(r)
    for (k,budget),batch in sorted(chosen.items()):
        assert sorted(x["draw"] for x in batch)==list(range(DRAWS))
        methods,corr,counts={},{},{}
        for key in ("ce","pearson","identity"):
            selections = [r[key][j] for r in batch] if key!="identity" else [0]*DRAWS
            freq = np.bincount(selections,minlength=27)/DRAWS
            methods[key] = np.einsum("chm,h->cm",losses,freq)
            selected = diags[selections]
            counts[key] = np.isfinite(selected).sum(axis=0)
            corr[key] = np.divide(np.nansum(selected,axis=0),counts[key],
                                  out=np.full(2,np.nan),where=counts[key]>0)
        prior = np.asarray([r["histogram"] for r in batch])
        methods["training_histogram"] = score(prior[None,:,:],h[:,None,:]).mean(axis=1)
        methods["uniform"] = score(np.full(6,1/6),h)
        for key in ("training_histogram","uniform"):
            corr[key],counts[key]=[np.nan,np.nan],[0,0]
        n=len(rows)
        for name in METHODS:
            absolute=methods[name].mean(axis=0)
            diff=methods[name]-methods["ce"]
            delta=diff.mean(axis=0)
            interval = np.quantile(weights@diff,[0.025,0.975],axis=0) if len({r["operation"] for r in rows})>1 else np.full((2,3),np.nan)
            common = {"judge":judge,"ratings_per_training_artifact":k,"training_labels":budget,"method":name}
            numeric={}
            for m,metric in enumerate(METRICS):
                numeric[metric]=absolute[m]
                numeric[metric+"_minus_ce_fitted"]=delta[m]
                numeric[metric+"_difference_conditional_p025"]=interval[0,m]
                numeric[metric+"_difference_conditional_p975"]=interval[1,m]
            yield {**common,**numeric,"rps_difference_lower_with_missing":(delta[0]*n-(scheduled-n))/scheduled,
                   "rps_difference_upper_with_missing":(delta[0]*n+(scheduled-n))/scheduled,
                   "pearson_of_means":corr[name][0],"pearson_draws_defined":counts[name][0],
                   "spearman_of_means":corr[name][1],"spearman_draws_defined":counts[name][1]}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("registered",type=Path)
    parser.add_argument("output",type=Path)
    parser.add_argument("--released-ratings",type=Path,help="Owner-released eligible individual ratings as specified by observations()")
    args=parser.parse_args()
    target,fits,forecast_digest=records(args.registered)
    obs=observations(args.released_ratings,target)
    report={"status":"no_released_ratings" if not obs else "scored_released_ratings",
            "registered_forecasts_sha256":forecast_digest,
            "released_input_sha256":sha(args.released_ratings.read_bytes()) if args.released_ratings else None,"panels":{}}
    results=[]
    for label,pool,distinct_only in PANELS:
        panel = [r for r in target if not distinct_only or r["distinct_operation"]]
        rows = [r for r in panel if r["source_item"] in obs]
        observed=[obs[r["source_item"]] for r in rows]
        report["panels"][label]={"scheduled_artifacts":len(panel),"observed_artifacts":len(rows),
            "scheduled_operation_groups":len({r["operation"] for r in panel}),"observed_operation_groups":len({r["operation"] for r in rows}),
            "released_ratings":sum(map(len,observed)),"distinct_raters":len({person for group in observed for person in group}),
            "min_ratings_per_evaluable_artifact":min(map(len,observed),default=None),
            "max_ratings_per_evaluable_artifact":max(map(len,observed),default=None)}
        if rows:
            weights=bootstrap_weights(rows)
            for judge in JUDGES:
                results.extend({"panel":label,**r} for r in evaluate(rows,len(panel),obs,fits,judge,pool,weights))
    args.output.mkdir(mode=0o700,parents=True,exist_ok=True)
    if results:
        dest=io.StringIO(newline="")
        writer=csv.DictWriter(dest,fieldnames=list(results[0]),lineterminator="\n")
        writer.writeheader()
        for r in results:
            writer.writerow({k:"" if isinstance(v,(float,np.floating)) and not np.isfinite(v) else v for k,v in r.items()})
        save_exact(args.output/"metrics.csv",dest.getvalue().encode())
    save_exact(args.output/"coverage.json",(json.dumps(report,indent=2,sort_keys=True)+"\n").encode())
    print(report["status"],"primary ratings:",report["panels"]["strict_all"]["released_ratings"],"metric rows:",len(results))


if __name__=="__main__":
    main()
