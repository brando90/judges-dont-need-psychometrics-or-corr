"""Check the complete threshold panel and replay three prespecified boundary cells via the full E23 engine."""
import argparse
import contextlib
import csv
import io
import tempfile
from pathlib import Path

import numpy as np

from run import ALPHAS, BUDGETS, HERE, COMPARATORS, METRICS, load, save
from analyze import run as complete_analysis


def read(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_directory",type=Path)
    parser.add_argument("--out",type=Path,default=HERE/"results")
    args = parser.parse_args()
    scores = read(args.out/"scores.csv")
    contrasts = read(args.out/"rps_contrasts.csv")
    thresholds = read(args.out/"sustained_thresholds.csv")
    assert len(scores) == 2*8*8*7 and len(contrasts) == 2*8*8*4
    assert len(thresholds) == 2*len(ALPHAS)
    for record in thresholds:
        subset = [r for r in contrasts if (r["criterion"], r["alpha"]) == (record["criterion"], record["alpha"])]
        comparators = {k:[k] for k in COMPARATORS}
        comparators.update(pearson_and_uniform=["pearson","uniform"],all_four=COMPARATORS)
        for key,selected in comparators.items():
            failures = [int(r["budget"]) for r in subset
                        if r["comparator"] in selected and float(r["simultaneous_lo"]) <= 0]
            threshold = next((str(b) for b in BUDGETS if b > max(failures,default=-1)), "")
            assert record[key] == threshold
    receipt = []
    # The two smallest positive settings check both sides of the relevance reversal, plus consistency at .01.
    for criterion,alpha in (("consistency",.01),("relevance",.01),("relevance",.05)):
        source = args.data_directory/f"summeval_{criterion}.npz"
        data = load(source)
        with np.load(source) as archive:
            total = archive["native_valid_votes"][:,None]
        data.native = (np.rint(data.native*total) + alpha)/(total+5*alpha)
        errors = []
        with tempfile.TemporaryDirectory() as scratch, contextlib.redirect_stdout(io.StringIO()):
            directory = Path(scratch)
            complete_analysis(data,"native",directory)
            for rows, filename, key, fields in [(scores,"scores.csv","method",METRICS),
                (contrasts,"contrasts.csv","comparator",("rps_gain","simultaneous_lo","simultaneous_hi","pointwise_lo","pointwise_hi"))]:
                original = {(r["budget"],r[key]):r for r in read(directory/filename) if r["depth"]=="1"}
                selected = [r for r in rows if r["criterion"]==criterion and float(r["alpha"])==alpha]
                assert len(original)==len(selected)
                for r in selected:
                    ref = original[r["budget"],r[key]]
                    errors.extend(abs(float(ref[field]) - float(r[field])) for field in fields)
            assert max(errors) < 1e-12
        receipt.append(dict(criterion=criterion,alpha=alpha,numeric_comparisons=len(errors),
                            maximum_absolute_error=max(errors)))
    save(args.out/"verification.csv",receipt)
    print("All sustained thresholds agree with all published lower bands.",receipt)


if __name__ == "__main__":
    main()
