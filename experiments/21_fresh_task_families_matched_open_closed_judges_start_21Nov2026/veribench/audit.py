"""Check task identities and the sealed-label-independent overlap of the two prompt sets."""
import argparse
import csv
from collections import Counter
import json
from pathlib import Path

from prepare import ARCHIVE_SHA, COLLECTION_REV, PACKET, git_bytes, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("manifest", type=Path)
    p.add_argument("--sister", type=Path, default=Path.home() / "trustworthy-theorem-judges")
    args = p.parse_args()
    rows = list(map(json.loads, args.manifest.read_text().splitlines()))
    archive, = Path(__file__).resolve().parents[2].glob("10_frontier*/results/judge_forecasts.json")
    assert sha(archive.read_bytes()) == ARCHIVE_SHA
    may = json.loads(archive.read_bytes())["artifacts"]
    test = [r for r in rows if r["split"] == "set_n_test"]
    assert len(test) == 100 and len({r["task"] for r in test}) == 25
    def basename(task):
        return task.split("__")[-1].casefold()

    for name, a, b in [
        ("Task source IDs (domain prefix removed, case folded)",
         {basename(r["task"]) for r in may}, {basename(r["task"]) for r in test}),
        ("Python source byte digests", {r["py_sha256"] for r in may},
         {r["file_sha256"]["python_source"] for r in test}),
        ("Candidate byte digests", {r["cand_sha256"] for r in may},
         {r["file_sha256"]["candidate_lean"] for r in test}),
    ]:
        print(f"{name}: May {len(a)}, Set N {len(b)}, shared {len(a & b)}")
        assert not (a & b)
    with Path(__file__).with_name("family_audit.csv").open(newline="") as f:
        screening = list(csv.DictReader(f))
    assert len(screening) == len({r["set_n_task"] for r in screening}) == 25
    assert {r["set_n_task"] for r in screening} == {r["task"] for r in test}
    print("Operation-level screening judgments, before test labels:", dict(Counter(r["may_relation"] for r in screening)))
    historical = {r["cand_sha256"] for r in may}
    warmup = [
        sha(git_bytes(args.sister, COLLECTION_REV, f"{PACKET}/warmup_task/{task}/agent_lean/agent_{candidate}.lean"))
        for task in ("warm_up_01_median", "warm_up_02_greatest_common_divisor")
        for candidate in (1, 2, 3)
    ]
    assert all(digest in historical for digest in warmup)
    print(f"Warm-up artifacts already in May: {sum(digest in historical for digest in warmup)}/{len(warmup)}")


if __name__ == "__main__":
    main()
