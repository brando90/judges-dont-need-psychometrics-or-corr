#!/usr/bin/env python3
"""Dataset facts behind the 11-26-2026 audit of the Gemini plan (standard library only).

Every count or statement that the audit ledger says about a dataset or a paper is printed by
this script from the files named on the command line, together with each input file's sha256,
so a reader can re-fetch the same bytes and compare. The raw data is NOT in this repository
(about 85 MB of DICES alone); README.md lists where each input came from.

Three kinds of lines are printed:
  FACT     a number or statement computed from an input file.
  PASS/FAIL  an integrity check (a pinned sha256 matches, a table in a paper matches the files).
             Any FAIL makes the exit code 1.
  VERDICT  what the plan said versus what the files show. These do not affect the exit code,
           because a plan statement being wrong is a finding, not a script failure.

Run (paths are the ones used on 11-26-2026; see README.md):
  python3 -I dataset_facts.py --facts-dir DIR --cache-dir DIR --popquorn-dir DIR \
      --popquorn-paper-txt FILE --summeval-paper-txt FILE --sister-dir DIR --exp36-data-io FILE

Paper text files were made with poppler's pdftotext: `pdftotext -layout` for the Kohli and POPQUORN PDFs (their
tables need the layout mode) and plain `pdftotext` (reading order) for the two-column SummEval PDF. Text hashes can
differ between poppler versions; the PDF hashes are the stable fingerprints.

Only aggregate counts are printed for the VeriBench formal specification coverage (FSC)
ratings; the sister-repo files used here already have rater identity removed, and this script
never prints a name or an individual rating.
"""
from __future__ import annotations

import argparse
import ast
import collections
import csv
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

csv.field_size_limit(10**9)

FAILS: list[str] = []


def fact(label: str, value) -> None:
    if not isinstance(value, str):
        value = json.dumps(value, sort_keys=True)
    print(f"FACT  {label}: {value}")


def check(label: str, ok: bool, detail: str = "") -> None:
    print(("PASS  " if ok else "FAIL  ") + label + (f" -- {detail}" if detail else ""))
    if not ok:
        FAILS.append(label)


def verdict(label: str, text: str) -> None:
    print(f"VERDICT  {label}: {text}")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def mean2(xs) -> str:
    return f"{sum(xs) / len(xs):.2f}"


def rng(a: int, b: int) -> str:
    return f"{a}" if a == b else f"{a} to {b}"


def head_of(repo: Path) -> str:
    try:
        out = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception:  # noqa: BLE001 - a missing git is reported, not fatal
        return "unknown"


# Pinned sha256 values copied from experiments/36_sanmi_distributional_pitch_anti_proper_corr_start_26Nov2026/data_io.py
# (commit 036cafa). The script also re-reads data_io.py when --exp36-data-io is given and checks the copy is exact.
PINNED = {
    "dices350.csv": "63cb8620eb04e64957f9e45b1f7cb3d079c4eed8e8b28909f46e0b7038ce4301",
    "dices990.csv": "14332b6391cdee80383b765c734de3bebf6d8553141ee7d2f47ef0910a12fcf7",
    "chaosNLI_snli.jsonl": "99f9015ddda7d85f66a087452bc30d53974314fe27e7d589e2f41ad44bd509c1",
    "chaosNLI_mnli_m.jsonl": "8eb49b589488e7b1a9ec95fb8a864bd50f0509b4f808eed0aed3f8167617664d",
    "chaosNLI_alphanli.jsonl": "f55817ecc11a02ba3030cd7d2c567d23843456809020723abad53f4fde024c9e",
    "model_annotations.aligned.jsonl": "f0d4166e0cdeb439b387c4449634b067b7e9f721b9ef4c2b722f6b7550d3d6ab",
}
REMOVED_350 = {
    "297514565398139", "297515609163939", "297515750682315", "297515617432733",
    "297541515566649", "297541515769980", "297515629971478", "297059995361243",
    "297541522412126", "297540556928761", "297541321453321", "297540562350921",
    "297540983991638", "297060365288109", "297514543980607", "297515729806999",
    "297541271027233", "296709611112092", "296709543131761",
}
REMOVED_990 = {
    "296768824962581", "296740172100277", "296729424498056", "296835612486158",
    "297212198239077", "296740254381555", "296740049152799", "297212002905763",
    "296740761150184", "296708974347177", "296770194464643", "296768399729683",
}


def section_inputs(args, files: dict[str, Path]) -> None:
    print("\n== 1. Input files (sha256 and size)")
    for name in sorted(files):
        p = files[name]
        digest = sha256(p)
        fact(f"sha256 {name}", f"{digest}  ({p.stat().st_size} bytes)")
        if name in PINNED:
            check(f"pinned sha256 of {name} equals the one in Experiment 36 data_io.py", digest == PINNED[name])
    if args.exp36_data_io:
        src = Path(args.exp36_data_io).read_text(encoding="utf-8")
        for var, mine in (("REMOVED_350", REMOVED_350), ("REMOVED_990", REMOVED_990)):
            m = re.search(var + r"\s*=\s*(\{.*?\})", src, flags=re.S)
            theirs = ast.literal_eval(m.group(1)) if m else None
            check(f"{var} in this script equals data_io.py", theirs == mine, f"{len(mine)} ids")
        for fname, digest in PINNED.items():
            check(f"data_io.py contains the pinned sha256 for {fname}", digest in src)
    fact("sister repo HEAD", head_of(Path(args.sister_dir)))
    fact("POPQUORN clone HEAD", head_of(Path(args.popquorn_dir)))


def section_dices(files: dict[str, Path]) -> dict:
    print("\n== 2. DICES (Aroyo et al. 2023), rater-level Q_overall")
    out = {}
    for tag, removed in (("350", REMOVED_350), ("990", REMOVED_990)):
        path = files[f"dices{tag}.csv"]
        per_item, raters = collections.defaultdict(list), set()
        per_item_f, raters_f = collections.defaultdict(list), set()
        labels, labels_f, rows = collections.Counter(), collections.Counter(), 0
        with path.open(newline="", encoding="utf-8", errors="replace") as handle:
            for row in csv.DictReader(handle):
                rows += 1
                rid, iid, q = str(row["rater_id"]), str(row["item_id"]), row["Q_overall"]
                per_item[iid].append(rid)
                raters.add(rid)
                labels[q] += 1
                if rid in removed or q not in {"No", "Unsure", "Yes"}:
                    continue
                per_item_f[iid].append(rid)
                raters_f.add(rid)
                labels_f[q] += 1
        n = [len(v) for v in per_item.values()]
        nf = [len(v) for v in per_item_f.values()]
        fact(f"DICES-{tag} rows / items / raters (as released)", [rows, len(per_item), len(raters)])
        fact(f"DICES-{tag} ratings per item as released (min, mean, max)", [min(n), mean2(n), max(n)])
        fact(f"DICES-{tag} README quality-filter list: ids listed / ids present in the file", [len(removed), len(removed & raters)])
        fact(f"DICES-{tag} after the filter: rows / items / raters", [sum(nf), len(per_item_f), len(raters_f)])
        fact(f"DICES-{tag} ratings per item after the filter (min, mean, max)", [min(nf), mean2(nf), max(nf)])
        fact(f"DICES-{tag} Q_overall counts as released", dict(labels))
        fact(f"DICES-{tag} Q_overall counts after the filter", dict(labels_f))
        out[tag] = {"min": min(n), "max": max(n), "min_f": min(nf), "max_f": max(nf)}
    return out


def section_chaos(files: dict[str, Path]) -> dict:
    print("\n== 3. ChaosNLI (Nie et al. 2020)")
    out = {}
    for name in ("chaosNLI_snli.jsonl", "chaosNLI_mnli_m.jsonl", "chaosNLI_alphanli.jsonl"):
        totals, widths = [], set()
        with files[name].open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                counts = [int(x) for x in row["label_count"]]
                totals.append(sum(counts))
                widths.add(len(counts))
        out[name] = len(totals)
        fact(f"{name} items / label_count width / votes per item (min, max)", [len(totals), sorted(widths), [min(totals), max(totals)]])
    three = out["chaosNLI_snli.jsonl"] + out["chaosNLI_mnli_m.jsonl"]
    fact("ChaosNLI three-class items (SNLI + MNLI-matched)", three)
    fact("ChaosNLI binary abductive NLI items", out["chaosNLI_alphanli.jsonl"])
    fact("ChaosNLI all three files", three + out["chaosNLI_alphanli.jsonl"])
    return {"three": three, "alpha": out["chaosNLI_alphanli.jsonl"], "all": three + out["chaosNLI_alphanli.jsonl"]}


def section_summeval(files: dict[str, Path]) -> None:
    print("\n== 4. SummEval (Fabbri et al. 2021)")
    arts, systems, n_exp, n_turk, rows = set(), set(), set(), set(), 0
    with files["model_annotations.aligned.jsonl"].open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows += 1
            arts.add(row["filepath"])
            systems.add(row["model_id"])
            n_exp.add(len(row["expert_annotations"]))
            n_turk.add(len(row["turker_annotations"]))
    fact("SummEval summaries (lines) / distinct articles / distinct systems", [rows, len(arts), len(systems)])
    fact("SummEval expert annotation records per summary", sorted(n_exp))
    fact("SummEval crowd (turker) annotation records per summary", sorted(n_turk))
    fact("SummEval item-aspect rows (summaries x 4 aspects)", rows * 4)


def section_summeval_paper(args, files: dict[str, Path]) -> None:
    print("\n== 4b. SummEval paper (arXiv 2007.12626): how the expert scores were produced")
    txt = Path(args.summeval_paper_txt).read_text(encoding="utf-8", errors="replace")
    flat = re.sub(r"\s+", " ", txt)
    checks = {
        "the paper reports two rounds of expert annotation": r"two rounds of annota",
        "second-round experts re-checked scores that differed by more than 2 points from another annotator": r"differed from another annotator by more than 2 points",
        "re-evaluating experts were allowed to see the other experts' first-round scores": r"judges were allowed to see scores assigned by other expert annotators",
        "Krippendorff alpha is 0.4132 for the first expert round and 0.7127 after the second": r"0\.4132[\s\S]{0,400}0\.7127",
    }
    for label, rx in checks.items():
        check(f"SummEval paper: {label}", bool(re.search(rx, flat)))
    fact("consequence", "SummEval expert ratings are not independent draws: the second round let experts see each other's scores, which pulls them together")


def table1_rows(paper_txt: Path) -> dict:
    text = paper_txt.read_text(encoding="utf-8", errors="replace")
    pat = {
        "offensiveness": r"Offensiveness rating\s+.*?Ruddit\s+([\d,]+)\s+(\d+)\s+([\d,]+)\s+([\d.]+)",
        "politeness": r"Politeness Rating\s+.*?Enron\s+([\d,]+)\s+(\d+)\s+([\d,]+)\s+([\d.]+)",
    }
    got = {}
    for k, rx in pat.items():
        m = re.search(rx, text, flags=re.S)
        if m:
            got[k] = {"annotations": int(m.group(1).replace(",", "")), "annotators": int(m.group(2)),
                      "instances": int(m.group(3).replace(",", "")), "avg": float(m.group(4))}
    return got


def section_popquorn(args, files: dict[str, Path]) -> dict:
    print("\n== 5. POPQUORN (Pei and Jurgens 2023), Potato-Prolific-Dataset")
    paper = Path(args.popquorn_paper_txt)
    text = paper.read_text(encoding="utf-8", errors="replace")
    tab = table1_rows(paper)
    out = {}
    for task, rating_col in (("offensiveness", "offensiveness"), ("politeness_rating", "politeness")):
        path = Path(args.popquorn_dir) / "dataset" / task / "raw_data.csv"
        per, ann, vals, rows = collections.defaultdict(set), set(), collections.Counter(), 0
        with path.open(newline="", encoding="utf-8", errors="replace") as handle:
            reader = csv.DictReader(handle)
            for col in ("instance_id", "user_id", rating_col):
                if col not in reader.fieldnames:
                    check(f"POPQUORN {task} has column {col}", False, str(reader.fieldnames))
            for row in reader:
                rows += 1
                per[row["instance_id"]].add(row["user_id"])
                ann.add(row["user_id"])
                vals[row[rating_col]] += 1
        n = [len(v) for v in per.values()]
        fact(f"POPQUORN {task} rows / items / annotators", [rows, len(per), len(ann)])
        fact(f"POPQUORN {task} annotators per item (min, mean, max)", [min(n), mean2(n), max(n)])
        fact(f"POPQUORN {task} rating value counts", dict(sorted(vals.items())))
        key = "offensiveness" if task == "offensiveness" else "politeness"
        t = tab.get(key)
        check(f"POPQUORN {task}: files match Table 1 of the paper (annotations, annotators, instances)",
              bool(t) and (t["annotations"], t["annotators"], t["instances"]) == (rows, len(ann), len(per)), json.dumps(t))
        check(f"POPQUORN {task}: Table 1 average labels per instance rounds to the files' mean",
              bool(t) and abs(t["avg"] - sum(n) / len(n)) < 0.06, f"paper {t['avg'] if t else None} vs files {mean2(n)}")
        out[key] = {"rows": rows, "items": len(per), "annotators": len(ann), "mean": sum(n) / len(n), "min": min(n), "max": max(n)}
    pay = re.search(r"paid \$12 per hour", text)
    timing = re.search(r"time of finishing 50 annotations is 13 minutes", text)
    check("POPQUORN paper states the $12 per hour rate", bool(pay))
    check("POPQUORN paper states 13 minutes for 50 annotations (in the offensiveness section)", bool(timing))
    fact("derived crowd price per offensiveness rating (arithmetic, not a published figure)", f"12 x (13/60) / 50 = {12 * 13 / 60 / 50:.4f} dollars")
    fact("note", "the 13-minute figure is stated for the offensiveness task only; the paper gives no timing for politeness")
    return out


def section_helpsteer2(facts_dir: Path) -> None:
    print("\n== 6. HelpSteer2 (Wang et al. 2024), dataset card")
    lines = (facts_dir / "helpsteer2_card.txt").read_text(encoding="utf-8", errors="replace").splitlines()
    wanted = {
        "adaptive annotators (three minimum, up to five, extra ones when the first do not agree)": r"minimum of three annotators and up to five annotators",
        "mean number of annotators is stated": r"mean of 3\.41 annotators",
        "individual annotations released as lists (disagreements split)": r"release the individual annotations",
        "outlier annotations are filtered when the mean is computed": r"after filtering outliers",
        "a few samples keep only two annotations (not-ratable labels)": r"only annotations by 2 annotators",
    }
    for label, rx in wanted.items():
        hits = [i + 1 for i, ln in enumerate(lines) if re.search(rx, ln)]
        check(f"HelpSteer2 card: {label}", bool(hits), f"line(s) {hits}")
    absent = [i + 1 for i, ln in enumerate(lines) if re.search(r"(?i)high[- ]disagreement|more than 2 points", ln)]
    fact("lines of the card that describe dropping items for disagreement of more than 2 points", absent if absent else "none found")
    # The 2-point figure does appear in the card, as the agreement rule that decides whether extra annotators are collected.
    rule = [i + 1 for i, ln in enumerate(lines)
            if re.search(r"up to five annotators, if the initial annotators do not agree with each other sufficiently \(2 points or less on helpfulness\)", ln)]
    check("HelpSteer2 card: the 2-point figure is the helpfulness agreement rule that decides whether extra annotators are added", bool(rule), f"line(s) {rule}")
    top3 = [i + 1 for i, ln in enumerate(lines) if re.search(r"mean of the three annotators who agree with each other most", ln)]
    check("HelpSteer2 card: the released label is the rounded mean of the three most-agreeing annotators", bool(top3), f"line(s) {top3}")
    qa = [i for i, ln in enumerate(lines) if re.search(r"retain only 20, ?324 samples", ln)]
    check("HelpSteer2 card: a quality-assurance step keeps 20,324 samples", bool(qa), f"line(s) {[i + 1 for i in qa]}")
    check("HelpSteer2 card: that quality-assurance sentence names no disagreement criterion",
          bool(qa) and not any(re.search(r"(?i)disagree", lines[i]) for i in qa))


def section_fsc(args) -> dict:
    print("\n== 7. VeriBench formal specification coverage (FSC), anonymized sister-repo files")
    base = next(Path(args.sister_dir, "experiments").glob("27_*")) / "data_11-24-2026"
    out = {}
    for label in ("old", "new"):
        path = base / f"human_ratings_{label}_11-24-2026.csv"
        items, tasks, dist, splits, tasks_by_split, vals, total = 0, set(), collections.Counter(), collections.Counter(), collections.defaultdict(set), collections.Counter(), 0
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                items += 1
                tasks.add(row["task"])
                k = int(row["n_ratings"])
                got = row["ratings_sorted_0to5_rater_identity_removed"].split()
                if len(got) != k:
                    check(f"FSC {label}: n_ratings equals the length of the rating list for {row['item']}", False)
                total += k
                dist[k] += 1
                splits[row["split"]] += 1
                tasks_by_split[row["split"]].add(row["task"])
                vals.update(got)
        fact(f"FSC {label} artifacts / tasks / ratings", [items, len(tasks), total])
        fact(f"FSC {label} artifacts by number of ratings", dict(sorted(dist.items())))
        fact(f"FSC {label} artifacts by split (train, val, test)", {k: splits[k] for k in sorted(splits)})
        fact(f"FSC {label} tasks by split", {k: len(tasks_by_split[k]) for k in sorted(tasks_by_split)})
        fact(f"FSC {label} rating value counts (0 to 5)", dict(sorted(vals.items())))
        out[label] = {"items": items, "tasks": len(tasks), "ratings": total, "dist": dict(dist)}
    fact("pooled artifacts / tasks / ratings", [out["old"]["items"] + out["new"]["items"], out["old"]["tasks"] + out["new"]["tasks"], out["old"]["ratings"] + out["new"]["ratings"]])
    recon = next(Path(args.sister_dir, "experiments", "archive").glob("07_*")) / "results.md"
    txt = recon.read_text(encoding="utf-8")
    check("sister repo reconciliation states 195 expert ratings and 345 five-rater ratings",
          "**195** ratings come from the three expert columns" in txt and "**345** ratings over five raters" in txt)
    check("sister repo reconciliation says 343 was not found in the files it searched", "I did not find 343" in txt)
    fact("note", "345 (five raters, 45 artifacts with 5 ratings and 30 with 4) is cited from that reconciliation; the anonymized files above hold only the 195 expert ratings of the old set")
    return out


def section_kohli(facts_dir: Path) -> None:
    print("\n== 8. Kohli, arXiv:2605.29797 (full text read from the PDF)")
    page = (facts_dir / "arxiv_kohli.txt").read_text(encoding="utf-8", errors="replace")
    authors = re.findall(r'<meta name="citation_author" content="(.*?)"', page)
    fact("authors listed on the arXiv page", len(authors))
    check("arXiv page lists version 2 dated 29 Oct 2026 and version 1 dated 28 May 2026", "29 Oct 2026" in page and "28 May 2026" in page)
    txt = (facts_dir / "kohli.txt").read_text(encoding="utf-8", errors="replace")
    checks = {
        "text identifies itself as arXiv:2605.29797v2": r"arXiv:2605\.29797v2",
        "ChaosNLI saturation table: N=10 row shows 91+-3% (KL) and 74+-6% (entropy correlation)": r"\b10\s+\.144±\.004\s+\.608±\.009\s+91±3%\s+74±6%",
        "ChaosNLI saturation table: N=20 row shows 94+-3% and 89+-5%": r"\b20\s+\.140±\.003\s+\.633±\.011\s+94±3%\s+89±5%",
        "DICES-990 table: N=3 row shows 90% (KL) and 62% (entropy correlation)": r"\b3\s+\.082±\.009\s+\.645±\.043\s+90%\s+62%",
        "Table 1 columns are Acc, ECE, Brier-soft, Dist-ECE, KL and Ent. r": r"Acc↑\s+ECE↓\s+Brier-soft↓\s+Dist-ECE↓\s+KL↓\s+Ent\. r↑",
        "compared training modes are Hard-label, label smoothing (LS) and Soft-label": r"Hard-label[\s\S]*LS α=0\.05[\s\S]*Soft-label",
    }
    for label, rx in checks.items():
        check(f"Kohli: {label}", bool(re.search(rx, txt)))
    fact("Kohli reports a Pearson-trained or correlation-trained judge", "not found; its 'entropy correlation' is a metric on predicted versus human entropy, and its trained models use hard, smoothed or soft label targets")


def section_verdicts(dices, chaos, pop, fsc) -> None:
    print("\n== 9. What the plan said versus what the files show (11-26-2026 plan, Gemini)")
    verdict("FSC 'R_v1 (343 items, 0-5 ordinal), ~5 domain experts/item'",
            f"not supported. Old set: {fsc['old']['items']} artifacts over {fsc['old']['tasks']} tasks with {fsc['old']['ratings']} expert ratings "
            f"({'/'.join(f'{k} ratings x {v}' for k, v in sorted(fsc['old']['dist'].items()))}); new set: {fsc['new']['items']} artifacts, {fsc['new']['ratings']} ratings; "
            "345 ratings over five raters is cited from the sister repo; 343 is in none of those files")
    verdict("DICES '60-112 raters/item'",
            f"not what the files show. DICES-350: {rng(dices['350']['min'], dices['350']['max'])} ratings per item ({rng(dices['350']['min_f'], dices['350']['max_f'])} after the README filter); "
            f"DICES-990: {rng(dices['990']['min'], dices['990']['max'])} ({rng(dices['990']['min_f'], dices['990']['max_f'])} after the filter)")
    verdict("ChaosNLI '3-class (4.6k items)'",
            f"the three-class set has {chaos['three']} items; {chaos['all']} counts the {chaos['alpha']} binary abductive-NLI items as well")
    verdict("POPQUORN 'politeness avg 6.7 and offensiveness avg 8.7 raters/item'",
            f"supported: files give {pop['politeness']['mean']:.2f} and {pop['offensiveness']['mean']:.2f} (maximum {pop['politeness']['max']} and {pop['offensiveness']['max']} per item, so a k-sweep past 8 to 10 is not possible)")
    verdict("POPQUORN '$12/hr, ~13 min per 50 ratings = ~$0.05/rating'",
            "supported as arithmetic on two sentences of the paper; the 13 minutes is for offensiveness only")
    verdict("HelpSteer2 'filtered out high-disagreement items >2 points'",
            "imprecise: the 2-point figure is real, but in the card it is the helpfulness agreement rule that decides whether extra annotators (up to five) are collected; the released label is the mean of the three most-agreeing annotators; the card describes no item filter based on disagreement (a quality-assurance step keeps 20,324 samples and names no disagreement criterion); per-annotator lists are released in the disagreements split")
    verdict("Kohli 'already runs subsampling saturation curves on ChaosNLI and DICES-990'",
            "supported by the paper's text; its 'correlation' is entropy correlation, and it compares soft-label, smoothed and hard-label training, not Pearson-trained judges")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--facts-dir", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--popquorn-dir", required=True)
    ap.add_argument("--popquorn-paper-txt", required=True)
    ap.add_argument("--sister-dir", required=True)
    ap.add_argument("--summeval-paper-txt", required=True)
    ap.add_argument("--exp36-data-io", default="")
    args = ap.parse_args()
    facts, cache = Path(args.facts_dir), Path(args.cache_dir)
    files = {
        "dices350.csv": facts / "diverse_safety_adversarial_dialog_350.csv",
        "dices990.csv": facts / "diverse_safety_adversarial_dialog_990.csv",
        "chaosNLI_snli.jsonl": cache / "chaosNLI_snli.jsonl",
        "chaosNLI_mnli_m.jsonl": cache / "chaosNLI_mnli_m.jsonl",
        "chaosNLI_alphanli.jsonl": cache / "chaosNLI_alphanli.jsonl",
        "model_annotations.aligned.jsonl": cache / "model_annotations.aligned.jsonl",
        "arxiv_kohli.txt": facts / "arxiv_kohli.txt",
        "kohli.txt (pdftotext of arXiv v2 PDF)": facts / "kohli.txt",
        "kohli.pdf": facts / "kohli.pdf",
        "helpsteer2_card.txt": facts / "helpsteer2_card.txt",
        "popquorn paper text (pdftotext of arXiv 2306.06826 PDF)": Path(args.popquorn_paper_txt),
    }
    files["summeval paper text (pdftotext in reading-order mode of arXiv 2007.12626 PDF)"] = Path(args.summeval_paper_txt)
    sepdf = Path(args.summeval_paper_txt).with_suffix(".pdf")
    if sepdf.exists():
        files["summeval.pdf (arXiv 2007.12626)"] = sepdf
    paper_pdf = Path(args.popquorn_paper_txt).with_suffix(".pdf")
    if paper_pdf.exists():
        files["popquorn.pdf (arXiv 2306.06826)"] = paper_pdf
    for extra in ("offensiveness", "politeness_rating"):
        files[f"popquorn {extra}/raw_data.csv"] = Path(args.popquorn_dir) / "dataset" / extra / "raw_data.csv"
    section_inputs(args, files)
    dices = section_dices(files)
    chaos = section_chaos(files)
    section_summeval(files)
    section_summeval_paper(args, files)
    pop = section_popquorn(args, files)
    section_helpsteer2(facts)
    fsc = section_fsc(args)
    section_kohli(facts)
    section_verdicts(dices, chaos, pop, fsc)
    print("\nRESULT: " + ("ALL INTEGRITY CHECKS PASS" if not FAILS else "FAILED: " + "; ".join(FAILS)))
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
