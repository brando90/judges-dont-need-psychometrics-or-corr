#!/usr/bin/env python3
"""Print the numbered Sources section (Markdown) of REPORT_11-27-2026.md from census.csv.

Deterministic: reads only census.csv, no randomness, no network. Each entry is keyed by its census id
(lineage-number), so a reader can look the row up in census.csv. Numbering is continuous across lineages.
Usage: python3 make_sources.py
"""
import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent
GROUPS = [
    ("L1", "Graded-score judges and metrics"),
    ("L2", "Pairwise-preference judges and reward models"),
    ("L3", "Benchmark and leaderboard ranking validation"),
    ("L4", "Frontier-lab and industry graders"),
    ("L5", "Retrieval relevance and annotation replacement"),
    ("L6", "Critique and distributional papers"),
    ("P", "Pre-LLM context rows"),
]


def authors_and_title(citation):
    if ". " in citation:
        authors, title = citation.split(". ", 1)
    else:
        authors, title = "", citation
    etal = "et al" in authors
    names = [a.strip() for a in authors.replace("et al", "").split(",") if a.strip()]
    if not names:
        who = ""
    elif len(names) > 2 or etal:
        who = f"{names[0]} et al."
    else:
        who = " and ".join(names)
    if who and not who.endswith("."):
        who += "."
    return who, title.rstrip(".")


def main():
    with open(HERE / "census.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    out = ["## 11. Sources", "",
           "Every row of `census.csv`, keyed by census id and numbered continuously. Titles and venues are as coded; the link is the page that was read.", ""]
    n = 0
    for lin, name in GROUPS:
        rs = sorted((r for r in rows if r["lineage"] == lin), key=lambda r: r["id"])
        note = "; excluded from every base rate" if lin in ("L6", "P") else ""
        out += [f"### {lin}. {name} ({len(rs)}{note})", ""]
        for r in rs:
            n += 1
            who, title = authors_and_title(r["citation"])
            out.append(f"{n}. [{r['id']}] {who} {title} ({r['year']}). {r['venue_or_type']}. <{r['source_url']}>")
        out.append("")
    print("\n".join(out))


if __name__ == "__main__":
    main()
