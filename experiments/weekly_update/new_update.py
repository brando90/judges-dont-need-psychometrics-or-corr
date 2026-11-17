#!/usr/bin/env python3
"""TLDR: Create the next weekly-update folder under experiments/weekly_update/ with the newest-first sort key (99999999 - YYYYMMDD) and a template post, or check that every existing folder's key matches its date."""
import argparse
import datetime
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
FOLDER_RE = re.compile(r"^(\d{8})_(\d{2}-\d{2}-\d{4})_(.+)$")

TEMPLATE = """# Judge-Objectives Weekly Update, {date}

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/weekly_update/{folder}/{file}>

**TLDR:** Draft weekly-update note for {date}: Part 1A gives the copy-paste additive update for item 3 in Brando's weekly updates Google Doc, Part 1B gives the compact Slack version, and Part 2 gives the full provenance and Q&A appendix.

---

## Part 1A. Paste-ready additive update for weekly updates Google Doc ({date})

...

---

## Part 1B. Compact Slack post

**Update {slack_date}**:

...

full update: https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/weekly_update/{folder}/{file}

---

## Part 2. Appendix (for Brando)

### A. Detailed answers to open questions

...

### B. Provenance sheet (numbers to repo paths)

| Number | Source |
|---|---|
| ... | `experiments/<full folder name>/...` |
"""


def key_for(day: datetime.date) -> int:
    return 99999999 - int(day.strftime("%Y%m%d"))


def check() -> int:
    bad = 0
    for folder in sorted(p for p in HERE.iterdir() if p.is_dir()):
        m = FOLDER_RE.match(folder.name)
        if not m:
            print(f"SKIP (not a dated folder): {folder.name}")
            continue
        key, date_text = int(m.group(1)), m.group(2)
        day = datetime.datetime.strptime(date_text, "%m-%d-%Y").date()
        expected = key_for(day)
        status = "ok" if key == expected else f"MISMATCH (expected {expected})"
        if key != expected:
            bad += 1
        print(f"{folder.name}: {status}")
    return bad


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--date", help="MM-DD-YYYY (default: today)")
    ap.add_argument("--slug", default="judges_ce_vs_corr_and_g_theory", help="folder and file slug")
    ap.add_argument("--check", action="store_true", help="verify every folder's key matches its date, create nothing")
    args = ap.parse_args()
    if args.check:
        sys.exit(1 if check() else 0)
    day = datetime.datetime.strptime(args.date, "%m-%d-%Y").date() if args.date else datetime.date.today()
    date_text = day.strftime("%m-%d-%Y")
    folder = HERE / f"{key_for(day)}_{date_text}_{args.slug}"
    if folder.exists():
        sys.exit(f"already exists: {folder}")
    folder.mkdir()
    post = folder / f"{args.slug}_{date_text}.md"
    post.write_text(TEMPLATE.format(date=date_text, slack_date=day.strftime("%b %-d, %Y"),
                                    folder=folder.name, file=post.name))
    print(folder)
    print(post)


if __name__ == "__main__":
    main()
