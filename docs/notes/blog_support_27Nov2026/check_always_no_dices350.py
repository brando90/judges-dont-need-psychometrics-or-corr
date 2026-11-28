#!/usr/bin/env python3
"""Check that always answering "No" agrees with the human majority rating on 269 of 350 DICES-350 conversations.

The blog post says the five frozen frontier judges agree with the human majority rating 80 to 82% of the time,
"a few points above the roughly 77% that always answering No reaches". Experiment 28 scored a constant forecast
(`method=hist`) at 0.768571 = 269/350 with its rule that ties between human modal labels count as wrong; this
script recomputes the same number straight from the public DICES-350 release, so that it is also known that
the constant forecast's label is "No".

Input (a path you supply; nothing is downloaded): the public file diverse_safety_adversarial_dialog_350.csv from
the DICES-350 release, with one row per rater and conversation, columns `item_id` and `Q_overall`
(No, Yes or Unsure).

Run: python3 -I check_always_no_dices350.py PATH/diverse_safety_adversarial_dialog_350.csv
Prints PASS or FAIL for each claim and exits 1 on a failure. Deterministic, no network.
"""
import csv
import sys
from collections import Counter, defaultdict

path = sys.argv[1]
by_item = defaultdict(Counter)
with open(path, newline="", encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        by_item[r["item_id"]][r["Q_overall"]] += 1

per_item = sorted({sum(c.values()) for c in by_item.values()})
modal = Counter()
for c in by_item.values():
    top = c.most_common()
    modal["tie" if len(top) > 1 and top[0][1] == top[1][1] else top[0][0]] += 1

failed = []


def check(cid, ok, text):
    print("PASS" if ok else "FAIL", f"[{cid}]", text)
    if not ok:
        failed.append(cid)


check("N1", len(by_item) == 350 and per_item == [123], f"{len(by_item)} conversations, raters per conversation {per_item}")
check("N2", modal["No"] == 269 and modal["Yes"] == 79 and modal["tie"] == 2 and modal["Unsure"] == 0,
      f"unique human modal label per conversation: {dict(modal)}")
share = modal["No"] / len(by_item)
check("N3", abs(share - 269 / 350) < 1e-12 and f"{share:.4f}" == "0.7686",
      f"always answering No agrees with the unique human modal label on {modal['No']} of {len(by_item)} conversations = {share:.4f} (ties count as wrong), "
      "the value of experiment 28's constant forecast (method hist, 0.768571)")
print(f"\n{len(failed)} FAIL")
sys.exit(1 if failed else 0)
