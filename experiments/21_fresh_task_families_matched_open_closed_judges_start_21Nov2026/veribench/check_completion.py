"""Report aggregate completion only. Reads sheet headers and two server-side counts per sheet."""
import argparse
import csv
import datetime
import io
import json
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote

import requests
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials

from prepare import COLLECTION, COLLECTION_REV, git_bytes


def letters(index):
    result = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        result = chr(65 + remainder) + result
    return result


def count(record, token):
    file_id, tab, category = record
    auth = {"Authorization": "Bearer " + token}
    h = requests.get(f"https://sheets.googleapis.com/v4/spreadsheets/{file_id}/values/" +
                     quote(f"'{tab}'!1:1", safe=""), headers=auth, timeout=25)
    h.raise_for_status()
    header = h.json()["values"][0]
    score = letters(1 + header.index("final_score_0_5"))
    item = letters(1 + header.index("item_id"))
    q = f"select count({score}), count({item}) label count({score}) 'filled', count({item}) 'assigned'"
    r = requests.get(f"https://docs.google.com/spreadsheets/d/{file_id}/gviz/tq",
                     params={"sheet":tab, "headers":1, "tqx":"out:csv", "tq":q}, headers=auth, timeout=25)
    r.raise_for_status()
    counts, = csv.DictReader(io.StringIO(r.content.decode("utf-8-sig")))
    assert set(counts) == {"filled", "assigned"}
    return category, int(counts["filled"]), int(counts["assigned"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sister", type=Path, default=Path.home()/"trustworthy-theorem-judges")
    parser.add_argument("--credentials", type=Path, default=Path.home()/"keys/gspread_authorized_user.json")
    args = parser.parse_args()
    source = json.loads(git_bytes(args.sister, COLLECTION_REV, COLLECTION + "/ratings/RAW_MANIFEST.json"))
    records = [(f["spreadsheet_id"], f["range"].split("!")[0].strip("'"), f["packet"]) for f in source["files"]]
    readme = git_bytes(args.sister, COLLECTION_REV, COLLECTION + "/README.md").decode()
    extra = re.search(r"\bT30\b.*?spreadsheets/d/([^/]+)", readme)
    assert extra
    records.append((extra[1], "Ratings", "native_main"))
    assert len({r[0] for r in records}) == len(records)
    credential = Credentials.from_authorized_user_file(args.credentials)
    credential.refresh(Request())
    totals = defaultdict(lambda: [0, 0, 0])
    with ThreadPoolExecutor(max_workers=4) as executor:
        for group, filled, assigned in executor.map(lambda r: count(r, credential.token), records):
            totals[group][0] += 1
            totals[group][1] += filled
            totals[group][2] += assigned
    print(json.dumps({"checked_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      "counts_order":["sheets","typed_score_count","assigned_rows"],
                      "rounds":dict(sorted(totals.items()))}, indent=2))


if __name__ == "__main__":
    main()
