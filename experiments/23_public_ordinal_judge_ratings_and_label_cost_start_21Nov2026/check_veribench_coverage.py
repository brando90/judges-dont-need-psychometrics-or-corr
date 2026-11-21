"""Count available official Set N ratings without downloading any individual outcomes."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
from pathlib import Path
import re
from urllib.parse import urlencode
from urllib.error import HTTPError
from urllib.request import urlopen


def query(sheet_id, statement):
    params = urlencode(dict(sheet="Ratings", headers=1, tqx="out:json", tq=statement))
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?{params}"
    try:
        with urlopen(url, timeout=35) as response:
            content = response.read().decode()
    except HTTPError as exc:
        if exc.code not in (401, 403):
            raise
        from google.auth.transport.requests import AuthorizedSession
        from google.oauth2.credentials import Credentials

        credentials = Credentials.from_authorized_user_file(args.google_credentials)
        with AuthorizedSession(credentials) as session:
            response = session.get(url, timeout=35)
            response.raise_for_status()
            content = response.text
    message = json.loads(content[content.index("{"):content.rindex("}")+1])
    assert message["status"] == "ok"
    return message["table"]


def coverage(sheet_id):
    schema = query(sheet_id, "select * limit 0")
    assert not schema["rows"]
    labels = {c["label"]: c["id"] for c in schema["cols"]}
    item, grade = labels["item_id"], labels["final_score_0_5"]
    table = query(sheet_id, f"select count({item}),count({grade})")
    assert len(table["rows"]) == 1
    return tuple(int(c["v"]) for c in table["rows"][0]["c"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("official_collection", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--as-of", required=True, help="Time of live aggregate check in Pacific time")
    parser.add_argument("--google-credentials", type=Path,
                        default=Path.home() / "keys/gspread_authorized_user.json",
                        help="Google authorization for source forms with restricted access")
    args = parser.parse_args()
    with (args.official_collection / "ratings/RAW_MANIFEST.json").open() as stream:
        files = json.load(stream)["files"]
    sheets = [f["spreadsheet_id"] for f in files if f["packet"] == "native_main"]
    roster = (args.official_collection / "README.md").read_text()
    line = next(s for s in roster.splitlines() if "Also rating:" in s and "T30" in s)
    sheets.append(re.search(r"docs.google.com/spreadsheets/d/([\w-]+)", line).group(1))
    assert len(sheets) == len(set(sheets)) == 34
    with ThreadPoolExecutor(max_workers=6) as pool:
        cells = list(pool.map(coverage, sheets))
    assert all(items == 100 for items, _ in cells)
    row = dict(as_of_pacific=args.as_of, sheets=len(sheets), assigned=sum(a for a,_ in cells),
               nonempty_score_cells=sum(b for _,b in cells), individual_outcomes_downloaded=0)
    with args.output.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=row, lineterminator="\n")
        writer.writeheader()
        writer.writerow(row)
    print(row)
