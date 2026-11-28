#!/usr/bin/env python3
"""Fetch (or copy) the six raw files used by Experiment 39 and verify their SHA-256 checksums.

Usage, from the experiment folder:
    python3 src/fetch_raw_data.py                # writes data/raw/
    python3 src/fetch_raw_data.py --out /tmp/x   # write somewhere else (for a dry check)

Sources (all public, no credentials, no model calls):
  * ChaosNLI SNLI / MNLI-m / alphaNLI jsonl: HuggingFace dataset earino/chaosnli at the pinned commit below.
  * DICES-350 and DICES-990 CSV: google-research-datasets/dices-dataset on GitHub (branch main; it is not
    commit-pinned, so the checksum below is what guarantees the bytes).
  * Experiment 28 frozen judge forecasts: copied from this repository (no download).
A file that already exists with the right checksum is left alone. A checksum mismatch is an error: nothing is
silently accepted.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT.parent

HF_COMMIT = "eb7cc18dbfa0f161d93a2b22c7856ca0faafcf2b"
HF_BASE = f"https://huggingface.co/datasets/earino/chaosnli/resolve/{HF_COMMIT}/raw/"
DICES_BASE = "https://raw.githubusercontent.com/google-research-datasets/dices-dataset/main/"

FILES = {
    "chaosNLI_snli.jsonl": HF_BASE + "chaosNLI_snli.jsonl",
    "chaosNLI_mnli_m.jsonl": HF_BASE + "chaosNLI_mnli_m.jsonl",
    "chaosNLI_alphanli.jsonl": HF_BASE + "chaosNLI_alphanli.jsonl",
    "diverse_safety_adversarial_dialog_350.csv": DICES_BASE + "350/diverse_safety_adversarial_dialog_350.csv",
    "diverse_safety_adversarial_dialog_990.csv": DICES_BASE + "990/diverse_safety_adversarial_dialog_990.csv",
}
FORECASTS_NAME = "forecasts_11-23-2026.csv"
FORECASTS_SRC = (
    EXPERIMENTS
    / "28_public_rating_sets_chaosnli_dices_frontier_judges_ce_vs_pearson_start_23Nov2026"
    / "forecasts"
    / FORECASTS_NAME
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_sums() -> dict[str, str]:
    sums = {}
    for line in (ROOT / "data" / "SHA256SUMS.txt").read_text().splitlines():
        if line.strip():
            digest, name = line.split()
            sums[name] = digest
    return sums


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "raw")
    args = ap.parse_args()
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    sums = read_sums()
    bad = 0
    for name, url in FILES.items():
        dest = out / name
        if dest.exists() and sha256(dest) == sums[name]:
            print(f"ok (already present) {name}")
            continue
        print(f"downloading {name} from {url}")
        with urllib.request.urlopen(url, timeout=120) as r, dest.open("wb") as f:
            shutil.copyfileobj(r, f)
        got = sha256(dest)
        if got != sums[name]:
            print(f"CHECKSUM MISMATCH {name}: expected {sums[name]} got {got}", file=sys.stderr)
            bad += 1
        else:
            print(f"ok {name}")
    dest = out / FORECASTS_NAME
    shutil.copyfile(FORECASTS_SRC, dest)
    got = sha256(dest)
    if got != sums[FORECASTS_NAME]:
        print(f"CHECKSUM MISMATCH {FORECASTS_NAME}: expected {sums[FORECASTS_NAME]} got {got}", file=sys.stderr)
        bad += 1
    else:
        print(f"ok {FORECASTS_NAME} (copied from the Experiment 28 folder)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
