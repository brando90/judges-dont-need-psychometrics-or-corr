#!/usr/bin/env python3
"""TLDR: Write FREEZE.json (SHA-256 of the protocol and measurement scripts) before any measurement.

Run once before executing Phase A/B/C:
  python experiments/04_g_theory_reliability_vs_distribution_accuracy_start_13Nov2026/scripts/freeze.py
Re-running refuses to overwrite; a later change must be recorded as an amendment (--amend "reason").
"""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
FILES = ['protocol.json', 'scripts/gtheory_lib.py', 'scripts/test_metrics.py', 'scripts/check_counterexamples.py',
         'scripts/run_d_study.py', 'scripts/run_historical.py']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--amend', default=None, help='reason for an amendment after the original freeze')
    a = ap.parse_args()
    path = HERE / 'FREEZE.json'
    hashes = {f: hashlib.sha256((HERE / f).read_bytes()).hexdigest() for f in FILES}
    now = dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    if path.exists():
        if not a.amend:
            raise SystemExit('FREEZE.json exists; pass --amend "reason" to record an amendment')
        old = json.loads(path.read_text())
        changed = sorted(f for f in FILES if old['hashes'].get(f) != hashes[f])
        old.setdefault('amendments', []).append({'utc': now, 'reason': a.amend, 'changed_files': changed, 'hashes': hashes})
        path.write_text(json.dumps(old, indent=2, sort_keys=True) + '\n')
        print(json.dumps({'amended': changed}))
        return
    path.write_text(json.dumps({'status': 'fixed before Phase A/B/C measurements; formula unit checks (test_metrics) already run',
                                'utc': now, 'hashes': hashes}, indent=2, sort_keys=True) + '\n')
    print(json.dumps(hashes, indent=1))


if __name__ == '__main__':
    main()
