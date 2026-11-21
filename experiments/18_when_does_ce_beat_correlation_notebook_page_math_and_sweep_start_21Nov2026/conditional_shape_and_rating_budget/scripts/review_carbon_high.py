#!/usr/bin/env python3
"""Run one Gemini Carbon High review from the account owner's Terminal."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory


STUDY = Path(__file__).resolve().parents[1]
MODEL = "gemini-4-carbon-high"
PACKET_SHA256 = "6f6564f6e687097ffb4659ccc7dc14d62482350aefeec84cffefa0a3501f5b65"
BRIEF = (
    "Perform ONE light scientific review. Check whether reported differences, denominators, loss definitions, "
    "math and finite-sample interpretation follow from the design and results. Check split handling, ties, "
    "conditional family uncertainty, stage of model fitting versus selection, exact versus empirical claims. "
    "Flag concrete errors with file and correction. Give a verdict PASS or FIX REQUIRED and severity. "
    "Do not edit files. Only the numeric, procedural and theoretical claims in this package are under review.\n"
)
SOURCES = (
    "README.md", "MATH_11-21-2026.md", "PROTOCOL.md", "results.md",
    "historical_rating_budget/PROTOCOL_11-21-2026.md",
    "historical_rating_budget/RESULTS_11-21-2026.md",
    "input_ablation/README.md", "scripts/run.py", "historical_rating_budget/analyze.py",
)


def build_packet():
    sections = [BRIEF.encode()]
    for name in SOURCES:
        sections.append(f"===== {name} =====\n".encode() + (STUDY / name).read_bytes())
    return b"\n\n".join(sections)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify the packet locally and exit")
    parser.add_argument("--packet", type=Path, help="read the archived packet instead of building it")
    parser.add_argument("--output", type=Path, default=STUDY / "carbon-high.json")
    args = parser.parse_args()

    packet = args.packet.read_bytes() if args.packet else build_packet()
    digest = hashlib.sha256(packet).hexdigest()
    if digest != PACKET_SHA256:
        parser.exit(1, f"Packet differs from the archived review input: {digest}\n")
    print(f"{MODEL} | {len(packet):,} bytes | SHA-256 {digest}", flush=True)
    if args.check:
        return
    if args.output.exists():
        parser.exit(1, f"Existing review output: {args.output}\n")

    # Pass the packet as a single literal argument. Keep the study files outside the CLI workspace.
    with TemporaryDirectory() as workspace:
        run = subprocess.run(
            ["agy", "--model", MODEL, "--output-format", "json", "-p", packet.decode("utf-8")],
            cwd=workspace, stdout=subprocess.PIPE,
        )
    if run.returncode:
        parser.exit(1, f"agy exited with status {run.returncode}.\n")
    result = json.loads(run.stdout)
    if result.get("status") != "SUCCESS" or not (result.get("response") or "").strip():
        reason = result.get("error") or f"status={result.get('status')}, response missing or incomplete"
        parser.exit(1, f"Gemini returned no usable review: {reason}\n")
    with args.output.open("xb") as output:
        output.write(run.stdout)
    print(f"Gemini response saved to {args.output}")


if __name__ == "__main__":
    main()
