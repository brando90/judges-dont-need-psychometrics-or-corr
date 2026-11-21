"""Count full content with the local Ollama model's vocabulary, without running inference."""
import argparse
import json
import re
import statistics
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from prepare import sha

VOCAB_MODEL = "sha256-02d45dc1cf451ba2475ac33b301c2dd8f985abe4c182ce04a1f2f5bf0260278d"


def count(text, gguf):
    output = subprocess.run(["llama-tokenize", "-m", str(gguf), "--stdin", "--no-bos", "--no-escape", "--show-count"],
                            input=text.encode(), capture_output=True, check=True).stdout
    return int(re.search(rb"Total number of tokens: (\d+)\s*$", output).group(1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--gguf", type=Path, default=Path.home()/".ollama/models/blobs"/VOCAB_MODEL)
    args = parser.parse_args()
    requests = list(map(json.loads, (args.directory / "requests.jsonl").read_text().splitlines()))
    records = {r["id"]: r for r in map(json.loads, (args.directory / "manifest.jsonl").read_text().splitlines())}
    assert {r["id"] for r in requests} == set(records)
    for r in requests:
        for key in ("system","user"):
            assert sha(r[key].encode()) == records[r["id"]][key + "_sha256"]
    system, = {r["system"] for r in requests}
    system_length = count(system, args.gguf)
    by_split = defaultdict(list)
    with ThreadPoolExecutor(max_workers=2) as executor:
        for r,n in zip(requests, executor.map(lambda r:count(r["user"],args.gguf), requests)):
            by_split[records[r["id"]]["split"]].append(n + system_length)
    print("System content tokens:", system_length)
    for split, vals in sorted(by_split.items()):
        print(f"{split}: total {sum(vals)}, min {min(vals)}, median {statistics.median(vals)}, max {max(vals)}")


if __name__ == "__main__":
    main()
