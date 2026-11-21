"""Check that both providers received the same prompts, then retain forecasts only."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def sha(data):
    return hashlib.sha256(data).hexdigest()


def parse(raw):
    text = raw.strip()
    if text.startswith(("```json\n", "```\n")) and text.endswith("```"):
        text = text.split("\n", 1)[1][:-3].strip()
    try:
        value = json.loads(text)["probabilities"]
        if len(value) != 6 or any(type(p) not in (int, float) or not math.isfinite(p)
                                  or not 0 <= p <= 1 for p in value):
            raise ValueError
        total = sum(value)
        if abs(total - 1) > 0.01 + 1e-12:
            raise ValueError
        return [p / total for p in value], "valid"
    except (ValueError, KeyError, TypeError):
        return [1 / 6] * 6, "invalid_uniform"


def main(requests, open_receipts, closed_receipts, output):
    data = requests.read_bytes()
    prompts = [json.loads(line) for line in data.splitlines()]
    closed_rows = [json.loads(line) for line in closed_receipts.read_bytes().splitlines()]
    closed = {r["id"]: r for r in closed_rows}
    assert len(closed_rows) == len(closed) == len(prompts)
    results = []
    for index, row in enumerate(prompts):
        local = json.loads((open_receipts / f"{index:05d}.json").read_text())
        hosted = closed[row["id"]]
        assert local["id"] == row["id"]
        assert local["prompts_sha256"] == hosted["requests_sha256"] == sha(data)
        messages = [{"role": role, "content": row[role]} for role in ("system", "user")]
        message_sha = sha(json.dumps(messages, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode())
        assert message_sha == local["messages_sha256"]
        for role in ("system", "user"):
            assert sha(row[role].encode()) == hosted[role + "_sha256"]
        for judge, receipt, raw, ok in (
            ("qwen", local, local.get("response", {}).get("message", {}).get("content", ""),
             local.get("response", {}).get("done_reason") == "stop"),
            ("claude", hosted, hosted.get("raw_response", ""), hosted["status"] == "ok"),
        ):
            probs, status = parse(raw)
            if not ok:
                probs, status = [1 / 6] * 6, "failure_uniform"
            results.append({"id": row["id"], "judge": judge, "messages_sha256": message_sha,
                            "response_sha256": sha(raw.encode()), "status": status,
                            "model": receipt.get("model") or local["response"]["model"],
                            **{f"p{i}": p for i, p in enumerate(probs)}})
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(results)
    for judge in ("qwen", "claude"):
        group = [r for r in results if r["judge"] == judge]
        print(judge, len(group), {s:sum(r["status"] == s for r in group)
                                 for s in sorted({r["status"] for r in group})})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("requests", "open_receipts", "closed_receipts", "output"):
        parser.add_argument(field, type=Path)
    main(**vars(parser.parse_args()))
