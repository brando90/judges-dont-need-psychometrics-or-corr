"""Turn raw judge receipts into frozen forecast rows and a cost table (11-23-2026).

Parsing (frozen in PROTOCOL): JSON {"probabilities": [K numbers]} optionally inside one Markdown fence, finite, in [0,1],
sum within 0.01 of one then renormalized; anything else, or a failed call, becomes the uniform forecast and is counted.
Usage: python3 collect_forecasts.py RECEIPT_DIR REQUESTS_JSONL FORECASTS.csv COSTS.csv
"""
import csv, json, math, sys
from pathlib import Path

JUDGES = {  # receipt file stem -> (label, input USD per M tokens, output USD per M tokens, rate source)
    "gpt-6-astra": ("gpt-6-astra", 10, 50, "gateway list rate used in experiment 26"),
    "claude-opus-5-5": ("claude-opus-5-5", 4, 20, "gateway list rate used in experiment 26"),
    "Kimi_K3": ("kimi-k3", 0, 0, "listed at 0 in the gateway provider"),
    "GLM_5.3": ("glm-5.3", 0, 0, "listed at 0 in the gateway provider"),
    "DeepSeek_V4.1_Flash_1017": ("deepseek-v4.1-flash", 0, 0, "listed at 0 in the gateway provider"),
}
REFERENCE = (1.4, 4.4)
KOF = lambda i: 2 if i.startswith("chaosnli_alphanli/") else 3


def parse(raw, K):
    text = (raw or "").strip()
    if text.startswith(("```json\n", "```\n")) and text.endswith("```"):
        text = text.split("\n", 1)[1][:-3].strip()
    try:
        v = json.loads(text)["probabilities"]
        if len(v) != K or any(type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1 for p in v):
            raise ValueError
        s = sum(v)
        if abs(s - 1) > 0.01 + 1e-12:
            raise ValueError
        return [p / s for p in v], "valid"
    except (ValueError, KeyError, TypeError, AttributeError):
        return [1 / K] * K, "invalid_uniform"


def main(receipts, requests, forecasts, costs):
    ids = [json.loads(l)["id"] for l in Path(requests).read_text().splitlines()]
    rows, cost_rows = [], []
    for stem, (label, rin, rout, src) in JUDGES.items():
        got = {}
        for line in (Path(receipts) / (stem + ".jsonl")).read_text().splitlines():
            r = json.loads(line)
            if r["status"] == "ok" or r["id"] not in got:
                got[r["id"]] = r
        tin = tout = ok = 0
        for i in ids:
            K = KOF(i)
            r = got.get(i)
            if r and r["status"] == "ok":
                probs, status = parse(r["raw_response"], K)
                u = r.get("usage") or {}
                tin += u.get("prompt_tokens", 0); tout += u.get("completion_tokens", 0); ok += 1
            else:
                probs, status = [1 / K] * K, "failed_call_uniform"
            rows.append([i, label, status] + [f"{p:.10g}" for p in probs] + [""] * (3 - K))
        cost_rows.append([label, len(ids), ok, tin, tout, round((tin * rin + tout * rout) / 1e6, 4), src,
                          round((tin * REFERENCE[0] + tout * REFERENCE[1]) / 1e6, 4)])
    with open(forecasts, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["id", "judge", "status", "p0", "p1", "p2"]); w.writerows(rows)
    with open(costs, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["judge", "calls", "ok_calls", "input_tokens", "output_tokens", "estimated_usd_at_configured_rate", "rate_source", "reference_usd_if_priced_at_1.4_and_4.4"])
        w.writerows(cost_rows)
    print(Path(costs).read_text())


if __name__ == "__main__":
    main(*sys.argv[1:5])
