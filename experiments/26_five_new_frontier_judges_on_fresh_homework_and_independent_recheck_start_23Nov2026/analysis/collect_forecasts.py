"""Turn raw judge receipts into frozen forecast rows and a cost table (11-23-2026).

Parsing is the experiment 21 rule: JSON {"probabilities": [6 numbers]} optionally inside one Markdown fence, finite,
within [0,1], sum within 0.01 of one then renormalized; anything else (or a failed call) becomes the uniform forecast
and is counted. Usage: python3 collect_forecasts.py RECEIPT_DIR FORECASTS.csv COSTS.csv
"""
import csv, hashlib, json, math, sys
from pathlib import Path

JUDGES = {  # receipt file stem -> (judge label, input USD per M tokens, output USD per M tokens, rate source)
    "gateway_gpt-6-astra": ("gpt-6-astra", 10, 50, "Pi gateway provider list rate"),
    "claude-opus-5-5": ("claude-opus-5-5", 4, 20, "Pi gateway provider list rate"),
    "Kimi_K3": ("kimi-k3", 0, 0, "listed at 0 in the gateway provider"),
    "GLM_5.3": ("glm-5.3", 0, 0, "listed at 0 in the gateway provider"),
    "DeepSeek_V4.1_Flash_1017": ("deepseek-v4.1-flash", 0, 0, "listed at 0 in the gateway provider"),
}
REFERENCE = (1.4, 4.4)  # GLM 5.3 list rate on another provider, used as a stand-in upper reference for the 0-rated models


def parse(raw):
    text = (raw or "").strip()
    if text.startswith(("```json\n", "```\n")) and text.endswith("```"):
        text = text.split("\n", 1)[1][:-3].strip()
    try:
        v = json.loads(text)["probabilities"]
        if len(v) != 6 or any(type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1 for p in v):
            raise ValueError
        s = sum(v)
        if abs(s - 1) > 0.01 + 1e-12:
            raise ValueError
        return [p / s for p in v], "valid"
    except (ValueError, KeyError, TypeError):
        return [1 / 6] * 6, "invalid_uniform"


def main(receipts, forecasts, costs):
    rows, cost_rows = [], []
    for stem, (label, rin, rout, src) in JUDGES.items():
        path = Path(receipts) / (stem + ".jsonl")
        got = {}
        for line in path.read_text().splitlines():
            r = json.loads(line)
            if r["status"] == "ok" or r["id"] not in got:
                got[r["id"]] = r
        tin = tout = ok = 0
        for r in got.values():
            if r["status"] == "ok":
                probs, status = parse(r["raw_response"])
                u = r.get("usage") or {}
                tin += u.get("prompt_tokens", 0); tout += u.get("completion_tokens", 0); ok += 1
                served, raw = r.get("served_model"), r["raw_response"]
            else:
                probs, status, served, raw = [1 / 6] * 6, "failure_uniform", None, ""
            rows.append({"id": r["id"], "judge": label, "messages_sha256": r["system_sha256"][:16] + r["user_sha256"][:16],
                         "response_sha256": hashlib.sha256(raw.encode()).hexdigest(), "status": status,
                         "model": served or label, **{f"p{i}": p for i, p in enumerate(probs)}})
        est = (tin * rin + tout * rout) / 1e6
        ref = (tin * REFERENCE[0] + tout * REFERENCE[1]) / 1e6
        cost_rows.append({"judge": label, "calls": len(got), "ok_calls": ok, "input_tokens": tin, "output_tokens": tout,
                          "estimated_usd_at_configured_rate": round(est, 4), "rate_source": src,
                          "reference_usd_if_priced_at_1.4_and_4.4": round(ref, 4)})
    for path, data in ((forecasts, rows), (costs, cost_rows)):
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]), lineterminator="\n")
            w.writeheader(); w.writerows(data)
    for c in cost_rows:
        print(c)
    for label in JUDGES.values():
        g = [r for r in rows if r["judge"] == label[0]]
        print(label[0], len(g), {s: sum(r["status"] == s for r in g) for s in sorted({r["status"] for r in g})})


if __name__ == "__main__":
    main(*sys.argv[1:4])
