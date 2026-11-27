"""Compare REPORT.md Study A table rows with a study_a_summary.json (argv[1]) and REPORT path (argv[2])."""
import json, re, sys
S = json.load(open(sys.argv[1])); rep = open(sys.argv[2]).read()
name = {"ChaosNLI SNLI":"chaosnli_snli","ChaosNLI MNLI-m":"chaosnli_mnli","ChaosNLI alphaNLI":"chaosnli_alphanli","DICES-350":"dices350","DICES-990":"dices990"}
def num(s):
    s = s.replace("$","").replace("\\times 10^{","e").replace("}","").replace(" ","").replace("+","")
    return float(s)
bad = 0; n = 0
for line in rep.splitlines():
    if not line.startswith("| ChaosNLI") and not line.startswith("| DICES"): continue
    cols = [c.strip() for c in line.strip("|").split("|")]
    if len(cols) != 11: continue
    ds = name[cols[0]]; B = cols[1]
    cell = S["datasets"][ds]["budgets"][B]; sel = cell["selection"]; pr = cell["paired"]["u1_minus_ps_rps"]
    n += 1
    def d(s): return s.replace("$","").replace("\\times","x").replace(" ","")
    got = {
      "u1": sel["u1"]["mode"], "ps": sel["ps"]["mode"], "rps_oracle": sel["rps_oracle"]["mode"], "item": sel["item_oracle"]["mode"],
      "mean": pr["mean"], "lo": pr["ci95"][0], "hi": pr["ci95"][1], "p": pr.get("paired_t_pvalue"),
      "ps_eq_rpsor": sel["ps"]["agree_rps_oracle"]["proportion"], "u1_eq_item": sel["u1"]["agree_item_oracle"]["proportion"],
      "tie_u1_u2": cell["primary_transform_tie_u1_u2"]["proportion"]}
    issues = []
    if d(cols[2]) != got["u1"]: issues.append(("u1", cols[2], got["u1"]))
    if d(cols[3]) != got["ps"] or got["ps"] != got["rps_oracle"]: issues.append(("ps/rps-oracle", cols[3], got["ps"], got["rps_oracle"]))
    if d(cols[4]) != got["item"]: issues.append(("item", cols[4], got["item"]))
    m = re.match(r"\$?([+-]?[0-9.]+)\$?\s*\$?\[([+-]?[0-9.]+),([+-]?[0-9.]+)\]\$?", cols[5].replace(" ",""))
    if m:
        mean, lo, hi = map(float, m.groups())
        for k, v in (("mean", mean), ("lo", lo), ("hi", hi)):
            dec = len(str(v).split(".")[1]) if "." in str(v) else 0
            if abs(round(got[k], 5) - v) > 1.01e-5 and abs(got[k]-v) > 6e-6: issues.append((k, v, got[k]))
    else:
        issues.append(("unparsed", cols[5]))
    if cols[6] not in ("—",):
        pv = num(cols[6])
        if got["p"] is None or abs(pv - got["p"]) / max(got["p"], 1e-300) > 0.02: issues.append(("p", pv, got["p"]))
    for k, c in (("ps_eq_rpsor", 8), ("u1_eq_item", 9), ("tie_u1_u2", 10)):
        if abs(float(cols[c]) - got[k]) > 0.0051: issues.append((k, cols[c], round(got[k], 4)))
    if issues: bad += 1; print(ds, B, issues)
print(f"rows checked {n}, rows with issues {bad}")
