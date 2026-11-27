"""Recompute Study A headline counts from a study_a_summary.json (path in argv[1])."""
import json, sys
import numpy as np
S = json.load(open(sys.argv[1]))
DS = ["chaosnli_snli", "chaosnli_mnli", "chaosnli_alphanli", "dices350", "dices990"]
def cells():
    for ds in DS:
        for B, cell in S["datasets"][ds]["budgets"].items():
            if len(cell["candidates"]) > 1:
                yield ds, int(B), cell
def tally(pair, metric):
    w = l = t = 0; rows = []
    for ds, B, cell in cells():
        r = cell["paired"][f"{pair}_{metric}"]
        # pair "a_minus_b": first_beats_second => a better
        if r["first_beats_second"]: l += 1   # a beats b
        elif r["second_beats_first"]: w += 1 # b beats a
        else: t += 1
        rows.append((ds, B, r["mean"], r["ci95"], r.get("paired_t_pvalue")))
    return w, l, t, rows
print("n comparable cells:", sum(1 for _ in cells()))
for pair, metric, label in [("u1_minus_ps","rps","PS beats U1 / U1 beats PS / tie (RPS)"),
                            ("u2_minus_ps","rps","PS beats U2 / U2 beats PS / tie (RPS)"),
                            ("u1_minus_ps_mix","rps_mix","ps_mix beats U1 / U1 beats ps_mix / tie (rps_mix)"),
                            ("u1_minus_u2","item_mse","U2 beats U1 / U1 beats U2 / tie (item MSE)"),
                            ("u1_minus_u2","grand_mse","U2 beats U1 / U1 beats U2 / tie (grand MSE)"),
                            ("u1_minus_ps","ce","PS beats U1 / U1 beats PS / tie (CE)"),
                            ("u1_minus_ps","tvd","PS beats U1 / U1 beats PS / tie (TVD)"),
                            ("u1_minus_u2","rps","U2 beats U1 / U1 beats U2 / tie (RPS)")]:
    w, l, t, rows = tally(pair, metric)
    print(f"{label}: {w} / {l} / {t}")
# multiplicity on u1 - ps RPS
_, _, _, rows = tally("u1_minus_ps", "rps")
ps = [(r[4], r[0], r[1], r[2]) for r in rows]
valid = [p for p in ps if p[0] is not None and np.isfinite(p[0])]
print("cells with defined p:", len(valid), " undefined:", [ (p[1],p[2]) for p in ps if p[0] is None or not np.isfinite(p[0])])
def holm(pvals, alpha=0.05):
    m = len(pvals); order = np.argsort(pvals); rej = np.zeros(m, bool)
    for i, j in enumerate(order):
        if pvals[j] <= alpha / (m - i): rej[j] = True
        else: break
    return rej
def bh(pvals, q=0.05):
    m = len(pvals); order = np.argsort(pvals); rej = np.zeros(m, bool); kmax = -1
    for i, j in enumerate(order):
        if pvals[j] <= (i + 1) / m * q: kmax = i
    for i in range(kmax + 1): rej[order[i]] = True
    return rej
for m_total in (len(valid), 20):
    pv = np.array([p[0] for p in valid] + [1.0] * (m_total - len(valid)))
    sig_raw = (pv < 0.05) & np.array([p[3] > 0 for p in valid] + [False] * (m_total - len(valid)))
    h = holm(pv); b = bh(pv)
    pos = np.array([p[3] > 0 for p in valid] + [False] * (m_total - len(valid)))
    print(f"m={m_total}: raw p<0.05 & PS better: {int(sig_raw.sum())}; Holm survivors (PS better): {int((h & pos).sum())}; BH q=0.05 survivors (PS better): {int((b & pos).sum())}")
    print("  Holm survivors:", [(valid[i][1], valid[i][2]) for i in range(len(valid)) if h[i]])
    print("  BH survivors:", [(valid[i][1], valid[i][2]) for i in range(len(valid)) if b[i]])
# U2 vs PS
_, _, _, rows2 = tally("u2_minus_ps", "rps")
for r in rows2:
    if r[3][0] is not None and not (r[3][0] <= 0 <= r[3][1]): print("U2-PS significant:", r)
# selection / agreement ranges
agree_item = []; agree_rps = []; u2_rps = []
for ds, B, cell in cells():
    s = cell["selection"]
    agree_item.append(s["u1"]["agree_item_oracle"]["proportion"])
    agree_rps.append(s["u1"]["agree_rps_oracle"]["proportion"])
    u2_rps.append((ds, B, s["u2"]["agree_rps_oracle"]["k"], s["u2"]["agree_rps_oracle"]["proportion"]))
    modes = {k: s[k]["mode"] for k in ("u1","u2","ps","ps_mix","rps_oracle","item_oracle","grand_oracle")}
    nmax_items = max(cell["candidates"], key=lambda c: int(c.split("x")[0]))
    nmax_raters = max(cell["candidates"], key=lambda c: int(c.split("x")[1]))
    flags = []
    if modes["item_oracle"] != modes["u1"]: flags.append("item_oracle!=u1")
    if modes["rps_oracle"] != nmax_items: flags.append("rps_oracle!=max_items")
    if modes["ps"] != nmax_items: flags.append("ps!=max_items")
    if modes["u2"] != nmax_items: flags.append("u2!=max_items")
    if modes["ps_mix"] != nmax_items: flags.append("ps_mix!=max_items")
    if s["u2"]["counts"].get(nmax_items, 0) != 150: flags.append(f"u2 not always max items {s['u2']['counts']}")
    if s["u1"]["counts"].get(nmax_raters, 0) != 150: flags.append("u1 not always max raters")
    if flags: print("FLAG", ds, B, flags, modes)
print("U1 agree item-oracle range: %.3f-%.3f" % (min(agree_item), max(agree_item)))
print("U1 agree RPS-oracle range: %.3f-%.3f" % (min(agree_rps), max(agree_rps)))
print("U2 agree RPS-oracle (SNLI 64):", [u for u in u2_rps if u[0]=="chaosnli_snli" and u[1]==64])
print("U2 agree RPS-oracle range: %.3f-%.3f" % (min(u[3] for u in u2_rps), max(u[3] for u in u2_rps)))
