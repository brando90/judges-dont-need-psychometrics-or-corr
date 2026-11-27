import json, sys, collections
import numpy as np
from scipy import stats
R = json.load(open(sys.argv[1]))
def tci(x):
    x = np.asarray(x, float); m = x.mean(); se = x.std(ddof=1) / np.sqrt(len(x)); t = stats.t.ppf(0.975, len(x) - 1)
    return m, m - t * se, m + t * se
tally = collections.Counter(); tally2 = collections.Counter(); tally_full = collections.Counter(); agree = []
print("dataset B | PS-pilot-only counts | U1-PSpilot RPS [CI] | U2-PSpilot [CI] | PSfull-PSpilot [CI] | same choice as full PS? n/a")
for ds, rec in R.items():
    for B, r in rec.items():
        if int(B) in (32, 1024): continue
        c = collections.Counter(r["ps_pilot"])
        u1 = np.array(r["u1_rps"]); u2 = np.array(r["u2_rps"]); pf = np.array(r["ps_full_rps"]); pp = np.array(r["ps_pilot_rps"])
        a = tci(u1 - pp); b = tci(u2 - pp); f = tci(pf - pp)
        def verdict(m, lo, hi, pos, neg):
            if lo > 0: return pos
            if hi < 0: return neg
            return "tie"
        if np.allclose(u1 - pp, 0): v1 = "tie"
        else: v1 = verdict(*a, "PSpilot beats U1", "U1 beats PSpilot")
        v2 = "tie" if np.allclose(u2 - pp, 0) else verdict(*b, "PSpilot beats U2", "U2 beats PSpilot")
        vf = "tie" if np.allclose(pf - pp, 0) else verdict(*f, "PSpilot beats PSfull", "PSfull beats PSpilot")
        tally[v1] += 1; tally2[v2] += 1; tally_full[vf] += 1
        print(f"{ds} {B} | {dict(c)} | {a[0]:+.5f} [{a[1]:+.5f},{a[2]:+.5f}] {v1} | {b[0]:+.5f} [{b[1]:+.5f},{b[2]:+.5f}] {v2} | {f[0]:+.5f} [{f[1]:+.5f},{f[2]:+.5f}] {vf}")
print("U1 vs PS-pilot-only:", dict(tally)); print("U2 vs PS-pilot-only:", dict(tally2)); print("PS-full vs PS-pilot-only:", dict(tally_full))
