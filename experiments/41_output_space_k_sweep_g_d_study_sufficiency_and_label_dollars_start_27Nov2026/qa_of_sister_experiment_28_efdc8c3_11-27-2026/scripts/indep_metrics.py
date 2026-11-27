"""Independent recomputation of RPS/CE/TVD/Brier from raw rating lists (each individual rating scored
against the forecast, then averaged within artifact, then equally across artifacts) for committed prediction
files: raw H4 lanes, Qwen logits, LoRA preds, and constants. Compares to committed final_results JSON.
Usage: python3 -I indep_metrics.py <E28 dir>"""
import csv, json, sys
from pathlib import Path
import numpy as np
E = Path(sys.argv[1])
K = 6
def load_rows():
    rows = []
    for st in ("old", "new"):
        for r in csv.DictReader((E / "data_11-26-2026" / f"human_ratings_{st}_11-26-2026.csv").open()):
            rows.append(dict(item=r["item"], task=r["task"], set=st, split=r["exp27_split"], fold=int(r["cv5_fold"]),
                             ratings=[int(x) for x in r["ratings_sorted_0to5_rater_identity_removed"].split()]))
    rows.sort(key=lambda r: r["item"]); return rows
def smooth(q, eps=1e-6):
    q = np.maximum(np.asarray(q, float), eps); return q / q.sum()
def rps_one(q, y):  # normalized RPS for a single rating y
    Q = np.cumsum(q)[:K-1]; O = (np.arange(K-1) >= y).astype(float); return np.sum((Q-O)**2)/(K-1)
def ce_one(q, y): return -np.log(smooth(q)[y])
def brier_one(q, y):
    e = np.zeros(K); e[y] = 1; return np.sum((q-e)**2)
def tvd(q, rats):
    p = np.bincount(rats, minlength=K)/len(rats); return 0.5*np.abs(q-p).sum()
def panel(qs, rr, weight="artifact"):
    out = {}
    for name, f in (("RPS", rps_one), ("CE", ce_one), ("Brier", brier_one)):
        if weight == "artifact":
            out[name] = float(np.mean([np.mean([f(q, y) for y in r]) for q, r in zip(qs, rr)]))
        else:
            out[name] = float(np.mean([f(q, y) for q, r in zip(qs, rr) for y in r]))
    out["TVD"] = float(np.mean([tvd(q, r) for q, r in zip(qs, rr)]))
    return out
rows = load_rows(); items = [r["item"] for r in rows]
print("n_items", len(rows), "n_ratings", sum(len(r["ratings"]) for r in rows),
      "old_items", sum(r["set"]=="old" for r in rows), "new_items", sum(r["set"]=="new" for r in rows))
print("ratings per item dist", {k: sum(len(r["ratings"])==k for r in rows) for k in (1,2,3,4,5)})
for st in ("old","new"):
    rr = [y for r in rows if r["set"]==st for y in r["ratings"]]; print(st, "n_ratings", len(rr), "share5", round(np.mean(np.array(rr)==5),4))
committed = json.loads((E/"results/final_results_11-26-2026.json").read_text())
# --- constants on exp27 test (train = non-test, counts pooled over ratings, smoothed) ---
def const(train_rows, test_rows, set_aware):
    def cnt(rs): return np.sum([np.bincount(r["ratings"], minlength=K) for r in rs], axis=0).astype(float)
    pg = smooth(cnt(train_rows)/cnt(train_rows).sum())
    if not set_aware: return [pg]*len(test_rows)
    out = []
    for r in test_rows:
        c = cnt([t for t in train_rows if t["set"]==r["set"]]); out.append(smooth(c/c.sum()))
    return out
tr = [r for r in rows if r["split"]!="test"]; te = [r for r in rows if r["split"]=="test"]
print("exp27 test items", len(te), "tasks", len({r['task'] for r in te}), "train+val tasks", len({r['task'] for r in tr}))
print("test/train task overlap:", len({r['task'] for r in te} & {r['task'] for r in tr}))
rr_te = [r["ratings"] for r in te]
cmp = committed["exp27_split_test"]
for nm, sa in (("constant_global", False), ("constant_set_aware", True)):
    mine = panel(const(tr, te, sa), rr_te); c = cmp[nm]["point"]
    print(f"exp27_test {nm}: mine", {k: round(v,5) for k,v in mine.items()}, "| committed", {k: round(c[k],5) for k in mine})
# cv5 constants
qs = [None]*len(rows)
for f in range(5):
    trf = [r for r in rows if r["fold"]!=f]; tef_idx = [i for i,r in enumerate(rows) if r["fold"]==f]
    for i, q in zip(tef_idx, const(trf, [rows[i] for i in tef_idx], True)): qs[i] = q
mine = panel(qs, [r["ratings"] for r in rows]); c = committed["cv5_nested_oof"]["constant_set_aware"]["point"]
print("cv5 constant_set_aware: mine", {k: round(v,5) for k,v in mine.items()}, "| committed", {k: round(c[k],5) for k in mine})
# fold integrity
for f in range(5):
    tk = {r["task"] for r in rows if r["fold"]==f}; ok = {r["task"] for r in rows if r["fold"]!=f}
    print(f"fold {f}: tasks {len(tk)} items {sum(r['fold']==f for r in rows)} old_tasks {len({r['task'] for r in rows if r['fold']==f and r['set']=='old'})} overlap_with_others {len(tk&ok)}")
# --- raw H4 lanes ---
def h4(lane):
    d = {}
    for l in (E/"results"/f"h4_{lane}_11-26-2026.jsonl").read_text().splitlines():
        o = json.loads(l)
        if o.get("status")=="ok": d.setdefault(o["id"].split("#r")[0], []).append(o["parsed"]["probs"])
    return [smooth(np.mean(d[it], axis=0)) for it in items]
rr_all = [r["ratings"] for r in rows]
for lane, key in (("codex_astra","h4_astra"),("agy_argon","h4_argon"),("agy_opus","h4_opus")):
    mine = panel(h4(lane), rr_all); c = committed["raw_dists_175"][key]["point"]
    print(f"raw {key}: mine", {k: round(v,5) for k,v in mine.items()}, "| committed", {k: round(c[k],5) for k in mine})
for key, fn in (("qwen3_4b","qwen3_4b_logit_6144.json"),("qwen3_8b","qwen3_8b_logit_6144.json")):
    d = {o["item"]: o for o in json.loads((E/"results"/fn).read_text())}
    mine = panel([smooth(d[it]["probs"]) for it in items], rr_all); c = committed["raw_dists_175"][key]["point"]
    print(f"raw {key}: mine", {k: round(v,5) for k,v in mine.items()}, "| committed", {k: round(c[k],5) for k in mine})
# --- LoRA ---
L = json.loads((E/"results/qwen3_4b_lora_ce_rps_pearson.json").read_text())
for split, key, rs in (("exp27_test","exp27_test_preds", te), ("cv5_oof","cv5_oof_preds", rows)):
    for obj in ("ce","rps","pearson"):
        mp = L[obj][key]; qs = [np.asarray(mp[r["item"]]["probs"], float) for r in rs]
        mine = panel(qs, [r["ratings"] for r in rs]); c = committed["qwen3_4b_lora"][split][obj]["point"]
        mineR = panel(qs, [r["ratings"] for r in rs], weight="rating")
        print(f"lora {split} {obj}: mine", {k: round(v,5) for k,v in mine.items()}, "| committed", {k: round(c[k],5) for k in mine}, "| rating-weighted RPS", round(mineR["RPS"],5))
    print(f"lora {split} keys n:", {o: len(L[o][key]) for o in L}, "| hp:", {o: L[o]["exp27_selected_hp"] for o in L} if split=="exp27_test" else "")
