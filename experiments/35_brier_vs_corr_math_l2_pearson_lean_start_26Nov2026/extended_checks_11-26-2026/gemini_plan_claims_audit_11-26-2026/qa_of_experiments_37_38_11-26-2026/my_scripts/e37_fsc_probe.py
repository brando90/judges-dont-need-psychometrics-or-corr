import sys, os, csv, collections
D="/tmp/exp37_audit/experiments/37_corr_first_moment_linear_cca_start_26Nov2026"
sys.path.insert(0, D)
os.chdir(D)
os.environ["VERIBENCH_FSC_DIR"]=os.environ.get("AUDIT_RATINGS_DIR","/tmp/ttj_f0d0/experiments/09_prepare_packets_and_collect_human_ratings_start_28Sep2026/ratings")
import numpy as np
import run_real as rr
data, status = rr.load_veribench()
print("loaded:", status["loaded"], "| items:", status["n_items"], "| ratings:", status["n_ratings"], "| raters:", status["n_raters"])
train, val, test = rr.data_io.split_by_group(data.groups, seed=rr.SEED)
print("split sizes (items): train", len(train), "val", len(val), "test", len(test))
groups=np.array(data.groups)
print("distinct task groups:", len(set(data.groups)), "| test task groups:", len(set(groups[test])), "| train:", len(set(groups[train])), "| val:", len(set(groups[val])))
ks=np.array([len(r) for r in data.ratings])
print("test ratings per item: min/median/max:", ks[test].min(), np.median(ks[test]), ks[test].max(), "| total test ratings:", int(ks[test].sum()))
# sealed Exp27 splits
base="/tmp/ttj_clone/repo/experiments/27_build_a_judge_that_beats_the_constant_forecast_on_held_out_tasks_old_new_ratings_start_24Nov2026/data_11-24-2026/"
sealed={r["task"]:(r["set"],r["split"]) for r in csv.DictReader(open(base+"task_splits_11-24-2026.csv",newline="",encoding="utf-8-sig"))}
print("sealed tasks:", len(sealed), "| example key format (length only):", [len(k) for k in list(sealed)[:3]])
tf=set(data.groups)
print("Exp37 task_folder values also present as sealed 'task' keys:", len(tf & set(sealed)), "of", len(tf))
# match on a normalized key if raw keys differ
import re
def norm(s): return re.sub(r'[^a-z0-9]','',s.lower())
sealed_n={norm(k):v for k,v in sealed.items()}
tf_n={norm(g):g for g in tf}
inter=set(sealed_n)&set(tf_n)
print("normalized overlap:", len(inter))
cross=collections.Counter()
for idx in test:
    g=data.groups[idx]
    cross[sealed.get(g, sealed_n.get(norm(g), ("unmatched","unmatched")))]+=1
print("Exp37 TEST items by sealed (set,split):", dict(cross))
cross2=collections.Counter()
for idx in train:
    g=data.groups[idx]
    cross2[sealed.get(g, sealed_n.get(norm(g), ("unmatched","unmatched")))]+=1
print("Exp37 TRAIN items by sealed (set,split):", dict(cross2))
cross3=collections.Counter()
for idx in val:
    g=data.groups[idx]
    cross3[sealed.get(g, sealed_n.get(norm(g), ("unmatched","unmatched")))]+=1
print("Exp37 VAL items by sealed (set,split):", dict(cross3))
