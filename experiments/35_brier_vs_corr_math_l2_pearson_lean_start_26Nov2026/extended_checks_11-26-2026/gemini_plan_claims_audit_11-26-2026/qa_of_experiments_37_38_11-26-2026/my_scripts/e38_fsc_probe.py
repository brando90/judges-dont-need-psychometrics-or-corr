import sys, os, csv, glob, collections
D="/tmp/exp38_audit/experiments/38_cka_opd_invariance_ladder_start_26Nov2026"
sys.path.insert(0, D); os.chdir(D)
os.environ["SANMI36_DATA"]="/tmp/exp37_audit/experiments/37_corr_first_moment_linear_cca_start_26Nov2026/data_cache"
import numpy as np
import run_real as rr
data, notes = rr.load_veribench()
print("items:",data.n_items,"| ratings:",int(sum(len(r) for r in data.ratings)))
train,val,test=rr.split_by_group(data.groups, seed=rr.SPLIT_SEED)
print("split sizes (items): train",len(train),"val",len(val),"test",len(test))
# item -> task from the Exp 09 CSVs (main files only)
root="/tmp/ttj/experiments/09_prepare_packets_and_collect_human_ratings_start_28Sep2026/ratings"
item2task={}
for f in glob.glob(root+"/*native_main*.csv"):
    for r in csv.DictReader(open(f,newline="",encoding="utf-8-sig")):
        if (r.get("final_score_0_5") or "").strip()!="":
            item2task[r["item_id"]]=r["task_folder"].strip()
tasks=np.array([item2task[g] for g in data.groups])
tt=set(tasks[test]); tr=set(tasks[train]); va=set(tasks[val])
print("distinct tasks: total",len(set(tasks)),"| test",len(tt),"| train",len(tr),"| val",len(va))
print("test tasks that ALSO appear in train:",len(tt&tr),"of",len(tt))
print("test ITEMS whose task also appears in train or val:",int(sum(1 for i in test if tasks[i] in (tr|va))),"of",len(test))
sealed={r["task"]:r["exp27_split"] for r in csv.DictReader(open("/tmp/ttj_clone/repo/experiments/28_multi_family_distributional_judge_and_open_weight_lora_on_new_ratings_start_26Nov2026/data_11-26-2026/task_splits_11-26-2026.csv",newline="",encoding="utf-8-sig"))}
cs=collections.Counter(sealed.get(tasks[i],"unmatched") for i in test)
print("Exp38 TEST items by sealed exp27_split of their task:",dict(cs))
cs=collections.Counter(sealed.get(tasks[i],"unmatched") for i in train)
print("Exp38 TRAIN items by sealed exp27_split of their task:",dict(cs))
# features: does the loader read the Lean candidate text?
lens=[len(t) for t in data.texts]
print("feature text length min/median/max:",min(lens),int(np.median(lens)),max(lens))
# constant-forecast reference on Exp 38's own split (train marginal, add-one)
K=6; cnt=np.ones(K)
for i in train:
    for y in data.ratings[i]: cnt[int(y)]+=1
q=cnt/cnt.sum()
ce=np.array([float(np.mean(-np.log(q[np.array(data.ratings[i],dtype=int)]))) for i in test])
rng=np.random.default_rng(20261126); bs=np.array([ce[rng.integers(0,len(ce),len(ce))].mean() for _ in range(1000)])
print("constant (train-marginal, add-one) test CE:",round(float(ce.mean()),4),"CI",np.round(np.percentile(bs,[2.5,97.5]),4).tolist(),"| uniform log6:",round(float(np.log(6)),4))
print("share of 5s: train",round(float(np.mean([y==5 for i in train for y in data.ratings[i]])),3),"test",round(float(np.mean([y==5 for i in test for y in data.ratings[i]])),3))
