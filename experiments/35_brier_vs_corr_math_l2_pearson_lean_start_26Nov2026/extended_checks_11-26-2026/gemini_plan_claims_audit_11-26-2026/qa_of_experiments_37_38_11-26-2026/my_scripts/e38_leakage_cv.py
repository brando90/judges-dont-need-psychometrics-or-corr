"""Audit script (Claude, 11-26-2026): does an item-level split of VeriBench FSC inflate held-out CE of Exp 38's head?
Uses Exp 38's own functions and hyperparameters. Only the 15 sealed TRAIN+VAL tasks of the new set are used (60 items);
no sealed-test task is touched. Two schemes, same head, same folds count: item-level 5-fold vs task-level 5-fold, 30 repeats."""
import sys, os, csv, glob, json
D="/tmp/exp38_audit/experiments/38_cka_opd_invariance_ladder_start_26Nov2026"
sys.path.insert(0, D); os.chdir(D)
os.environ["SANMI36_DATA"]="/tmp/exp37_audit/experiments/37_corr_first_moment_linear_cca_start_26Nov2026/data_cache"
import numpy as np
import run_real as rr
data,_=rr.load_veribench()
root="/tmp/ttj/experiments/09_prepare_packets_and_collect_human_ratings_start_28Sep2026/ratings"
item2task={}
for f in glob.glob(root+"/*native_main*.csv"):
    for r in csv.DictReader(open(f,newline="",encoding="utf-8-sig")):
        if (r.get("final_score_0_5") or "").strip()!="": item2task[r["item_id"]]=r["task_folder"].strip()
sealed={r["task"]:r["exp27_split"] for r in csv.DictReader(open("/tmp/ttj_clone/repo/experiments/28_multi_family_distributional_judge_and_open_weight_lora_on_new_ratings_start_26Nov2026/data_11-26-2026/task_splits_11-26-2026.csv",newline="",encoding="utf-8-sig"))}
tasks=np.array([item2task[g] for g in data.groups])
keep=np.array([i for i in range(data.n_items) if sealed[tasks[i]] in ("train","val")])
print("items kept (sealed train+val tasks only):",len(keep),"| tasks:",len(set(tasks[keep])),"| sealed-test items touched: 0")
feats=rr.hash_features([data.texts[i] for i in keep])
K=6
ratings=[data.ratings[i] for i in keep]; tk=tasks[keep]
def hist(i):
    c=np.bincount(np.asarray(ratings[i],dtype=int),minlength=K).astype(float); return c/c.sum()
def run_fold(tr,te):
    xtr,(xte,),_=rr.svd_project(feats[tr],[feats[te]])
    W,_=rr.fit_ce(xtr,np.stack([hist(i) for i in tr]))
    prob=rr.softmax(xte@W)
    ce_head=np.array([float(np.mean(-np.log(np.clip(prob[r][np.asarray(ratings[i],dtype=int)],1e-12,1)))) for r,i in enumerate(te)])
    cnt=np.ones(K)
    for i in tr:
        for y in ratings[i]: cnt[int(y)]+=1
    q=cnt/cnt.sum()
    ce_const=np.array([float(np.mean(-np.log(q[np.asarray(ratings[i],dtype=int)]))) for i in te])
    return ce_head,ce_const
def cv(scheme,seed,nf=5):
    rng=np.random.default_rng(seed); n=len(keep)
    head=np.zeros(n); const=np.zeros(n)
    if scheme=="item":
        perm=rng.permutation(n); folds=np.array_split(perm,nf)
    else:
        ut=np.array(sorted(set(tk))); ut=ut[rng.permutation(len(ut))]; groups=np.array_split(ut,nf)
        folds=[np.array([i for i in range(n) if tk[i] in set(g)]) for g in groups]
    for te in folds:
        tr=np.setdiff1d(np.arange(n),te)
        h,c=run_fold(tr,te); head[te]=h; const[te]=c
    return head,const
out={}
for scheme in ("item","task"):
    H=[];C=[]
    for s in range(30):
        h,c=cv(scheme,s); H.append(h); C.append(c)
    H=np.array(H); C=np.array(C)
    diff_per_repeat=(H-C).mean(axis=1)           # head minus constant, mean over items, per repeat
    item_diff=(H-C).mean(axis=0)                 # per item averaged over repeats
    rng=np.random.default_rng(20261126); bs=np.array([item_diff[rng.integers(0,len(item_diff),len(item_diff))].mean() for _ in range(1000)])
    out[scheme]={"head_CE_mean":float(H.mean()),"constant_CE_mean":float(C.mean()),"head_minus_constant_mean":float(diff_per_repeat.mean()),
                 "sd_over_repeats":float(diff_per_repeat.std(ddof=1)),"item_bootstrap_95":[float(x) for x in np.percentile(bs,[2.5,97.5])]}
    print(f"{scheme:4s}-level 5-fold x30: head CE {H.mean():.4f} | constant CE {C.mean():.4f} | head minus constant {diff_per_repeat.mean():+.4f} (SD over repeats {diff_per_repeat.std(ddof=1):.4f}; item-bootstrap 95% [{np.percentile(bs,2.5):+.4f}, {np.percentile(bs,97.5):+.4f}])")
json.dump(out,open("/tmp/ds_run/e38_leakage_cv.json","w"),indent=1)
