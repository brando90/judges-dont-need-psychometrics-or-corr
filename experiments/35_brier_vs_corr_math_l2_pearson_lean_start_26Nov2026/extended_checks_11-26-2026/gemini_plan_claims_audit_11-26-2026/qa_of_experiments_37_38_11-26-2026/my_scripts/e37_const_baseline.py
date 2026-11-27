import sys, os
D="/tmp/exp37_audit/experiments/37_corr_first_moment_linear_cca_start_26Nov2026"
sys.path.insert(0, D); os.chdir(D)
os.environ["VERIBENCH_FSC_DIR"]=os.environ.get("AUDIT_RATINGS_DIR","/tmp/ttj_f0d0/experiments/09_prepare_packets_and_collect_human_ratings_start_28Sep2026/ratings")
import numpy as np
import run_real as rr
data,_=rr.load_veribench()
train,val,test=rr.data_io.split_by_group(data.groups, seed=rr.SEED)
K=6
# training marginal over individual ratings (what a constant forecast would use), add-one smoothed
cnt=np.ones(K)
for i in train:
    for y in data.ratings[i]: cnt[int(y)]+=1
q=cnt/cnt.sum()
print("train-marginal (add-one) forecast:",np.round(q,3).tolist())
def per_item_ce(q,idx):
    out=[]
    for i in idx:
        ys=np.array([int(y) for y in data.ratings[i]])
        out.append(float(np.mean(-np.log(q[ys]))))
    return np.array(out)
ce=per_item_ce(q,test)
print("constant-forecast test CE (item-averaged, nats):",round(float(ce.mean()),4),"| uniform log6:",round(float(np.log(6)),4))
# bootstrap by item
rng=np.random.default_rng(37); B=1000
bs=np.array([ce[rng.integers(0,len(ce),len(ce))].mean() for _ in range(B)])
print("item-bootstrap 95% CI:",np.round(np.percentile(bs,[2.5,97.5]),4).tolist())
# share of 5s among train ratings and test ratings
tr=[int(y) for i in train for y in data.ratings[i]]; te=[int(y) for i in test for y in data.ratings[i]]
print("share of grade 5: train",round(np.mean(np.array(tr)==5),3),"test",round(np.mean(np.array(te)==5),3),"| n train ratings",len(tr),"n test ratings",len(te))
