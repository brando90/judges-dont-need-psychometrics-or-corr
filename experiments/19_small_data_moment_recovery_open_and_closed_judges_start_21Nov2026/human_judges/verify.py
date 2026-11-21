"""Replay the calibration grid, selections, archived scores and selected bootstrap draws."""
import csv
import itertools
from pathlib import Path

import numpy as np

import analyze as a


def near(actual, expected, atol=1e-11):
    np.testing.assert_allclose(actual,expected,rtol=0,atol=atol)


def reference_grid(base):
    u=(np.arange(6)-2.5)/2.5
    q=np.stack([base**t*np.exp(b*u+c*u*u) for t,b,c in a.COEFS],axis=2)
    return q/q.sum(axis=-1,keepdims=True)


def direct_losses(q, ys):
    """Integrate the individual-rating definitions, including each rating separately."""
    rows=[]
    f=q.cumsum(-1)[...,:5]
    m=q @ a.POWERS[:,:5]
    for i,y in enumerate(ys):
        z=y/5
        rps=np.mean((f[i][:,:,None,:]-(y[:,None]<=np.arange(5)))**2,axis=(-2,-1))
        ce=-np.mean(np.log(np.take(q[i],y,axis=-1)),axis=-1)
        mm=np.mean((m[i][:,:,None,:]-z[:,None]**np.arange(1,6))**2,axis=-2)
        v=(m[i,...,1]-m[i,...,0]**2-(np.var(z,ddof=1) if len(z)>1 else np.nan))**2
        rows.append(np.concatenate([rps[...,None],ce[...,None],mm,v[...,None]],axis=-1))
    return np.array(rows)


def first_tie(s):
    return np.argmax(s<=np.min(s,axis=-1,keepdims=True)+a.TOL,axis=-1)


def independent_choices(q, y, lt, weight):
    mu=q@a.R
    w=weight/weight.sum()
    u=mu-np.average(mu,axis=0,weights=w)
    t=y-np.average(y,weights=w)
    numerator=np.einsum("n,ngc,n->gc",w,u,t)
    denominator=np.sqrt(np.einsum("n,ngc,ngc->gc",w,u,u)*(w@(t*t)))
    ok=(np.ptp(mu,axis=0)>a.TOL)&(np.ptp(y)>a.TOL)
    corr=np.divide(numerator,denominator,out=np.full_like(numerator,-np.inf),where=ok)
    tie=corr>=corr.max(-1,keepdims=True)-a.TOL
    tie[~ok.any(-1)]=False
    tie[~ok.any(-1),0]=True
    ce,rps=np.average(lt[...,1],axis=0,weights=w),np.average(lt[...,0],axis=0,weights=w)
    return np.stack([first_tie(ce),tie.argmax(-1),first_tie(np.where(tie,rps,np.inf))],axis=-1)


def main():
    keys,base,items,labels,_=a.load()
    # At the identity head, perturbing either shape coefficient must give its
    # own moment residual as the derivative of one observed rating's log loss.
    u = (np.arange(6) - 2.5) / 2.5
    step = 1e-5
    for feature in (u, u**2):
        losses = []
        for delta in (-step, step):
            shifted = base * np.exp(delta * feature)
            losses.append(-np.log(shifted / shifted.sum(axis=-1, keepdims=True)))
        derivative = (losses[1] - losses[0]) / (2 * step)
        near(derivative, (base @ feature)[..., None] - feature, atol=5e-9)
    print("Verified both per-rating CE moment gradients for every archived input and rating category.")
    family=np.array([item["family"] for item in items]); _,fid=np.unique(family,return_inverse=True)
    q=reference_grid(base)
    alt_grid=a.grid(base)
    near(q,alt_grid);near(q[:,:,0],base)
    full, m, v=a.target(labels)
    test_loss=direct_losses(q,labels)
    near(a.loss_panel(q,full,m,v),test_loss)
    weights=np.random.default_rng(20261123).exponential(size=(3,19))
    archived=a.history
    original,=a.EXPERIMENTS.glob("10*/scripts")
    import sys
    sys.path.insert(0,str(original))
    from analyze_all import build_250_grouped_splits
    old_splits=build_250_grouped_splits(items)
    scores={b:[] for b in a.BUDGETS}
    g=np.arange(len(keys))[:,None]
    weighted_checks=0
    for idx, (train,heldout) in enumerate(archived.splits(items)):
        assert set(old_splits[idx]["test_families"])==heldout
        assert np.array_equal(old_splits[idx]["train_families_ordered"],train)
        te=np.flatnonzero(np.isin(family,list(heldout)))
        rng=np.random.default_rng(np.random.SeedSequence([20261121,190,idx]))
        selected_ratings=[rng.permutation(y) for y in labels]
        for k in (1,2):
            short=[y[:k] for y in selected_ratings]
            # Direct individual-rating CE and RPS by lookup; score only observed classes.
            tr_ce=np.stack([-np.take(np.log(q[i]),t,axis=-1).mean(-1) for i,t in enumerate(short)])
            c=q.cumsum(-1)[...,:5]
            tr_rps=np.stack([np.mean((c[i][:,:,None,:]-(t[:,None]<=np.arange(5)))**2,axis=(-2,-1))
                             for i,t in enumerate(short)])
            panel=np.stack([tr_rps,tr_ce],axis=-1)
            for n in (3,7,15):
                tr=np.flatnonzero(np.isin(family,train[:n])); y=np.array([short[i].mean() for i in tr])
                uniform=np.ones(len(tr))
                choice=independent_choices(q[tr],y,panel[tr],uniform)
                keep=np.concatenate([choice,np.zeros_like(choice[:,:1])],axis=1)
                test=test_loss[te].mean(0)[g,keep]
                const_counts=np.bincount(np.concatenate([short[i] for i in tr]),minlength=6)+.5
                const=const_counts/const_counts.sum()
                q_const=np.broadcast_to(const,(len(te),1,1,6))
                const_metric=direct_losses(q_const,[labels[i] for i in te]).mean(0)[0,0]
                scores[n,k].append(np.concatenate([test,np.broadcast_to(const_metric,(len(keys),1,8))],axis=1))
                if idx not in (0,99,249):continue
                wt=weights[:,fid[te]]; wt/=wt.sum(1,keepdims=True)
                w=weights[:,fid[tr]]; w/=w.sum(1,keepdims=True)
                expected=np.stack([independent_choices(q[tr],y,panel[tr],wi) for wi in w])
                actual,_,_=a.select(w,alt_grid[tr]@a.R,y,tr_ce[tr],tr_rps[tr])
                np.testing.assert_array_equal(actual,expected)
                p=(w@a.history.histograms([short[i] for i in tr]))*(k*len(tr))+.5
                p/=p.sum(1,keepdims=True)
                fast=a.constant_losses(p,wt,full[te],m[te],v[te])
                for j,r in enumerate(p):
                    literal=direct_losses(np.broadcast_to(r,(len(te),1,1,6)),[labels[i] for i in te])
                    near(np.average(literal,weights=wt[j],axis=0)[0,0],fast[j])
                weighted_checks+=len(w)
    by_budget={b:np.mean(vals,axis=0) for b,vals in scores.items()}
    with (a.ROOT/'scores.csv').open() as stream:
        for row in csv.DictReader(stream):
            if row['metric'] not in a.LOSS:continue
            b=int(row['n']),int(row['k'])
            near(float(row['mean']),by_budget[b][keys.index(row['input']),a.METHODS.index(row['method']),a.LOSS.index(row['metric'])])
    contrasts=0
    with (a.ROOT/'contrasts.csv').open() as stream:
        for row in csv.DictReader(stream):
            b=int(row['n']),int(row['k']); j=keys.index(row['input']); m=a.LOSS.index(row['metric'])
            hat=by_budget[b][j,a.METHODS.index(row['method']),m]-by_budget[b][j,a.METHODS.index(row['subtract']),m]
            near(float(row['difference']),hat)
            assert float(row['lower_95'])<=float(row['upper_95'])
            contrasts+=1
    print(f"Original 250 splits matched. Reproduced all 1,440 absolute held-out loss estimates and {contrasts} contrast means.")
    print(f"{weighted_checks} weighted fit settings × 6 inputs × 3 selectors matched independent centered correlations and losses; weighted constants reproduced.")


if __name__=='__main__':main()
