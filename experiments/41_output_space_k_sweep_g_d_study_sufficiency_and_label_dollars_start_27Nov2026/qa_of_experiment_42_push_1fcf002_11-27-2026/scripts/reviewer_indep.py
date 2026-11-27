# Independent recomputation for Experiment 42 review. Does not import or copy the experiment code.
import json, math, sys
from fractions import Fraction as Fr
import numpy as np
D = json.load(open(sys.argv[1]))
vb = np.array(D["veribench_native_main"]["marginal_counts"], float); vb /= vb.sum()
ch3 = np.array(D["chaosnli_snli_mnli"]["marginal_counts"], float)
di3 = np.array(D["dices990_q_overall"]["marginal_counts"], float); di3 /= di3.sum()
ys = np.arange(6.0)
def tilt(T, target):
    k = np.arange(len(T), dtype=float)
    def m(b):
        w = T*np.exp(b*(k - k.mean())); w = w/w.sum(); return w @ k, w
    lo, hi = -60.0, 60.0
    for _ in range(300):
        mid = 0.5*(lo+hi); mm, _ = m(mid)
        if mm < target: lo = mid
        else: hi = mid
    mm, w = m(0.5*(lo+hi)); assert abs(mm-target) < 1e-9, (mm, target); return w
B = np.array([1.6,2.0,2.4,2.8,3.2,3.6])
refl = vb[::-1].copy()
strict = np.array([tilt(refl, m-0.8) for m in B])
lenient = np.array([tilt(vb, m+0.8) for m in B])
low = tilt(refl, 0.5); high = tilt(vb, 4.5)
w = (B-0.5)/4.0
pol = (1-w)[:,None]*low + w[:,None]*high
chT = np.zeros(6); chT[[0,2,5]] = ch3; chT /= chT.sum()
chaos = np.array([tilt(chT, m) for m in B])
mean = lambda P: P @ ys
var = lambda P: P @ ys**2 - (P @ ys)**2
print("means strict", mean(strict).round(9)); print("means lenient", mean(lenient).round(9))
print("means pol", mean(pol).round(9), "chaos", mean(chaos).round(9))
print("pole means", low@ys, high@ys, "w", w)
def cdf(P): return np.cumsum(P, axis=-1)[..., :-1]
def rps(Q, P):  # expected normalized RPS, mean over rows
    F, G = cdf(P), cdf(Q); return float((((G-F)**2).sum(-1) + (F*(1-F)).sum(-1)).mean()/5)
def tvd(Q,P): return float((0.5*np.abs(Q-P).sum(-1)).mean())
def ce(Q,P,eps=None):
    if eps: Q = (1-6*eps)*Q + eps
    with np.errstate(divide="ignore"):
        t = np.where(P>0, -P*np.log(np.where(Q>0,Q,1e-300)), 0.0)
    bad = ((P>0)&(Q<=0)).any()
    return float("inf") if bad else float(t.sum(-1).mean())
def erho(P):
    vu = np.mean((mean(P)-mean(P).mean())**2); ve = np.mean(var(P)); return vu/(vu+ve)
def binom(ms):
    from math import comb
    return np.array([[comb(5,k)*(m/5)**k*(1-m/5)**(5-k) for k in range(6)] for m in ms])
def point(locs):
    Q = np.zeros((len(locs),6)); Q[np.arange(len(locs)), locs] = 1; return Q
train = (strict+lenient+pol+chaos)/4
bn = binom(B)
sp = point(np.array([0,1,2,3,4,5])); lp = point(np.clip(np.floor(B+0.5).astype(int)+1,0,5))
print("lenient locs", np.clip(np.floor(B+0.5).astype(int)+1,0,5))
for name,Q in [("oracle",pol),("trainmix",train),("binom",bn),("lenient_law",lenient),("spearman_pt",sp),("lenient_pt",lp)]:
    mq = mean(Q)
    print(f"{name:12s} RPS {rps(Q,pol):.6f} TVD {tvd(Q,pol):.6f} CE {ce(Q,pol):.6f} CEs {ce(Q,pol,1e-6):.6f} bias {np.mean(mq-B):+.6f} pearson {np.corrcoef(mq,B)[0,1]:.6f} erho {erho(Q):.6f}")
from scipy.stats import spearmanr
print("lenient pt spearman", spearmanr(mean(lp),B).correlation)
# Exact expected RPS of add-one with N ratings per bin from law P, scored on target T
def addone_exact(N, P_src_list, T):
    # P_src_list: list of laws, each contributing N ratings per bin
    tot = len(P_src_list)*N; F = cdf(T); acc = 0.0
    EG = (sum(N*cdf(P) for P in P_src_list) + (np.arange(5)+1))/(tot+6)
    VG = sum(N*cdf(P)*(1-cdf(P)) for P in P_src_list)/(tot+6)**2
    return float((((EG-F)**2 + VG + F*(1-F)).sum(-1)).mean()/5)
for n in (4,16,64,256):
    pe = addone_exact(n, [pol], pol); po = addone_exact(n, [strict,lenient,pol,chaos], pol)
    print(f"n={n} personaCE {pe:.6f} pooledCE {po:.6f} pooled-persona {po-pe:+.6f} persona-oracle {pe-rps(pol,pol):+.6f}")
# Binomial at pooled sample mean: own MC, seed 7, 200000 reps of the per-bin pooled mean
rng = np.random.default_rng(7)
def binom_mc(n, R=200000, target=pol):
    out = np.zeros(R)
    tot = 0
    ms = np.zeros((R,6))
    for b in range(6):
        s = np.zeros(R)
        for P in (strict,lenient,pol,chaos):
            c = rng.multinomial(n, P[b], size=R); s += c @ ys
        ms[:,b] = s/(4*n)
    th = np.clip(ms/5, 0.02, 0.98)
    from math import comb
    Q = np.stack([comb(5,k)*th**k*(1-th)**(5-k) for k in range(6)], -1)
    F = cdf(target)[None]; G = np.cumsum(Q,-1)[..., :-1]
    r = ((((G-F)**2).sum(-1) + (F*(1-F)).sum(-1)).mean(-1))/5
    return r.mean(), r.std()/math.sqrt(R)
for n in (4,256):
    mb, se = binom_mc(n)
    pe = addone_exact(n,[pol],pol)
    print(f"n={n} binomial(pooled mean) E[RPS] {mb:.6f} +- {1.96*se:.6f}; gap binom-personaCE {mb-pe:.6f}")
# strict target
print("strict: oracle RPS", round(rps(strict,strict),6), "binom RPS", round(rps(bn,strict),6), "TVD", round(tvd(bn,strict),6), "bias", round(np.mean(mean(bn)-mean(strict)),6), "CEs", round(ce(bn,strict,1e-6),6), "oracle CEs", round(ce(strict,strict,1e-6),6))
# shift mixture
sh = 0.05*strict+0.05*lenient+0.8*pol+0.1*chaos
print("shift: oracle", round(rps(sh,sh),6), "trainmix", round(rps(train,sh),6), round(tvd(train,sh),6), "binom", round(rps(bn,sh),6), round(tvd(bn,sh),6), "mean shift", mean(sh).round(9))
print("Phi own (=erho) by persona: strict %.6f lenient %.6f pol %.6f chaos %.6f" % (erho(strict), erho(lenient), erho(pol), erho(chaos)))
# Equal-moment panel, exact
def F_(xs): return [Fr(x) for x in xs]
M1 = [Fr(x,16) for x in (5,0,3,3,0,5)]; M2 = [Fr(x,32) for x in (7,9,0,0,9,7)]
A = [Fr(x,24) for x in (11,0,3,10,0,0)]; Bl = [Fr(x,24) for x in (10,3,0,11,0,0)]
U = [A, M1, A[::-1]]; V = [Bl, M2, Bl[::-1]]
def mu(p): return sum(k*p[k] for k in range(6))
def vr(p): return sum(k*k*p[k] for k in range(6)) - mu(p)**2
def cd(p): 
    s=Fr(0); out=[]
    for k in range(5): s+=p[k]; out.append(s)
    return out
def rpsF(q,p): 
    G,F = cd(q),cd(p); return sum((g-f)**2 + f*(1-f) for g,f in zip(G,F))/5
def tvdF(q,p): return sum(abs(a-b) for a,b in zip(q,p))/2
print("means U", [mu(p) for p in U], "vars U", [vr(p) for p in U]); print("means V", [mu(p) for p in V], "vars V", [vr(p) for p in V])
def erhoF(P, div=None):
    ms=[mu(p) for p in P]; mbar=sum(ms)/len(ms); d = div or len(ms)
    vu=sum((m-mbar)**2 for m in ms)/d; ve=sum(vr(p) for p in P)/len(P); return vu/(vu+ve)
print("Erho2 U,V (divisor N)", erhoF(U), erhoF(V), " (divisor N-1)", erhoF(U, 2), erhoF(V, 2))
print("center RPS gap", rpsF(M2,M1)-rpsF(M1,M1), "center TVD", tvdF(M2,M1))
gap = sum(rpsF(v,u)-rpsF(u,u) for u,v in zip(U,V))/3
print("panel avg RPS gap", gap, float(gap), "panel TVD", sum(tvdF(v,u) for u,v in zip(U,V))/3, float(sum(tvdF(v,u) for u,v in zip(U,V))/3))
print("rps U", float(sum(rpsF(u,u) for u in U)/3), "rps V on U", float(sum(rpsF(v,u) for u,v in zip(U,V))/3))
# native side table
d3 = np.array([tilt(di3, m) for m in (0.7,1.0,1.3)]); c3 = np.array([tilt(ch3/ch3.sum(), m) for m in (0.7,1.0,1.3)])
k3=np.arange(3.0)
def rps3(Q,P):
    F=np.cumsum(P,-1)[:,:-1]; G=np.cumsum(Q,-1)[:,:-1]; return float((((G-F)**2+F*(1-F)).sum(-1)).mean()/2)
def erho3(P):
    m=P@k3; v=P@k3**2-m**2; vu=np.mean((m-m.mean())**2); return vu/(vu+v.mean())
print("native: TVD", round(float((0.5*np.abs(c3-d3).sum(-1)).mean()),6), "RPS d/d", round(rps3(d3,d3),6), "c/d", round(rps3(c3,d3),6), "erho d", round(erho3(d3),6), "c", round(erho3(c3),6))
