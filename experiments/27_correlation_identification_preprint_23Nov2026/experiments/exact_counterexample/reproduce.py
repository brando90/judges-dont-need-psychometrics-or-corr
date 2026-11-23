#!/usr/bin/env python3
"""Reproduce the paper's exact three-item distribution counterexample.

Uses only Python's standard library; run from any working directory:
python3 experiments/exact_counterexample/reproduce.py
All probability, CDF, TVD, and RPS calculations use exact rational arithmetic.
Logarithmic quantities are printed as double-precision evaluations of exact inputs.
No fitting, sampling, or model calls occur.
"""
from fractions import Fraction as F
from math import log, sqrt

P = [(F(3,5), F(3,10), F(1,10)),
     (F(1,5), F(3,5), F(1,5)),
     (F(1,10), F(3,10), F(3,5))]
Q = [(F(13,20), F(1,5), F(3,20)),
     (F(2,5), F(1,5), F(2,5)),
     (F(3,20), F(1,5), F(13,20))]

def mean(p):
    return sum(F(r)*v for r,v in enumerate(p))

def cdf(p):
    return [sum(p[:k+1]) for k in range(len(p)-1)]

def tvd(p,q):
    return sum(abs(a-b) for a,b in zip(p,q))/2

def excess_rps(p,q):
    return sum((a-b)**2 for a,b in zip(cdf(p),cdf(q)))/(len(p)-1)

def noise_rps(p):
    return sum(a*(1-a) for a in cdf(p))/(len(p)-1)

def ce(p,q):
    return -sum(float(a)*log(float(b)) for a,b in zip(p,q))

def avg(values):
    return sum(values)/len(values)

def corr(x,y):
    xc=[a-avg(x) for a in x]
    yc=[a-avg(y) for a in y]
    return float(sum(a*b for a,b in zip(xc,yc)))/sqrt(float(sum(a*a for a in xc)*sum(b*b for b in yc)))

for p,q in zip(P,Q):
    assert sum(p)==sum(q)==1
    assert min(p)>0 and min(q)>0
    assert mean(p)==mean(q)
    assert sum(a-b for a,b in zip(cdf(p),cdf(q)))==mean(q)-mean(p)
t=[mean(p) for p in P]
m=[mean(q) for q in Q]
assert t==m==[F(1,2),F(1),F(3,2)]
assert avg([tvd(p,q) for p,q in zip(P,Q)])==F(1,5)
assert avg([excess_rps(p,q) for p,q in zip(P,Q)])==F(3,200)
assert avg([noise_rps(p) for p in P])==F(49,300)
assert corr(m,t)==1.0
print('K = 3; n = 3; no fitted model; exact finite illustration')
print('target means:', ', '.join(str(v) for v in t))
print('forecast means:', ', '.join(str(v) for v in m))
print('correlation:',corr(m,t))
for name, forecasts in [('True distribution',P),('Changed shape, same mean',Q)]:
    tv=avg([tvd(p,q) for p,q in zip(P,forecasts)])
    ex=avg([excess_rps(p,q) for p,q in zip(P,forecasts)])
    risk=avg([noise_rps(p)+excess_rps(p,q) for p,q in zip(P,forecasts)])
    cross=avg([ce(p,q) for p,q in zip(P,forecasts)])
    kl=avg([ce(p,q)-ce(p,p) for p,q in zip(P,forecasts)])
    print(f'{name}: TVD={tv}; excess RPS={ex}; expected RPS={risk}; CE={cross:.9f} nats; excess CE={kl:.9f} nats')
print('All exact rational assertions passed.')
