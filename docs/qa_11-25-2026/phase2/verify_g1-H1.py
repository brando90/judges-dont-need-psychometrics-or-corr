"""QA g1-H1: forecast construction verification (11-26-2026). CPU only."""
import json
from fractions import Fraction
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
ROOT = Path(__file__).resolve().parents[3]
def exp(n): return next((ROOT/'experiments').glob(f'{n:02}_*'))
a = json.loads((exp(4)/'results/phase_a.json').read_text())['A1']['per_artifact']
p = np.array([[float(Fraction(v)) for v in row['p']] for row in a]); y = np.arange(6)
m = p @ y
q = np.full_like(p, .001); q[:,0] += (5-m)/5; q[:,5] += m/5; q /= q.sum(1)[:,None]
error = abs(q@y-m); ce = float(-(p*np.log(q)).sum(1).mean())
kl = float((p*(np.log(p)-np.log(q))).sum(1).mean())
rps = lambda f: float(((np.cumsum(f,axis=1)[:,:-1]-np.cumsum(p,axis=1)[:,:-1])**2).mean())
tvd = lambda f: float(.5*np.abs(f-p).sum(1).mean())
saved = json.loads((exp(5)/'results/gap_check.json').read_text())['A1_two_point_judge']
assert abs(saved['mean_abs_mean_error']-error.mean()) < 1e-12
assert abs(saved['CE_nats_mean']-ce) < 1e-12 and abs(saved['KL_nats_mean']-kl) < 1e-12
assert np.all((q>0).sum(1)==6)
print('QA g1-H1 (11-26-2026): CONFIRMED')
print(f'11 artifacts; positive categories each = {(q>0).sum(1).tolist()}')
print(f'Mean absolute mean error = {error.mean():.12f}; maximum = {error.max():.12f} rating points')
print(f'RPS excess = {rps(q):.12f}; CE = {ce:.12f}; TVD = {tvd(q):.12f}; KL = {kl:.12f}; Pearson = {np.corrcoef(m,q@y)[0,1]:.12f}')
print('The 0.001 category masses were normalized; no recentering operation exists.')
