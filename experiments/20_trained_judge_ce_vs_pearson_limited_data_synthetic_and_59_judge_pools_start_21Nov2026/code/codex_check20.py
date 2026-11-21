"""Experiment 20, cross-check (11-21-2026): independent re-implementation of the experiment-19 (Codex) calibration-grid comparison, to see whether its RPS numbers reproduce.
TLDR: inputs = four judges' P0 six-category probabilities (Qwen3-4B, Qwen2.5-Coder-14B open; gpt-5.4-mini, gpt-6.1-sol closed) and the two pair mixtures; 27-point calibration grid q ∝ exp(t log p + b u + c u^2);
fit by training CE or by training Pearson of candidate means; same 250 family splits, same rating subsample seeds. Prints held-out RPS for CE-fit / Pearson-fit / identity / constant for the six (families, ratings) budgets
and compares with the numbers in experiment 19's scores.csv. Written from the experiment-19 PROTOCOL text and the splits of experiment 18, not copied from its analyze.py. CPU only.
Usage: python3 codex_check20.py --open_json open_weight_forecasts.json --scores_csv scores_exp19.csv --out check.json   (run from the folder holding real20.py, dataset_pool.json, judge_forecasts.json)"""
import sys, json, csv, itertools, argparse
ap = argparse.ArgumentParser(); ap.add_argument('--open_json', required=True); ap.add_argument('--scores_csv', required=True); ap.add_argument('--out', default='check.json'); b = ap.parse_args()
sys.argv = ['real20.py']
import numpy as np
import real20 as r20
from lib20 import R, U
EPS = r20.EPS; ratings = r20.ratings; full_hist = r20.full_hist; fam_col = r20.fam_col
od = json.load(open(b.open_json)); order = od['artifact_order']; keys = r20.keys; posn = [order.index(k) for k in keys]
cl = {s['judge_id']: s for s in r20.pj['arm_a_specs']}
def native(jid): d = {r['item']: r for r in r20.pj['native_prob_forecasts'][jid]}; return np.array([d[k]['probabilities'] for k in keys])
raw = {'qwen3_4b': np.asarray(od['verbal_forecasts']['q3_4b_2507__P0_human_mirror'])[posn], 'qwen25_coder_14b': np.asarray(od['verbal_forecasts']['q25c_14b__P0_human_mirror'])[posn],
       'gpt54_mini': native('gpt54_mini__P0'), 'gpt61_sol': native('gpt61_sol__P0')}
raw = {k: v / v.sum(1, keepdims=True) for k, v in raw.items()}
raw['open_pair'] = (raw['qwen3_4b'] + raw['qwen25_coder_14b']) / 2; raw['closed_pair'] = (raw['gpt54_mini'] + raw['gpt61_sol']) / 2
inputs = {k: (1 - 6 * EPS) * v + EPS for k, v in raw.items()}
u = U; COEFS = sorted(itertools.product((0.5, 1, 2), (-1, 0, 1), (-1, 0, 1)), key=lambda p: (np.log2(p[0]) ** 2 + p[1] ** 2 + p[2] ** 2, p))
def grid(p):  # n x 6 -> n x 27 x 6
    L = np.stack([t * np.log(p) + b_ * u + c * u ** 2 for t, b_, c in COEFS], 1); L -= L.max(-1, keepdims=True); q = np.exp(L); return q / q.sum(-1, keepdims=True)
def rps_art(Q, P):  # Q: n x ... x 6 ; P: n x 6
    Pe = P.reshape(P.shape[0], *([1] * (Q.ndim - 2)), 6); cq, cp = Q.cumsum(-1)[..., :-1], Pe.cumsum(-1)[..., :-1]; return (((cq - cp) ** 2 + cp * (1 - cp)).mean(-1))
def ce_art(Q, P): return -(np.log(Q) * P.reshape(P.shape[0], *([1] * (Q.ndim - 2)), 6)).sum(-1)
res = {}
for name, p in inputs.items():
    G = grid(p); mu = G @ R  # n x 27
    for n in (3, 7, 15):
        for k in (1, 2):
            tot = {m: [] for m in ('ce', 'pearson', 'identity', 'constant')}
            for s, train, te, test in r20.SPL:
                rng = np.random.default_rng(np.random.SeedSequence([20261121, 190, s])); perms = [rng.permutation(r) for r in ratings]
                tr = np.flatnonzero(np.isin(fam_col, train[:n])); Ctr = np.array([np.bincount(perms[i][:k], minlength=6) for i in tr], float); Htr = Ctr / Ctr.sum(1, keepdims=True)
                ce_tr = ce_art(G[tr], Htr).mean(0); jc = int(np.flatnonzero(ce_tr <= ce_tr.min() + 1e-12)[0])
                y = Htr @ R; m_ = mu[tr]; ok = (np.ptp(m_, 0) > 1e-12) & (np.ptp(y) > 1e-12)
                if ok.any():
                    mc = m_ - m_.mean(0); yc = y - y.mean(); c = np.full(27, -np.inf); c[ok] = ((mc * yc[:, None]).sum(0) / np.sqrt((mc ** 2).sum(0) * (yc ** 2).sum()))[ok]; jp = int(np.flatnonzero(c >= c.max() - 1e-12)[0])
                else: jp = 0
                const = (Ctr.sum(0) + 0.5) / (Ctr.sum() + 3.0); Pte = full_hist[te]
                tot['ce'].append(rps_art(G[te, jc], Pte).mean()); tot['pearson'].append(rps_art(G[te, jp], Pte).mean()); tot['identity'].append(rps_art(G[te, 0], Pte).mean()); tot['constant'].append(rps_art(np.tile(const, (len(te), 1)), Pte).mean())
            for m, v in tot.items(): res[(name, n, k, m)] = float(np.mean(v))
ref = {}
for r in csv.DictReader(open(b.scores_csv)):
    if r['metric'] == 'rps': ref[(r['input'], int(r['n']), int(r['k']), r['method'])] = float(r['mean'])
rows = []; mx = 0
for key, v in res.items():
    if key in ref: rows.append({'input': key[0], 'families': key[1], 'ratings': key[2], 'method': key[3], 'mine': v, 'exp19': ref[key], 'abs_diff': abs(v - ref[key])}); mx = max(mx, abs(v - ref[key]))
json.dump({'rows': rows, 'max_abs_diff': mx, 'n_compared': len(rows)}, open(b.out, 'w'), indent=1)
for r in rows:
    if r['families'] == 3 and r['ratings'] == 1 and r['input'] in ('closed_pair', 'open_pair'): print(r)
print('compared', len(rows), 'max abs diff', mx)
