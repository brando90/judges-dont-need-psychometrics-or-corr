import csv, sys, collections, numpy as np
from scipy import stats
art = list(csv.DictReader(open(sys.argv[1])))
h2 = collections.defaultdict(lambda: collections.defaultdict(list))
for r in csv.DictReader(open(sys.argv[2])): h2[r['item']][r['judge']].append(int(r['score_0to5']))
S = np.arange(6.)
out = []
for r in art:
    if r['set'] != 'new': continue
    mem = []
    for j in ('H2_astra', 'H2_opus'):
        c = np.bincount(h2[r['item']][j], minlength=6).astype(float); mem.append(c/c.sum())
    for pre in ('h3', 'h4_astra', 'h4_argon', 'h4_opus'):
        p = np.array([float(r[f'{pre}_p{k}']) for k in range(6)]); mem.append(p/p.sum())
    mu = np.array([m @ S for m in mem]); D = mu.std(ddof=1)
    q = np.mean(mem, 0); q = q/q.sum(); H = -(q[q>0]*np.log(q[q>0])).sum()/np.log(6)
    n = int(r['n_all']); rat = [int(x) for x in r['ratings_all'].split()]
    score = D + H*(1-q[5]) + 0.5/(1+n)
    out.append((-score, r['item'], r['task'], r['split'], n, sum(x==5 for x in rat), mu, len(h2[r['item']]['H2_astra']), len(h2[r['item']]['H2_opus'])))
out.sort(key=lambda t: (t[0], t[1]))
top, bot = out[:25], out[-25:]
f = lambda L: (sum(t[5] for t in L), sum(t[4] for t in L))
print('n new', len(out), 'top25 fives/ratings', f(top), 'bottom25', f(bot), 'all', f(out))
print('top item', out[0][1], out[0][2], 'score', round(-out[0][0],4), 'member means', np.round(out[0][6],3), 'h2 calls', out[0][7], out[0][8])
print('top5', [(t[1][-7:], round(-t[0],3)) for t in out[:5]])
print('top25 split counts', collections.Counter(t[3] for t in top))
fr = np.array([t[5]/t[4] for t in out]); sc = np.array([-t[0] for t in out])
print('spearman score vs five-rate', round(stats.spearmanr(sc, fr).statistic, 4))
# what the rank captures: ensemble ceiling mass vs 'judges think 5' -- check correlation of rank with ensemble expected score
print('min/max H2 calls per judge', min(t[7] for t in out), max(t[7] for t in out), min(t[8] for t in out), max(t[8] for t in out))
# Rank-score sensitivity: drop the 0.5/(1+n) term
out2 = sorted(out, key=lambda t: (t[0] + 0.5/(1+t[4]), t[1])); print('without count term: top25', f(out2[:25]), 'bottom25', f(out2[-25:]))
# Null comparison: rank purely by ensemble expected score (lowest first)
