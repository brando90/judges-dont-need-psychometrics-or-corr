import os, sys, json
os.environ["SANMI36_DATA"] = "/tmp/qa55/data"
sys.path.insert(0, "/tmp/qa55/code/exp36")
import numpy as np
from data_io import load_chaos_3way, load_summeval, split_by_group
d = load_chaos_3way()
tr, va, te = split_by_group(d.groups, seed=0)
prem = lambda i: d.texts[i].split(' hypothesis: ')[0]
trp = {prem(i) for i in tr}
n_shared = sum(1 for i in te if prem(i) in trp)
print(f"ChaosNLI SNLI+MNLI: {n_shared} of {len(te)} test items share their premise text with a train item")
s = load_summeval('expert')
print('SummEval expert items', s.n_items, 'distinct groups (source articles)', len(set(s.groups)))
tr, va, te = split_by_group(s.groups, seed=0)
print('SummEval split articles', len({s.groups[i] for i in tr}), len({s.groups[i] for i in va}), len({s.groups[i] for i in te}))
