import os, sys, collections
EXP = sys.argv[1]
sys.path.insert(0, EXP)
import data_io
from data_io import load_chaos_3way, split_by_group, assert_no_prompt_leakage
d = load_chaos_3way()
tr, va, te = split_by_group(d.groups, seed=0)
assert_no_prompt_leakage(d.groups, d.rater_ids, tr, va, te)
def premise(t):
    return t.split(' hypothesis: ')[0]
prem_trval = set(premise(d.texts[i]) for i in list(tr) + list(va))
leak = [i for i in te if premise(d.texts[i]) in prem_trval]
print('items', d.n_items, 'test items', len(te), 'test items sharing a premise with train/val:', len(leak))
# also snli image-id grouping as in Exp 29
snli_leak = 0
g_trval = set()
for i in list(tr) + list(va):
    g = d.groups[i]
    if g.startswith('chaosNLI_snli::'):
        g_trval.add(g.split('::', 1)[1].split('#')[0])
for i in te:
    g = d.groups[i]
    if g.startswith('chaosNLI_snli::') and g.split('::', 1)[1].split('#')[0] in g_trval:
        snli_leak += 1
print('SNLI test items whose image/caption id (uid before #) is also in train/val:', snli_leak)
