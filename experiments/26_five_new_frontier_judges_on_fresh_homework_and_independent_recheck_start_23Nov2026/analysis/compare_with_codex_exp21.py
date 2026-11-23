import csv,sys
R='repo/experiments/21_fresh_task_families_matched_open_closed_judges_start_21Nov2026/short_answers/results/'
mine={ (r['judge'],int(r['ratings_per_answer']),int(r['training_ratings']),r['method'],r['metric']):r for r in csv.DictReader(open('work/recheck21/scores.csv'))}
theirs=list(csv.DictReader(open(R+'scores.csv')))
mx=0;n=0;bad=[]
for t in theirs:
    k=(t['judge'],int(t['ratings_per_answer']),int(t['training_ratings']),t['method'],t['metric'])
    if k not in mine: bad.append(('missing',k)); continue
    for f in ('mean','lower95','upper95'):
        d=abs(float(t[f])-float(mine[k][f])); mx=max(mx,d); n+=1
        if d>1e-9: bad.append((k,f,t[f],mine[k][f]))
print('scores compared',n,'max abs diff',mx,'bad',len(bad)); print(bad[:6])
mc={ (r['judge'],int(r['ratings_per_answer']),int(r['training_ratings']),r['comparator'],r['metric']):r for r in csv.DictReader(open('work/recheck21/contrasts.csv'))}
tc=list(csv.DictReader(open(R+'contrasts.csv')))
mx=0;n=0;bad=[]
for t in tc:
    k=(t['judge'],int(t['ratings_per_answer']),int(t['training_ratings']),t['comparator'],t['metric'])
    if k not in mc: bad.append(('missing',k)); continue
    for a,b in (('improvement','improvement'),('lower_one_sided_975','lower_975'),('upper_two_sided_95','upper_95')):
        d=abs(float(t[a])-float(mc[k][b])); mx=max(mx,d); n+=1
        if d>1e-9: bad.append((k,a,t[a],mc[k][b]))
    if int(t['folds_favor_ce'])!=int(mc[k]['folds_favor_ce']): bad.append((k,'folds'))
print('contrasts compared',n,'max abs diff',mx,'bad',len(bad)); print(bad[:6])
