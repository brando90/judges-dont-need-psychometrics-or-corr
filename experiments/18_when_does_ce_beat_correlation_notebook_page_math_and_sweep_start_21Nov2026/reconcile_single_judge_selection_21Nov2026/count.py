import json
r=json.load(open('reconcile.json'))['cells']
for pn in ['N15','N15s3','N15+H44','H44','H5']:
    for pr in 'CM':
        cs=[c for c in r if c['pool']==pn and c['proto']==pr]
        pos=sum(c['pearson_minus_ce']>0 for c in cs); neg=sum(c['pearson_minus_ce']<0 for c in cs)
        sp=sum(c['ci_lo']>0 for c in cs); sn=sum(c['ci_hi']<0 for c in cs)
        print(pn,pr,len(cs),'CE better pts',pos,'Pearson better pts',neg,'CE sig',sp,'Pearson sig',sn)
import statistics
# protocol effect: N15 C vs M and k effect
c=lambda p,pr,n,k:[x for x in r if x['pool']==p and x['proto']==pr and x['n_train_units']==n and x['k']==k][0]['pearson_minus_ce']
print('k spread N15 C', max(abs(c('N15','C',n,1)-c('N15','C',n,2)) for n in (3,7,15)), max(abs(c('N15','C',n,1)-c('N15','C',n,'all')) for n in (3,7,15)))
print('k spread H44 M', max(abs(c('H44','M',n,1)-c('H44','M',n,'all')) for n in (4,8,12,16)))
