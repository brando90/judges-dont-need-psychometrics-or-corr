"""Diagnose exact-string replay failure without altering scored files (11-26-2026)."""
from pathlib import Path
import csv,gzip,sys
import numpy as np
root=Path(__file__).resolve().parents[3]
e18=next((root/'experiments').glob('18_*'))/'conditional_shape_and_rating_budget'
sys.path.insert(0,str(e18/'scripts'))
import run
rows=list(csv.DictReader(gzip.open(run.OUTPUT/'raw.csv.gz','rt')))
maximum=0.;count=0
for index in (0,run.REPLICATIONS-1):
    saved=[r for r in rows if int(r['rep'])==index]
    for old,new in zip(saved,run.simulation(index),strict=True):
        assert set(old)==set(new)
        for key,value in new.items():
            if old[key]==str(value):continue
            count+=1
            a,b=float(old[key]),float(value)
            if np.isnan(a) and np.isnan(b):continue
            assert np.isfinite(a) and np.isfinite(b)
            maximum=max(maximum,abs(a-b))
print('E18 exact-string replay diagnosis (11-26-2026)')
print('First and last replication serialized numeric differences:',count)
print('Maximum absolute difference:',maximum)
assert maximum<1e-12
print('PASS numerical tolerance 1e-12. The original verifier still exits 1 on exact-string equality; this diagnosis does not relabel it as passing.')
