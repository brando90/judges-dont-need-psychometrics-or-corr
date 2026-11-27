import json, sys
r = json.load(open(sys.argv[1]))
M = r['matrices']
for name, m in M.items():
    print(f"== {name}  n_train={m['n_train']} n_val={m['n_val']} n_test={m['n_test']} K={m['summary']['n_classes']}")
    print("   keys:", list(m.keys()))
    for obj, row in m['rows'].items():
        p = row['point']
        ci = row.get('ci', {})
        def f(k):
            v = p.get(k)
            c = ci.get(k, {})
            return f"{v:.4f}[{c.get('lo', float('nan')):.4f},{c.get('hi', float('nan')):.4f}]" if isinstance(v, (int, float)) else str(v)
        print(f"   {obj:8s} ce={f('ce')} rpsn={f('rps_norm')} brier={f('brier')} r={f('pearson_r')} opt={ {k: row['optimizer'].get(k) for k in ('success','nit','objective')} }")
    for extra in ('marginal', 'uniform', 'shuffle_ce'):
        v = m.get(extra)
        if isinstance(v, dict):
            pt = v.get('point', v)
            print(f"   {extra:8s}", {k: (round(x, 4) if isinstance(x, float) else x) for k, x in pt.items() if not isinstance(x, dict)})
        else:
            print(f"   {extra:8s}", v)
    tr = m.get('trace')
    if tr is not None:
        print("   trace:", str(tr)[:300])
