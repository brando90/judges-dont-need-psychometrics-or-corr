import csv, sys, collections, hashlib
for path, col in ((sys.argv[1], 'offensiveness'), (sys.argv[2], 'politeness')):
    rows = list(csv.DictReader(open(path, newline='', encoding='utf-8')))
    print('==', path.split('/')[-1], 'rows', len(rows), 'columns', list(rows[0].keys())[:12], '...', len(rows[0]))
    vals = collections.Counter(r[col] for r in rows)
    print('  value counter:', dict(sorted(vals.items())[:12]))
    pairs = collections.Counter((r['user_id'], r['instance_id']) for r in rows)
    dup = {k: v for k, v in pairs.items() if v > 1}
    print('  (user,instance) duplicate pairs:', len(dup), 'extra rows:', sum(v - 1 for v in dup.values()))
    inst = collections.Counter(r['instance_id'] for r in rows)
    print('  instances:', len(inst), 'k min/max', min(inst.values()), max(inst.values()))
    # text duplicates across instance ids
    text_to_inst = collections.defaultdict(set)
    for r in rows:
        text_to_inst[r['text'].strip()].add(r['instance_id'])
    multi = {t: s for t, s in text_to_inst.items() if len(s) > 1}
    print('  texts shared by >1 instance_id:', len(multi), 'instances involved:', sum(len(s) for s in multi.values()))
    inst_texts = collections.defaultdict(set)
    for r in rows:
        inst_texts[r['instance_id']].add(r['text'].strip())
    print('  instance_ids with >1 distinct text:', sum(1 for s in inst_texts.values() if len(s) > 1))
    # Is there a 'user_id' field and any demographic columns? (do not print values)
    print('  users:', len(set(r['user_id'] for r in rows)))
    # invalid values
    bad = 0
    for r in rows:
        try:
            v = float(r[col]); 
            if v != int(v) or not 1 <= v <= 5: bad += 1
        except Exception:
            bad += 1
    print('  invalid/non-integer rows:', bad)
