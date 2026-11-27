import json, re, sys
S = json.load(open(sys.argv[1]))
leaves = []
def walk(d):
    if isinstance(d, dict):
        for v in d.values(): walk(v)
    elif isinstance(d, list):
        for v in d: walk(v)
    elif isinstance(d, (int, float)) and not isinstance(d, bool):
        leaves.append(float(d))
walk(S)
extra = [float(x) for x in sys.argv[3:]]
pool = leaves + extra
num = re.compile(r'(?<![\w.])[-−+]?\d+\.\d+(?:×10[⁻-]?[⁰¹²³⁴⁵⁶⁷⁸⁹\d]+)?')
for path in [sys.argv[2]]:
    for ln, line in enumerate(open(path, encoding="utf-8"), 1):
        for m in num.finditer(line):
            tok = m.group(0)
            if "×" in tok: continue
            s = tok.replace("−", "-").replace("+", "")
            val = float(s); d = len(s.split(".")[1])
            ok = any(round(abs(x), d) == round(abs(val), d) or abs(abs(x)-abs(val)) <= 0.5*10**-d + 1e-12 for x in pool)
            if not ok:
                print(f"{path.split('/')[-1]}:{ln}: UNMATCHED {tok}")
