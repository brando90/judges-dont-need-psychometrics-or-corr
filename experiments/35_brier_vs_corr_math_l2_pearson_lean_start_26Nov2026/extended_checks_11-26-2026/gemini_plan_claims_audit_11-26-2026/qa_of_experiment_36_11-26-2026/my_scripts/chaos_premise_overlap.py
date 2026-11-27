import os, sys
exp36_dir, data_dir = sys.argv[1], sys.argv[2]
os.environ["SANMI36_DATA"] = data_dir
sys.path.insert(0, exp36_dir)
import re
from data_io import load_chaos_3way
from run_all import prepare
d = load_chaos_3way()
print("example text:", repr(d.texts[0])[:300])
prep = prepare(d)
# the text packs premise and hypothesis; split on the committed delimiter if present
def premise(t):
    m = re.split(r"\s*hypothesis\s*:\s*", t, maxsplit=1, flags=re.I)
    return re.sub(r"^\s*premise\s*:\s*", "", m[0], flags=re.I).strip().lower()
tr = {premise(d.texts[i]) for i in prep["train"]}
te = [premise(d.texts[i]) for i in prep["test"]]
shared = sum(1 for p in te if p in tr)
print("n_train", len(prep["train"]), "n_val", len(prep["val"]), "n_test", len(prep["test"]))
print("test items whose premise also occurs in a train item:", shared, "of", len(te))
