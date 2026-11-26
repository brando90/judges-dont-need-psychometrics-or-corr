"""Cheap source/output verification for g5 and g6 MEDIUM/LOW, 11-26-2026.

Missing raw-label inputs are reported as limitations. Saved score agreement
does not establish a new raw-rating replication or validate stored intervals.
"""
import ast
import csv
import hashlib
import json
import runpy
import subprocess
from collections import Counter
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
E = {n: next((ROOT / "experiments").glob(f"{n}_*")) for n in range(24, 34)}
read = lambda p: list(csv.DictReader(p.open()))
print("g5/g6 MEDIUM and LOW verification (11-26-2026)")
tracked = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines()
for n in (24, 25, 29):
    manifest = json.loads((E[n] / "results/data_manifest.json").read_text())
    path = Path(manifest["prepared_bundle_path"])
    matching = [p for p in tracked if Path(p).name == path.name]
    print(f"E{n} raw-label bundle exists={path.exists()} tracked equivalents={matching}")
    for archive in sorted((E[n] / "results").glob("*predictions.npz")):
        z = np.load(archive, allow_pickle=False)
        label_keys = [k for k in z.files if any(x in k.lower() for x in ("vote", "label", "prompt", "split"))]
        print(archive.relative_to(ROOT), "keys", len(z.files), "label/prompt/split keys", label_keys)
for n in (24, 25):
    ss = read(E[n] / "results/scores.csv")
    for r in ss:
        if r["method"] in ("ce_calibrated", "pearson_calibrated") and r["depth"] == "1" and int(r["b_total"]) == (8192 if r.get("study") == "qwen3_4b" else 2048):
            print(f"E{n} saved headline", r.get("study", "qwen"), r["method"],
                  "RPS/CE/TVD", *(r[k] for k in ("rps", "ce_nats", "tvd")))
for r in read(E26 := E[26] / "results/new5/scores.csv"):
    if r["judge"] == "gpt-6-astra" and r["ratings_per_answer"] == "2" and r["training_ratings"] == "256" and r["metric"] == "rps" and r["method"] in ("ce_fit", "pearson_fit"):
        print("E26 saved RPS", r["method"], r["mean"])
print("E26 label-like tracked files", [p for p in tracked if "26_" in p and any(s in Path(p).name.lower() for s in ("label", "rating"))])
print("g5-M4 actual scorer probability at an originally zero category:")
for n, name in ((24, "calibrate_and_score.py"), (25, "calibrate_and_score_exp25.py")):
    tree = ast.parse((E[n] / "scripts" / name).read_text())
    fn = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == "ce_per_item")
    module = ast.Module(body=[fn], type_ignores=[])
    scope = {"np": np}
    exec(compile(module, name, "exec"), scope)
    for eps in (6e-6, 1e-2):
        loss = scope["ce_per_item"](np.array([[0, 1., 0, 0, 0]]), np.array([[1., 0, 0, 0, 0]]), eps)
        print(n, "uniform-mixture weight", eps, "category floor", float(np.exp(-loss[0])))
for n, name in ((24, "training_provenance.json"), (25, "qwen3_4b_provenance.json"), (25, "llama31_8b_provenance.json")):
    j = json.loads((E[n] / "results" / name).read_text())
    print("g5-L1 selected runs", name, len(j["runs"]))
    if name.startswith("llama"):
        print("g5-M5 protocol count 3428357; recorded count", j["n_trainable_params"], "difference", 3428357-j["n_trainable_params"])
pdf = E[27] / "PAPER_11-23-2026.pdf"
print("g5-M6 PDF SHA256", hashlib.sha256(pdf.read_bytes()).hexdigest())
print("g5-M6 deposited ZIP local candidates", [p for p in tracked if "correlation-identification-sources" in p])
for label, nums in (("g5-L2", (24,25,26,27)), ("g6-L5", (28,29,30,31,32,33))):
    print(label, "HOW_TO_READ_RESULTS present", [n for n in nums if (E[n]/"HOW_TO_READ_RESULTS.md").exists()], "of", len(nums))
ckpt = (E[25] / "CKPT_exp25_scale_and_llama.md").read_text()
print("g5-L3 completed header and stale next steps both present", "synced results to Google Drive Desktop" in ckpt, "Run `verify_exp25.py`, sync Drive Desktop" in ckpt)
print("g5-L4 exact toy results:")
toy = runpy.run_path(str(E[27] / "experiments/exact_counterexample/reproduce.py"), run_name="qa_toy")
for name, q in (("truth", toy["P"]), ("same-mean forecast", toy["Q"])):
    rps = toy["avg"]([toy["noise_rps"](p)+toy["excess_rps"](p,v) for p,v in zip(toy["P"],q)])
    ce = toy["avg"]([toy["ce"](p,v) for p,v in zip(toy["P"],q)])
    tvd = toy["avg"]([toy["tvd"](p,v) for p,v in zip(toy["P"],q)])
    print(name, "RPS", str(rps), "CE", ce, "TVD", str(tvd), "Pearson", toy["corr"]([toy["mean"](p) for p in toy["P"]], [toy["mean"](v) for v in q]))
br = read(E[28] / "results/breakeven_summary.csv")
print("g6-M1 saved joint break-even pairs", Counter(r["d"] for r in br if r["N_both"]))
winning = [r for r in read(E[28]/"results/comparator_wins.csv") if r["metric"] == "rps" and r["comparator"] == "pearson"]
print("g6-M1/M6 significant Pearson RPS wins", len(winning), "by (ratings, items)", dict(Counter((int(r["B"]), int(r["B"])//int(r["d"])) for r in winning)))
fits = read(E[31]/"results/fits_11-25-2026.csv")
rr = [r for r in fits if r["exp"] == "28" and r["level"] == "cell" and r["metric"] == "rps" and r["objective"] == "CE" and r["panel_id"].endswith("_d1")]
unidentified = [r for r in rr if r["floor_identified"] == "False"]
print("g6-M3 unidentified", len(unidentified), "of", len(rr), "zero lower", sum(float(r["floor_lo"]) == 0 for r in unidentified), "positive lower", sum(float(r["floor_lo"]) > 0 for r in unidentified))
print("g6-M3 MNLI Claude stored floor lower", next(r["floor_lo"] for r in rr if "mnli_claude" in r["panel_id"]))
s28 = read(E[28]/"results/scores_absolute.csv")
print("g6-M4 max budget by ratings/item", {d:max(int(r["B"]) for r in s28 if r["d"]==d) for d in ("1","5")})
j = json.loads((E[29]/"results/data_manifest.json").read_text())
print("g6-M5 saved human LOVO RPS", {k:(v["lovo_test"] if k=="hs2" else v["test_set"]["lovo"])["rps"] for k,v in j["suites"].items()})
tree = ast.parse((E[33]/"verify_moment_derivation.py").read_text())
fd = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name=="fd")
print("g6-M7 finite-difference default step", ast.literal_eval(fd.args.defaults[-1]))
cor = read(E[28]/"results/correlations.csv")
print("g6-L1 correlation-only records", len(cor), "columns", list(cor[0]))
report = (E[31]/"results_11-25-2026.md").read_text().split("## Table 5.")[1].split("## Table 6.")[0]
print("g6-L2 correlation-only table data rows", len([x for x in report.splitlines() if x.startswith("| ")])-1)
fig = list((E[31]/"figures").glob("*.png"))
print("g6-L4 figures total", len(fig), "E26 rpa1", sum("rpa1" in f.name for f in fig), "E26 rpa2", sum("rpa2" in f.name for f in fig))
print("Limit: saved summaries and absence checks do not regenerate scores, bootstrap bands, ratings or split identity.")
