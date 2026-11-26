"""Run existing CPU checks without overwriting scored outputs, 11-26-2026."""
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
E = {n: next((ROOT / "experiments").glob(f"{n}_*")) for n in range(24, 34)}
tests = [("exp24_verifier", ["python3", str(E[24]/"scripts/verify.py")]),
         ("exp25_verifier", ["python3", str(E[25]/"scripts/verify_exp25.py")]),
         ("exp29_verifier", ["python3", str(E[29]/"scripts/verify_exp29.py")]),
         ("exp27_exact", ["python3", str(E[27]/"experiments/exact_counterexample/reproduce.py")]),
         ("exp27_identities", ["python3", str(E[27]/"experiments/verify_identities.py")])]
code33 = "import runpy; n=runpy.run_path(" + repr(str(E[33]/"verify_moment_derivation.py")) + ",run_name='qa_checks'); print('finite difference step',n['fd'].__defaults__); "
code33 += "; ".join(f"n[{name!r}]()" for name in ("check_tower", "check_bounds", "check_vandermonde", "check_polynomial_family", "check_gradients", "check_pearson_squared", "check_threshold"))
tests.append(("exp33_seven_checks_without_writes", ["python3", "-c", code33]))
for n in (30, 31):
    code = "import ast,subprocess,pathlib; "
    code += f"p=pathlib.Path({str(E[n])!r}); "
    if n == 30:
        code += "src=p/'build_results_md.py'; tree=ast.parse(src.read_text()); "
        code += "tree.body=[x for x in tree.body if not (isinstance(x,ast.Expr) and isinstance(x.value,ast.Call) and isinstance(x.value.func,ast.Attribute) and x.value.func.attr=='write_text')]; "
        code += "scope={'__file__':str(src),'__name__':'qa_builder'}; exec(compile(tree,str(src),'exec'),scope); generated=scope['md']; "
    else:
        code += "r=subprocess.run(['python3',str(p/'build_results_md.py')],capture_output=True,text=True,check=True); generated=r.stdout; "
    code += "tables=lambda s:[l for l in s.splitlines() if l.startswith('|')]; "
    code += "assert tables(generated)==tables((p/'results_11-25-2026.md').read_text()); print('Generated scored table rows match preserved report:',len(tables(generated)))"
    tests.append((f"exp{n}_generated_table_parity", ["python3", "-c", code]))

def run(test):
    name, command = test
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                            env={**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"})
    log = OUT/f"test_{name}_11-26-2026.txt"
    log.write_text(f"{name} (11-26-2026)\n" + result.stdout + result.stderr + f"\nExit code: {result.returncode}\n")
    return dict(name=name, date="11-26-2026", command=command, exit_code=result.returncode,
                log=str(log.relative_to(ROOT)), note="Existing CPU check; no raw outputs overwritten.")

with ThreadPoolExecutor(max_workers=3) as pool:
    results = list(pool.map(run, tests))
(ROOT.parent/"g5_g6_tests_11-26-2026.json").write_text(json.dumps(results, indent=2)+"\n")
for r in results:
    print(r["name"], "exit", r["exit_code"], r["log"])
print("No pytest test files exist in experiments 24–33; available CPU verifiers used. Exp 28 label-dependent verifier not run because required raw labels are absent.")
